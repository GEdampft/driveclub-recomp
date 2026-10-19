// SPDX-License-Identifier: GPL-2.0-or-later
// VR (dcport-vr): see vr_multiview.h.

#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <array>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "video_core/buffer_cache/buffer.h"
#include "video_core/renderer_vulkan/vk_runtime.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/renderer_vulkan/vr/vr_fixes.h"
#include "video_core/renderer_vulkan/vr/vr_multiview.h"
#include "video_core/texture_cache/image.h"
#include "video_core/texture_cache/image_view.h"

namespace Vulkan::Vr::Multiview {
namespace {

bool Requested() {
    static const bool on = [] {
        const char* v = std::getenv("BB_VR_STEREO");
        return v && std::strcmp(v, "mv") == 0;
    }();
    return on;
}

bool g_enabled = false;
VkDevice g_device{};

std::mutex g_pipelines_mutex;
std::unordered_map<VkPipeline, VkPipeline> g_twins; // pipeline -> its multiview twin

float g_shift = 0.f;     // the eyes' clip shift (left +, right -)
bool g_writes_stereo = false; // the bindings so far write a two-layer image
// Development audit (BB_VR_AUDIT=1): which passes make one picture for both eyes out of per-eye pictures. Every
// draw and dispatch's inputs and outputs (images by guest range, storage buffers by range) are noted; a pass that runs
// for the left eye only (one-layer targets, a compute dispatched once) while it reads a two-layer image or something
// such a pass wrote makes its outputs "left eye only", and a later pass for both eyes that reads one of them shows the
// left eye's data in the right eye. Each finding is printed once per shader.
struct AuditRes {
    u64 address;
    u64 size;
    std::string name;
};
bool g_audit = false;
std::vector<AuditRes> g_audit_reads, g_audit_writes, g_audit_targets;
bool g_audit_reads_stereo = false;
std::string g_audit_left_reads; // left-eye-only resources the bindings read
bool g_audit_pass_stereo = false;
std::vector<AuditRes> g_left_only; // resources written for the left eye only from per-eye inputs
std::mutex g_audit_mutex;

std::string AuditName(const VideoCore::Image& image) {
    char text[128];
    std::snprintf(text, sizeof(text), "%ux%u fmt %d %s at %#llx", image.info.size.width, image.info.size.height,
                  int(image.info.pixel_format), Stereo(image) ? "2 layers" : "1 layer",
                  (unsigned long long)image.info.guest_address);
    return text;
}

void AuditRead(u64 address, u64 size, const std::string& name) {
    for (const auto& res : g_left_only) {
        if (address < res.address + res.size && res.address < address + size) {
            g_audit_left_reads += " [" + res.name + "]";
            return;
        }
    }
    (void)name;
}

std::string AuditList(const std::vector<AuditRes>& list) {
    std::string text;
    for (const auto& res : list) {
        text += " [" + res.name + "]";
    }
    return text;
}

// One draw or dispatch done: `per_eye` when it produced both eyes (a multiview pass, a dispatch run per eye).
void AuditFinish(bool compute, u64 hash, bool per_eye) {
    std::vector<AuditRes> outputs = g_audit_writes;
    if (!compute) {
        outputs.insert(outputs.end(), g_audit_targets.begin(), g_audit_targets.end());
    }
    const bool from_eyes = g_audit_reads_stereo || !g_audit_left_reads.empty();
    static std::unordered_map<u64, bool> seen;
    if (!per_eye && from_eyes && !outputs.empty()) {
        for (const auto& out : outputs) {
            std::erase_if(g_left_only, [&](const AuditRes& r) { return r.address == out.address; });
            AuditRes marked = out;
            char tag[32];
            std::snprintf(tag, sizeof(tag), " (by %#llx)", (unsigned long long)hash);
            marked.name += tag;
            g_left_only.push_back(marked);
        }
        if (seen.emplace(hash ^ 1, true).second) {
            std::printf("VR audit: ONE EYE %s %#llx writes%s from%s%s\n", compute ? "compute" : "draw vs",
                        (unsigned long long)hash, AuditList(outputs).c_str(), AuditList(g_audit_reads).c_str(),
                        g_audit_left_reads.empty() ? "" : (" | left-eye-only:" + g_audit_left_reads).c_str());
            std::fflush(stdout);
        }
    } else {
        for (const auto& out : outputs) {
            std::erase_if(g_left_only, [&](const AuditRes& r) { return r.address == out.address; });
        }
    }
    if (per_eye && !g_audit_left_reads.empty() && seen.emplace(hash ^ 2, true).second) {
        std::printf("VR audit: BOTH EYES READ LEFT %s %#llx reads%s; writes%s; also reads%s\n",
                    compute ? "compute" : "draw vs", (unsigned long long)hash, g_audit_left_reads.c_str(),
                    AuditList(outputs).c_str(), AuditList(g_audit_reads).c_str());
        std::fflush(stdout);
    }
    g_audit_reads.clear();
    g_audit_writes.clear();
    g_audit_reads_stereo = false;
    g_audit_left_reads.clear();
}
u64 g_draw_hash = 0;        // the draw's vertex shader (BB_VR_MONO_COMPUTE)
bool g_second_eye = false;    // the dispatch being recorded runs again for the right eye

void Log(const char* text) {
    std::printf("VR multiview: %s\n", text);
    std::fflush(stdout);
}

} // namespace

bool Enabled() {
    return g_enabled;
}

const char* CacheSuffix() {
    // The cache opens before the game's render size can be read, and one shader holds that size (Fixes::PatchShader):
    // other per-eye sizes than 2880 (the launcher's BB_VR_EYE_RES) keep a cache of their own.
    static const std::string suffix = [] {
        const char* res = std::getenv("BB_VR_EYE_RES");
        // _mv2: guest buffers are per-eye arrays in the shaders (EyeBufferInfo).
        return res && res[0] && std::strcmp(res, "2880") != 0 ? std::string("_mv2_") + res : std::string("_mv2");
    }();
    return Requested() ? suffix.c_str() : "";
}

void OnDevice(VkDevice device, bool supported) {
    g_device = device;
    g_enabled = Requested() && supported;
    g_audit = g_enabled && std::getenv("BB_VR_AUDIT") != nullptr;
    if (Requested()) {
        Log(supported ? "on" : "requested, but the device has no multiview");
    }
}

u32 BackingLayers(const VideoCore::ImageInfo& info, u32 layers) {
    static int logged = 0;
    static const bool log = std::getenv("BB_VR_HASHES") != nullptr;
    if (log && g_enabled && info.size.width >= 1400 && !info.props.is_block && logged < 300) { // development: large images
        ++logged;
        std::printf("VR multiview: image %ux%u fmt %d layers %u levels %u samples %u type %d block %d at %#llx\n",
                    info.size.width, info.size.height, int(info.pixel_format), layers, info.resources.levels,
                    info.num_samples, int(info.type), int(info.props.is_block), (unsigned long long)info.guest_address);
        std::fflush(stdout);
    }
    // Multisampled ones too (Driveclub draws a half-size 4x MSAA pass, 1440x1440 for 2880 eyes: the lights' glows shrink
    // without it. One layer, it was drawn for the left eye only). Their resolves take both layers (ResolveLayers).
    if (!g_enabled || layers != 1 || info.props.is_block || info.type != AmdGpu::ImageType::Color2D ||
        info.size.depth != 1) {
        return layers;
    }
    // The lit scene's 1024x896 copy (vs 0xa556e185) and its blurs (B10G11R11 and R16G16B16A16F, about ten passes),
    // which the final composite (vs 0x8d52ab95) adds to both eyes: one layer, they were made from the left eye's
    // picture and the right eye got its glow in other places (the fire by the lake blocky and bright, BB_VR_AUDIT).
    if (info.size.width == 1024 && info.size.height == 896 &&
        (info.pixel_format == vk::Format::eB10G11R11UfloatPack32 || info.pixel_format == vk::Format::eR16G16B16A16Sfloat)) {
        return 2;
    }
    // The render size and the output size (larger when DLSS upscales) and their halvings (rounded up), down to a
    // sixteenth.
    u32 sizes[2][2]{};
    if (!Fixes::RenderSize(sizes[0][0], sizes[0][1])) {
        return layers;
    }
    if (!Fixes::OutputSize(sizes[1][0], sizes[1][1])) {
        sizes[1][0] = sizes[0][0];
        sizes[1][1] = sizes[0][1];
    }
    for (const auto& size : sizes) {
        for (u32 k = 0; k <= 4; ++k) {
            const u32 w = (size[0] + (1u << k) - 1) >> k, h = (size[1] + (1u << k) - 1) >> k;
            if (info.size.width == w && info.size.height == h) {
                return 2;
            }
        }
    }
    // The bloom: 4/15 of the render size (2880 -> 768, within a pixel: the game's rounding), one image with its mips
    // (formats R16G16B16A16 float and B10G11R11). One layer, it was built from the left eye alone and the right eye got
    // the left eye's glow (headlights, glints on the cars' paint in other places in each eye). Only that size: with
    // DLSS Quality it is 512, and its halvings were the game's own textures (64x64, 128x128...).
    const s64 bloom_w = s64(sizes[0][0] * 4 / 15), bloom_h = s64(sizes[0][1] * 4 / 15);
    if (std::abs(s64(info.size.width) - bloom_w) <= 1 && std::abs(s64(info.size.height) - bloom_h) <= 1 &&
        (info.pixel_format == vk::Format::eR16G16B16A16Sfloat ||
         info.pixel_format == vk::Format::eB10G11R11UfloatPack32)) {
        return 2;
    }
    return layers;
}

bool Stereo(const VideoCore::Image& image) {
    return g_enabled && image.backing && image.backing->image.image_ci.arrayLayers == 2 && image.info.resources.layers == 1;
}

bool ArrayView(AmdGpu::ImageType type) {
    return Requested() && type == AmdGpu::ImageType::Color2D;
}

void AdjustView(const VideoCore::Image& image, const VideoCore::ImageViewInfo& info, vk::ImageViewType& type,
                u32& layer_count) {
    if (!info.vr_array || type != vk::ImageViewType::e2D) {
        return;
    }
    type = vk::ImageViewType::e2DArray;
    if (Stereo(image) && info.range.base.layer == 0 && info.range.extent.layers == 1) {
        layer_count = 2;
    }
}

u16 ViewMask(const VideoCore::Image* const* images, u32 count) {
    if (!g_enabled) {
        return 0;
    }
    u32 stereo = 0, mono = 0;
    for (u32 i = 0; i < count; ++i) {
        if (images[i]) {
            ++(Stereo(*images[i]) ? stereo : mono);
        }
    }
    static int logged = 0;
    if (stereo && mono && logged < 40) { // development: passes that mix two-layer and one-layer targets draw mono
        ++logged;
        std::printf("VR multiview: mixed pass:");
        for (u32 i = 0; i < count; ++i) {
            if (images[i]) {
                const auto& info = images[i]->info;
                std::printf(" [%ux%u fmt %d levels %u %s]", info.size.width, info.size.height, int(info.pixel_format),
                            info.resources.levels, Stereo(*images[i]) ? "stereo" : "mono");
            }
        }
        std::printf("\n");
        std::fflush(stdout);
    }
    if (g_audit) {
        std::scoped_lock lock{g_audit_mutex};
        g_audit_targets.clear();
        for (u32 i = 0; i < count; ++i) {
            if (images[i]) {
                g_audit_targets.push_back(
                    {images[i]->info.guest_address, images[i]->info.guest_size, AuditName(*images[i])});
            }
        }
        g_audit_pass_stereo = stereo && !mono;
    }
    return stereo && !mono ? 0b11 : 0;
}

void OnPipelineCreated(vk::Device device, vk::PipelineCache cache, const vk::GraphicsPipelineCreateInfo& info,
                       const vk::PipelineRenderingCreateInfo& rendering, vk::Pipeline pipeline) {
    if (!g_enabled || !pipeline) {
        return;
    }
    vk::PipelineRenderingCreateInfo multiview = rendering;
    multiview.viewMask = 0b11;
    vk::GraphicsPipelineCreateInfo twin_info = info;
    twin_info.pNext = &multiview;
    // Specialisation constant 200 = true: the vertex shader shifts each view (emit_spirv_vr.h).
    static const VkBool32 on = VK_TRUE;
    static const vk::SpecializationMapEntry entry{200, 0, sizeof(VkBool32)};
    static const vk::SpecializationInfo specialization{1, &entry, sizeof(on), &on};
    std::array<vk::PipelineShaderStageCreateInfo, 8> stages{};
    const u32 stage_count = std::min<u32>(info.stageCount, u32(stages.size()));
    for (u32 i = 0; i < stage_count; ++i) {
        stages[i] = info.pStages[i];
        stages[i].pSpecializationInfo = &specialization;
    }
    twin_info.pStages = stages.data();
    static int logged = 0;
    for (u32 i = 0; i < stage_count && logged < 30; ++i) {
        if (stages[i].stage & (vk::ShaderStageFlagBits::eGeometry | vk::ShaderStageFlagBits::eTessellationEvaluation)) {
            ++logged; // development: their positions come after the vertex stage (no eye shift there yet)
            std::printf("VR multiview: pipeline with %s stage\n",
                        stages[i].stage == vk::ShaderStageFlagBits::eGeometry ? "geometry" : "tessellation");
            std::fflush(stdout);
            break;
        }
    }
    auto [result, twin] = device.createGraphicsPipeline(cache, twin_info);
    if (result != vk::Result::eSuccess) {
        static int logged = 0;
        if (logged++ < 5) {
            std::printf("VR multiview: twin pipeline failed (%s)\n", vk::to_string(result).c_str());
            std::fflush(stdout);
        }
        return;
    }
    std::scoped_lock lock{g_pipelines_mutex};
    g_twins[VkPipeline(pipeline)] = VkPipeline(twin);
}

vk::Pipeline Variant(vk::Pipeline pipeline, u16 view_mask) {
    if (!view_mask || !pipeline) {
        return pipeline;
    }
    // Twins never change once made: each recording thread keeps its own copy of the ones it found.
    thread_local std::unordered_map<VkPipeline, VkPipeline> cache;
    if (const auto hit = cache.find(VkPipeline(pipeline)); hit != cache.end()) {
        return vk::Pipeline(hit->second);
    }
    std::scoped_lock lock{g_pipelines_mutex};
    const auto it = g_twins.find(VkPipeline(pipeline));
    if (it == g_twins.end()) {
        static int logged = 0;
        if (logged++ < 5) {
            Log("a multiview pass without its twin pipeline (draws the left eye only)");
        }
        return pipeline;
    }
    cache.emplace(it->first, it->second);
    return vk::Pipeline(it->second);
}

float Shift() {
    return g_shift;
}

void SetShift(float shift) {
    static int logged = 0;
    if (logged++ % 900 == 0) {
        std::printf("VR multiview: eye clip shift %.4f\n", shift);
        std::fflush(stdout);
    }
    g_shift = shift;
}

void NoteImage(const VideoCore::Image& image, bool storage) {
    if (storage && Stereo(image)) {
        g_writes_stereo = true;
    }
    if (g_audit) {
        std::scoped_lock lock{g_audit_mutex};
        AuditRes res{image.info.guest_address, image.info.guest_size, AuditName(image)};
        if (storage) {
            g_audit_writes.push_back(res);
        } else {
            g_audit_reads_stereo |= Stereo(image);
            AuditRead(res.address, res.size, res.name);
            g_audit_reads.push_back(std::move(res));
        }
    }
}

u64 g_dispatch_hash = 0;
void SetDispatchHash(u64 hash) {
    g_dispatch_hash = hash;
}

void AuditCopy(const char* what, const VideoCore::Image& src, const VideoCore::Image& dst) {
    if (!g_audit || (!Stereo(src) && !Stereo(dst))) {
        return;
    }
    std::scoped_lock lock{g_audit_mutex};
    const std::string key = std::string(what) + AuditName(src) + AuditName(dst);
    static std::unordered_map<std::string, bool> seen;
    if (seen.emplace(key, true).second) {
        std::printf("VR audit: COPY %s from [%s samples %u] to [%s samples %u]\n", what, AuditName(src).c_str(),
                    src.info.num_samples, AuditName(dst).c_str(), dst.info.num_samples);
        std::fflush(stdout);
    }
}

void SetDrawHash(u64 hash) {
    g_draw_hash = hash;
}

// Per-eye buffers (vr_multiview.h): the computes that run per eye (BB_VR_EYE_COMPUTES=hex,hex to replace the list,
// 0 for none), the buffers they write and their right-eye copies, and the binding being made.
struct EyeBuffer {
    u64 base;
    u64 size;
    u64 shadow_base; // the copy holds [shadow_base, base + size): the same alignment within 256 bytes as the guest's
    std::unique_ptr<VideoCore::Buffer> shadow;
};
std::vector<EyeBuffer> g_eye_buffers;
std::mutex g_eye_mutex;
bool g_eye_dispatch = false; // the dispatch being recorded writes a per-eye buffer
thread_local struct {
    u64 address, size, adjust;
    bool valid;
    bool eye_write; // written by a per-eye compute: the right eye's copy starts from the buffer as it is now
} t_eye_binding{};

bool EyeCompute(u64 hash) {
    static const std::vector<u64> list = [] {
        // The tiled lighting's light data (from depth and normals). Not 0x7ceec0f3 (256 KB from depth, albedo, normals):
        // a simulation that reads its own last state (fire, rain): run per eye, the eyes' copies drifted apart.
        std::vector<u64> out{0x409b1cc4};
        if (const char* v = std::getenv("BB_VR_EYE_COMPUTES")) {
            out.clear();
            for (const char* p = v; *p;) {
                char* end = nullptr;
                const u64 value = std::strtoull(p, &end, 16);
                if (value) {
                    out.push_back(value);
                }
                p = (*end == ',') ? end + 1 : end;
                if (end == p && *p) {
                    break;
                }
            }
        }
        return out;
    }();
    return std::find(list.begin(), list.end(), hash) != list.end();
}

void NoteBuffer(u64 address, u64 size, bool written, u64 adjust) {
    t_eye_binding = {address, size, adjust, true, false};
    if (g_enabled && written && size && EyeCompute(g_dispatch_hash)) {
        t_eye_binding.eye_write = true;
        std::scoped_lock lock{g_eye_mutex};
        const bool known = std::any_of(g_eye_buffers.begin(), g_eye_buffers.end(), [&](const EyeBuffer& e) {
            return address >= e.base && address + size <= e.base + e.size;
        });
        if (!known) {
            g_eye_buffers.push_back({address, size, address & ~u64{0xff}, nullptr});
            std::printf("VR multiview: per-eye buffer %#llx+%#llx (compute %#llx)\n", (unsigned long long)address,
                        (unsigned long long)size, (unsigned long long)g_dispatch_hash);
            std::fflush(stdout);
        }
        g_eye_dispatch = true;
    }
    if (!g_audit) {
        return;
    }
    std::scoped_lock lock{g_audit_mutex};
    char text[64];
    std::snprintf(text, sizeof(text), "buffer %#llx+%#llx", (unsigned long long)address, (unsigned long long)size);
    if (written) {
        g_audit_writes.push_back({address, size, text});
    } else {
        AuditRead(address, size, text);
    }
}

u32 BufferDescriptors(bool special) {
    return Requested() && !special ? 2 : 1;
}

u32 ExtraDescriptors(std::span<const vk::DescriptorSetLayoutBinding> bindings) {
    u32 extra = 0;
    for (const auto& binding : bindings) {
        if (binding.descriptorType == vk::DescriptorType::eStorageBuffer && binding.descriptorCount > 1) {
            extra += binding.descriptorCount - 1;
        }
    }
    return extra;
}

std::optional<vk::DescriptorBufferInfo> EyeBufferInfo(const Instance& instance, Runtime& runtime, bool special,
                                                      const vk::DescriptorBufferInfo& first) {
    const auto binding = std::exchange(t_eye_binding, {});
    if (!Requested() || special) {
        return std::nullopt;
    }
    vk::DescriptorBufferInfo second = first;
    if (!g_enabled || !binding.valid || !first.buffer) {
        return second;
    }
    std::scoped_lock lock{g_eye_mutex};
    for (auto& e : g_eye_buffers) {
        if (binding.address < e.base || binding.address + binding.size > e.base + e.size) {
            continue;
        }
        if (!e.shadow) {
            const u64 bytes = e.base + e.size - e.shadow_base;
            e.shadow = std::make_unique<VideoCore::Buffer>(instance, 0, bytes, VideoCore::MemoryType::DeviceLocal,
                                                           "VR right eye buffer");
            runtime.FillBuffer(e.shadow.get(), 0, bytes, 0);
        }
        second.buffer = e.shadow->Handle();
        second.offset = binding.address - e.shadow_base - binding.adjust;
        if (binding.eye_write) {
            // What the game put in the buffer before this dispatch (counters reset, data from the CPU) reaches only
            // the left eye's: the right eye's copy is seeded with it, else it drifted (lit particles broken in the
            // right eye).
            const vk::BufferCopy copy{first.offset + binding.adjust, second.offset + binding.adjust, binding.size};
            runtime.GetScheduler().EndRendering();
            runtime.GetScheduler().Record([src = first.buffer, dst = second.buffer, copy](vk::CommandBuffer cmdbuf) {
                const vk::MemoryBarrier2 before{
                    .srcStageMask = vk::PipelineStageFlagBits2::eAllCommands,
                    .srcAccessMask = vk::AccessFlagBits2::eMemoryWrite,
                    .dstStageMask = vk::PipelineStageFlagBits2::eCopy,
                    .dstAccessMask = vk::AccessFlagBits2::eTransferRead | vk::AccessFlagBits2::eTransferWrite,
                };
                cmdbuf.pipelineBarrier2(vk::DependencyInfo{.memoryBarrierCount = 1, .pMemoryBarriers = &before});
                cmdbuf.copyBuffer(src, dst, copy);
                const vk::MemoryBarrier2 after{
                    .srcStageMask = vk::PipelineStageFlagBits2::eCopy,
                    .srcAccessMask = vk::AccessFlagBits2::eTransferWrite,
                    .dstStageMask = vk::PipelineStageFlagBits2::eAllCommands,
                    .dstAccessMask = vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite,
                };
                cmdbuf.pipelineBarrier2(vk::DependencyInfo{.memoryBarrierCount = 1, .pMemoryBarriers = &after});
            });
        }
        break;
    }
    return second;
}

std::span<const vk::BufferImageCopy> StereoUploads(const VideoCore::Image& dst,
                                                   std::span<const vk::BufferImageCopy> copies) {
    if (!Stereo(dst)) {
        return copies;
    }
    thread_local std::vector<vk::BufferImageCopy> both;
    both.assign(copies.begin(), copies.end());
    for (const auto& copy : copies) {
        if (copy.imageSubresource.baseArrayLayer == 0 && copy.imageSubresource.layerCount == 1) {
            auto right = copy;
            right.imageSubresource.baseArrayLayer = 1;
            both.push_back(right);
        }
    }
    return both;
}

Shader::PushData Adjust(const Shader::PushData& push_data, bool compute) {
    Shader::PushData out = push_data;
    if (!g_enabled) {
        return out;
    }
    out.vr_eye = 0;
    if (compute) {
        out.vr_shift = {0.f, 0.f};
        g_second_eye = g_writes_stereo || g_eye_dispatch; // a per-eye buffer: the right eye's dispatch writes its copy
        if (g_audit) {
            std::scoped_lock lock{g_audit_mutex};
            AuditFinish(true, g_dispatch_hash, g_second_eye);
        }
        g_dispatch_hash = 0; // draws bind with no compute's hash (EyeCompute)
        static const bool log = std::getenv("BB_VR_HASHES") != nullptr;
        static std::unordered_map<u64, bool> seen;
        if (log && seen.emplace(g_dispatch_hash ^ (u64(g_second_eye) << 63), true).second) {
            std::printf("VR multiview: compute %#llx %s\n", (unsigned long long)g_dispatch_hash,
                        g_second_eye ? "both eyes" : "once");
            std::fflush(stdout);
        }
    } else {
        out.vr_shift = {g_shift, -g_shift};
        if (g_audit) {
            std::scoped_lock lock{g_audit_mutex};
            AuditFinish(false, g_draw_hash, g_audit_pass_stereo);
        }
    }
    g_writes_stereo = false;
    return out;
}

void SecondEye(Scheduler& scheduler, vk::PipelineLayout layout, const Shader::PushData& push_data, vk::Pipeline handle,
               u32 x, u32 y, u32 z, vk::Buffer indirect, u64 indirect_offset) {
    if (!g_enabled || !std::exchange(g_second_eye, false)) {
        return;
    }
    Shader::PushData right = push_data;
    right.vr_shift = {0.f, 0.f};
    right.vr_eye = 1;
    // The right eye's copies of per-eye buffers are not tracked by the buffer cache: their reads before and after.
    const bool eye_buffers = std::exchange(g_eye_dispatch, false);
    scheduler.Record([=](vk::CommandBuffer cmdbuf) {
        const vk::MemoryBarrier2 barrier{
            .srcStageMask = vk::PipelineStageFlagBits2::eAllCommands,
            .srcAccessMask = vk::AccessFlagBits2::eShaderWrite | vk::AccessFlagBits2::eShaderRead,
            .dstStageMask = vk::PipelineStageFlagBits2::eAllCommands,
            .dstAccessMask = vk::AccessFlagBits2::eShaderWrite | vk::AccessFlagBits2::eShaderRead,
        };
        if (eye_buffers) {
            cmdbuf.pipelineBarrier2(vk::DependencyInfo{.memoryBarrierCount = 1, .pMemoryBarriers = &barrier});
        }
        cmdbuf.pushConstants(layout, vk::ShaderStageFlagBits::eCompute, 0u, sizeof(right), &right);
        cmdbuf.bindPipeline(vk::PipelineBindPoint::eCompute, handle);
        if (indirect) {
            cmdbuf.dispatchIndirect(indirect, indirect_offset);
        } else {
            cmdbuf.dispatch(x, y, z);
        }
        if (eye_buffers) {
            cmdbuf.pipelineBarrier2(vk::DependencyInfo{.memoryBarrierCount = 1, .pMemoryBarriers = &barrier});
        }
    });
}

} // namespace Vulkan::Vr::Multiview
