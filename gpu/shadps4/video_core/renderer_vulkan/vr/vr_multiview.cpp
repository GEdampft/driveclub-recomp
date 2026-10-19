// SPDX-License-Identifier: GPL-2.0-or-later
// VR (dcport-vr): see vr_multiview.h.

#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <array>
#include <cstring>
#include <mutex>
#include <string>
#include <unordered_map>

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
// Development (BB_VR_MONO_COMPUTE): the bindings read a two-layer image and write one-layer storage images (their
// sizes): such a compute runs once, for the left eye, and both eyes then use its result.
bool g_reads_stereo = false;
std::string g_mono_writes;
std::string g_mono_pass; // the current pass's targets when it draws one layer (BB_VR_MONO_COMPUTE)
std::string g_stereo_reads; // the two-layer images the bindings sample (BB_VR_MONO_COMPUTE)
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
        return res && res[0] && std::strcmp(res, "2880") != 0 ? std::string("_mv") + res : std::string("_mv");
    }();
    return Requested() ? suffix.c_str() : "";
}

void OnDevice(VkDevice device, bool supported) {
    g_device = device;
    g_enabled = Requested() && supported;
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
    static const bool mono_log = std::getenv("BB_VR_MONO_COMPUTE") != nullptr;
    if (mono_log) {
        g_mono_pass.clear();
        if (!(stereo && !mono)) {
            for (u32 i = 0; i < count; ++i) {
                if (images[i]) {
                    g_mono_pass += " " + std::to_string(images[i]->info.size.width) + "x" +
                                   std::to_string(images[i]->info.size.height) + " fmt " +
                                   std::to_string(int(images[i]->info.pixel_format)) + " backing layers " +
                                   std::to_string(images[i]->backing ? images[i]->backing->image.image_ci.arrayLayers : 0) +
                                   " info layers " + std::to_string(images[i]->info.resources.layers) + " at " +
                                   std::to_string(images[i]->info.guest_address & 0xffffffffull);
                }
            }
        }
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
    static const bool mono_log = std::getenv("BB_VR_MONO_COMPUTE") != nullptr;
    if (mono_log) {
        if (!storage && Stereo(image)) {
            g_reads_stereo = true;
            char text[96];
            std::snprintf(text, sizeof(text), " %ux%u fmt %d at %#llx", image.info.size.width, image.info.size.height,
                          int(image.info.pixel_format), (unsigned long long)image.info.guest_address);
            g_stereo_reads += text;
        } else if (storage && !Stereo(image)) {
            g_mono_writes += " " + std::to_string(image.info.size.width) + "x" + std::to_string(image.info.size.height) +
                             " fmt " + std::to_string(int(image.info.pixel_format));
        }
    }
}

u64 g_dispatch_hash = 0;
void SetDispatchHash(u64 hash) {
    g_dispatch_hash = hash;
}

void SetDrawHash(u64 hash) {
    g_draw_hash = hash;
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
        g_second_eye = g_writes_stereo;
        static std::unordered_map<u64, bool> mono_seen;
        if (g_reads_stereo && !g_writes_stereo && !g_mono_writes.empty() &&
            mono_seen.emplace(g_dispatch_hash, true).second) {
            std::printf("VR multiview: compute %#llx reads both eyes, writes one-layer images only:%s\n",
                        (unsigned long long)g_dispatch_hash, g_mono_writes.c_str());
            std::fflush(stdout);
        }
        g_reads_stereo = false;
        g_mono_writes.clear();
        static const bool log = std::getenv("BB_VR_HASHES") != nullptr;
        static std::unordered_map<u64, bool> seen;
        if (log && seen.emplace(g_dispatch_hash ^ (u64(g_second_eye) << 63), true).second) {
            std::printf("VR multiview: compute %#llx %s\n", (unsigned long long)g_dispatch_hash,
                        g_second_eye ? "both eyes" : "once");
            std::fflush(stdout);
        }
    } else {
        out.vr_shift = {g_shift, -g_shift};
        static std::unordered_map<std::string, bool> mono_draw_seen;
        if (g_reads_stereo && !g_mono_pass.empty() &&
            mono_draw_seen.emplace(g_mono_pass + std::to_string(g_draw_hash), true).second) {
            std::printf("VR multiview: a one-layer pass samples both eyes' images (drawn for the left eye only):%s; vs "
                        "%#llx reads%s\n",
                        g_mono_pass.c_str(), (unsigned long long)g_draw_hash, g_stereo_reads.c_str());
            std::fflush(stdout);
        }
    }
    g_reads_stereo = false;
    g_mono_writes.clear();
    g_stereo_reads.clear();
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
    scheduler.Record([=](vk::CommandBuffer cmdbuf) {
        cmdbuf.pushConstants(layout, vk::ShaderStageFlagBits::eCompute, 0u, sizeof(right), &right);
        cmdbuf.bindPipeline(vk::PipelineBindPoint::eCompute, handle);
        if (indirect) {
            cmdbuf.dispatchIndirect(indirect, indirect_offset);
        } else {
            cmdbuf.dispatch(x, y, z);
        }
    });
}

} // namespace Vulkan::Vr::Multiview
