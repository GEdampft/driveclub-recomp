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
    if (!g_enabled || layers != 1 || info.num_samples != 1 || info.props.is_block ||
        info.type != AmdGpu::ImageType::Color2D || info.size.depth != 1) {
        return layers;
    }
    u32 width = 0, height = 0;
    if (!Fixes::RenderSize(width, height)) {
        return layers;
    }
    // The render size and its halvings (rounded up), down to a sixteenth.
    for (u32 k = 0; k <= 4; ++k) {
        const u32 w = (width + (1u << k) - 1) >> k, h = (height + (1u << k) - 1) >> k;
        if (info.size.width == w && info.size.height == h) {
            return 2;
        }
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
}

u64 g_dispatch_hash = 0;
void SetDispatchHash(u64 hash) {
    g_dispatch_hash = hash;
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
        static const bool log = std::getenv("BB_VR_HASHES") != nullptr;
        static std::unordered_map<u64, bool> seen;
        if (log && seen.emplace(g_dispatch_hash ^ (u64(g_second_eye) << 63), true).second) {
            std::printf("VR multiview: compute %#llx %s\n", (unsigned long long)g_dispatch_hash,
                        g_second_eye ? "both eyes" : "once");
            std::fflush(stdout);
        }
    } else {
        out.vr_shift = {g_shift, -g_shift};
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
