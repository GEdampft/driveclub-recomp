// SPDX-License-Identifier: GPL-2.0-or-later
// VR (dcport-vr): true stereo rendering with Vulkan multiview (BB_VR_STEREO=mv). See also
// shader_recompiler/backend/spirv/emit_spirv_vr.h (the shader side).
//
// Main-view images (2D, one layer, one sample, at the game's render size or a power-of-two fraction of it; mip chains too)
// get a second array layer: the right eye. Layer 0 stays the guest's image (uploads, downloads, the presenter). A
// render pass whose attachments all have two layers draws with view mask 0b11 (each graphics pipeline has a multiview
// twin, bound for such passes); guest views of plain 2D images are 2D arrays (shaders read layer = view). In the twins,
// perspective vertices get a clip-space x shift per view (the eyes either side of the player's camera); compute
// dispatches that write a two-layer storage image run again for the right eye (push data vr_eye = 1).

#pragma once

#include "common/types.h"
#include "shader_recompiler/resource.h"
#include "video_core/amdgpu/resource.h"
#include "video_core/renderer_vulkan/vk_common.h"
#include "video_core/texture_cache/image.h"

namespace VideoCore {
struct Image;
struct ImageInfo;
struct ImageViewInfo;
} // namespace VideoCore

namespace Vulkan {
class Scheduler;
}

namespace Vulkan::Vr::Multiview {

/// BB_VR_STEREO=mv on a device with multiview (fixed once the device is made).
bool Enabled();

/// The pipeline cache directory's suffix: multiview shaders are cached apart ("_mv").
const char* CacheSuffix();

/// Device creation: whether the device has (and now enables) multiview.
void OnDevice(VkDevice device, bool supported);

/// Image creation: the backing's array layers (2 for main-view images).
u32 BackingLayers(const VideoCore::ImageInfo& info, u32 layers);

/// The image has the right eye's layer.
bool Stereo(const VideoCore::Image& image);

/// Guest views of this type are 2D arrays (the shaders declare them so).
bool ArrayView(AmdGpu::ImageType type);

/// View creation: a guest 2D view becomes a 2D array, over both layers of a two-layer image.
void AdjustView(const VideoCore::Image& image, const VideoCore::ImageViewInfo& info, vk::ImageViewType& type,
                u32& layer_count);

/// A render pass's view mask: 0b11 when every attachment has two layers (`images`, null entries skipped).
u16 ViewMask(const VideoCore::Image* const* images, u32 count);

/// ViewMask of a pass's colour targets (texture cache ids in `colors[i].first`) and depth target.
template <typename Cache, typename Colors, typename Depth>
u16 PassViewMask(Cache& cache, const Colors& colors, u32 num_colors, const Depth& depth) {
    if (!Enabled()) {
        return 0;
    }
    const VideoCore::Image* images[16]{};
    u32 count = 0;
    for (u32 i = 0; i < num_colors && count < 15; ++i) {
        if (colors[i].first) {
            images[count++] = &cache.GetImage(colors[i].first);
        }
    }
    if (depth.first) {
        images[count++] = &cache.GetImage(depth.first);
    }
    return ViewMask(images, count);
}

/// Graphics pipeline creation: builds the multiview twin of `pipeline`.
void OnPipelineCreated(vk::Device device, vk::PipelineCache cache, const vk::GraphicsPipelineCreateInfo& info,
                       const vk::PipelineRenderingCreateInfo& rendering, vk::Pipeline pipeline);

/// The pipeline to bind for a pass with `view_mask`.
vk::Pipeline Variant(vk::Pipeline pipeline, u16 view_mask);

/// Each frame: the eyes' clip x shift (left eye +shift, right eye -shift; P00 x half the eye distance).
void SetShift(float shift);
/// The left eye's clip x shift (the right eye's is its negative).
float Shift();

/// Binding: an image bound to the next draw or dispatch (storage: written).
void NoteImage(const VideoCore::Image& image, bool storage);

/// Development (BB_VR_HASHES): the dispatch about to bind (its hash, for the second-eye log).
void SetDispatchHash(u64 hash);

/// Push data as recorded: the draw's shift, or eye 0 for a dispatch (which then knows whether it writes two layers).
Shader::PushData Adjust(const Shader::PushData& push_data, bool compute);

/// Image copies (`regions`, vk::ImageCopy) from `src` to `dst`: a two-layer destination gets its right eye too, from
/// the source's right eye or, for a one-layer source, the same picture.
template <typename Regions>
void StereoCopies(const VideoCore::Image& src, const VideoCore::Image& dst, Regions& regions) {
    if (!Enabled() || !Stereo(dst)) {
        return;
    }
    const bool stereo_src = Stereo(src);
    const size_t count = regions.size();
    for (size_t i = 0; i < count; ++i) {
        auto& region = regions[i];
        if (region.dstSubresource.baseArrayLayer != 0 || region.dstSubresource.layerCount != 1 ||
            region.srcSubresource.layerCount != 1) {
            continue;
        }
        if (stereo_src && region.srcSubresource.baseArrayLayer == 0) {
            region.srcSubresource.layerCount = region.dstSubresource.layerCount = 2;
        } else {
            auto right = region;
            right.dstSubresource.baseArrayLayer = 1;
            regions.push_back(right);
        }
    }
}

/// After a dispatch is recorded: the right eye's dispatch when it writes a two-layer image (`indirect`: null for a
/// direct dispatch of x, y, z groups).
void SecondEye(Scheduler& scheduler, vk::PipelineLayout layout, const Shader::PushData& push_data, vk::Pipeline handle,
               u32 x, u32 y, u32 z, vk::Buffer indirect = {}, u64 indirect_offset = 0);

} // namespace Vulkan::Vr::Multiview
