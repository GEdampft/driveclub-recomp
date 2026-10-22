// SPDX-License-Identifier: GPL-2.0-or-later
// bbport: object motion vectors for temporal upscaling (docs/ROADMAP.md, step 1).
//
// G-buffer shaders preserve clip positions for the exact indexed range of each draw.
// Matching includes vertex streams, index contents and base offsets. A missing match falls
// back to camera motion. See motion_history.h for CPU bookkeeping and its regression test.

#pragma once

#include <array>
#include <vector>
#include "video_core/renderer_vulkan/motion_history.h"

#include "common/types.h"
#include "video_core/renderer_vulkan/vk_common.h"
#include "video_core/texture_cache/image.h"

namespace Vulkan {

class Instance;
class Scheduler;
struct RenderState;

class ObjectMotion {
public:
    ObjectMotion(const Instance& instance, Scheduler& scheduler);
    ~ObjectMotion();

    [[nodiscard]] bool Enabled() const noexcept {
        return enabled;
    }

    using DrawInfo = Motion::Draw;
    u32 PrepareDraw(const DrawInfo& draw);
    /// Small skeletons: whether the bone palette changed (Motion::History::Moving).
    bool Moving(const Motion::History::GateKey& key, u64 palette) {
        return history.Moving(key, palette);
    }
    /// Index range and topology hash of an indexed draw, reused across frames.
    template <class Scan>
    Motion::IndexRangeCache::Result IndexRange(const Motion::IndexRangeCache::Key& key,
                                               Scan&& scan) {
        return index_ranges.Get(key, frame, scan);
    }

    /// Rendering of a motion pipeline: the motion attachment (render-target size).
    void Attach(RenderState& state, u32 width, u32 height);

    /// Start of a frame (display pass).
    void OnFrameStart();
    void InvalidateHistory() { history.NextFrame(); history.NextFrame(); }

    /// The motion image for the compose pass (layout General after this call) and whether it
    /// holds this frame's vectors at `width` x `height`.
    vk::ImageView PrepareRead(vk::CommandBuffer cmdbuf, u32 width, u32 height, bool& valid);
    /// The image of the last PrepareRead (layout General), or null.
    [[nodiscard]] vk::ImageView View() const noexcept {
        return view ? *view : vk::ImageView{};
    }
    /// VR fork: one layer (an eye) of the last PrepareRead's image, or null.
    [[nodiscard]] vk::ImageView LayerView(u32 layer) const noexcept {
        return layer < image_layers && layer < layer_views.size() && layer_views[layer] ? *layer_views[layer]
                                                                                     : vk::ImageView{};
    }
    [[nodiscard]] u32 Layers() const noexcept {
        return image_layers;
    }
    [[nodiscard]] vk::Image Image(u32 width, u32 height) const noexcept {
        return written && width == image_width && height == image_height
            ? vk::Image(image) : vk::Image{};
    }

private:
    void EnsureImage(u32 width, u32 height, u32 layers = 1);
    /// VR fork: makes the kept image of this size current (layers 0: any count); false when there is none.
    bool SwapIn(u32 width, u32 height, u32 layers);

    const Instance& instance;
    Scheduler& scheduler;
    bool enabled = false;

    // Parameter ring (host visible): FrameSlots frames of ParamsPerFrame entries (two u32x4),
    // element 0 all zero (motion off).
    static constexpr u32 FrameSlots = 4;
    static constexpr u32 ParamsPerFrame = 8192;
    // Positions (device local): two halves (current/previous frame) of vec4; element 0 is reserved.
    static constexpr u32 PositionsPerFrame = 4u << 20;
    Motion::History history{PositionsPerFrame};
    Motion::IndexRangeCache index_ranges;
    std::array<u64, FrameSlots> params_ticks{};
    vk::Buffer params_buffer{};
    VmaAllocation params_allocation{};
    u32* params_mapped{};
    vk::Buffer positions_buffer{};
    VmaAllocation positions_allocation{};

    u64 frame = 0;
    u32 params_used = 0;
    VideoCore::UniqueImage image;
    vk::UniqueImageView view;
    u32 image_width = 0, image_height = 0;
    u32 image_layers = 1;                         ///< VR fork: an eye each in multiview passes
    std::array<vk::UniqueImageView, 2> layer_views; ///< VR fork
    bool written = false; ///< this frame's vectors are in the image
    vk::ImageLayout image_layout = vk::ImageLayout::eUndefined;
    /// VR fork: the images of the other sizes, kept. With VR DLSS the G-buffer passes alternate between the scene
    /// (two layers at the render size) and the rear-view mirror (768x512, one layer): the image was recreated at every
    /// switch, each time after a wait for the GPU to finish everything (a fifth of the GPU command thread's time,
    /// 30-45 FPS at the race start at 3200 per eye).
    struct KeptImage {
        VideoCore::UniqueImage image;
        vk::UniqueImageView view;
        u32 width = 0, height = 0, layers = 1;
        std::array<vk::UniqueImageView, 2> layer_views;
        bool written = false;
        vk::ImageLayout layout = vk::ImageLayout::eUndefined;
    };
    std::vector<KeptImage> kept_images;
};

} // namespace Vulkan
