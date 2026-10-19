// SPDX-License-Identifier: GPL-2.0-or-later
// VR (dcport-vr) development hooks, driven by trigger files in %TEMP%\dcvr\ (written by the tools in
// tools/vr/): buffer dumps of Driveclub's TAA resolve constants and a hardware write watch on the game's memory.

#pragma once

#include "video_core/renderer_vulkan/vr/vr_openxr.h"

#include "common/types.h"
#include "video_core/renderer_vulkan/vr/vr_multiview.h"

namespace Vulkan::Vr::Dev {

/// Once per presented frame: reads the trigger files.
void Poll();

/// The TAA resolve's three buffers (frame, camera, objects: guest addresses are host addresses) this frame.
void OnDcTaa(u64 frame, u64 frame_size, u64 camera, u64 camera_size, u64 objects, u64 objects_size);

/// Every compute dispatch (the rasterizer's shader info): picks out Driveclub's TAA resolve (0x403f86d4 in menus, 0xfdd72cf1 in races).
/// BB_VR_HASHES=1: each compute shader's hash and buffer count, once.
void NoteDispatch(u64 hash, size_t buffers, u32 groups_x, u32 groups_y, const u32 (&image_sizes)[4]);

/// While a DUMPALL is running: one dispatch's buffers (index, address, size).
bool CapturingDispatches();
void CaptureDispatch(u64 hash, u32 index, u64 address, u64 size);

template <typename Info>
void OnDispatch(const Info& cs, u32 groups_x = 0, u32 groups_y = 0) {
    if (!Vr::Active()) {
        return; // flat play: nothing to note
    }
    u32 image_sizes[4]{}; // the first two images: width << 16 | height
    for (u32 i = 0; i < cs.images.size() && i < 2; ++i) {
        const auto image = cs.images[i].GetSharp(cs);
        image_sizes[i] = u32(image.width + 1) << 16 | u32(image.height + 1);
    }
    NoteDispatch(cs.pgm_hash, cs.buffers.size(), groups_x, groups_y, image_sizes);
    Multiview::SetDispatchHash(cs.pgm_hash);
    if (CapturingDispatches()) {
        for (u32 i = 0; i < cs.buffers.size(); ++i) {
            const auto b = cs.buffers[i].GetSharp(cs);
            CaptureDispatch(cs.pgm_hash, i, b.base_address, b.GetSize());
        }
    }
    if ((cs.pgm_hash == 0x403f86d4ULL || cs.pgm_hash == 0xfdd72cf1ULL) && cs.buffers.size() >= 3) {
        const auto b0 = cs.buffers[0].GetSharp(cs), b1 = cs.buffers[1].GetSharp(cs), b2 = cs.buffers[2].GetSharp(cs);
        OnDcTaa(b0.base_address, b0.GetSize(), b1.base_address, b1.GetSize(), b2.base_address, b2.GetSize());
    }
}

} // namespace Vulkan::Vr::Dev
