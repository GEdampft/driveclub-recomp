// SPDX-License-Identifier: GPL-2.0-or-later
// VR fork: object motion vectors for VR DLSS (vk_object_motion.h, upstream's per-draw clip position history).
//
// VR DLSS took its motion vectors from the depth and the camera only (vr_dlss.cpp, DriveMotion): every pixel moved as
// if it were fixed in the world, so the other cars (and the car you sit in) came out blurred and smeared. Upstream's
// object motion stores each instrumented draw's clip positions and compares them with the same draw's previous frame;
// it ran only with the flat temporal upscaler and chose skinned models by their bone palettes. In VR with DLSS it runs
// for Driveclub's cars instead: the G-buffer draws that write the cars' stencil bits (0x82, as the motion pass's car
// mask reads them). The positions are stored before the eye's clip shift (the same shift both frames: it cancels in the
// motion), the motion attachment has a layer per eye, and the motion pass takes the draw's own vectors where they are
// valid and still on top. BB_VR_OBJECT_MOTION=0 turns it off.
#pragma once

#include <cstdlib>
#include <cstring>

#include "video_core/amdgpu/regs.h"

namespace Vulkan {
class ObjectMotion;
}

namespace Vulkan::Vr::ObjectMotion {

/// The rasterizer's object motion (set once it is made), for the VR DLSS motion pass.
inline Vulkan::ObjectMotion* g_object_motion = nullptr;

/// VR with DLSS (the launcher's settings), and not turned off.
inline bool Wanted() {
    static const bool wanted = [] {
        const char* eye = std::getenv("BB_VR_EYE_RES");
        const char* aa = std::getenv("BB_AA");
        const char* off = std::getenv("BB_VR_OBJECT_MOTION");
        return eye && eye[0] && eye[0] != '0' && aa && std::strcmp(aa, "dlss") == 0 && !(off && off[0] == '0');
    }();
    return wanted;
}

/// The stencil bits the game writes for the cars (and the hands): drive_motion.comp's car mask.
constexpr u32 CarStencilBits = 0x82;

/// A draw that writes the cars' stencil bits where it passes the depth test.
inline bool CarDraw(const AmdGpu::Regs& regs) {
    if (!regs.depth_control.stencil_enable || !regs.depth_buffer.StencilValid()) {
        return false;
    }
    const auto op = regs.stencil_control.stencil_zpass_front;
    const auto& ref = regs.stencil_ref_front;
    u32 value = 0;
    if (op == AmdGpu::StencilFunc::ReplaceTest) {
        value = ref.stencil_test_val;
    } else if (op == AmdGpu::StencilFunc::ReplaceOp || op == AmdGpu::StencilFunc::Or) {
        value = ref.stencil_op_val;
    } else {
        return false;
    }
    return (value & ref.stencil_write_mask & CarStencilBits) != 0;
}

} // namespace Vulkan::Vr::ObjectMotion
