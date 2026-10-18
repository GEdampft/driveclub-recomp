// SPDX-License-Identifier: GPL-2.0-or-later
// VR (dcport-vr): see vr_fixes.h.

#include <cstdio>
#include <cstdlib>

#include "video_core/renderer_vulkan/vr/vr_fixes.h"

namespace Vulkan::Vr::Fixes {
namespace {

constexpr u64 ImageBase = 0x800000000ULL;
constexpr u64 RenderWidth = 0x12d0544, RenderHeight = 0x12d0554; // eboot 0x16d0544 / 0x16d0554 (u32)

// Every OpConstant of a 32-bit type holding `from` becomes `to`; the number changed.
int ReplaceConstant(std::vector<u32>& spirv, u32 from, u32 to) {
    int changed = 0;
    for (size_t i = 5; i < spirv.size();) {
        const u32 words = spirv[i] >> 16, op = spirv[i] & 0xffff;
        if (words == 0) {
            break;
        }
        if (op == 43 /* OpConstant */ && words == 4 && spirv[i + 3] == from) {
            spirv[i + 3] = to;
            ++changed;
        }
        i += words;
    }
    return changed;
}

} // namespace

bool RenderSize(u32& width, u32& height) {
    width = *reinterpret_cast<const u32*>(ImageBase + RenderWidth);
    height = *reinterpret_cast<const u32*>(ImageBase + RenderHeight);
    return width >= 64 && height >= 64 && width <= 16384 && height <= 16384;
}

void PatchShader(u64 hash, std::vector<u32>& spirv) {
    static const bool off = std::getenv("BB_VR_NO_FIXES") != nullptr; // development: A/B
    if (hash != 0x2d9c66a5ULL || off) {
        return;
    }
    const u32 width = *reinterpret_cast<const u32*>(ImageBase + RenderWidth);
    const u32 height = *reinterpret_cast<const u32*>(ImageBase + RenderHeight);
    if (width < 64 || height < 64 || width > 16384 || height > 16384 || (width == 1920 && height == 1080)) {
        return;
    }
    const int changed = ReplaceConstant(spirv, 960, width / 2) + ReplaceConstant(spirv, 540, height / 2);
    std::printf("VR fixes: half-resolution downsample bounds %ux%u (%d constants)\n", width / 2, height / 2, changed);
    std::fflush(stdout);
}

} // namespace Vulkan::Vr::Fixes
