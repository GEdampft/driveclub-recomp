// SPDX-License-Identifier: GPL-2.0-or-later
// VR (dcport-vr): fixes to Driveclub's shaders for render sizes other than 1920x1080 (the VR render is 2880x2880).

#pragma once

#include <vector>

#include "common/types.h"

namespace Vulkan::Vr::Fixes {

/// Pipeline cache, after a guest shader's SPIR-V is made: rewrites constants the game's shaders hold for 1920x1080.
/// Compute 0x2d9c66a5 (a 2x2 downsample into a half-resolution target) stops at 960x540 while its dispatch covers the
/// whole target: those bounds become half the game's render size (eboot 0x16d0544/0x16d0554, which the resolution
/// patches change). The "depth bug" of the community resolution patches.
void PatchShader(u64 hash, std::vector<u32>& spirv);

/// The game's render size (eboot 0x16d0544/0x16d0554); false if unreadable.
bool RenderSize(u32& width, u32& height);

} // namespace Vulkan::Vr::Fixes
