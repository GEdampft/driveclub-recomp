// SPDX-License-Identifier: GPL-2.0-or-later
// VR (dcport-vr): Driveclub's camera. The game's view setters (image +0x736650 and +0x7376e0, SysV: rdi = the view,
// rsi = the camera's world matrix, rdx = its projection) are hooked; for the player's view the world matrix gets the
// head pose and the projection the headset's field of view, so every pass the engine derives (culling, lighting,
// motion vectors) follows the head.

#pragma once

#include "common/types.h"

namespace Vulkan::Vr::Camera {

/// Installs the hooks once the game image is in memory (checks the setters' bytes first). Safe to call every frame.
void Install();

/// Advance learned camera view slot on the game camera-cycle button (five slots).
void CameraModeAdvance();

/// The head pose to apply (row-vector 4x4: rows are the head's axes and position in the recentred tracking space,
/// metres), and the field of view to render (tangents of the half angles). From the frame loop.
/// `layer_pose` (orientation x, y, z, w, position x, y, z): the same head in the headset layer's space.
void SetHead(const float (&pose)[16], float tan_left, float tan_right, float tan_up, float tan_down,
             const float (&layer_pose)[7]);

/// The player camera's near plane (reversed infinite projection: P[3][2]), 0 before the first frame.
float Near();

/// Development: logs the next `count` setter calls (view, projection, position).
void LogNext(int count);

/// Development: the recent rendered camera positions (newest last), at most `max`.
u32 RecentPositions(float (*positions)[3], u32 max);

/// The game's screen aspect (the HUD's canvas; the render size may differ: a square render keeps 16:9 here).
float ScreenAspect();

/// The pose and field of view the last player view was rendered with (for the headset layer).
bool LastRendered(float (&layer_pose)[7], float (&tangents)[4]);

/// The pose and field of view of the recent player view rendered from `position` (the camera's world position as the GPU
/// sees it, the head's offset included); `distance` gets the closest match's distance.
bool RenderedAt(const float* position, float (&layer_pose)[7], float (&tangents)[4], float& distance);

} // namespace Vulkan::Vr::Camera
