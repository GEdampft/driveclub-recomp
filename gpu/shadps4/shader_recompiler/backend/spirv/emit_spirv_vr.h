// SPDX-License-Identifier: GPL-2.0-or-later
// VR (dcport-vr): stereo rendering with Vulkan multiview (BB_VR_STEREO=mv).
//
// Main-view render targets carry a second array layer (the right eye; layer 0 stays the guest's image). Graphics passes
// into them run with view mask 0b11; compute passes that write them run once per eye with push data vr_eye. In this
// mode every guest shader declares its plain 2D images as 2D arrays and reads/writes layer `Layer()`: gl_ViewIndex in
// graphics stages, vr_eye in compute. One-layer images give both eyes their only layer (sampling clamps the layer;
// fetches and storage access clamp it explicitly). The hardware vertex shader adds vr_shift[view] to clip x: an eye
// moved sideways by e changes clip x by -P00 * e for every vertex (zero for draws that are not the player's camera).

#pragma once

#include "shader_recompiler/backend/spirv/spirv_emit_context.h"

namespace Shader::Backend::SPIRV::Vr {

/// Multiview shaders are compiled (BB_VR_STEREO=mv; fixed for the session).
bool MultiviewShaders();

/// Whether an image of this view type is declared as a 2D array.
bool ArrayedImage(AmdGpu::ImageType view_type);

/// After the interfaces: the MultiView capability and the view index input (graphics stages).
void Define(EmitContext& ctx);

/// `coords` with the eye's layer appended when the image is a VR array (float for sampling, integer otherwise;
/// integer layers are clamped to the image's layers, `image` being the loaded image or, for atomics, null).
Id Coords(EmitContext& ctx, const EmitContext::TextureDefinition& texture, Id image, Id coords, bool integer);

/// End of the hardware vertex shader, in multiview pipelines (specialisation constant 200): the position's x plus
/// vr_shift[view] when w is not 1 (perspective geometry: the player's view in a main-view pass; full-screen passes draw
/// at w = 1 and stay put).
void Epilogue(EmitContext& ctx);

/// Geometry shaders: the same shift for each vertex they emit.
void BeforeEmitVertex(EmitContext& ctx);

} // namespace Shader::Backend::SPIRV::Vr
