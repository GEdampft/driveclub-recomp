// SPDX-License-Identifier: GPL-2.0-or-later
// VR (dcport-vr): see emit_spirv_vr.h.

#include <cstdlib>
#include <cstring>

#include "shader_recompiler/backend/spirv/emit_spirv_vr.h"

namespace Shader::Backend::SPIRV::Vr {

bool MultiviewShaders() {
    static const bool on = [] {
        const char* v = std::getenv("BB_VR_STEREO");
        return v && std::strcmp(v, "mv") == 0;
    }();
    return on;
}

bool ArrayedImage(AmdGpu::ImageType view_type) {
    return MultiviewShaders() && view_type == AmdGpu::ImageType::Color2D;
}

bool EyeBufferArrays() {
    return MultiviewShaders();
}

void Define(EmitContext& ctx) {
    if (!MultiviewShaders() || ctx.sw_stage == SwStage::Compute) {
        return;
    }
    ctx.AddExtension("SPV_KHR_multiview");
    ctx.AddCapability(spv::Capability::MultiView);
    ctx.vr_view_index = ctx.DefineVariable(ctx.U32[1], spv::BuiltIn::ViewIndex, spv::StorageClass::Input);
    if (ctx.sw_stage == SwStage::Fragment) {
        ctx.Decorate(ctx.vr_view_index, spv::Decoration::Flat);
    }
    ctx.Name(ctx.vr_view_index, "vr_view_index");
}

namespace {

// The eye this invocation draws: the view index in graphics stages, push data vr_eye in compute.
Id Layer(EmitContext& ctx) {
    if (ctx.sw_stage == SwStage::Compute) {
        const Id pointer = ctx.TypePointer(spv::StorageClass::PushConstant, ctx.U32[1]);
        return ctx.OpLoad(ctx.U32[1], ctx.OpAccessChain(pointer, ctx.push_data_block,
                                                        ctx.ConstU32(PushData::VrEyeIndex)));
    }
    return ctx.OpLoad(ctx.U32[1], ctx.vr_view_index);
}

} // namespace

void EyeBuffers(EmitContext& ctx) {
    if (!EyeBufferArrays()) {
        return;
    }
    // The block types as DefineBuffer made them (Sirit gives the same ids for the same declarations).
    const auto data_type = [&](u32 alias) -> Id {
        switch (EmitContext::PointerType(alias)) {
        case EmitContext::PointerType::U8:
            return ctx.U8;
        case EmitContext::PointerType::U16:
            return ctx.U16;
        case EmitContext::PointerType::U32:
            return ctx.U32[1];
        case EmitContext::PointerType::F32:
            return ctx.F32[1];
        case EmitContext::PointerType::U64:
            return ctx.U64;
        default:
            return Id{};
        }
    };
    Id eye{};
    for (auto& buffer : ctx.buffers) {
        if (buffer.buffer_type != BufferType::Guest) {
            continue;
        }
        for (u32 i = 0; i < buffer.aliases.size(); ++i) {
            auto& alias = buffer.aliases[i];
            const Id type = data_type(i);
            if (!Sirit::ValidId(alias.id) || !Sirit::ValidId(type)) {
                continue;
            }
            if (!Sirit::ValidId(eye)) {
                eye = Layer(ctx);
            }
            const Id block = ctx.TypeStruct(ctx.TypeRuntimeArray(type));
            alias.id = ctx.OpAccessChain(ctx.TypePointer(spv::StorageClass::StorageBuffer, block), alias.id, eye);
        }
    }
}

Id Coords(EmitContext& ctx, const EmitContext::TextureDefinition& texture, Id image, Id coords, bool integer) {
    if (!texture.vr_array) {
        return coords;
    }
    Id layer = Layer(ctx);
    if (!integer) {
        return ctx.OpCompositeConstruct(ctx.F32[3], ctx.OpCompositeExtract(ctx.F32[1], coords, 0),
                                        ctx.OpCompositeExtract(ctx.F32[1], coords, 1),
                                        ctx.OpConvertUToF(ctx.F32[1], layer));
    }
    // Fetches and storage access out of the layers are undefined: one-layer images give layer 0.
    if (!Sirit::ValidId(image)) {
        image = ctx.OpLoad(texture.image_type, texture.id);
    }
    const Id size = texture.is_storage ? ctx.OpImageQuerySize(ctx.U32[3], image)
                                       : ctx.OpImageQuerySizeLod(ctx.U32[3], image, ctx.u32_zero_value);
    const Id last = ctx.OpISub(ctx.U32[1], ctx.OpCompositeExtract(ctx.U32[1], size, 2), ctx.ConstU32(1U));
    layer = ctx.OpUMin(ctx.U32[1], layer, last);
    return ctx.OpCompositeConstruct(ctx.U32[3], ctx.OpCompositeExtract(ctx.U32[1], coords, 0),
                                    ctx.OpCompositeExtract(ctx.U32[1], coords, 1), layer);
}

namespace {

// The position's x plus vr_shift[view] when w is not 1, in multiview pipelines (specialisation constant 200).
void ShiftPosition(EmitContext& ctx) {
    const Id pointer = ctx.TypePointer(spv::StorageClass::PushConstant, ctx.F32[1]);
    const Id shift = ctx.OpLoad(ctx.F32[1], ctx.OpAccessChain(pointer, ctx.push_data_block,
                                                              ctx.ConstU32(PushData::VrShiftIndex),
                                                              ctx.OpLoad(ctx.U32[1], ctx.vr_view_index)));
    const Id x_pointer = ctx.OpAccessChain(ctx.output_f32, ctx.output_position, ctx.ConstU32(0U));
    const Id w = ctx.OpLoad(ctx.F32[1], ctx.OpAccessChain(ctx.output_f32, ctx.output_position, ctx.ConstU32(3U)));
    // Specialisation constant 200: the pipeline is the multiview twin (mono passes keep their positions).
    if (!Sirit::ValidId(ctx.vr_multiview)) {
        ctx.vr_multiview = ctx.SpecConstantFalse(ctx.U1[1]);
        ctx.Decorate(ctx.vr_multiview, spv::Decoration::SpecId, 200U);
        ctx.Name(ctx.vr_multiview, "vr_multiview");
    }
    const Id multiview = ctx.vr_multiview;
    const Id perspective =
        ctx.OpLogicalAnd(ctx.U1[1], multiview, ctx.OpFOrdNotEqual(ctx.U1[1], w, ctx.ConstF32(1.f)));
    const Id x = ctx.OpLoad(ctx.F32[1], x_pointer);
    ctx.OpStore(x_pointer, ctx.OpFAdd(ctx.F32[1], x, ctx.OpSelect(ctx.F32[1], perspective, shift, ctx.f32_zero_value)));
}

} // namespace

void Epilogue(EmitContext& ctx) {
    // The stage that feeds the rasteriser: the hardware VS (a vertex or tessellation evaluation shader).
    if (!MultiviewShaders() || ctx.hw_stage != HwStage::Vertex ||
        (ctx.sw_stage != SwStage::Vertex && ctx.sw_stage != SwStage::TessellationEval) ||
        ctx.runtime_info.hw.vs.clip_disable || !Sirit::ValidId(ctx.output_position)) {
        return;
    }
    static const bool no_tess = std::getenv("BB_VR_NO_TESS_SHIFT") != nullptr; // development: A/B
    if (no_tess && ctx.sw_stage == SwStage::TessellationEval) {
        return;
    }
    ShiftPosition(ctx);
}

void BeforeEmitVertex(EmitContext& ctx) {
    if (!MultiviewShaders() || ctx.sw_stage != SwStage::Geometry || !Sirit::ValidId(ctx.output_position)) {
        return;
    }
    ShiftPosition(ctx);
}

} // namespace Shader::Backend::SPIRV::Vr
