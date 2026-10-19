// SPDX-License-Identifier: GPL-2.0-or-later
// VR (dcport-vr): see emit_spirv_vr.h.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "shader_recompiler/backend/spirv/emit_spirv_vr.h"
#include "shader_recompiler/ir/program.h"

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

// ScanScreenParams' result for the shader being emitted (one shader per thread at a time).
struct ScreenParam {
    u32 index;   // output parameter
    u32 element; // its component
    float scale; // the output is scale * x / w + a constant
};
thread_local std::vector<ScreenParam> t_screen_params;

IR::Inst* Resolve(const IR::Value& value) {
    return value.IsImmediate() ? nullptr : value.TryInst();
}

// a / b as a * (1 / b) or a / b: {a, b}, else nulls.
std::pair<IR::Inst*, IR::Inst*> Divide(IR::Inst* inst) {
    if (!inst) {
        return {};
    }
    if (inst->GetOpcode() == IR::Opcode::FPDiv32) {
        return {Resolve(inst->Arg(0)), Resolve(inst->Arg(1))};
    }
    if (inst->GetOpcode() != IR::Opcode::FPMul32) {
        return {};
    }
    for (int i = 0; i < 2; ++i) {
        IR::Inst* recip = Resolve(inst->Arg(1 - i));
        if (recip && recip->GetOpcode() == IR::Opcode::FPRecip32) {
            return {Resolve(inst->Arg(i)), Resolve(recip->Arg(0))};
        }
    }
    return {};
}

// `inst` as scale * (a / b) + c with constant scale and c: the divide and the scale.
std::pair<IR::Inst*, float> ScaledDivide(IR::Inst* inst) {
    if (!inst) {
        return {};
    }
    if (Divide(inst).first) {
        return {inst, 1.f};
    }
    const auto op = inst->GetOpcode();
    if (op == IR::Opcode::FPMul32 || op == IR::Opcode::FPFma32) {
        for (int i = 0; i < 2; ++i) {
            IR::Inst* div = Resolve(inst->Arg(i));
            const IR::Value other = inst->Arg(1 - i);
            if (Divide(div).first && other.IsImmediate() &&
                (op == IR::Opcode::FPMul32 || inst->Arg(2).IsImmediate())) {
                return {div, other.F32()};
            }
        }
    } else if (op == IR::Opcode::FPAdd32) {
        for (int i = 0; i < 2; ++i) {
            if (inst->Arg(1 - i).IsImmediate()) {
                return ScaledDivide(Resolve(inst->Arg(i)));
            }
        }
    }
    return {};
}

} // namespace

void ScanScreenParams(const IR::Program& program) {
    t_screen_params.clear();
    const Info& info = program.info;
    if (!MultiviewShaders() || info.hw_stage != HwStage::Vertex || info.sw_stage != SwStage::Vertex) {
        return;
    }
    IR::Inst* pos[4]{};
    std::vector<std::pair<const IR::Inst*, IR::Inst*>> params; // (store, value) of each output parameter
    for (const IR::Block* block : program.blocks) {
        for (const IR::Inst& inst : block->Instructions()) {
            if (inst.GetOpcode() != IR::Opcode::SetAttribute) {
                continue;
            }
            const IR::Attribute attr = inst.Arg(0).Attribute();
            const u32 element = inst.Arg(2).U32();
            if (attr == IR::Attribute::Position0 && element < 4) {
                pos[element] = Resolve(inst.Arg(1));
            } else if (IR::IsParam(attr)) {
                params.emplace_back(&inst, Resolve(inst.Arg(1)));
            }
        }
    }
    if (!pos[0] || !pos[3]) {
        return;
    }
    std::vector<ScreenParam> found;
    u32 others = 0; // outputs divided by something else (view rays, another frame's position)
    for (const auto& [store, value] : params) {
        const auto [div, scale] = ScaledDivide(value);
        if (!div) {
            continue;
        }
        const auto [a, b] = Divide(div);
        others += b != pos[3];
        if (b == pos[3] && a == pos[0]) {
            found.push_back({u32(store->Arg(0).Attribute()) - u32(IR::Attribute::Param0), store->Arg(2).U32(), scale});
        }
    }
    static const bool off = std::getenv("BB_VR_NO_SCREEN_PARAMS") != nullptr; // development: A/B
    if (off) {
        return;
    }
    static const bool log = std::getenv("BB_VR_LOG_SCREEN_PARAMS") != nullptr;
    if (log && !found.empty()) {
        std::printf("VR screen params: vs %#llx param %u.%u scale %g (%zu; %u other divides)\n",
                    (unsigned long long)info.pgm_hash, found[0].index, found[0].element, found[0].scale, found.size(),
                    others);
        std::fflush(stdout);
    }
    t_screen_params = std::move(found);
}

namespace {

// The position's x plus vr_shift[view] when w is not 1, in multiview pipelines (specialisation constant 200); outputs
// holding x / w (ScanScreenParams) plus scale * vr_shift[view] / w.
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
    const Id eye_shift = ctx.OpSelect(ctx.F32[1], perspective, shift, ctx.f32_zero_value);
    ctx.OpStore(x_pointer, ctx.OpFAdd(ctx.F32[1], x, eye_shift));
    if (ctx.sw_stage != SwStage::Vertex) {
        return;
    }
    for (const ScreenParam& param : t_screen_params) {
        const auto& out = ctx.output_params.at(param.index);
        if (!Sirit::ValidId(out.id) || param.element >= out.num_components) {
            continue;
        }
        const Id pointer =
            out.num_components == 1 ? out.id : ctx.OpAccessChain(out.pointer_type, out.id, ctx.ConstU32(param.element));
        const Id moved = ctx.OpFMul(ctx.F32[1], ctx.OpFDiv(ctx.F32[1], eye_shift, w), ctx.ConstF32(param.scale));
        ctx.OpStore(pointer, ctx.OpFAdd(ctx.F32[1], ctx.OpLoad(ctx.F32[1], pointer), moved));
    }
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
