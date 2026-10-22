// SPDX-License-Identifier: GPL-2.0-or-later
// VR (dcport-vr): Driveclub's camera hooks. See vr_camera.h.
//
// The setters copy the world matrix (rsi) and the projection (rdx) into the view (rdi). Their first 21 bytes are three
// position-independent stores; they are replaced by "mov rax, stub; jmp rax". The stub saves the caller-saved
// registers, calls Handler (Windows ABI) with the saved block, which may point rsi and rdx at modified copies, then
// runs the displaced 21 bytes and jumps back.
//
// Row-vector matrices, as the game keeps them: v_world = v_local * W (rows: right, up, back, position; the camera
// looks down -Z, Y up: OpenXR's convention), clip = v_view * P (P[2][3] = -1, reversed infinite depth: P[3][2] = near).
//
// BB_VR_HEAD=0: hooks on, matrices untouched. BB_VR_CAMLOG=1: each view object's projection and position, once.

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <vector>

#include <xbyak/xbyak.h>

#include "video_core/renderer_vulkan/vr/vr_camera.h"
#include "video_core/renderer_vulkan/vr/vr_user_config.h"
#include "video_core/renderer_vulkan/vr/vr_multiview.h"

#ifdef _WIN32
#include <windows.h>
#endif

namespace Vulkan::Vr::Camera {
namespace {

constexpr u64 ImageBase = 0x800000000ULL;

extern "C" void SceneHandler(u64* regs);
extern "C" void ViewHandler(u64* regs);
extern "C" void CameraSetHandler(u64* regs);

struct Site {
    u64 offset;
    std::vector<u8> bytes; // the displaced instructions (position-independent), at least 12 bytes
    void (*handler)(u64*);
};

// Scene::SetCamera(scene, camera {world 4x4, projection 4x4, near, far, ...}): the scene keeps the camera (+0x4040)
// that every view and the next frame's "previous camera" come from, so the head goes in here. Then SetView(view,
// world, projection, ...) and its twin, for the development log.
const Site Sites[] = {
    // Camera::Set(camera, world, projection): copies both into the camera and derives its culling (frustum, detail)
    // from them, before the scene gets the camera. See CameraSetHandler.
    {0x72d9e0, {0x55, 0x48, 0x89, 0xe5, 0x41, 0x57, 0x41, 0x56, 0x53, 0x50, 0x49, 0x89, 0xd6}, &CameraSetHandler},
    {0x5e7a20, {0x55, 0x48, 0x89, 0xe5, 0x41, 0x57, 0x41, 0x56, 0x41, 0x55, 0x41, 0x54, 0x53}, &SceneHandler},
    {0x736650, {0xc5, 0xf8, 0x28, 0x46, 0x30, 0xc5, 0x78, 0x5c, 0x87, 0x40, 0x05, 0x00, 0x00, 0xc5, 0xf8, 0x29, 0x87, 0x40,
                0x05, 0x00, 0x00}, &ViewHandler},
    {0x7376e0, {0xc5, 0xf8, 0x28, 0x46, 0x30, 0xc5, 0xf8, 0x29, 0x87, 0x40, 0x05, 0x00, 0x00, 0xc5, 0xf8, 0x29, 0x87, 0x80,
                0x05, 0x00, 0x00}, &ViewHandler},
};

struct alignas(16) Mat {
    float m[16];
};

// The saved register block (see Stub): lowest address first.
enum Reg { R11, R10, R9, R8, RDI, RSI, RDX, RCX, RAX, RegCount };

struct Head {
    float pose[16];
    float tan[4]; // left (negative), right, up, down (negative)
    float layer[7];
};

std::mutex g_head_mutex;
Head g_head{};
bool g_head_valid = false;
Head g_rendered{};
bool g_rendered_valid = false;
// The last player views' rendered positions with their heads (RenderedAt): the GPU renders a frame or two behind.
struct Rendered {
    float position[3];
    Head head;
};
constexpr u32 HistorySize = 32;
Rendered g_history[HistorySize];
u32 g_history_count = 0;

// Copies handed to the setters: a ring, as the setters may run on more than one thread and the game copies them at once.
constexpr u32 RingSize = 64;
struct alignas(16) CameraDesc {
    u8 bytes[0x1000]; // the description runs past +0x2d0 (nested view settings); copied whole
};
CameraDesc g_ring[RingSize];
std::atomic<u32> g_ring_next{0};

bool g_head_enabled = true;
bool g_camlog = false;
int g_fov_mode = 2; // BB_VR_FOV: 0 the game's projection, 1 symmetric headset field of view, 2 the headset's (asymmetric)
std::atomic<int> g_log_next{0};

// The VR camera offset remains a manual toggle. Short gamepad L3 requests it;
// F10 is the keyboard alternative (F6 controls FOV in DriveClub 0.2.5).
std::atomic<bool> g_pad_toggle{false};
void RequestCameraToggle() {
    g_pad_toggle.store(true, std::memory_order_release);
}
bool FarCameraActive() {
    if (!UserConfig::Get().far_enabled) return false;
#ifdef _WIN32
    static bool down = false, active = false;
    const bool pressed = (GetAsyncKeyState(VK_F10) & 0x8000) != 0;
    if ((pressed && !down) || g_pad_toggle.exchange(false, std::memory_order_acq_rel)) {
        active = !active;
        std::printf("VR camera offset: %s (gamepad or F10)\n", active ? "ON" : "OFF");
        std::fflush(stdout);
    }
    down = pressed;
    return active;
#else
    return false;
#endif
}

std::atomic<float> g_near{0.f};

void Multiply(const float* a, const float* b, float* out) {
    for (int r = 0; r < 4; ++r) {
        for (int c = 0; c < 4; ++c) {
            out[r * 4 + c] = a[r * 4 + 0] * b[0 * 4 + c] + a[r * 4 + 1] * b[1 * 4 + c] + a[r * 4 + 2] * b[2 * 4 + c] +
                             a[r * 4 + 3] * b[3 * 4 + c];
        }
    }
}

// The player's view: a perspective projection of the screen's aspect, wider than the mirrors' (P00 below 2: 0.97 at
// 16:9, 1.725 square; the mirror 2.53).
bool IsPlayerView(const float* proj) {
    const float sx = proj[0], sy = proj[5];
    const float ratio = sy / sx / ScreenAspect();
    return sx > 0.3f && sx < 2.f && ratio > 0.96f && ratio < 1.045f && proj[11] == -1.f;
}

void LogView(u64 view, const float* world, const float* proj) {
    static std::mutex mutex;
    static std::vector<u64> seen;
    std::scoped_lock lock{mutex};
    if (std::find(seen.begin(), seen.end(), view) != seen.end() || seen.size() > 64) {
        return;
    }
    seen.push_back(view);
    std::printf("VR camera: view %#llx proj %.4g %.4g %.4g %.4g | %.4g %.4g pos %.5g %.5g %.5g%s\n",
                (unsigned long long)view, proj[0], proj[5], proj[10], proj[14], proj[8], proj[9], world[12], world[13],
                world[14], IsPlayerView(proj) ? " (player)" : "");
    std::fflush(stdout);
}

extern "C" void ViewHandler(u64* regs) {
    const auto* world = reinterpret_cast<const float*>(regs[RSI]);
    const auto* proj = reinterpret_cast<const float*>(regs[RDX]);
    if (g_camlog) {
        LogView(regs[RDI], world, proj);
    }
    if (g_log_next.load(std::memory_order_relaxed) > 0 && g_log_next.fetch_sub(1) > 0) {
        std::printf("VR camera: call view %#llx proj %.4g %.4g %.4g %.4g | %.4g %.4g pos %.5g %.5g %.5g fwd %.3g %.3g %.3g%s\n",
                    (unsigned long long)regs[RDI], proj[0], proj[5], proj[10], proj[14], proj[8], proj[9], world[12],
                    world[13], world[14], -world[8], -world[9], -world[10], IsPlayerView(proj) ? " (player)" : "");
        std::fflush(stdout);
    }
}

// The game culls with the frustum of the camera it sets here (Camera::Set(camera, world, block): the block holds the
// projection, then at +0x60 six view-space planes, outward normals, inside where n.p + d <= 0: near, left, right,
// bottom, top, far). Its own camera's planes face forward, +-46 degrees across: as the head turned, parts of the car
// beside the driver and of cars alongside were culled (Matt 2026-10-19: the passenger door). The player camera's
// planes are the headset's field of view turned with the head instead; the world and the projection stay the game's
// (the detail it picks is unchanged), and the block is a copy (the game's is rebuilt every frame).
// BB_VR_CULL_HEAD=0 turns it off.
alignas(16) float g_cull_block[0x110 / 4];

void HeadPlane(const float* pose, float nx, float ny, float nz, float d, float* out) {
    const float inv = 1.f / std::sqrt(nx * nx + ny * ny + nz * nz);
    nx *= inv, ny *= inv, nz *= inv, d *= inv;
    // Row vectors: a head direction v goes to v * R (the pose's upper 3x3). The planes stay at the eye: the head's
    // position moves them by centimetres in the car (the margin covers it), and in the menus it is metres away.
    const float cx = nx * pose[0] + ny * pose[4] + nz * pose[8];
    const float cy = nx * pose[1] + ny * pose[5] + nz * pose[9];
    const float cz = nx * pose[2] + ny * pose[6] + nz * pose[10];
    out[0] = cx, out[1] = cy, out[2] = cz;
    out[3] = d;
}

extern "C" void CameraSetHandler(u64* regs) {
    static const bool enabled = !(std::getenv("BB_VR_CULL_HEAD") && std::getenv("BB_VR_CULL_HEAD")[0] == '0');
    const auto* block = reinterpret_cast<const float*>(regs[RDX]);
    if (!enabled || !g_head_enabled || !IsPlayerView(block)) {
        return;
    }
    Head head;
    {
        std::scoped_lock lock{g_head_mutex};
        if (!g_head_valid) {
            return;
        }
        head = g_head;
    }
    std::memcpy(g_cull_block, block, sizeof(g_cull_block));
    // Both eyes: the headset's field of view as rendered (the multiview overscan), and a little more for the eyes'
    // offsets.
    const float margin = (Multiview::Enabled() ? 0.08f : 0.f) + 0.05f;
    const float l = head.tan[0] - margin, r = head.tan[1] + margin, u = head.tan[2] + margin, d = head.tan[3] - margin;
    float* planes = g_cull_block + 0x60 / 4;
    const float near_d = planes[3], far_d = planes[23];
    HeadPlane(head.pose, 0.f, 0.f, 1.f, near_d, planes + 0);
    HeadPlane(head.pose, -1.f, 0.f, -l, 0.f, planes + 4);
    HeadPlane(head.pose, 1.f, 0.f, r, 0.f, planes + 8);
    HeadPlane(head.pose, 0.f, -1.f, -d, 0.f, planes + 12);
    HeadPlane(head.pose, 0.f, 1.f, u, 0.f, planes + 16);
    HeadPlane(head.pose, 0.f, 0.f, -1.f, far_d, planes + 20);
    static bool told = false;
    if (!told) {
        told = true;
        std::printf("VR camera: culling with the head (the game's left plane %.4f %.4f %.4f %.4g)\n", block[28],
                    block[29], block[30], block[31]);
        std::fflush(stdout);
    }
    regs[RDX] = reinterpret_cast<u64>(g_cull_block);
}

extern "C" void SceneHandler(u64* regs) {
    const auto* desc = reinterpret_cast<const float*>(regs[RSI]);
    const float* world = desc;
    const float* proj = desc + 16;
    if (g_log_next.load(std::memory_order_relaxed) > 0) {
        std::printf("VR camera: scene %#llx desc %#llx rdx %#llx rcx %#llx ret %#llx proj %.4g %.4g pos %.5g %.5g %.5g\n",
                    (unsigned long long)regs[RDI], (unsigned long long)regs[RSI], (unsigned long long)regs[RDX],
                    (unsigned long long)regs[RCX], (unsigned long long)regs[RegCount], proj[0], proj[5], world[12],
                    world[13], world[14]);
        std::fflush(stdout);
    }
    if (!g_head_enabled || !IsPlayerView(proj)) {
        return;
    }
    // Development (BB_VR_LOOKLOG=1): the game's player camera yaw (its forward on the ground plane, degrees) per call.
    static const bool look_log = std::getenv("BB_VR_LOOKLOG") != nullptr;
    if (look_log) {
        static u64 calls = 0;
        std::printf("VR look: %llu yaw %.4f fwd %.5f %.5f %.5f proj %.4f %.4f\n", (unsigned long long)calls++,
                    std::atan2(-world[8], -world[10]) * 57.29578f, -world[8], -world[9], -world[10], proj[0], proj[5]);
        std::fflush(stdout);
    }
    Head head;
    {
        std::scoped_lock lock{g_head_mutex};
        if (!g_head_valid) {
            return;
        }
        head = g_head;
        g_rendered = g_head;
        g_rendered_valid = true;
    }
    // Optional manual offset: shifts the current player camera (no camera-mode detection).
    float custom_world[16];
    const float* camera_world = world;
    if (FarCameraActive()) {
        const auto& cfg = UserConfig::Get();
        std::memcpy(custom_world, world, sizeof(custom_world));
        // Row-vector world: third row is local backward, last row is position.
        for (int i = 0; i < 3; ++i)
            custom_world[12 + i] += world[8 + i] * cfg.far_distance_offset;
        custom_world[13] += cfg.far_height_offset;
        if (cfg.far_pitch_offset != 0.f) {
            const float radians = -cfg.far_pitch_offset * 0.01745329251994329577f;
            const float co = std::cos(radians), si = std::sin(radians);
            for (int i = 0; i < 3; ++i) {
                custom_world[4 + i] = co * world[4 + i] + si * world[8 + i];
                custom_world[8 + i] = -si * world[4 + i] + co * world[8 + i];
            }
        }
        camera_world = custom_world;
    }
    g_near = proj[14];
    CameraDesc& copy = g_ring[g_ring_next.fetch_add(1) % RingSize];
    std::memcpy(copy.bytes, desc, sizeof(copy.bytes));
    auto* new_world = reinterpret_cast<float*>(copy.bytes);
    auto* new_proj = new_world + 16;
    Multiply(head.pose, camera_world, new_world);

    // The headset's field of view, the game's depth terms kept (reversed infinite: P[3][2] = near).
    float l = head.tan[0], r = head.tan[1], u = head.tan[2], d = head.tan[3];
    if (g_fov_mode == 1) {
        const float x = std::max(-l, r), y = std::max(u, -d);
        l = -x, r = x, u = y, d = -y;
    }
    if (Multiview::Enabled()) {
        // Each eye is this view shifted sideways: near surfaces move inwards, leaving the outer edge of each eye
        // unrendered. Rendering a little wider (the layer says so) puts that edge outside what the headset shows.
        static const float overscan = [] {
            const char* v = std::getenv("BB_VR_OVERSCAN");
            return v ? float(std::atof(v)) : 0.08f;
        }();
        l -= overscan;
        r += overscan;
    }
    if (g_fov_mode != 0) {
        new_proj[0] = 2.f / (r - l);
        new_proj[5] = 2.f / (u - d);
        new_proj[8] = (r + l) / (r - l);
        new_proj[9] = (u + d) / (u - d);
    }
    {
        std::scoped_lock lock{g_head_mutex};
        // What the layer must say: the field of view actually rendered.
        if (g_fov_mode == 0) {
            g_rendered.tan[0] = (-1.f - new_proj[8]) / new_proj[0];
            g_rendered.tan[1] = (1.f - new_proj[8]) / new_proj[0];
            g_rendered.tan[2] = (1.f - new_proj[9]) / new_proj[5];
            g_rendered.tan[3] = (-1.f - new_proj[9]) / new_proj[5];
        } else {
            g_rendered.tan[0] = l, g_rendered.tan[1] = r, g_rendered.tan[2] = u, g_rendered.tan[3] = d;
        }
        Rendered& entry = g_history[g_history_count++ % HistorySize];
        std::memcpy(entry.position, new_world + 12, sizeof(entry.position));
        entry.head = g_rendered;
    }
    regs[RSI] = reinterpret_cast<u64>(copy.bytes);
}

struct Stub : Xbyak::CodeGenerator {
    Stub(const Site& site) : Xbyak::CodeGenerator(4096) {
        using namespace Xbyak::util;
        push(rax);
        push(rcx);
        push(rdx);
        push(rsi);
        push(rdi);
        push(r8);
        push(r9);
        push(r10);
        push(r11);
        // Entry rsp = 8 mod 16; 9 pushes make it 16-aligned. xmm0-7 (argument registers) and the shadow space.
        sub(rsp, 0xa0);
        for (int i = 0; i < 8; ++i) {
            movdqu(ptr[rsp + 0x20 + i * 16], Xbyak::Xmm(i));
        }
        lea(rcx, ptr[rsp + 0xa0]);
        mov(rax, reinterpret_cast<u64>(site.handler));
        call(rax);
        for (int i = 0; i < 8; ++i) {
            movdqu(Xbyak::Xmm(i), ptr[rsp + 0x20 + i * 16]);
        }
        add(rsp, 0xa0);
        pop(r11);
        pop(r10);
        pop(r9);
        pop(r8);
        pop(rdi);
        pop(rsi);
        pop(rdx);
        pop(rcx);
        pop(rax);
        db(site.bytes.data(), site.bytes.size());
        jmp(ptr[rip]);
        dq(ImageBase + site.offset + site.bytes.size());
    }
};

std::once_flag g_install_once;
std::vector<Stub*> g_stubs;

void DoInstall() {
#ifdef _WIN32
    if (const char* v = std::getenv("BB_VR"); !v || v[0] != '1') { // opt-in, see Vr::Init
        return;
    }
    g_head_enabled = !(std::getenv("BB_VR_HEAD") && std::getenv("BB_VR_HEAD")[0] == '0');
    g_camlog = std::getenv("BB_VR_CAMLOG") != nullptr;
    if (const char* v = std::getenv("BB_VR_FOV")) {
        g_fov_mode = std::atoi(v);
    }
    for (const Site& site : Sites) {
        auto* code = reinterpret_cast<u8*>(ImageBase + site.offset);
        MEMORY_BASIC_INFORMATION mbi{};
        if (!VirtualQuery(code, &mbi, sizeof(mbi)) || mbi.State != MEM_COMMIT ||
            std::memcmp(code, site.bytes.data(), site.bytes.size()) != 0) {
            std::printf("VR camera: the view setter at +%#llx is not as expected: camera hooks off\n",
                        (unsigned long long)site.offset);
            std::fflush(stdout);
            return;
        }
    }
    const bool cull_head = !(std::getenv("BB_VR_CULL_HEAD") && std::getenv("BB_VR_CULL_HEAD")[0] == '0');
    for (const Site& site : Sites) {
        if (site.handler == &CameraSetHandler && !cull_head) {
            continue;
        }
        auto* stub = new Stub(site);
        stub->ready();
        g_stubs.push_back(stub);
        auto* code = reinterpret_cast<u8*>(ImageBase + site.offset);
        std::vector<u8> patch(site.bytes.size(), 0x90);
        patch[0] = 0x48; // mov rax, imm64
        patch[1] = 0xb8;
        const u64 target = reinterpret_cast<u64>(stub->getCode());
        std::memcpy(patch.data() + 2, &target, 8);
        patch[10] = 0xff; // jmp rax
        patch[11] = 0xe0;
        DWORD old = 0;
        VirtualProtect(code, patch.size(), PAGE_EXECUTE_READWRITE, &old);
        std::memcpy(code, patch.data(), patch.size());
        VirtualProtect(code, patch.size(), old, &old);
        FlushInstructionCache(GetCurrentProcess(), code, patch.size());
    }
    std::printf("VR camera: view setters hooked%s\n", g_head_enabled ? "" : " (BB_VR_HEAD=0: matrices untouched)");
    std::fflush(stdout);
#endif
}

} // namespace

std::atomic<bool> g_image_mapped{false}; // the game's image is mapped (Install saw it)

float ScreenAspect() {
    // The headset's thread can ask before the game is loaded, and the image's data is mapped after its code (which
    // Install checks): the value's own page is checked until it has been read once (a constant of the eboot).
    static std::atomic<float> known{0.f};
    if (const float aspect = known.load(std::memory_order_relaxed); aspect != 0.f) {
        return aspect;
    }
    const u64 address = ImageBase + 0x729978; // eboot 0xb29978
    if (!g_image_mapped.load(std::memory_order_acquire)) {
        return 16.f / 9.f;
    }
#ifdef _WIN32
    MEMORY_BASIC_INFORMATION mbi{};
    constexpr DWORD readable = PAGE_READONLY | PAGE_READWRITE | PAGE_WRITECOPY | PAGE_EXECUTE_READ |
                               PAGE_EXECUTE_READWRITE | PAGE_EXECUTE_WRITECOPY;
    if (!VirtualQuery(reinterpret_cast<void*>(address), &mbi, sizeof(mbi)) || mbi.State != MEM_COMMIT ||
        !(mbi.Protect & readable) || (mbi.Protect & PAGE_GUARD)) {
        return 16.f / 9.f;
    }
#endif
    const float aspect = *reinterpret_cast<const float*>(address);
    if (aspect > 0.5f && aspect < 4.f) {
        known.store(aspect, std::memory_order_relaxed);
        return aspect;
    }
    return 16.f / 9.f;
}

void Install() {
#ifdef _WIN32
    // The game image is mapped before the first frame; check it is there.
    MEMORY_BASIC_INFORMATION mbi{};
    if (!VirtualQuery(reinterpret_cast<void*>(ImageBase + Sites[0].offset), &mbi, sizeof(mbi)) ||
        mbi.State != MEM_COMMIT) {
        return;
    }
#endif
    g_image_mapped.store(true, std::memory_order_release);
    std::call_once(g_install_once, DoInstall);
}

float Near() {
    return g_near.load();
}

void LogNext(int count) {
    g_log_next = count;
}

void SetHead(const float (&pose)[16], float tan_left, float tan_right, float tan_up, float tan_down,
             const float (&layer_pose)[7]) {
    std::scoped_lock lock{g_head_mutex};
    std::memcpy(g_head.pose, pose, sizeof(g_head.pose));
    std::memcpy(g_head.layer, layer_pose, sizeof(g_head.layer));
    g_head.tan[0] = tan_left;
    g_head.tan[1] = tan_right;
    g_head.tan[2] = tan_up;
    g_head.tan[3] = tan_down;
    g_head_valid = true;
}

bool LastRendered(float (&layer_pose)[7], float (&tangents)[4]) {
    std::scoped_lock lock{g_head_mutex};
    if (!g_rendered_valid) {
        return false;
    }
    std::memcpy(layer_pose, g_rendered.layer, sizeof(layer_pose));
    std::memcpy(tangents, g_rendered.tan, sizeof(tangents));
    return true;
}

u32 RecentPositions(float (*positions)[3], u32 max) {
    std::scoped_lock lock{g_head_mutex};
    const u32 count = std::min({g_history_count, HistorySize, max});
    for (u32 i = 0; i < count; ++i) {
        std::memcpy(positions[i], g_history[(g_history_count - count + i) % HistorySize].position, sizeof(positions[i]));
    }
    return count;
}

bool RenderedAt(const float* position, float (&layer_pose)[7], float (&tangents)[4], float& distance) {
    std::scoped_lock lock{g_head_mutex};
    if (std::abs(position[0]) + std::abs(position[1]) + std::abs(position[2]) < 1e-3f) {
        return false; // the origin: loading screens' cameras, or not a camera at all
    }
    const u32 count = std::min(g_history_count, HistorySize);
    const Rendered* best = nullptr;
    float best_d2 = 0.f;
    for (u32 n = 0; n < count; ++n) { // oldest to newest: the newest of equal matches wins
        const Rendered& e = g_history[(g_history_count - count + n) % HistorySize];
        const float dx = e.position[0] - position[0], dy = e.position[1] - position[1], dz = e.position[2] - position[2];
        const float d2 = dx * dx + dy * dy + dz * dz;
        if (!best || d2 <= best_d2) {
            best = &e;
            best_d2 = d2;
        }
    }
    if (!best) {
        return false;
    }
    distance = std::sqrt(best_d2);
    std::memcpy(layer_pose, best->head.layer, sizeof(layer_pose));
    std::memcpy(tangents, best->head.tan, sizeof(tangents));
    return true;
}

int CameraToggleButton() {
    const auto& c = UserConfig::Get();
    if (!c.far_enabled || c.camera_toggle_button == "off") return 0;
    if (c.camera_toggle_button == "pad_up") return 2;
    if (c.camera_toggle_button == "pad_down") return 3;
    if (c.camera_toggle_button == "pad_left") return 4;
    if (c.camera_toggle_button == "pad_right") return 5;
    return 1; // L3
}
int CameraHoldMs() { return UserConfig::Get().camera_hold_ms; }
void GamepadCameraToggle() { RequestCameraToggle(); }

} // namespace Vulkan::Vr::Camera

extern "C" int bbgpu_vr_camera_button(void) {
    return Vulkan::Vr::Camera::CameraToggleButton();
}
extern "C" int bbgpu_vr_camera_hold_ms(void) {
    return Vulkan::Vr::Camera::CameraHoldMs();
}
extern "C" void bbgpu_vr_camera_toggle(void) {
    Vulkan::Vr::Camera::GamepadCameraToggle();
}
