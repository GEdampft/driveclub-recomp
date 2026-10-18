// SPDX-License-Identifier: GPL-2.0-or-later
// VR (dcport-vr) development hooks. See vr_dev.h. Trigger files (in %TEMP%\dcvr\), each removed when read:
//   DUMP      the TAA resolve's buffers of the next frame -> taa_<n>_{frame,camera,objects}.bin and taa_<n>.txt
//   WATCH     "<hex address> [length 1|2|4|8] [w|rw]": hardware data watch (DR0) on every thread; each hit logs the
//             instruction after the access (absolute, and relative to its allocation: the game image) for 300 hits.
//             An empty WATCH file removes the watch.

#include <algorithm>
#include <atomic>
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <string>

#include "video_core/renderer_vulkan/vr/vr_camera.h"
#include "video_core/renderer_vulkan/vr/vr_dev.h"
#include "video_core/renderer_vulkan/vr/vr_openxr.h"
#include "video_core/renderer_vulkan/vr/vr_stereo.h"

#ifdef _WIN32
#include <windows.h>
#include <tlhelp32.h>
#endif

namespace Vulkan::Vr::Dev {
namespace {

std::filesystem::path Dir() {
    const char* temp = std::getenv("TEMP");
    std::filesystem::path dir = std::filesystem::path(temp ? temp : ".") / "dcvr";
    std::error_code ec;
    std::filesystem::create_directories(dir, ec);
    return dir;
}

bool Take(const char* name, std::string* content = nullptr) {
    const auto path = Dir() / name;
    std::error_code ec;
    if (!std::filesystem::exists(path, ec)) {
        return false;
    }
    if (content) {
        std::ifstream f(path, std::ios::binary);
        content->assign(std::istreambuf_iterator<char>(f), {});
    }
    std::filesystem::remove(path, ec);
    return true;
}

std::atomic<bool> g_dump_pending{false};
std::atomic<int> g_dump_index{0};
std::atomic<bool> g_findcam_pending{false};
std::atomic<int> g_capture_left{0};
std::mutex g_capture_mutex;
FILE* g_capture_list = nullptr;
int g_capture_seq = 0;

void WriteFile(const std::filesystem::path& path, u64 address, u64 size) {
    std::ofstream f(path, std::ios::binary);
    f.write(reinterpret_cast<const char*>(address), std::streamsize(size));
}

#ifdef _WIN32
std::atomic<u64> g_watch_address{0};
std::atomic<int> g_watch_hits{0};
PVOID g_handler = nullptr;

LONG CALLBACK WatchHandler(EXCEPTION_POINTERS* info) {
    if (info->ExceptionRecord->ExceptionCode != EXCEPTION_SINGLE_STEP || !(info->ContextRecord->Dr6 & 1)) {
        return EXCEPTION_CONTINUE_SEARCH;
    }
    info->ContextRecord->Dr6 = 0;
    const int hit = g_watch_hits.fetch_add(1);
    if (hit < 300) {
        const u64 rip = info->ContextRecord->Rip;
        MEMORY_BASIC_INFORMATION mbi{};
        VirtualQuery(reinterpret_cast<void*>(rip), &mbi, sizeof(mbi));
        const u64 base = u64(mbi.AllocationBase);
        u64 value = 0;
        std::memcpy(&value, reinterpret_cast<const void*>(g_watch_address.load()), 8);
        std::printf("VR watch: hit %d at rip %#llx (base %#llx + %#llx) thread %lu value %#llx rsp %#llx\n", hit,
                    (unsigned long long)rip, (unsigned long long)base, (unsigned long long)(rip - base),
                    GetCurrentThreadId(), (unsigned long long)value,
                    (unsigned long long)info->ContextRecord->Rsp);
        std::fflush(stdout);
    }
    return EXCEPTION_CONTINUE_EXECUTION;
}

// PWATCH: the watched page made read-only; a write faults here, is logged (when it touches [address, address+length)),
// then single-stepped with the page writable and the protection put back on the step.
std::atomic<u64> g_pwatch_page{0}, g_pwatch_address{0}, g_pwatch_length{0};
std::atomic<int> g_pwatch_hits{0};
thread_local bool t_pwatch_stepping = false;

LONG CALLBACK PageWatchHandler(EXCEPTION_POINTERS* info) {
    const DWORD code = info->ExceptionRecord->ExceptionCode;
    const u64 page = g_pwatch_page.load();
    if (code == EXCEPTION_ACCESS_VIOLATION && page && info->ExceptionRecord->NumberParameters >= 2 &&
        info->ExceptionRecord->ExceptionInformation[0] == 1) {
        const u64 at = info->ExceptionRecord->ExceptionInformation[1];
        if ((at & ~0xfffull) != page) {
            return EXCEPTION_CONTINUE_SEARCH;
        }
        const u64 a = g_pwatch_address.load(), len = g_pwatch_length.load();
        if (at + 16 > a && at < a + len) {
            const int hit = g_pwatch_hits.fetch_add(1);
            if (hit < 200) {
                const u64 rip = info->ContextRecord->Rip;
                MEMORY_BASIC_INFORMATION mbi{};
                VirtualQuery(reinterpret_cast<void*>(rip), &mbi, sizeof(mbi));
                const u64 base = u64(mbi.AllocationBase);
                const u64* stack = reinterpret_cast<const u64*>(info->ContextRecord->Rsp);
                std::printf("VR pwatch: hit %d write %#llx at rip %#llx (base %#llx + %#llx) thread %lu stack %#llx %#llx "
                            "%#llx %#llx\n",
                            hit, (unsigned long long)at, (unsigned long long)rip, (unsigned long long)base,
                            (unsigned long long)(rip - base), GetCurrentThreadId(), (unsigned long long)stack[0],
                            (unsigned long long)stack[1], (unsigned long long)stack[2], (unsigned long long)stack[3]);
                std::fflush(stdout);
            }
        }
        DWORD old = 0;
        VirtualProtect(reinterpret_cast<void*>(page), 0x1000, PAGE_READWRITE, &old);
        t_pwatch_stepping = true;
        info->ContextRecord->EFlags |= 0x100; // trap flag: one instruction, then protect again
        return EXCEPTION_CONTINUE_EXECUTION;
    }
    if (code == EXCEPTION_SINGLE_STEP && t_pwatch_stepping) {
        t_pwatch_stepping = false;
        if (page) {
            DWORD old = 0;
            VirtualProtect(reinterpret_cast<void*>(page), 0x1000, PAGE_READONLY, &old);
        }
        return EXCEPTION_CONTINUE_EXECUTION;
    }
    return EXCEPTION_CONTINUE_SEARCH;
}

void SetPageWatch(u64 address, u32 length) {
    static PVOID handler = AddVectoredExceptionHandler(1, PageWatchHandler);
    (void)handler;
    if (const u64 old_page = g_pwatch_page.exchange(0)) {
        DWORD old = 0;
        VirtualProtect(reinterpret_cast<void*>(old_page), 0x1000, PAGE_READWRITE, &old);
    }
    g_pwatch_hits = 0;
    g_pwatch_address = address;
    g_pwatch_length = length;
    if (address) {
        const u64 page = address & ~0xfffull;
        g_pwatch_page = page;
        DWORD old = 0;
        const BOOL ok = VirtualProtect(reinterpret_cast<void*>(page), 0x1000, PAGE_READONLY, &old);
        std::printf("VR pwatch: page %#llx read-only (%s, was %#lx) for %#llx+%u\n", (unsigned long long)page,
                    ok ? "ok" : "failed", old, (unsigned long long)address, length);
        std::fflush(stdout);
    }
}

void SetWatch(u64 address, u32 length, bool reads) {
    if (!g_handler) {
        g_handler = AddVectoredExceptionHandler(1, WatchHandler);
    }
    g_watch_address = address;
    g_watch_hits = 0;
    const u64 len_bits = length == 8 ? 2 : length == 4 ? 3 : length == 2 ? 1 : 0;
    const u64 rw_bits = reads ? 3 : 1;
    const DWORD self = GetCurrentThreadId();
    HANDLE snap = CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0);
    THREADENTRY32 entry{sizeof(entry)};
    int threads = 0;
    for (BOOL ok = Thread32First(snap, &entry); ok; ok = Thread32Next(snap, &entry)) {
        if (entry.th32OwnerProcessID != GetCurrentProcessId() || entry.th32ThreadID == self) {
            continue;
        }
        HANDLE thread =
            OpenThread(THREAD_GET_CONTEXT | THREAD_SET_CONTEXT | THREAD_SUSPEND_RESUME, FALSE, entry.th32ThreadID);
        if (!thread) {
            continue;
        }
        SuspendThread(thread);
        CONTEXT ctx{};
        ctx.ContextFlags = CONTEXT_DEBUG_REGISTERS;
        if (GetThreadContext(thread, &ctx)) {
            ctx.Dr0 = address;
            ctx.Dr7 = address ? (1ull | (rw_bits << 16) | (len_bits << 18)) : 0;
            SetThreadContext(thread, &ctx);
            ++threads;
        }
        ResumeThread(thread);
        CloseHandle(thread);
    }
    CloseHandle(snap);
    std::printf("VR watch: %#llx length %u %s on %d threads\n", (unsigned long long)address, length,
                reads ? "read/write" : "write", threads);
    std::fflush(stdout);
}
#endif

} // namespace

void Poll() {
    if (Take("DUMP")) {
        g_dump_pending = true;
    }
    std::string shot;
    if (Take("TSHOT", &shot)) {
        Stereo::ShootTarget(std::strtoull(shot.c_str(), nullptr, 16));
    }
    if (Take("EYES")) {
        Stereo::DumpEyes();
    }
    if (Take("DRAWLOG")) {
        Stereo::LogDraws();
    }
    if (Take("RECENTER")) {
        Recenter();
    }
    if (Take("CAMLOG")) {
        Camera::LogNext(80);
    }
    if (Take("FINDCAM")) {
        g_findcam_pending = true;
    }
    if (Take("DUMPALL")) {
        std::scoped_lock lock{g_capture_mutex};
        g_capture_list = std::fopen((Dir() / "dispatches.txt").string().c_str(), "w");
        g_capture_seq = 0;
        g_capture_left = 400; // dispatches (about two frames)
    }
#ifdef _WIN32
    std::string spec;
    if (Take("WATCH", &spec)) {
        u64 address = 0;
        unsigned length = 4;
        char mode[8] = "w";
        std::sscanf(spec.c_str(), "%llx %u %7s", reinterpret_cast<unsigned long long*>(&address), &length, mode);
        SetWatch(address, length, std::strcmp(mode, "rw") == 0);
    }
    if (Take("PWATCH", &spec)) {
        u64 address = 0;
        unsigned length = 4;
        std::sscanf(spec.c_str(), "%llx %u", reinterpret_cast<unsigned long long*>(&address), &length);
        SetPageWatch(address, length);
    }
#endif
}

bool CapturingDispatches() {
    return g_capture_left.load(std::memory_order_relaxed) > 0;
}

void CaptureDispatch(u64 hash, u32 index, u64 address, u64 size) {
    std::scoped_lock lock{g_capture_mutex};
    if (!g_capture_list || g_capture_left <= 0) {
        return;
    }
    const int seq = g_capture_seq;
    if (index == 0 && --g_capture_left <= 0) {
        std::fclose(g_capture_list);
        g_capture_list = nullptr;
        std::printf("VR dev: dispatch capture done\n");
        std::fflush(stdout);
        return;
    }
    if (index == 0) {
        ++g_capture_seq;
    }
    std::fprintf(g_capture_list, "%d %llx %u %llx %llx\n", seq, (unsigned long long)hash, index,
                 (unsigned long long)address, (unsigned long long)size);
    if (address && size && size <= 0x10000) {
        char name[96];
        std::snprintf(name, sizeof(name), "d%04d_%llx_b%u.bin", seq, (unsigned long long)hash, index);
        WriteFile(Dir() / name, address, size);
    }
}

void NoteDispatch(u64 hash, size_t buffers, u32 groups_x, u32 groups_y, const u32 (&image_sizes)[4]) {
    static const bool enabled = std::getenv("BB_VR_HASHES") != nullptr;
    if (!enabled) {
        return;
    }
    static std::mutex mutex;
    static std::vector<u64> seen;
    std::scoped_lock lock{mutex};
    const u64 key = hash ^ (u64(groups_x) << 32 | groups_y) * 0x9e3779b97f4a7c15ull;
    if (std::find(seen.begin(), seen.end(), key) == seen.end()) {
        seen.push_back(key);
        std::printf("VR dev: compute %#llx, %zu buffers, %ux%u groups, images %ux%u %ux%u\n", (unsigned long long)hash,
                    buffers, groups_x, groups_y, image_sizes[0] >> 16, image_sizes[0] & 0xffff, image_sizes[1] >> 16,
                    image_sizes[1] & 0xffff);
        std::fflush(stdout);
    }
}

#ifdef _WIN32
namespace {
// Every committed, readable, non-write-combined region (the GPU upload memory is write-combined): copies of `want`
// (floats within tolerance), skipping [skip, skip + skip_size).
void ScanFloats(const char* label, const float* want, int count, u64 skip, u64 skip_size) {
    int hits = 0;
    MEMORY_BASIC_INFORMATION mbi{};
    for (u64 address = 0x10000; address < (1ull << 47) && hits < 60; address = u64(mbi.BaseAddress) + mbi.RegionSize) {
        if (!VirtualQuery(reinterpret_cast<void*>(address), &mbi, sizeof(mbi))) {
            break;
        }
        if (mbi.State != MEM_COMMIT || (mbi.Protect & (PAGE_GUARD | PAGE_NOACCESS | PAGE_WRITECOMBINE)) ||
            !(mbi.Protect & (PAGE_READONLY | PAGE_READWRITE | PAGE_EXECUTE_READ | PAGE_EXECUTE_READWRITE))) {
            continue;
        }
        const auto* f = reinterpret_cast<const float*>(mbi.BaseAddress);
        const size_t n = mbi.RegionSize / 4;
        for (size_t i = 0; i + count <= n && hits < 60; ++i) {
            bool match = true;
            for (int k = 0; k < count && match; ++k) {
                const float d = f[i + k] - want[k];
                match = d <= 2e-4f * (want[k] < 0 ? -want[k] : want[k]) + 1e-6f &&
                        -d <= 2e-4f * (want[k] < 0 ? -want[k] : want[k]) + 1e-6f;
            }
            const u64 at = u64(&f[i]);
            if (match && !(at >= skip && at < skip + skip_size)) {
                std::printf("VR findcam %s: %#llx (region %#llx+%#llx alloc %#llx):", label, (unsigned long long)at,
                            (unsigned long long)u64(mbi.BaseAddress), (unsigned long long)mbi.RegionSize,
                            (unsigned long long)u64(mbi.AllocationBase));
                for (int k = -4; k < 16; ++k) {
                    if (i + k < n) {
                        std::printf(" %.5g", f[i + k]);
                    }
                }
                std::printf("\n");
                ++hits;
            }
        }
    }
    std::fflush(stdout);
}
} // namespace
#endif

void OnDcTaa(u64 frame, u64 frame_size, u64 camera, u64 camera_size, u64 objects, u64 objects_size) {
    if (camera && camera_size >= 111 * 4) {
        Stereo::OnFrameCamera(reinterpret_cast<const float*>(camera) + 108); // the camera's world position
    }
#ifdef _WIN32
    if (g_findcam_pending.exchange(false)) {
        const auto* c = reinterpret_cast<const float*>(camera);
        ScanFloats("view row 0", c + 16, 3, camera, camera_size);
        ScanFloats("view column 0", c + 32, 3, camera, camera_size);
        ScanFloats("position", c + 108, 3, camera, camera_size);
    }
#endif
    static std::atomic<u64> calls{0};
    if (const u64 n = calls.fetch_add(1); n % 600 == 0) {
        std::printf("VR dev: TAA resolve call %llu (frame %#llx camera %#llx objects %#llx)\n", (unsigned long long)n,
                    (unsigned long long)frame, (unsigned long long)camera, (unsigned long long)objects);
        std::fflush(stdout);
    }
    if (!g_dump_pending.exchange(false)) {
        return;
    }
    const int n = g_dump_index++;
    const auto dir = Dir();
    const auto stem = "taa_" + std::to_string(n);
    WriteFile(dir / (stem + "_frame.bin"), frame, frame_size);
    WriteFile(dir / (stem + "_camera.bin"), camera, camera_size);
    WriteFile(dir / (stem + "_objects.bin"), objects, objects_size);
    std::ofstream(dir / (stem + ".txt")) << std::hex << "frame " << frame << " " << frame_size << "\ncamera " << camera
                                         << " " << camera_size << "\nobjects " << objects << " " << objects_size
                                         << "\n";
    std::printf("VR dev: TAA buffers dumped (%s): frame %#llx, camera %#llx, objects %#llx\n", stem.c_str(),
                (unsigned long long)frame, (unsigned long long)camera, (unsigned long long)objects);
    std::fflush(stdout);
}

} // namespace Vulkan::Vr::Dev
