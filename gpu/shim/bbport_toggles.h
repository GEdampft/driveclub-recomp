// bbport: optimizations that can be switched off while the game runs (BB_TOGGLE_FILE,
// see runtime_memory.c), to find which one changes rendering without restarting.
#pragma once
#include <atomic>
#include <csetjmp>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#ifdef _WIN32
// bbport (Windows): no signals. The loader's exception handler resumes the faulting thread
// in bb_longjmp (src/compat_win.c: no unwinding through the faulting frames).
typedef unsigned long long sigjmp_buf[32];
extern "C" __attribute__((returns_twice)) int bb_setjmp(sigjmp_buf buffer);
#define sigsetjmp(buffer, save) bb_setjmp(buffer)
#endif

extern "C" std::uint64_t runtime_disabled_optimizations;
/// Recovery point for speculative guest memory reads on this thread (runtime_memory.c).
extern "C" __thread sigjmp_buf* runtime_fault_recover;

namespace BbToggle {
enum : std::uint64_t {
    RegionCache = 1,
    FetchShaderCache = 2,
    PageTrackingEarlyExit = 4,
    PendingPollLimit = 8,
    ThreadedRecording = 16,
    ImageDescCache = 32,
    LockFreeUploadCheck = 64,
    FindImageCache = 128,
    DeferredUploads = 256,
    AccessMemo = 512,
    TextureBindingMemo = 1024,
    CoarseReadTracking = 2048,
    PreparedResources = 4096,
    DrawPreparation = 8192,
    DeferredStreamCopies = 16384,
    HotPages = 32768,
    FaultWindow = 65536,
    ParallelCopies = 131072,
    AsyncFences = 262144,
    PoolSmallCopies = 524288,
    RecordPrefetch = 1ull << 32,
    TextureViewMemo = 1ull << 33,
    TextureBindHelper = 1ull << 34,
    EarlyDrawInputs = 1ull << 35,
    ConstantRing = 1ull << 36,
    DrawPipeline = 1ull << 37,
    PipelinedTasks = 1ull << 38,
    PendingFenceWaits = 1ull << 39,
    PipelinedDispatch = 1ull << 40,
    RecorderFences = 1ull << 41,
    PipelinedMemoryWrites = 1ull << 42,
    MultiCopyShader = 1ull << 43,
    RenderStateMemo = 1ull << 44,
    TextureSetMemo = 1ull << 45,
    PipelinedIndirectDraws = 1ull << 46,
    SceneAttachmentsOnly = 1ull << 47,
    SampleSceneProxies = 1ull << 48,
    OrderedGuestWrites = 1ull << 49,
    SceneHalfRes = 1ull << 50, ///< live scaling also reduces the 960x540 post targets
    UpdateImageFastPath = 1u << 30,
    // TAA A/B in one run: optional techniques, off by default (no measured gain, 2026-10-02).
    TaaTonemapBlend = 1ull << 51,
    TaaClip = 1ull << 52,
    TaaVariance = 1ull << 53,
    TaaFilter = 1ull << 54,
    // On by default (bit set: off): history of a thin feature this jitter phase missed is kept
    // when nothing moves. Static-camera flicker of railings/window bars p99.9 -45% (2026-10-03).
    TaaKeepNearerHistory = 1ull << 55,
    SceneMipBias = 1ull << 57, ///< negative LOD bias of G-buffer samplers at reduced scene sizes
    MappedRangeCopies = 1ull << 58, ///< set: IsMapped takes the shared lock again (A/B)
    DepthFormSwap = 1ull << 59, ///< an image used as depth and as color keeps both forms (texture_cache.cpp)
    // Bits 20-29 are used as raw debug toggles by the camera/object motion and the upscaler.
};
inline bool Disabled(std::uint64_t bit) {
    return (__atomic_load_n(&runtime_disabled_optimizations, __ATOMIC_RELAXED) & bit) != 0;
}
} // namespace BbToggle

namespace BbStats {
/// Guest writes caught by page protection, and pages currently left unprotected as hot.
inline std::atomic<std::uint64_t> tracker_faults{0};
inline std::atomic<std::int64_t> hot_pages{0};
/// Stall diagnostics (BB_FRAME_STATS): per-frame deltas printed for frames over 40 ms.
inline std::atomic<std::uint64_t> images_registered{0};
inline std::atomic<std::uint64_t> image_upload_bytes{0};
inline std::atomic<std::uint64_t> buffer_upload_bytes{0};
inline std::atomic<int> gpu_thread_clock{-1}; ///< clockid_t of the GPU command thread
inline std::atomic<std::uint64_t> draws{0}, dispatches{0}, submissions{0};
/// Frames the GPU command thread has started (display pass), for per-frame diagnostics.
inline std::atomic<std::uint64_t> gpu_frames{0};
/// Wall time spent in operations suspected of stalls (ns, all threads).
inline std::atomic<std::uint64_t> t_resident{0}, t_protect{0}, t_image_create{0}, t_refresh{0},
    t_staging{0}, t_host_wait{0}, t_copy{0}, copy_bytes{0}, t_read_faults{0}, read_faults{0},
    t_write_faults{0}, t_copy_cpu{0}, copy_sys_us{0}, copy_minflt{0};
/// Diagnostics are collected only with BB_FRAME_STATS=1.
inline const bool enabled = [] {
    const char* env = std::getenv("BB_FRAME_STATS");
    return env && env[0] == '1';
}();
struct Timer {
    std::atomic<std::uint64_t>& total;
    std::chrono::steady_clock::time_point start =
        enabled ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    ~Timer() {
        if (!enabled) {
            return;
        }
        total.fetch_add(std::chrono::duration_cast<std::chrono::nanoseconds>(
                            std::chrono::steady_clock::now() - start)
                            .count(),
                        std::memory_order_relaxed);
    }
};
/// The phase and packet timers cost a clock read each, many per draw: BB_PHASES=1 only.
inline const bool phases_enabled = [] {
    const char* env = std::getenv("BB_PHASES");
    return enabled && env && env[0] == '1';
}();
struct PhaseTimer {
    std::atomic<std::uint64_t>& total;
    std::chrono::steady_clock::time_point start =
        phases_enabled ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    ~PhaseTimer() {
        if (!phases_enabled) {
            return;
        }
        total.fetch_add(std::chrono::duration_cast<std::chrono::nanoseconds>(
                            std::chrono::steady_clock::now() - start)
                            .count(),
                        std::memory_order_relaxed);
    }
};
/// Time blocked in Scheduler::Wait by call site (ns): 0 other, 1 buffer watch, 2 fault buffer,
/// 3 present pacing, 4 Finish, 5 WaitSubmitted.
inline std::atomic<std::uint64_t> wait_site_ns[8]{};
/// Guest read paths into BufferCache::ReadMemory: 0 offloaded, 1 window split, 2 unresolved (sync fallback),
/// 3 locked caller (sync), 4 offload disabled.
inline std::atomic<std::uint64_t> readback_paths[8]{};
/// Read faults taken by the GPU command thread itself, by faulting instruction (BB_FRAME_STATS).
inline std::atomic<std::uint64_t> gpu_fault_rip[32]{}, gpu_fault_count[32]{};
inline void NoteGpuReadFault(std::uint64_t rip) {
    for (unsigned i = 0, at = static_cast<unsigned>((rip >> 2) * 0x9E3779B1u) & 31; i < 32; ++i, at = (at + 1) & 31) {
        std::uint64_t expected = 0;
        if (gpu_fault_rip[at].load(std::memory_order_relaxed) == rip ||
            gpu_fault_rip[at].compare_exchange_strong(expected, rip) || expected == rip) {
            gpu_fault_count[at].fetch_add(1, std::memory_order_relaxed);
            return;
        }
    }
}
/// Device memory created while running: 0 cache buffers, 1 images, 2 runtime temporary buffers.
inline std::atomic<std::uint64_t> alloc_count[4]{}, alloc_bytes[4]{};
/// Draws into the small mirror scene, and how many of them were skipped (BB_MIRROR_RATE).
inline std::atomic<std::uint64_t> mirror_seen{0}, mirror_skipped{0}, mirror_targets{0}, mirror_depth_only{0};
/// The mirror scene's render targets (guest address), whether each is being skipped this frame, and the
/// shaders that sample them: shader hash -> draws, and how many of those came while the target was stale.
inline std::atomic<std::uint64_t> mirror_keys[8]{};
inline std::atomic<bool> mirror_stale[8]{};
inline std::atomic<std::uint64_t> mirror_consumer_hash[16]{}, mirror_consumer_count[16]{}, mirror_consumer_stale[16]{};
/// True while the mirror scene is being skipped (BB_MIRROR_RATE other than 1); images refreshed from guest memory
/// that were blocked because they are one of its targets.
inline std::atomic<bool> mirror_skipping{false};
/// Small-scene color draws by index count: <100, <300, <1000, <3000, <10000, more; and with several instances.
inline std::atomic<std::uint64_t> mirror_size_hist[8]{};
inline std::atomic<std::uint64_t> mirror_refresh_blocked{0};
inline std::atomic<std::uint64_t> mirror_dispatch_skipped{0};
/// 768x512 images created (0 overlap recreate, 1 expand, 2 new) and freed, in total and while their target was
/// being skipped, to see whether the mirror pictures are rebuilt (and lost) between frames.
inline std::atomic<std::uint64_t> mirror_img_created[3]{}, mirror_img_created_stale{0}, mirror_img_freed{0},
    mirror_img_freed_stale{0};
/// Shaders of draws into the mirror scene that sample the mirror scene's own pictures (a feedback pass such as a
/// water and wiper simulation, which reads last frame's picture): those draws are never skipped.
inline std::atomic<std::uint64_t> mirror_feedback[64]{};
inline std::atomic<std::uint32_t> mirror_feedback_count{0};
inline std::atomic<std::uint64_t> mirror_feedback_kept{0};
inline void NoteMirrorFeedback(std::uint64_t hash) {
    if (!hash) {
        return;
    }
    for (unsigned i = 0, at = static_cast<unsigned>(hash * 0x9E3779B1u >> 9) & 63; i < 64; ++i, at = (at + 1) & 63) {
        std::uint64_t expected = 0;
        if (mirror_feedback[at].load(std::memory_order_relaxed) == hash) {
            return;
        }
        if (mirror_feedback[at].compare_exchange_strong(expected, hash)) {
            mirror_feedback_count.fetch_add(1, std::memory_order_relaxed);
            return;
        }
        if (expected == hash) {
            return;
        }
    }
}
inline bool IsMirrorFeedback(std::uint64_t hash) {
    for (unsigned i = 0, at = static_cast<unsigned>(hash * 0x9E3779B1u >> 9) & 63; i < 64; ++i, at = (at + 1) & 63) {
        const std::uint64_t v = mirror_feedback[at].load(std::memory_order_relaxed);
        if (v == hash) {
            return true;
        }
        if (v == 0) {
            return false;
        }
    }
    return false;
}
inline bool MirrorStale(std::uint64_t address) {
    for (unsigned i = 0; i < 8; ++i) {
        if (mirror_keys[i].load(std::memory_order_relaxed) == address) {
            return mirror_stale[i].load(std::memory_order_relaxed);
        }
    }
    return false;
}
inline bool IsMirrorTarget(std::uint64_t address) {
    for (unsigned i = 0; i < 8; ++i) {
        if (mirror_keys[i].load(std::memory_order_relaxed) == address) {
            return true;
        }
    }
    return false;
}
inline void NoteMirrorConsumer(std::uint64_t address, std::uint64_t hash) {
    for (unsigned i = 0; i < 8; ++i) {
        if (mirror_keys[i].load(std::memory_order_relaxed) != address) {
            continue;
        }
        for (unsigned j = 0, at = static_cast<unsigned>(hash * 0x9E3779B1u >> 7) & 15; j < 16; ++j, at = (at + 1) & 15) {
            std::uint64_t expected = 0;
            if (mirror_consumer_hash[at].load(std::memory_order_relaxed) == hash ||
                mirror_consumer_hash[at].compare_exchange_strong(expected, hash) || expected == hash) {
                mirror_consumer_count[at].fetch_add(1, std::memory_order_relaxed);
                mirror_consumer_stale[at].fetch_add(mirror_stale[i].load(std::memory_order_relaxed), std::memory_order_relaxed);
                return;
            }
        }
        return;
    }
}
/// Texture cache churn: images created by site (0 overlap recreate, 1 expand, 2 new, 3 stencil), page tracking
/// changes (0 track, 1 untrack) and the code that frees images, by return address.
inline std::atomic<std::uint64_t> image_creates[4]{}, image_tracking[2]{};
inline std::atomic<std::uint64_t> image_free_rip[16]{}, image_free_count[16]{};
inline void NoteImageFree(std::uint64_t rip) {
    for (unsigned i = 0, at = static_cast<unsigned>((rip >> 2) * 0x9E3779B1u) & 15; i < 16; ++i, at = (at + 1) & 15) {
        std::uint64_t expected = 0;
        if (image_free_rip[at].load(std::memory_order_relaxed) == rip ||
            image_free_rip[at].compare_exchange_strong(expected, rip) || expected == rip) {
            image_free_count[at].fetch_add(1, std::memory_order_relaxed);
            return;
        }
    }
}
/// Wall time the GPU command thread waited for guest submissions (ns).
inline std::atomic<std::uint64_t> gpu_idle_ns{0};
/// Draws recorded into the reduced scene targets, and draws after the scene started.
inline std::atomic<std::uint64_t> reduced_draws{0}, scene_draws{0};
/// Wall time spent blocked in the scheduler (ns): waiting for the recording thread to drain,
/// for host copies before a submission or fence, and for GPU ticks.
inline std::atomic<std::uint64_t> sync_recording_ns{0}, host_copies_wait_ns{0}, tick_wait_ns{0},
    copy_threads_wait_ns{0}, host_copy_waits{0};
struct WaitTimer {
    std::atomic<std::uint64_t>& total;
    std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
    ~WaitTimer() {
        total.fetch_add(std::chrono::duration_cast<std::chrono::nanoseconds>(
                            std::chrono::steady_clock::now() - start).count(),
                        std::memory_order_relaxed);
    }
};
/// Per-phase wall time of the GPU command thread draw path (ns), BB_FRAME_STATS only:
/// 0 filter, 1 pipeline lookup, 2 render state, 3 bind resources, 4 begin rendering, 5 emit + bind, 6 record + flush.
inline std::atomic<std::uint64_t> phase_ns[8]{};
/// Wall time per graphics packet opcode on the GPU command thread (ns), BB_FRAME_STATS only.
inline std::atomic<std::uint64_t> opcode_ns[256]{};
/// GPU thread rusage, refreshed after each graphics submission.
inline std::atomic<std::uint64_t> gpu_sys_us{0}, gpu_user_us{0}, gpu_invol_switches{0},
    gpu_vol_switches{0}, gpu_minor_faults{0};
/// Protection faults (signals) taken by the GPU thread itself.
inline std::atomic<std::uint64_t> gpu_signal_faults{0};
/// Protection changes: calls and pages, those removing write access (TLB shootdowns) apart.
inline std::atomic<std::uint64_t> protect_calls{0}, protect_pages{0}, protect_revoke_calls{0},
    protect_revoke_pages{0};
} // namespace BbStats
