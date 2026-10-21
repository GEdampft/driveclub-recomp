// SPDX-License-Identifier: GPL-2.0-or-later
// VR fork: the fence after a GDS counter copy is written when the GPU has done the copy.
//
// Driveclub's vegetation culling counts visible clusters into GDS, copies the counters to memory (DMA_DATA, GDS to
// memory) and follows each copy with an EVENT_WRITE_EOP fence into the next slot of a ring the game polls: when the
// slot's sequence label matches, it reads the count. Fences are written here as soon as the packet is decoded, before
// the GPU has run the copy, so the game read counts a frame or more old and drew clusters from stale slots (clumps of
// grass and bushes floating in the sky). Keeping the counter pages exact (every poll a readback wait) fixed that at
// 6 FPS, and the game hammers those pages with its own data, so protecting them costs faults too.
//
// Here each GDS copy also copies the fresh bytes into host-visible staging, and a worker waits for the copy's
// submission on the timeline semaphore and writes the counter bytes into guest memory (the fast page's shadow merge,
// like small_writeback.cpp). The ring's EOP fence (the one written into the slot itself, 16 bytes after the count) is
// deferred to after that, so the label never says done before the count is there, in the order the real GPU would.
// Nothing is protected or waited on by the game's threads; the counter bytes are late outputs (readback_hybrid.h) so
// an access to them never reads back.
//
// The game's other GDS copies each frame (two on the graphics queue, two on the compute queue, to fixed slots) are
// followed by frame fences written elsewhere, which the game spins on at once: deferring those too serialised the
// game with the GPU (50 FPS against 74 at 3200), so they are written at decode time as before, and only their bytes
// come back through the worker. BB_GDS_FENCE=0 turns all of it off; =2 defers every fence after a GDS copy; =3 defers
// none (the worker still writes the bytes back); BB_GDS_FENCE_LOG=1 prints the deferred fences and their latency
// every 5 s.

#include <bit>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <mutex>
#include <thread>
#include <vector>

#include "common/thread.h"
#include "core/memory.h"
#include "video_core/amdgpu/liverpool.h"
#include "video_core/buffer_cache/buffer_cache.h"
#include "video_core/buffer_cache/readback_hybrid.h"
#include "video_core/renderer_vulkan/vk_runtime.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/renderer_vulkan/vk_staging_buffer_pool.h"

namespace VideoCore {
namespace {

struct GdsCopy {
    Vulkan::StagingBufferRef staging;
    VAddr addr;
    u32 size;
    u64 tick; ///< the submission that runs the copy
    std::chrono::steady_clock::time_point noted;
};

struct DeferredFence {
    std::vector<GdsCopy> copies;
    std::function<void()> signal;
    bool deferred;
    std::chrono::steady_clock::time_point decoded;
};

std::mutex g_mutex;
std::condition_variable g_cv;
std::vector<GdsCopy> g_pending; // copies since the last fence (GPU thread order)
std::deque<DeferredFence> g_queue;
bool g_worker_started = false;

bool Log() {
    static const bool on = std::getenv("BB_GDS_FENCE_LOG") && std::getenv("BB_GDS_FENCE_LOG")[0] == '1';
    return on;
}

int Mode() {
    static const int mode = std::getenv("BB_GDS_FENCE") ? std::atoi(std::getenv("BB_GDS_FENCE")) : 1;
    return mode;
}

/// A fence written into the record a copy filled (the ring slot's label, 16 bytes after its count).
bool FenceOfCopy(VAddr fence, const GdsCopy& copy) {
    return fence >= copy.addr && fence < copy.addr + 64;
}

} // namespace

bool BufferCache::GdsFencesEnabled() {
    static const bool on = [] {
        const char* env = std::getenv("BB_GDS_FENCE");
        return !(env && env[0] == '0') && Mode() != 0 && ReadbackHybrid::Enabled() && !ReadbackHybrid::GdsExact();
    }();
    return on;
}

// GPU thread, right after the GDS-to-memory copy was recorded (dst_buffer holds the fresh bytes at dst_offset once it
// runs): the bytes are copied on into staging in the same submission.
void BufferCache::NoteGdsCopy(const Buffer* dst_buffer, u64 dst_offset, VAddr dst, u32 size) {
    if (!GdsFencesEnabled() || size == 0) {
        return;
    }
    const auto staging = staging_pool.Request(size, MemoryType::HostCached, 0, true);
    const vk::BufferCopy copy = {
        .srcOffset = dst_offset,
        .dstOffset = staging.offset,
        .size = size,
    };
    runtime.CopyBuffer(dst_buffer, staging.buffer, std::span{&copy, 1});
    std::scoped_lock lk{g_mutex};
    g_pending.push_back({staging, dst, size, scheduler.CurrentTick(), std::chrono::steady_clock::now()});
}

// The EOP fence at `fence` that follows GDS copies: true when it is taken over (signal runs once the copies are done
// and their bytes are in guest memory); false when no copy is pending, or the fence is not a copy's own label, and
// the caller writes it now as usual (the worker still writes the copies' bytes back).
bool BufferCache::DeferGdsFence(VAddr fence, std::function<void()> signal) {
    if (!GdsFencesEnabled()) {
        return false;
    }
    std::unique_lock lk{g_mutex};
    if (g_pending.empty()) {
        return false;
    }
    bool defer = Mode() == 2;
    for (const GdsCopy& copy : g_pending) {
        defer |= Mode() != 3 && FenceOfCopy(fence, copy);
    }
    g_queue.push_back({std::move(g_pending), defer ? std::move(signal) : std::function<void()>{[] {}}, defer,
                       std::chrono::steady_clock::now()});
    g_pending.clear();
    if (!g_worker_started) {
        g_worker_started = true;
        std::thread([this] { GdsFenceWorker(); }).detach();
    }
    lk.unlock();
    g_cv.notify_one();
    return defer;
}

void BufferCache::GdsFenceWorker() {
    Common::SetCurrentThreadName("GdsFences");
    u64 fences = 0, copies = 0, latency_us = 0, max_us = 0;
    auto report = std::chrono::steady_clock::now();
    for (;;) {
        DeferredFence fence;
        {
            std::unique_lock lk{g_mutex};
            g_cv.wait(lk, [] { return !g_queue.empty(); });
            fence = std::move(g_queue.front());
            g_queue.pop_front();
        }
        for (GdsCopy& copy : fence.copies) {
            scheduler.GetWorkSemaphore()->Wait(copy.tick);
            copy.staging.Invalidate();
            // The GPU's count is the newer value whatever the game wrote to the slot before (it resets a slot, then
            // submits the copy): written as is, and into a fast page's shadow, which a merge would have kept the
            // game's zero over (small_writeback.cpp MergeCopyBack; the cabin vanished for a frame).
            memory->TryWriteBacking(std::bit_cast<u8*>(copy.addr), copy.staging.mapped, copy.size);
            for (u64 k = 0; k < copy.size;) {
                const VAddr at = copy.addr + k;
                const VAddr page = at & ~VAddr(4095);
                const u64 seg = std::min<u64>(copy.size - k, page + 4096 - at);
                ReadbackHybrid::WithShadow(page, [&](u8* shadow) { std::memcpy(shadow + (at - page), copy.staging.mapped + k, seg); });
                k += seg;
            }
            liverpool->SendCommand([this, staging = copy.staging] { staging_pool.FreeDeferred(staging); });
        }
        fence.signal();
        if (Log() && fence.deferred) {
            const auto now = std::chrono::steady_clock::now();
            const u64 us = std::chrono::duration_cast<std::chrono::microseconds>(now - fence.decoded).count();
            ++fences;
            copies += fence.copies.size();
            latency_us += us;
            max_us = std::max(max_us, us);
            if (now - report >= std::chrono::seconds(5)) {
                std::printf("GDS fences: %llu deferred (%llu copies), label %.1f ms after decode on average, max %.1f ms (5 s)\n",
                            (unsigned long long)fences, (unsigned long long)copies, latency_us / 1000.0 / fences,
                            max_us / 1000.0);
                std::fflush(stdout);
                fences = copies = latency_us = max_us = 0;
                report = now;
            }
        }
    }
}

} // namespace VideoCore
