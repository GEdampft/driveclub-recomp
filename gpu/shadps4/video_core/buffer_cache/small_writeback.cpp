// SPDX-License-Identifier: GPL-2.0-or-later
// VR fork: small GPU writes written back to guest memory without readbacks.
//
// VR runs with readbacks off (BB_READBACKS=0): the game keeps CPU data in the same 4 KB pages as a few tiny GPU writes,
// and precise readbacks made the CPU wait for the GPU about once a frame (10-17% of the frame rate). With them off the
// CPU never sees those GPU writes, and in Driveclub's first race the picture went black at dawn (about 45 s in) until
// a restart. Small written ranges (bindings of up to 1 KB) are copied out at each submit instead, and written into
// guest memory once the GPU has finished them (no page protection, no waiting): the CPU reads them a frame or so late.
// BB_READBACK_SMALL=0 turns it off; BB_READBACK_SMALL_LOG=1 prints the ranges and bytes per second.

#include <algorithm>
#include <bit>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <mutex>
#include <vector>

#include "core/emulator_settings.h"
#include "core/memory.h"
#include "video_core/buffer_cache/buffer.h"
#include "video_core/buffer_cache/buffer_cache.h"
#include "video_core/renderer_vulkan/vk_runtime.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/renderer_vulkan/vk_staging_buffer_pool.h"

namespace VideoCore {
namespace {

constexpr u64 MaxSmallWrite = 1024;

struct Range {
    VAddr addr;
    u64 size;
};

struct Job {
    Vulkan::StagingBufferRef staging;
    std::vector<Range> ranges; // guest ranges, in staging order (each at a 16-byte aligned offset)
    std::vector<u64> offsets;
    u64 tick;
};

std::mutex g_mutex;
std::vector<Range> g_noted;
std::deque<Job> g_jobs;

bool Enabled() {
    static const bool enabled = [] {
        const char* env = std::getenv("BB_READBACK_SMALL");
        return EmulatorSettings.GetReadbacksMode() == GpuReadbacksMode::Disabled && !(env && env[0] == '0');
    }();
    return enabled;
}

} // namespace

void BufferCache::NoteSmallGpuWrite(VAddr device_addr, u64 size) {
    if (size == 0 || size > MaxSmallWrite || !Enabled()) {
        return;
    }
    std::scoped_lock lk{g_mutex};
    g_noted.push_back({device_addr, size});
}

void BufferCache::WriteBackSmallWrites() {
    if (!Enabled()) {
        return;
    }
    static const bool log = std::getenv("BB_READBACK_SMALL_LOG") != nullptr;
    static u64 log_ranges = 0, log_bytes = 0, log_submits = 0;
    static auto log_start = std::chrono::steady_clock::now();
    // Finished copies: their bytes into guest memory.
    while (!g_jobs.empty() && scheduler.IsFree(g_jobs.front().tick)) {
        Job& job = g_jobs.front();
        job.staging.buffer->Invalidate(job.staging.offset, job.staging.size);
        for (size_t i = 0; i < job.ranges.size(); ++i) {
            memory->TryWriteBacking(std::bit_cast<u8*>(job.ranges[i].addr), job.staging.mapped + job.offsets[i],
                                    job.ranges[i].size);
        }
        staging_pool.FreeDeferred(job.staging);
        g_jobs.pop_front();
    }
    std::vector<Range> ranges;
    {
        std::scoped_lock lk{g_mutex};
        ranges.swap(g_noted);
    }
    if (ranges.empty()) {
        return;
    }
    std::ranges::sort(ranges, [](const Range& a, const Range& b) {
        return a.addr != b.addr ? a.addr < b.addr : a.size > b.size;
    });
    // One copy per distinct start (the largest size there); ranges inside an earlier one are skipped.
    Job job{};
    VAddr covered_end = 0;
    u64 total = 0;
    for (const Range& range : ranges) {
        if (range.addr + range.size <= covered_end) {
            continue;
        }
        job.ranges.push_back(range);
        job.offsets.push_back(total);
        total += (range.size + 15) & ~u64{15};
        covered_end = std::max(covered_end, range.addr + range.size);
    }
    job.staging = staging_pool.Request(total, MemoryType::HostCached, 0, true);
    // Copies grouped by arena (a range never crosses one: it was bound from one).
    size_t i = 0;
    while (i < job.ranges.size()) {
        const auto* arena = GetArena(job.ranges[i].addr >> block_shift, job.ranges[i].addr >> block_shift);
        DownloadCopies copies;
        for (; i < job.ranges.size(); ++i) {
            const Range& range = job.ranges[i];
            if (range.addr < arena->cpu_addr || range.addr + range.size > arena->cpu_addr + arena->size_bytes) {
                break;
            }
            copies.push_back(vk::BufferCopy{
                .srcOffset = range.addr - arena->cpu_addr,
                .dstOffset = job.staging.offset + job.offsets[i],
                .size = range.size,
            });
        }
        if (copies.empty()) { // outside every arena: dropped
            job.ranges.erase(job.ranges.begin() + i);
            job.offsets.erase(job.offsets.begin() + i);
            continue;
        }
        runtime.CopyBuffer(arena, job.staging.buffer, copies);
    }
    job.tick = scheduler.CurrentTick();
    if (log) {
        log_ranges += job.ranges.size();
        log_bytes += total;
        ++log_submits;
        const auto now = std::chrono::steady_clock::now();
        if (now - log_start >= std::chrono::seconds(5)) {
            const double s = std::chrono::duration<double>(now - log_start).count();
            std::printf("Small GPU writes written back: %.0f ranges/s, %.0f bytes/s, %.0f submits/s; first %#llx +%llu\n",
                        log_ranges / s, log_bytes / s, log_submits / s, (unsigned long long)job.ranges[0].addr,
                        (unsigned long long)job.ranges[0].size);
            std::fflush(stdout);
            log_ranges = log_bytes = log_submits = 0;
            log_start = now;
        }
    }
    g_jobs.push_back(std::move(job));
}

} // namespace VideoCore
