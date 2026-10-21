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
#include "video_core/buffer_cache/readback_hybrid.h"
#include "video_core/renderer_vulkan/vk_runtime.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/renderer_vulkan/vk_staging_buffer_pool.h"

namespace VideoCore {
namespace {

// Development: BB_READBACK_SMALL_MAX=<KB> raises the size limit.
const u64 MaxSmallWrite = [] {
    const char* env = std::getenv("BB_READBACK_SMALL_MAX");
    return env ? std::strtoull(env, nullptr, 10) * 1024 : u64(1024);
}();

struct Range {
    VAddr addr;
    u64 size;
    bool late = false; ///< readback_hybrid.h: a late output's bytes, written back on any page
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
        return (EmulatorSettings.GetReadbacksMode() == GpuReadbacksMode::Disabled || ReadbackHybrid::Enabled()) &&
               !(env && env[0] == '0');
    }();
    return enabled;
}

} // namespace

void BufferCache::NoteSmallGpuWrite(VAddr device_addr, u64 size) {
    // Development (BB_READBACK_SMALL_PAGES=<hex page>,...): GPU-written ranges touching those 4 KB pages, once each.
    static const std::vector<VAddr> watch_pages = [] {
        std::vector<VAddr> pages;
        if (const char* env = std::getenv("BB_READBACK_SMALL_PAGES")) {
            for (const char* s = env; *s;) {
                char* end = nullptr;
                const VAddr v = std::strtoull(s, &end, 16);
                if (end == s) {
                    ++s;
                    continue;
                }
                pages.push_back(v & ~VAddr{0xfff});
                s = end;
            }
        }
        return pages;
    }();
    if (!watch_pages.empty() && size) {
        static std::mutex seen_mutex;
        static std::vector<std::pair<VAddr, u64>> seen;
        for (const VAddr page : watch_pages) {
            if (device_addr < page + 0x1000 && page < device_addr + size) {
                std::scoped_lock lk{seen_mutex};
                if (std::find(seen.begin(), seen.end(), std::pair{device_addr, size}) == seen.end() && seen.size() < 400) {
                    seen.emplace_back(device_addr, size);
                    std::printf("Small GPU write in a watched page: %#llx +%#llx\n", (unsigned long long)device_addr,
                                (unsigned long long)size);
                    std::fflush(stdout);
                }
                break;
            }
        }
    }
    if (size == 0 || !Enabled()) {
        return;
    }
    if (ReadbackHybrid::t_late_writer) { // readback_hybrid.h: its bytes on its first and last pages
        const VAddr end = device_addr + size;
        const VAddr first_end = std::min(end, (device_addr | 4095) + 1);
        std::scoped_lock lk{g_mutex};
        g_noted.push_back({device_addr, first_end - device_addr, true});
        if (end > first_end) {
            const VAddr last_begin = std::max(first_end, (end - 1) & ~VAddr(4095));
            g_noted.push_back({last_begin, end - last_begin, true});
        }
        return;
    }
    if (size > MaxSmallWrite) {
        // readback_hybrid.h: a larger range's first and last pages when learned fast (not protected): their GPU bytes.
        if (!ReadbackHybrid::Enabled()) {
            return;
        }
        const VAddr end = device_addr + size;
        const VAddr first_end = std::min(end, (device_addr | 4095) + 1);
        const VAddr last_begin = std::max(first_end, (end - 1) & ~VAddr(4095));
        std::scoped_lock lk{g_mutex};
        if (ReadbackHybrid::IsFastPage(device_addr)) {
            g_noted.push_back({device_addr, first_end - device_addr});
        }
        if (last_begin < end && last_begin >= first_end && ReadbackHybrid::IsFastPage(last_begin)) {
            g_noted.push_back({last_begin, end - last_begin});
        }
        return;
    }
    // readback_hybrid.h: only ranges on its fast pages.
    if (ReadbackHybrid::Enabled() && !ReadbackHybrid::SmallFast() &&
        !(ReadbackHybrid::IsFastPage(device_addr) &&
          ((device_addr ^ (device_addr + size - 1)) < 4096 || ReadbackHybrid::IsFastPage(device_addr + size - 1)))) {
        return;
    }
    std::scoped_lock lk{g_mutex};
    g_noted.push_back({device_addr, size});
}

// readback_hybrid.h: a fast page's GPU bytes go into guest memory only where the GPU changed them since the shadow
// and the game did not (both changed: the game's value stays and goes up with the next upload). Other pages: all of
// them with `other_pages` (a readback), none without (a copy-back: a page no longer fast gets its bytes by readback).
void BufferCache::MergeCopyBack(VAddr addr, const u8* gpu, u64 size, bool other_pages) {
    // Page by page: pages that are not fast take the GPU's bytes as they are (or none), a fast page merges.
    for (u64 k = 0; k < size;) {
        const VAddr at = addr + k;
        const VAddr page = at & ~VAddr(4095);
        const u64 seg = std::min<u64>(size - k, page + 4096 - at);
        u8 guest[4096], merged[4096];
        bool fast = false;
        if (ReadbackHybrid::IsFastPage(page)) {
            memory->CopySparseMemory(at, guest, seg);
            std::memcpy(merged, guest, seg);
            fast = ReadbackHybrid::WithShadow(page, [&](u8* shadow) {
                u8* sh = shadow + (at - page);
                for (u64 j = 0; j < seg; ++j) {
                    const u8 v = gpu[k + j];
                    if (v != sh[j]) {
                        if (guest[j] == sh[j]) {
                            merged[j] = v;
                        }
                        sh[j] = v;
                    }
                }
            });
        }
        if (fast) {
            for (u64 j = 0; j < seg;) {
                if (merged[j] == guest[j]) {
                    ++j;
                    continue;
                }
                u64 e = j;
                while (e < seg && merged[e] != guest[e]) {
                    ++e;
                }
                memory->TryWriteBacking(std::bit_cast<u8*>(at + j), merged + j, e - j);
                j = e;
            }
        } else if (other_pages) {
            memory->TryWriteBacking(std::bit_cast<u8*>(at), gpu + k, seg);
        }
        k += seg;
    }
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
            const u8* gpu = job.staging.mapped + job.offsets[i];
            if (!ReadbackHybrid::Enabled()) {
                memory->TryWriteBacking(std::bit_cast<u8*>(job.ranges[i].addr), gpu, job.ranges[i].size);
                continue;
            }
            MergeCopyBack(job.ranges[i].addr, gpu, job.ranges[i].size, job.ranges[i].late);
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
            job.ranges.back().late |= range.late;
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
