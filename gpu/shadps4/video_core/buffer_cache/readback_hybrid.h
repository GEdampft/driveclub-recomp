// SPDX-License-Identifier: GPL-2.0-or-later
// VR fork: precise readbacks without most of their cost (BB_READBACK_HYBRID, on by default with precise readbacks).
//
// With precise readbacks every page the GPU writes is protected, and a guest access to it waits for the GPU to finish
// the work that wrote it. Driveclub keeps its own hot data in the same 4 KB pages as some of the GPU's output: small
// counters and flags, and the partial first and last pages of larger buffers. Those accesses never touch a GPU byte,
// yet each waited for the GPU (about 350 waits a second at 3200 per eye, 15-20% of the frame rate).
//
// Pages whose guest faults only ever hit the game's own bytes are learned (LearnFault) and marked "fast": they stay
// GPU modified but are not protected (RegionManager::small_gpu), and small_writeback.cpp copies their GPU bytes back
// after each submit, as with readbacks off. Written ranges of up to 1 KB are fast from the start. A page where a guest
// access touched GPU bytes (the crowd's ground heights) stays exact for good. BB_READBACK_HYBRID=0 turns it off;
// BB_READBACK_CLASSIFY=1 logs the faults per page.
//
// Driveclub's page 0x400220f000 is exact from the start (DefaultExact): its 4-byte GPU flag at 0x400220ff50, which
// the game reads once a slot's label from two frames back has passed (game +0x758a2d), arrived a submit or more late
// when fast, and grass clumps were drawn floating in random places. The game's own bytes on that page fault too, but
// cheaply: its GPU work has long finished by then (no frame rate change). 0x300042f000 (a 1 KB range read right
// after the GPU writes it) is left fast: exact, each read waited for the GPU (72 to 50 FPS at 3200 per eye).

#pragma once

#include <array>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <mutex>
#include <set>
#include <vector>

#include "common/types.h"
#include "core/emulator_settings.h"
#include "video_core/buffer_cache/range_set.h"

namespace VideoCore::ReadbackHybrid {

/// Written ranges up to this size are fast from the start (counters, flags).
constexpr u64 SmallWrite = 1024;
/// A page is learned as fast after this many guest faults, none of them on GPU bytes.
constexpr u32 LearnFaults = 32;

inline bool Enabled() {
    static const bool on = [] {
        const char* v = std::getenv("BB_READBACK_HYBRID");
        return !(v && v[0] == '0') && EmulatorSettings.GetReadbacksMode() != GpuReadbacksMode::Disabled;
    }();
    return on;
}

// Development switches: BB_HYBRID_SMALL=0 (small ranges are not fast from the start), BB_HYBRID_LEARN=0 (no page is
// learned fast), BB_HYBRID_EXACT=<hex page,...> (these pages are never fast, instead of DefaultExact; 0 for none).
constexpr VAddr DefaultExact = 0x400220f000;
inline bool SmallFast() {
    static const bool on = !(std::getenv("BB_HYBRID_SMALL") && std::getenv("BB_HYBRID_SMALL")[0] == '0');
    return on;
}
inline bool Learning() {
    static const bool on = !(std::getenv("BB_HYBRID_LEARN") && std::getenv("BB_HYBRID_LEARN")[0] == '0');
    return on;
}
inline bool ExactPage(VAddr address) {
    static const std::vector<VAddr> pages = [] {
        std::vector<VAddr> list;
        const char* v = std::getenv("BB_HYBRID_EXACT");
        if (!v) {
            list.push_back(DefaultExact);
        }
        while (v && *v) {
            char* end = nullptr;
            const unsigned long long page = std::strtoull(v, &end, 16);
            if (end == v) {
                break;
            }
            list.push_back(VAddr(page) & ~VAddr(4095));
            v = *end ? end + 1 : end;
        }
        return list;
    }();
    if (pages.empty()) {
        return false;
    }
    const VAddr page = address & ~VAddr(4095);
    for (const VAddr p : pages) {
        if (p == page) {
            return true;
        }
    }
    return false;
}

/// BB_READBACK_CLASSIFY: the pages a small range makes fast, each the first time.
inline void NoteSmallFast(VAddr address, u64 size) {
    static const bool log = std::getenv("BB_READBACK_CLASSIFY") != nullptr;
    if (!log) {
        return;
    }
    static std::mutex m;
    static std::set<VAddr> seen;
    std::scoped_lock lk(m);
    if (seen.size() < 4000 && seen.insert(address & ~VAddr(4095)).second) {
        std::printf("Readback hybrid: small range %#llx +%llu fast (page %#llx)\n", (unsigned long long)address,
                    (unsigned long long)size, (unsigned long long)(address & ~VAddr(4095)));
        std::fflush(stdout);
    }
}

/// A small range kept exact by ExactPage, logged the first time per page (the log shows DefaultExact still matches).
inline void NoteSmallExact(VAddr address, u64 size) {
    static std::mutex m;
    static std::set<VAddr> seen;
    std::scoped_lock lk(m);
    if (seen.size() < 16 && seen.insert(address & ~VAddr(4095)).second) {
        std::printf("Readback hybrid: small range %#llx +%llu exact (page %#llx)\n", (unsigned long long)address,
                    (unsigned long long)size, (unsigned long long)(address & ~VAddr(4095)));
        std::fflush(stdout);
    }
}

// The GPU's written bytes not yet read back (BufferCache::gpu_modified_ranges, copied under a lock for guest threads).
inline std::mutex g_written_mutex;
inline RangeSet g_written;

inline void NoteGpuWritten(VAddr address, u64 size, bool written) {
    if (!size || !Enabled()) {
        return;
    }
    std::scoped_lock lk{g_written_mutex};
    if (written) {
        g_written.Add(address, size);
    } else {
        g_written.Subtract(address, size);
    }
}

inline bool GpuWroteBytes(VAddr address, u64 size) {
    std::scoped_lock lk{g_written_mutex};
    return g_written.Intersects(address, size);
}

// Learned fast pages: an open-addressed table of page numbers (+1), read without a lock by the GPU thread.
constexpr size_t TableSize = 1024;
inline std::array<std::atomic<u64>, TableSize> g_fast{};

inline bool IsFastPage(VAddr address) {
    if (ExactPage(address)) {
        return false;
    }
    const u64 key = (address >> 12) + 1;
    for (size_t i = 0, slot = (key * 0x9E3779B97F4A7C15ull) >> 54; i < TableSize; ++i, slot = (slot + 1) % TableSize) {
        const u64 v = g_fast[slot].load(std::memory_order_relaxed);
        if (v == key) {
            return true;
        }
        if (v == 0) {
            return false;
        }
    }
    return false;
}

inline void AddFastPage(VAddr address) {
    const u64 key = (address >> 12) + 1;
    for (size_t i = 0, slot = (key * 0x9E3779B97F4A7C15ull) >> 54; i < TableSize; ++i, slot = (slot + 1) % TableSize) {
        u64 expected = 0;
        if (g_fast[slot].compare_exchange_strong(expected, key) || expected == key) {
            return;
        }
    }
}

/// A guest fault on a GPU modified page (a read, or a write that would read back first): learns the page as fast
/// once enough faults hit only the game's own bytes; a fault on GPU bytes keeps it exact.
inline void LearnFault(VAddr address, bool write) {
    if (!Enabled()) {
        return;
    }
    const VAddr page = address & ~VAddr(4095);
    // The bytes the instruction accesses: from the fault address (its first byte), at most 32 (AVX), on this page.
    const VAddr begin = address;
    const VAddr end = address + 32 > page + 4096 ? page + 4096 : address + 32;
    const bool gpu = GpuWroteBytes(begin, end - begin);
    static const bool log = std::getenv("BB_READBACK_CLASSIFY") != nullptr;
    static std::mutex m;
    struct Count {
        u32 own = 0, gpu = 0;
        bool exact = false;
    };
    static std::map<VAddr, Count> pages;
    static auto start = std::chrono::steady_clock::now();
    std::scoped_lock lk(m);
    Count& c = pages[page];
    if (gpu) {
        ++c.gpu;
        c.exact = true;
    } else if (++c.own == LearnFaults && !c.exact && Learning() && !ExactPage(page)) {
        AddFastPage(page);
        if (log) {
            std::printf("Readback hybrid: page %#llx learned fast (%s faults)\n", (unsigned long long)page,
                        write ? "write" : "read");
            std::fflush(stdout);
        }
    }
    const auto now = std::chrono::steady_clock::now();
    if (log && now - start > std::chrono::seconds(5)) {
        std::printf("Readback hybrid: guest faults by page (own/GPU bytes):");
        for (const auto& [p, n] : pages) {
            std::printf(" %#llx=%u/%u%s", (unsigned long long)p, n.own, n.gpu, IsFastPage(p) ? "*" : "");
        }
        std::printf("\n");
        std::fflush(stdout);
        start = now;
    }
}

} // namespace VideoCore::ReadbackHybrid
