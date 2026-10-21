// SPDX-License-Identifier: GPL-2.0-or-later
// VR fork: precise readbacks without most of their cost (BB_READBACK_HYBRID; on with precise readbacks).
//
// With precise readbacks every page the GPU writes is protected, and a guest access to it waits for the GPU to finish
// the work that wrote it. Driveclub keeps its own hot data in the same 4 KB pages as some of the GPU's output: small
// counters and flags, and the partial first and last pages of larger buffers. Those accesses never touch a GPU byte,
// yet each waited for the GPU (about 350 waits a second at 3200 per eye, 15-20% of the frame rate).
//
// A guest access that touches only the game's own bytes on such a page (its size decoded from the instruction) needs
// no GPU data: it runs with the page open for that one instruction (trap flag, PageManager::StepOwnAccess) and the
// protection is put back, with no readback and no wait. A write marks the page CPU modified as usual. A read of GPU
// bytes reads back and waits, and makes the page exact for good; an exact page steps 64 own-byte accesses in a row,
// then reads back once.
//
// Stepping costs two exceptions and two protection changes, too much for the hottest pages (~165k accesses a second at
// the race start: 20 FPS). Because every access to a protected page is seen, a page is made fast (unprotected, its GPU
// bytes copied back after each submit by small_writeback.cpp) after 2048 own-byte accesses over 4 s with no read of
// GPU bytes (LearnStep), right after a readback that gives its shadow.
//
// The game and the GPU write some of the same bytes (counters the game resets and the GPU advances), so on a fast page
// neither side's copy may overwrite the other's. The shadow holds the bytes as last agreed: copy-backs and readbacks
// write only bytes the GPU changed since it and the game did not (BufferCache::MergeCopyBack), uploads send only GPU
// bytes the game changed since it (the rest of the page goes up as usual), and on exact pages uploads leave out the
// GPU's bytes not yet read back (ForEachNotGpuWritten).
//
// The first version (3e3b4c7) learned from faults alone (only the first access after each GPU write faults), made
// ranges up to 1 KB fast from the start and copied whole ranges back. Labels are written when the work is recorded, not
// when the GPU finishes it, so the game read GPU data on fast pages stale (grass clumps floated: a GPU flag at
// 0x400220ff50), and the copy-backs put the GPU's copy over values the game wrote: 300 such clobbers before a crash
// (game +0x8825a0 walking a vector whose begin pointer had been zeroed, 0x3000436470), on race load, on quitting a race
// and in the menus. Development switches: BB_READBACK_HYBRID=0, BB_HYBRID_SMALL=1 and BB_HYBRID_LEARN=1 (the first
// version's fast pages), BB_HYBRID_STEP_LEARN=0, BB_HYBRID_CERTIFY=1 (demote a page on a GPU range not seen when it was
// learned), BB_READBACK_CLASSIFY=1 (faults, steps, learned and exact pages).

#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
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

// On by default (BB_READBACK_HYBRID=0 turns it off): the pages GDS counters are copied to (the flags the game polls,
// the grass clumps' among them) are exact for good (MarkGdsCopyTarget), and the grass clumps are not drawn in VR
// (vr_stereo.cpp). A page learned fast can still hold other GPU bytes the game reads later; nothing seen from that.
inline bool Enabled() {
    static const bool on = [] {
        const char* v = std::getenv("BB_READBACK_HYBRID");
        return !(v && v[0] == '0') && EmulatorSettings.GetReadbacksMode() != GpuReadbacksMode::Disabled;
    }();
    return on;
}

// Development switches: BB_HYBRID_SMALL=1 (small ranges are fast from the start: the game read GPU flags there stale,
// floating grass clumps), BB_HYBRID_LEARN=0 (no page is learned fast), BB_HYBRID_EXACT=<hex page,...> (these pages
// are never fast).
inline bool SmallFast() {
    static const bool on = std::getenv("BB_HYBRID_SMALL") && std::getenv("BB_HYBRID_SMALL")[0] == '1';
    return on;
}
inline bool Learning() {
    static const bool on = std::getenv("BB_HYBRID_LEARN") && std::getenv("BB_HYBRID_LEARN")[0] == '1';
    return on;
}
inline bool ExactPage(VAddr address) {
    static const std::vector<VAddr> pages = [] {
        std::vector<VAddr> list;
        const char* v = std::getenv("BB_HYBRID_EXACT");
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

/// The parts of [address, address + size) without GPU bytes still to be read back. An upload of a CPU modified fast
/// page leaves the GPU's bytes alone: guest memory may hold older ones (they come back after the GPU finishes), and
/// uploading them would undo the GPU's writes. The game does not write them: a page is learned fast only when its
/// faults hit the game's own bytes.
template <typename Func>
inline void ForEachNotGpuWritten(VAddr address, u64 size, Func&& func) {
    std::scoped_lock lk{g_written_mutex};
    g_written.ForEachNotInRange(address, size, func);
}

/// The size of the faulting access, from its instruction (PageManager); 32 (an AVX access) when unknown.
inline thread_local u32 t_access_size = 32;

// Late outputs: a few shaders write small results the game reads on the CPU every frame or two, and each read was a
// wait for the GPU to finish the newest work (~120 waits a second, ~0.45 s of game-thread waiting a second at 3200).
// The game only steers by them (Driveclub: 0xeea18ba7 the scene's average luminance for auto exposure, 0xf558c2c1
// two probe positions, 0x832138ea/0x89ea5c34 and 0x2f6a12bc small lighting tables), so a frame-old value does: their
// bytes count as the game's own (an access to only those is stepped, no readback), and they are copied back into
// guest memory once the GPU finishes them (small_writeback.cpp), like a fast page's but without making the page fast:
// other GPU bytes on it (GDS counters) stay exact. A write by any other GPU work over such bytes makes them exact
// again. Ignoring the four outputs entirely gave 84 FPS against 79.5 at 3200. BB_HYBRID_LATE=<hex,...> replaces the
// list, 0 turns it off.
inline bool LateWriter(u64 shader_hash) {
    static const std::vector<u64> hashes = [] {
        std::vector<u64> list;
        const char* v = std::getenv("BB_HYBRID_LATE");
        if (!v) {
            v = "eea18ba7,f558c2c1,832138ea,89ea5c34,2f6a12bc";
        }
        while (*v) {
            char* end = nullptr;
            const u64 h = std::strtoull(v, &end, 16);
            if (end == v) {
                break;
            }
            if (h) {
                list.push_back(h);
            }
            v = *end ? end + 1 : end;
        }
        return list;
    }();
    return Enabled() && std::find(hashes.begin(), hashes.end(), shader_hash) != hashes.end();
}
/// Set by the rasterizer around a late output's ObtainBuffer.
inline thread_local bool t_late_writer = false;
inline std::mutex g_late_mutex;
inline std::map<VAddr, std::vector<std::pair<u16, u16>>> g_late; // page -> late output ranges [begin, end) on it
inline std::atomic<u32> g_late_pages{0};

/// Every GPU-written range (BufferCache::ObtainBuffer): a late output's ranges on its first and last pages are noted
/// (the pages between are read back as usual); any other write takes the noted ranges it overlaps away.
inline void NoteWriter(VAddr address, u64 size) {
    if (!size || (!t_late_writer && g_late_pages.load(std::memory_order_relaxed) == 0)) {
        return;
    }
    const VAddr end = address + size;
    std::scoped_lock lk{g_late_mutex};
    if (t_late_writer) {
        const auto note = [&](VAddr begin, VAddr stop) {
            const VAddr page = begin & ~VAddr(4095);
            auto& ranges = g_late[page];
            const std::pair<u16, u16> r{u16(begin - page), u16(stop - page)}; // an end of 4096 fits
            if (std::find(ranges.begin(), ranges.end(), r) == ranges.end() && ranges.size() < 16) {
                ranges.push_back(r);
            }
        };
        const VAddr first_end = std::min(end, (address | 4095) + 1);
        note(address, first_end);
        if (end > first_end) {
            note(std::max(first_end, (end - 1) & ~VAddr(4095)), end);
        }
    } else {
        for (auto it = g_late.lower_bound(address & ~VAddr(4095)); it != g_late.end() && it->first < end;) {
            const VAddr page = it->first;
            std::erase_if(it->second, [&](const std::pair<u16, u16>& r) {
                return page + r.first < end && address < page + r.second;
            });
            it = it->second.empty() ? g_late.erase(it) : std::next(it);
        }
    }
    g_late_pages.store(u32(g_late.size()), std::memory_order_relaxed);
}

/// [begin, end) (within one page) lies in a late output's range.
inline bool LateBytes(VAddr begin, VAddr end) {
    if (g_late_pages.load(std::memory_order_relaxed) == 0) {
        return false;
    }
    const VAddr page = begin & ~VAddr(4095);
    std::scoped_lock lk{g_late_mutex};
    const auto it = g_late.find(page);
    if (it == g_late.end()) {
        return false;
    }
    for (const auto& [b, e] : it->second) {
        if (begin - page >= b && end - page <= e) {
            return true;
        }
    }
    return false;
}

/// The bytes a faulting access touches (from its first byte, t_access_size, on its page) hold no GPU data still to be
/// read back, or only late outputs.
inline bool OwnBytes(VAddr address) {
    const VAddr end = std::min<VAddr>(address + t_access_size, (address & ~VAddr(4095)) + 4096);
    return !GpuWroteBytes(address, end - address) || LateBytes(address, end);
}

/// Set by the fault handler around a guest write to its own bytes: the tracker marks the page CPU modified without a
/// readback (MemoryTracker::InvalidateRegion).
inline thread_local bool t_own_write = false;

/// Pages learned from stepped accesses (PageManager::StepOwnAccess): one is made fast after StepLearnAccesses own-byte
/// accesses over at least StepLearnMs with no access to GPU bytes (MarkExact). Every access to the page is seen while
/// it learns (each faults), unlike the first version's learning, which saw only the first access after a GPU write.
// 2048 over 4 s (2026-10-21 night) stepped a hot page 150k times before trusting it (31k trap-flag exceptions a
// second on the game thread, the GPU thread stalled by the protection changes): no faster than the hybrid off on a
// 7-car grid. vr1.1's hybrid trusted a page after 32 faults at once (64-67 FPS there): back to that, with the pages
// GDS counters are copied to (MarkGdsCopyTarget) exact for good. BB_HYBRID_LEARN_STEPS / BB_HYBRID_LEARN_MS override.
inline const u64 StepLearnAccesses = std::getenv("BB_HYBRID_LEARN_STEPS") ? std::strtoull(std::getenv("BB_HYBRID_LEARN_STEPS"), nullptr, 10) : 64;
inline const s64 StepLearnMs = std::getenv("BB_HYBRID_LEARN_MS") ? std::atoll(std::getenv("BB_HYBRID_LEARN_MS")) : 0;
/// An exact page (the game reads its GPU bytes too) steps at most this many own-byte accesses in a row; the next one
/// reads back instead (one wait), and the page needs nothing until the GPU writes it again.
constexpr u32 ExactStepBudget = 64;
struct PageLearn {
    u64 steps = 0;
    s64 first_ms = 0;
    u32 exact_steps = 0;
    bool exact = false;
    bool fast = false;
};
inline std::mutex g_learn_mutex;
inline std::map<VAddr, PageLearn> g_learn;

/// The faulting instruction (PageManager), for the log.
inline thread_local u64 t_fault_rip = 0;

/// A page the GPU copies GDS counters into (Rasterizer::CopyBuffer from GDS): the flags and counters the game polls
/// right after the GPU writes them (the grass flag at 0x400220ff50 among them). Such a page is never learned fast: a
/// page learned while the game was not reading its GPU bytes (the countdown on the grid) read them a frame late once
/// it did (floating grass clumps in the sky past the start). Demotes a page already fast.
inline void DemoteFastPage(VAddr page, VAddr begin, VAddr end);
// On by default (BB_HYBRID_GDS_EXACT=0 serves them a frame late, as vr1.1 did): read late, every vegetation type
// that takes its clusters' visibility bits from the counter floats (Matt, 2026-10-21 afternoon, one corner in: the
// grass clump skip covers one shader only). Each read is a readback wait; Rasterizer::CopyBuffer submits right after
// a GDS copy so that wait covers the copy, not the frame queued behind it.
// TEMP measurement: the scheduler tick of the latest GDS copy into each page, and how the game's polls find it.
inline std::mutex g_gds_mutex;
inline std::map<VAddr, u64> g_gds_tick;
inline std::atomic<u64> g_gds_poll_done{0}, g_gds_poll_wait{0};
inline void NoteGdsCopyTick(VAddr address, u64 size, u64 tick) {
    std::scoped_lock lk{g_gds_mutex};
    for (VAddr page = address & ~VAddr(4095); page < address + size; page += 4096) {
        g_gds_tick[page] = tick;
    }
}
/// The tick of the latest GDS copy into the page of `address`, 0 when none.
inline u64 GdsCopyTick(VAddr address) {
    std::scoped_lock lk{g_gds_mutex};
    const auto it = g_gds_tick.find(address & ~VAddr(4095));
    return it == g_gds_tick.end() ? 0 : it->second;
}

inline void MarkGdsCopyTarget(VAddr address, u64 size) {
    static const bool exact = !(std::getenv("BB_HYBRID_GDS_EXACT") && std::getenv("BB_HYBRID_GDS_EXACT")[0] == '0');
    if (!exact || !Enabled() || !size) {
        return;
    }
    static const bool log = std::getenv("BB_READBACK_CLASSIFY") != nullptr;
    for (VAddr page = address & ~VAddr(4095); page < address + size; page += 4096) {
        bool was_fast = false;
        {
            std::scoped_lock lk{g_learn_mutex};
            PageLearn& p = g_learn[page];
            was_fast = p.fast;
            if (!p.exact && log) {
                std::printf("Readback hybrid: page %#llx exact: GDS counters are copied to %#llx +%llu\n",
                            (unsigned long long)page, (unsigned long long)address, (unsigned long long)size);
                std::fflush(stdout);
            }
            p.exact = true;
        }
        if (was_fast) {
            DemoteFastPage(page, address, address + size);
        }
    }
}

inline void MarkExact(VAddr address, bool write) {
    std::scoped_lock lk{g_learn_mutex};
    PageLearn& p = g_learn[address & ~VAddr(4095)];
    static const bool log = std::getenv("BB_READBACK_CLASSIFY") != nullptr;
    if (!p.exact && log) {
        const u64 rip = t_fault_rip;
        std::printf("Readback hybrid: page %#llx exact: %s of %u bytes at %#llx by %s+%#llx after %llu own-byte "
                    "accesses\n",
                    (unsigned long long)(address & ~VAddr(4095)), write ? "write" : "read", t_access_size,
                    (unsigned long long)address, rip >= 0x800000000ull && rip < 0x810000000ull ? "game" : "host",
                    (unsigned long long)(rip >= 0x800000000ull && rip < 0x810000000ull ? rip - 0x800000000ull : rip),
                    (unsigned long long)p.steps);
        std::fflush(stdout);
    }
    p.exact = true;
}

inline void AddFastPage(VAddr address);

/// Before stepping an own-byte access: false when it should read back instead (an exact page over its budget).
/// What StepOwnAccess does with an own-byte access.
enum class StepAction { Step, ReadBack, Promote };

inline StepAction LearnStep(VAddr address) {
    static const bool learn = !(std::getenv("BB_HYBRID_STEP_LEARN") && std::getenv("BB_HYBRID_STEP_LEARN")[0] == '0');
    const VAddr page = address & ~VAddr(4095);
    const s64 now = std::chrono::duration_cast<std::chrono::milliseconds>(
                        std::chrono::steady_clock::now().time_since_epoch())
                        .count();
    std::scoped_lock lk{g_learn_mutex};
    PageLearn& p = g_learn[page];
    if (p.exact) {
        if (++p.exact_steps > ExactStepBudget) {
            p.exact_steps = 0;
            return StepAction::ReadBack;
        }
        return StepAction::Step;
    }
    if (p.fast || !learn) {
        return StepAction::Step;
    }
    if (p.steps++ == 0) {
        p.first_ms = now;
    }
    if (p.steps >= StepLearnAccesses && now - p.first_ms >= StepLearnMs && !ExactPage(page)) {
        p.fast = true;
        static const bool log = std::getenv("BB_READBACK_CLASSIFY") != nullptr;
        if (log) {
            std::printf("Readback hybrid: page %#llx learned fast (%llu own-byte accesses in %lld ms)\n",
                        (unsigned long long)page, (unsigned long long)p.steps, (long long)(now - p.first_ms));
            std::fflush(stdout);
        }
        return StepAction::Promote;
    }
    return StepAction::Step;
}

/// BB_READBACK_CLASSIFY: stepped accesses per second, and the most stepped pages.
inline void NoteStep(bool write, VAddr address = 0) {
    static const bool log = std::getenv("BB_READBACK_CLASSIFY") != nullptr;
    if (!log) {
        return;
    }
    static std::mutex page_mutex;
    static std::map<VAddr, u64> page_steps;
    {
        std::scoped_lock lk{page_mutex};
        ++page_steps[address & ~VAddr(4095)];
    }
    static std::atomic<u64> reads{0}, writes{0};
    static std::atomic<s64> last{0};
    (write ? writes : reads).fetch_add(1, std::memory_order_relaxed);
    const s64 now = std::chrono::steady_clock::now().time_since_epoch().count() / 1000000000;
    s64 seen = last.load(std::memory_order_relaxed);
    if (now - seen >= 5 && last.compare_exchange_strong(seen, now) && seen) {
        std::printf("Readback hybrid: own-byte accesses stepped: %llu reads/s, %llu writes/s\n",
                    (unsigned long long)(reads.exchange(0) / (now - seen)),
                    (unsigned long long)(writes.exchange(0) / (now - seen)));
        std::vector<std::pair<u64, VAddr>> top;
        {
            std::scoped_lock lk{page_mutex};
            for (const auto& [page, n] : page_steps) {
                top.emplace_back(n, page);
            }
            page_steps.clear();
        }
        std::sort(top.rbegin(), top.rend());
        std::printf("Readback hybrid: most stepped pages:");
        for (size_t i = 0; i < top.size() && i < 16; ++i) {
            std::printf(" %#llx=%llu", (unsigned long long)top[i].second, (unsigned long long)top[i].first);
            if (i < 8) { // the GPU's ranges on the page
                std::scoped_lock lk{g_written_mutex};
                std::printf("[");
                g_written.ForEachInRange(top[i].second, 4096, [](VAddr begin, VAddr end) {
                    std::printf(" +%#llx..%#llx", (unsigned long long)(begin & 4095),
                                (unsigned long long)(((end - 1) & 4095) + 1));
                });
                std::printf(" ]");
            }
        }
        std::printf(" (%zu pages)\n", top.size());
        std::fflush(stdout);
    }
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

/// A removed page's slot keeps a tombstone, so probes for later keys go on past it.
constexpr u64 FastTombstone = ~u64{0};
inline void RemoveFastPage(VAddr address) {
    const u64 key = (address >> 12) + 1;
    for (size_t i = 0, slot = (key * 0x9E3779B97F4A7C15ull) >> 54; i < TableSize; ++i, slot = (slot + 1) % TableSize) {
        const u64 v = g_fast[slot].load(std::memory_order_relaxed);
        if (v == key) {
            g_fast[slot].store(FastTombstone, std::memory_order_relaxed);
            return;
        }
        if (v == 0) {
            return;
        }
    }
}

// A fast page's bytes as last agreed by the game and the GPU (its shadow), and the GPU's ranges on it when it was
// learned. Both sides may write the same bytes (counters the game resets and the GPU advances), so neither side's copy
// may overwrite the other's: a copy-back writes only bytes the GPU changed since the shadow, an upload sends only
// GPU-range bytes the game changed since it. A GPU write to a range not seen when the page was learned (memory reused
// while a race loads) demotes the page (FastFor).
struct FastState {
    std::array<u8, 4096> shadow;
    std::vector<std::pair<u16, u16>> ranges; // [begin, end) on the page; end may be 4096
};
inline std::mutex g_fast_mutex;
inline std::map<VAddr, std::unique_ptr<FastState>> g_fast_state;

/// Makes `page` fast; `guest` is its content right after a readback (the shadow).
inline void PromoteFastPage(VAddr page, const u8* guest) {
    auto state = std::make_unique<FastState>();
    std::memcpy(state->shadow.data(), guest, 4096);
    {
        std::scoped_lock lk{g_written_mutex};
        g_written.ForEachInRange(page, 4096, [&](VAddr begin, VAddr end) {
            state->ranges.emplace_back(u16(begin - page), u16(std::min<VAddr>(end, page + 4096) - page));
        });
    }
    {
        std::scoped_lock lk{g_fast_mutex};
        g_fast_state[page] = std::move(state);
    }
    AddFastPage(page);
}

inline void DemoteFastPage(VAddr page, VAddr begin, VAddr end) {
    RemoveFastPage(page);
    {
        std::scoped_lock lk{g_fast_mutex};
        g_fast_state.erase(page);
    }
    {
        std::scoped_lock lk{g_learn_mutex};
        PageLearn& p = g_learn[page];
        p.fast = false;
        p.steps = 0;
    }
    static const bool log = std::getenv("BB_READBACK_CLASSIFY") != nullptr;
    if (log) {
        std::printf("Readback hybrid: page %#llx exact again: GPU range +%#llx..%#llx not seen when learned\n",
                    (unsigned long long)page, (unsigned long long)(begin - page), (unsigned long long)(end - page));
        std::fflush(stdout);
    }
}

/// [begin, end) (within one page) is a GPU write the fast page knows; a new one demotes the page.
inline bool FastFor(VAddr begin, VAddr end) {
    const VAddr page = begin & ~VAddr(4095);
    if (!IsFastPage(page)) {
        return false;
    }
    // Development (BB_HYBRID_CERTIFY=1): demote on a range not seen when learned. Off: the GPU's ranges on these pages
    // vary from frame to frame, and every demotion meant 4 s of stepping again (45-60 FPS at 3200).
    static const bool certify = std::getenv("BB_HYBRID_CERTIFY") && std::getenv("BB_HYBRID_CERTIFY")[0] == '1';
    if (!certify) {
        return true;
    }
    end = std::min<VAddr>(end, page + 4096);
    {
        std::scoped_lock lk{g_fast_mutex};
        const auto it = g_fast_state.find(page);
        if (it != g_fast_state.end()) {
            for (const auto& [b, e] : it->second->ranges) {
                if (begin - page >= b && end - page <= e) {
                    return true;
                }
            }
        }
    }
    DemoteFastPage(page, begin, end);
    return false;
}

/// Runs f(shadow) for a fast page's shadow under its lock; false when the page has none.
template <typename Func>
inline bool WithShadow(VAddr page, Func&& f) {
    std::scoped_lock lk{g_fast_mutex};
    const auto it = g_fast_state.find(page);
    if (it == g_fast_state.end()) {
        return false;
    }
    f(it->second->shadow.data());
    return true;
}

/// The GPU's bytes not yet read back in [address, address + size), as [begin, end) parts.
inline std::vector<std::pair<VAddr, VAddr>> GpuWrittenParts(VAddr address, u64 size) {
    std::vector<std::pair<VAddr, VAddr>> parts;
    std::scoped_lock lk{g_written_mutex};
    g_written.ForEachInRange(address, size, [&](VAddr begin, VAddr end) { parts.emplace_back(begin, end); });
    return parts;
}

/// A guest fault on a GPU modified page (a read, or a write that would read back first): learns the page as fast
/// once enough faults hit only the game's own bytes; a fault on GPU bytes keeps it exact.
inline void LearnFault(VAddr address, bool write) {
    if (!Enabled()) {
        return;
    }
    const VAddr page = address & ~VAddr(4095);
    // The bytes the instruction accesses: from the fault address (its first byte), t_access_size, on this page.
    const VAddr begin = address;
    const VAddr end = std::min<VAddr>(address + t_access_size, page + 4096);
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
    if (gpu && write) {
        ++c.gpu; // a game write to GPU bytes: a fast page merges it (copy-backs and uploads)
    } else if (gpu) {
        ++c.gpu;
        c.exact = true;
        MarkExact(address, write);
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
