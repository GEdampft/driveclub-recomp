// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <chrono>
#include <array>
#include <map>
#include <unordered_map>
#include <algorithm>
#include <bit>
#include <cstdlib>
#include <cstring>
#include <string>
#include <magic_enum/magic_enum.hpp>
#include "bbport_copy.h"
#include "bbport_toggles.h"
#include "common/alignment.h"
#include "core/memory.h"
#include "video_core/amdgpu/liverpool.h"
#include "video_core/buffer_cache/buffer.h"
#include "video_core/buffer_cache/buffer_cache.h"
#include "video_core/buffer_cache/readback_hybrid.h" // VR fork
#include "video_core/buffer_cache/memory_tracker.h"
#include "video_core/renderer_vulkan/vk_graphics_pipeline.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_runtime.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/texture_cache/texture_cache.h"

#include <vk_mem_alloc.h>

extern "C" void bbgpu_print_recent_passes(void);
extern "C" void bbgpu_note_device_lost(void);

namespace VideoCore {

static constexpr size_t GDS_BUFFER_SIZE = 64_KB;
static constexpr size_t STREAM_BUFFER_SIZE = 128_MB;

static constexpr auto ARENA_USAGE =
    vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst |
    vk::BufferUsageFlagBits::eUniformBuffer | vk::BufferUsageFlagBits::eStorageBuffer |
    vk::BufferUsageFlagBits::eIndexBuffer | vk::BufferUsageFlagBits::eVertexBuffer |
    vk::BufferUsageFlagBits::eIndirectBuffer | vk::BufferUsageFlagBits::eShaderDeviceAddress;

std::optional<u32> FindMemoryType(const vk::PhysicalDeviceMemoryProperties& properties,
                                  vk::MemoryPropertyFlags wanted, u32 memory_type_bits) {
    for (u32 i = 0; i < properties.memoryTypeCount; ++i) {
        if (((memory_type_bits >> i) & 1) == 0) {
            continue;
        }
        const auto flags = properties.memoryTypes[i].propertyFlags;
        if ((flags & wanted) == wanted) {
            return i;
        }
    }
    return std::nullopt;
}

// Memory for the arenas that the CPU can map, chosen from what the driver reports: memory that is both device local
// and host visible. On a discrete card that only exists with resizable BAR (and then its heap is the card's whole
// video memory); a small BAR window of a few hundred MiB is left alone. Cards that share memory with the CPU always
// have it. BB_ARENA_MEMORY: auto (default), device (never), hostvisible (any such type).
static std::optional<u32> FindHostVisibleArenaType(const Vulkan::Instance& instance, u32 memory_type_bits) {
    using Flags = vk::MemoryPropertyFlagBits;
    const char* env = std::getenv("BB_ARENA_MEMORY");
    const std::string mode = env ? env : "auto";
    if (mode == "device") {
        return std::nullopt;
    }
    const auto& props = instance.GetMemoryProperties();
    const auto type = FindMemoryType(props, Flags::eDeviceLocal | Flags::eHostVisible | Flags::eHostCoherent,
                                     memory_type_bits);
    if (!type || mode == "hostvisible" || instance.IsIntegrated()) {
        return type;
    }
    u64 largest_local_heap = 0;
    for (u32 i = 0; i < props.memoryHeapCount; ++i) {
        if ((props.memoryHeaps[i].flags & vk::MemoryHeapFlagBits::eDeviceLocal) != vk::MemoryHeapFlags{}) {
            largest_local_heap = std::max<u64>(largest_local_heap, props.memoryHeaps[i].size);
        }
    }
    const u64 heap_size = props.memoryHeaps[props.memoryTypes[*type].heapIndex].size;
    if (heap_size >= 1_GB && heap_size == largest_local_heap) {
        return type;
    }
    return std::nullopt;
}

BufferCache::BufferCache(const Vulkan::Instance& instance_, Vulkan::Scheduler& scheduler_,
                         Vulkan::Runtime& runtime_, AmdGpu::Liverpool* liverpool_,
                         TextureCache& texture_cache_, PageManager& tracker)
    : instance{instance_}, scheduler{scheduler_}, runtime{runtime_},
      staging_pool{runtime_.GetStagingPool()}, liverpool{liverpool_},
      memory{Core::Memory::Instance()}, texture_cache{texture_cache_},
      memory_tracker{std::make_unique<MemoryTracker>(tracker)},
      stream_buffer{instance, scheduler, MemoryType::Stream, STREAM_BUFFER_SIZE},
      gds_buffer{instance, 0, GDS_BUFFER_SIZE, MemoryType::Stream, "GDS Buffer"},
      memory_semaphore{instance} {
    while (arena_page_bits > MIN_ARENA_PAGE_BITS &&
           (u64{2} << arena_page_bits) > instance.GetMaxBufferSize()) {
        --arena_page_bits;
    }
    arena_page_size = u64{1} << arena_page_bits;
    num_arena_pages = u64{1} << (ADDRESS_SPACE_BITS - arena_page_bits);
    if (arena_page_bits != MAX_ARENA_PAGE_BITS) {
        LOG_INFO(Render_Vulkan, "Buffer arenas of {} MiB (driver buffer limit {} MiB)",
                 arena_page_size >> 20, instance.GetMaxBufferSize() >> 20);
    }
    const vk::BufferCreateInfo probe_ci = {
        .flags =
            vk::BufferCreateFlagBits::eSparseBinding | vk::BufferCreateFlagBits::eSparseResidency,
        .size = arena_page_size,
        .usage = ARENA_USAGE,
        .sharingMode = vk::SharingMode::eExclusive,
    };
    const vk::DeviceBufferMemoryRequirements req_info = {
        .pCreateInfo = &probe_ci,
    };
    const auto device = instance.GetDevice();
    const auto reqs = device.getBufferMemoryRequirements(req_info).memoryRequirements;
    block_size = Common::AlignUp(std::max<u64>(reqs.alignment, MIN_BLOCK_SIZE), reqs.alignment);
    ASSERT_MSG(std::popcount(block_size) == 1, "Sparse block size {} is not a power of 2",
               block_size);
    block_shift = std::bit_width(block_size) - 1;
    blocks_per_arena_page = arena_page_size / block_size;
    blocks_per_arena_page_shift = arena_page_bits - block_shift;
    arena_memory_type_index =
        FindMemoryType(instance.GetMemoryProperties(), vk::MemoryPropertyFlagBits::eDeviceLocal,
                       reqs.memoryTypeBits)
            .value();
    {
        const char* direct = std::getenv("BB_READBACK_DIRECT");
        // Reading GPU results straight from host visible memory (resizable BAR) is untested on AMD cards, which
        // show device losses and broken frames with it: off there unless BB_READBACK_DIRECT=1 asks for it.
        const bool amd = instance.GetVendorID() == 0x1002;
        direct_readback = direct ? direct[0] != '0' : !amd;
        if (const auto host_type = FindHostVisibleArenaType(instance, reqs.memoryTypeBits); host_type && direct_readback) {
            arena_memory_type_index = *host_type;
            arena_host_visible = true;
        }
        std::printf("GPU: buffer arena memory type %u, host visible (direct guest reads): %s\n", arena_memory_type_index,
                    arena_host_visible ? "yes" : "no");
        // What the driver offers, to see whether resizable BAR is on: a CPU visible type in the card's own memory
        // that is as large as the card's memory means it is.
        const auto& mem = instance.GetMemoryProperties();
        for (u32 h = 0; h < mem.memoryHeapCount; ++h) {
            const bool local =
                (mem.memoryHeaps[h].flags & vk::MemoryHeapFlagBits::eDeviceLocal) != vk::MemoryHeapFlags{};
            std::string visible_types;
            for (u32 t = 0; t < mem.memoryTypeCount; ++t) {
                const auto f = mem.memoryTypes[t].propertyFlags;
                if (mem.memoryTypes[t].heapIndex == h && (f & vk::MemoryPropertyFlagBits::eHostVisible)) {
                    visible_types += (visible_types.empty() ? "" : ",") + std::to_string(t) +
                                     ((f & vk::MemoryPropertyFlagBits::eHostCoherent) ? "c" : "");
                }
            }
            std::printf("GPU: memory heap %u: %llu MiB%s, CPU visible types: %s\n", h,
                        static_cast<unsigned long long>(mem.memoryHeaps[h].size >> 20), local ? ", device local" : "",
                        visible_types.empty() ? "none" : visible_types.c_str());
        }
    }

    const u64 bda_pagetable_size =
        (blocks_per_arena_page * num_arena_pages) * sizeof(vk::DeviceAddress);
    fault_manager = std::make_unique<FaultManager>(instance, scheduler, *this, block_shift,
                                                   blocks_per_arena_page * num_arena_pages);
    bda_pagetable_buffer = std::make_unique<Buffer>(
        instance, 0, bda_pagetable_size, MemoryType::DeviceLocal, "BDA Page Table Buffer");
    runtime.FillBuffer(bda_pagetable_buffer.get(), 0u, bda_pagetable_size, 0u);
}

BufferCache::~BufferCache() = default;

void BufferCache::InvalidateMemory(VAddr device_addr, u64 size, bool assume_locks) {
    memory_tracker->InvalidateRegion(device_addr, size, [this, device_addr, size, assume_locks] {
        ReadMemory(device_addr, size, true, assume_locks);
    });
}

// bbport: the game fills its per-frame buffers (constants, skinning output) sequentially and
// every 4 KiB page cost a protection fault (~110k/s in Hunter's Nightmare, a fifth of each
// render worker's time in the kernel). A fault unprotects the aligned window around it instead;
// pages marked CPU-modified without being written only cost an upload when bound.
// Item: context = BufferCache, source = guest address, destination = host pointer, size,
// extra = the Buffer whose mapping holds the destination (flushed after the copy).
void BufferCache::RunGuestCopy(const BbCopy::Item& item) {
    auto* cache = static_cast<BufferCache*>(item.context);
    auto* dst = reinterpret_cast<u8*>(item.destination);
    cache->memory->CopySparseMemory(item.source, dst, item.size);
    auto* buffer = reinterpret_cast<Buffer*>(item.extra);
    buffer->Flush(dst - buffer->mapped_data.data(), item.size);
}

// Small guest copies run on the recording thread (it spins for work: no wakeup, and it is
// idle most of the time); PoolSmallCopies (toggle 524288) batches them for the copy threads.
void BufferCache::SmallGuestCopy(const BbCopy::Item& item) {
    if (scheduler.IsRecordingDeferred() && !BbToggle::Disabled(BbToggle::PoolSmallCopies)) {
        scheduler.RecordHostCopy([item] { item.run(item); });
        return;
    }
    BbCopy::QueueCopy(item);
}

void BufferCache::ExtendWriteFault(VAddr device_addr) {
    static const u64 window = [] {
        const char* env = std::getenv("BB_FAULT_WINDOW");
        const u64 kib = env ? std::strtoull(env, nullptr, 10) : 256;
        return std::bit_ceil(std::clamp<u64>(kib, 4, 1024)) * 1024;
    }();
    if (window <= TRACKER_BYTES_PER_PAGE || BbToggle::Disabled(BbToggle::FaultWindow)) {
        return;
    }
    memory_tracker->ExtendWriteFault(Common::AlignDown(device_addr, window), window);
}

static constexpr u64 READBACK_WINDOW_SIZE = 512_KB;

void BufferCache::ReadMemory(VAddr device_addr, u64 size, bool is_write, bool assume_locks) {
    // Development (BB_READBACK_LOG=1): each CPU access that needs GPU data, the first 200000.
    static const bool log_reads = std::getenv("BB_READBACK_LOG") != nullptr;
    static std::atomic<int> logged{0};
    if (log_reads && logged.fetch_add(1) < 200000) {
        std::printf("Readback: %#llx +%#llx %s\n", (unsigned long long)device_addr, (unsigned long long)size,
                    is_write ? "write" : "read");
        std::fflush(stdout);
    }
    // The faulting thread waits for the download instead of the GPU command thread: that thread
    // keeps recording the frame while the copy runs.
    static const bool offload_enabled = [] {
        const char* env = std::getenv("BB_READBACK_OFFLOAD");
        return !(env && env[0] == '0');
    }();
    if (assume_locks) {
        BbStats::readback_paths[3].fetch_add(1, std::memory_order_relaxed);
    } else if (!offload_enabled || !readback_offload) {
        BbStats::readback_paths[4].fetch_add(1, std::memory_order_relaxed);
    } else {
        // A request over several windows is read window by window: each is an offloaded copy
        // of its own instead of one synchronous download of the whole arena.
        bool all_offloaded = true;
        const VAddr end = device_addr + size;
        for (VAddr at = device_addr; at < end && all_offloaded;) {
            const VAddr window_end = Common::AlignDown(at, READBACK_WINDOW_SIZE) + READBACK_WINDOW_SIZE;
            const VAddr part_end = std::min<VAddr>(end, window_end);
            all_offloaded = OffloadReadback(at, part_end - at, is_write);
            at = part_end;
        }
        if (all_offloaded) {
            BbStats::readback_paths[0].fetch_add(1, std::memory_order_relaxed);
            return;
        }
        BbStats::readback_paths[2].fetch_add(1, std::memory_order_relaxed);
    }
    const auto flush_request = [this, device_addr, size, is_write] {
        const u32 first_block = device_addr >> block_shift;
        const u32 last_block = (device_addr + size - 1) >> block_shift;
        const auto* arena = GetArena(first_block, last_block);

        // GPU-modified ranges come as many small scattered islands,
        // so the download is widened to a window around the request
        constexpr u64 WindowSize = READBACK_WINDOW_SIZE;
        const VAddr arena_end = arena->cpu_addr + arena->size_bytes;
        // Every sync with the GPU costs a full frame of waiting, so one read downloads all the
        // GPU-modified data of the arena at once instead of one window per fault.
        static const bool whole_arena = [] {
            const char* env = std::getenv("BB_READBACK_WINDOW");
            return !(env && env[0] == '1');
        }();
        const VAddr window_start =
            whole_arena ? arena->cpu_addr
                        : std::max<VAddr>(Common::AlignDown(device_addr, WindowSize), arena->cpu_addr);
        const VAddr window_end =
            whole_arena ? arena_end
                        : std::min<VAddr>(std::max<VAddr>(window_start + WindowSize, device_addr + size),
                                          arena_end);
        DownloadMemory(arena, window_start, window_end - window_start);
        if (is_write) {
            memory_tracker->MarkRegionAsCpuModified(device_addr, size);
        }
    };
    if (assume_locks) {
        flush_request();
    } else {
        liverpool->SendCommand<true>(std::move(flush_request));
    }
}

u64 BufferCache::CollectDownloads(const Buffer* arena, VAddr device_addr, u64 size,
                                  DownloadCopies& copies) {
    u64 total_size_bytes = 0;
    const VAddr arena_base = arena->cpu_addr;
    memory_tracker->ForEachDownloadRange<false>(device_addr, size, [&](u64 address, u64 size) {
        const auto add_download = [&](VAddr start, VAddr end) {
            const u64 new_offset = start - arena_base;
            const u64 new_size = end - start;
            copies.push_back(vk::BufferCopy{
                .srcOffset = new_offset,
                .dstOffset = total_size_bytes,
                .size = new_size,
            });
            // Align up to avoid cache conflicts
            constexpr u64 align = 64ULL;
            constexpr u64 mask = ~(align - 1ULL);
            total_size_bytes += (new_size + align - 1) & mask;
        };
        gpu_modified_ranges.ForEachInRange(address, size, add_download);
        gpu_modified_ranges.Subtract(address, size);
        ReadbackHybrid::NoteGpuWritten(address, size, false); // VR fork
    });
    return total_size_bytes;
}

void BufferCache::DownloadMemory(const Buffer* arena, VAddr device_addr, u64 size) {
    {
        // A pending offloaded readback owns the GPU modified pages of its window until it finishes.
        std::unique_lock lk{readback_mutex};
        readback_cv.wait(lk, [&] { return !IsReadbackPending(device_addr, size); });
        MergeReadbackReturns();
    }
    DownloadCopies copies;
    const VAddr arena_base = arena->cpu_addr;
    const u64 total_size_bytes = CollectDownloads(arena, device_addr, size, copies);
    if (total_size_bytes == 0) {
        return;
    }
    const auto download = staging_pool.Request(total_size_bytes, VideoCore::MemoryType::HostCached);
    for (auto& copy : copies) {
        copy.dstOffset += download.offset;
    }
    runtime.CopyBuffer(arena, download.buffer, copies);
    scheduler.Finish();

    download.buffer->Invalidate(download.offset, download.size);
    for (const auto& copy : copies) {
        auto* dst_addr = std::bit_cast<u8*>(arena_base + copy.srcOffset);
        const u8* src = download.mapped + (copy.dstOffset - download.offset);
        if (ReadbackHybrid::Enabled()) { // VR fork: fast pages merge (the game may have written GPU bytes there)
            MergeCopyBack(arena_base + copy.srcOffset, src, copy.size, true);
        } else {
            memory->TryWriteBacking(dst_addr, src, copy.size);
        }
    }
    memory_tracker->UnmarkRegionAsGpuModified(device_addr, size);
}

// The tick that completes the last GPU write into each readback window (hashed): a direct read waits for it.
void BufferCache::StampGpuWrite(VAddr device_addr, u64 size) {
    if (!arena_host_visible || size == 0) {
        return;
    }
    const u64 tick = scheduler.CurrentTick();
    for (u64 w = device_addr / READBACK_WINDOW_SIZE; w <= (device_addr + size - 1) / READBACK_WINDOW_SIZE; ++w) {
        window_write_tick[w & 4095] = tick;
    }
    scheduler.RequestHostReadBarrier();
}

struct DirectSpan {
    const u8* src;
    VAddr dst;
    u64 size;
};

struct BufferCache::Readback {
    VAddr window{};
    DownloadCopies copies{};
    Vulkan::StagingBufferRef staging{};
    VAddr arena_base{};
    u64 write_seq{};
    std::vector<u64> page_seq; ///< VR fork: per page, for TryUnmarkPagesAsGpuModified
    bool one_recorder{};       ///< VR fork: draws recorded on this thread too (no two-thread draw path)
    u64 tick{};
    bool busy{};
    /// Read from the arena mapping once the tick has passed; verify: the usual copy is made too and compared.
    bool direct{};
    bool verify{};
    std::vector<DirectSpan> spans;
};

bool BufferCache::OffloadReadback(VAddr device_addr, u64 size, bool is_write) {
    const VAddr window = Common::AlignDown(device_addr, READBACK_WINDOW_SIZE);
    if (device_addr + size > window + READBACK_WINDOW_SIZE) {
        BbStats::readback_paths[1].fetch_add(1, std::memory_order_relaxed);
        return false;
    }
    // A request still GPU modified after two rounds of its own takes the synchronous download,
    // which copies and unmarks in one GPU thread command.
    u32 attempts = 0;
    while (attempts < 2) {
        Readback job{.window = window};
        liverpool->SendCommand<true>([this, &job] { RecordReadback(job); });
        if (job.busy) {
            std::unique_lock lk{readback_mutex};
            readback_cv.wait(lk, [&] { return !IsReadbackPending(window, READBACK_WINDOW_SIZE); });
        } else {
            ++attempts;
            if (!job.copies.empty()) {
                scheduler.GetWorkSemaphore()->Wait(job.tick);
                FinishReadback(job);
            }
        }
        bool resolved = true;
        if (is_write) {
            // Marks the range CPU modified if it is GPU clean.
            memory_tracker->InvalidateRegion(device_addr, size, [&resolved] { resolved = false; });
        } else {
            resolved = !memory_tracker->IsRegionGpuModified(device_addr, size);
        }
        if (resolved) {
            return true;
        }
    }
    return false;
}

void BufferCache::RecordReadback(Readback& job) {
    {
        std::scoped_lock lk{readback_mutex};
        if (IsReadbackPending(job.window, READBACK_WINDOW_SIZE)) {
            job.busy = true;
            return;
        }
        MergeReadbackReturns();
    }
    const u64 block = job.window >> block_shift;
    const auto* arena = GetArena(block, block);
    const u64 total_size_bytes =
        CollectDownloads(arena, job.window, READBACK_WINDOW_SIZE, job.copies);
    if (total_size_bytes == 0) {
        return;
    }
    job.arena_base = arena->cpu_addr;
    job.write_seq = memory_tracker->GpuWriteSeq(job.window, READBACK_WINDOW_SIZE);
    memory_tracker->GpuWriteSeqs(job.window, READBACK_WINDOW_SIZE, job.page_seq); // VR fork
    job.one_recorder = BbToggle::Disabled(BbToggle::DrawPipeline);

    // Direct path: the data is already in memory the CPU can read once the GPU has finished writing it, so no copy
    // is recorded. Small requests only (reads of video memory through the mapping are slow), and only when every
    // piece can be found in a mapped allocation.
    if (arena_host_visible && direct_readback.load(std::memory_order_relaxed) && total_size_bytes <= 1_MB) {
        bool ok = true;
        for (const auto& copy : job.copies) {
            const u64 first = (job.arena_base + copy.srcOffset) >> block_shift;
            const u64 last = (job.arena_base + copy.srcOffset + copy.size - 1) >> block_shift;
            u64 covered = 0;
            resident_ranges.ForEachInRange(first, last + 1, [&](const Backing& backing) {
                if (!backing.base) {
                    return;
                }
                const u64 from = std::max<u64>(first, backing.start);
                const u64 to = std::min<u64>(last + 1, backing.end);
                const VAddr range_begin = std::max<VAddr>(job.arena_base + copy.srcOffset, from << block_shift);
                const VAddr range_end = std::min<VAddr>(job.arena_base + copy.srcOffset + copy.size, to << block_shift);
                if (range_end <= range_begin) {
                    return;
                }
                job.spans.push_back({backing.base + backing.offset + (range_begin - (u64{backing.start} << block_shift)),
                                     range_begin, range_end - range_begin});
                covered += range_end - range_begin;
            });
            ok = ok && covered == copy.size;
        }
        if (ok) {
            job.direct = true;
            // A tenth of the early direct reads (then one in a hundred and twenty-eight) also make the usual copy
            // and compare the two: a driver or memory type that behaves differently switches the direct path off.
            ++direct_reads;
            const u32 every = direct_reads <= 512 ? 8u : 128u;
            const char* verify_env = std::getenv("BB_READBACK_VERIFY");
            job.verify = verify_env ? (std::atoi(verify_env) > 0 && (direct_reads % std::atoi(verify_env)) == 0)
                                    : (direct_reads % every) == 0;
            const u64 stamped = window_write_tick[(job.window / READBACK_WINDOW_SIZE) & 4095];
            job.tick = stamped ? stamped : scheduler.CurrentTick();
            scheduler.RequestHostReadBarrier();
        } else {
            job.spans.clear();
        }
    }
    if (!job.direct || job.verify) {
        // Held until FinishReadback releases it; the pool reuses it once the GPU passes that release.
        job.staging =
            staging_pool.Request(total_size_bytes, VideoCore::MemoryType::HostCached, 0, true);
        for (auto& copy : job.copies) {
            copy.dstOffset += job.staging.offset;
        }
        runtime.CopyBuffer(arena, job.staging.buffer, job.copies);
        job.tick = scheduler.CurrentTick();
    }
    if (job.tick >= scheduler.CurrentTick()) {
        scheduler.Flush();
    }
    std::scoped_lock lk{readback_mutex};
    pending_readbacks.push_back(job.window);
}

void BufferCache::FinishReadback(const Readback& job) {
    bool use_direct = job.direct;
    if (job.staging.buffer) {
        job.staging.buffer->Invalidate(job.staging.offset, job.staging.size);
    }
    if (use_direct && job.verify) {
        // Both copies hold the same GPU writes unless the GPU wrote the window again in between (then the window is
        // not unmarked below and comparing says nothing).
        const bool rewritten = memory_tracker->GpuWriteSeq(job.window, READBACK_WINDOW_SIZE) != job.write_seq;
        u64 pos = 0;
        bool same = true;
        for (const auto& copy : job.copies) {
            const u8* staged = job.staging.mapped + (copy.dstOffset - job.staging.offset);
            u64 done = 0;
            for (const auto& span : job.spans) {
                if (span.dst >= job.arena_base + copy.srcOffset && span.dst < job.arena_base + copy.srcOffset + copy.size) {
                    same = same && std::memcmp(span.src, staged + (span.dst - (job.arena_base + copy.srcOffset)), span.size) == 0;
                    done += span.size;
                }
            }
            pos += done;
        }
        if (!same && !rewritten) {
            BbStats::readback_paths[6].fetch_add(1, std::memory_order_relaxed);
            std::printf("GPU: a direct read of video memory differs from the copy made by the GPU: direct reads are "
                        "switched off\n");
            direct_readback.store(false, std::memory_order_relaxed);
            use_direct = false;
        }
    }
    // The write-back runs ahead of the verdict: other downloads of the window wait while this one
    // is pending, and a vetoed window stays GPU modified.
    // VR fork: fast pages merge (readback_hybrid.h: the game may have written GPU bytes there).
    const bool merge = ReadbackHybrid::Enabled();
    if (use_direct) {
        BbStats::readback_paths[5].fetch_add(1, std::memory_order_relaxed);
        for (const auto& span : job.spans) {
            if (merge) {
                MergeCopyBack(span.dst, span.src, span.size, true);
            } else {
                memory->TryWriteBacking(std::bit_cast<u8*>(span.dst), span.src, span.size);
            }
        }
    } else {
        for (const auto& copy : job.copies) {
            auto* dst_addr = std::bit_cast<u8*>(job.arena_base + copy.srcOffset);
            const u8* src = job.staging.mapped + (copy.dstOffset - job.staging.offset);
            if (merge) {
                MergeCopyBack(job.arena_base + copy.srcOffset, src, copy.size, true);
            } else {
                memory->TryWriteBacking(dst_addr, src, copy.size);
            }
        }
    }
    const bool unmarked = memory_tracker->TryUnmarkRegionAsGpuModified(
        job.window, READBACK_WINDOW_SIZE, job.write_seq);
    // VR fork: the pages the GPU wrote again stay GPU modified, the others are unmarked. With one recording thread
    // the copied ranges are not returned at all: a write marked after the copy was recorded runs after it on the GPU
    // and marks its own ranges (returning the rest kept bytes the GPU wrote once, long ago, on a page it writes every
    // frame - GDS counter copies next to the game's own data - reading back again at every access).
    std::vector<bool> kept;
    if (!unmarked) {
        memory_tracker->TryUnmarkPagesAsGpuModified(job.window, READBACK_WINDOW_SIZE, job.page_seq, kept);
        if (job.one_recorder && BbToggle::Disabled(BbToggle::DrawPipeline)) {
            kept.assign(kept.size(), false);
        }
    }
    {
        std::scoped_lock lk{readback_mutex};
        if (!unmarked) {
            for (const auto& copy : job.copies) {
                const VAddr begin = job.arena_base + copy.srcOffset, end = begin + copy.size;
                for (VAddr at = begin; at < end;) {
                    const VAddr part_end = std::min<VAddr>(end, Common::AlignDown(at, 4096) + 4096);
                    const size_t page = (at - job.window) / 4096;
                    if (page >= kept.size() || kept[page]) {
                        readback_returns.emplace_back(at, part_end - at);
                    }
                    at = part_end;
                }
            }
        }
        std::erase(pending_readbacks, job.window);
    }
    readback_cv.notify_all();
    // The staging pool belongs to the GPU thread.
    if (job.staging.buffer) {
        liverpool->SendCommand([this, staging = job.staging] { staging_pool.FreeDeferred(staging); });
    }
}

bool BufferCache::IsReadbackPending(VAddr addr, u64 size) const {
    return std::ranges::any_of(pending_readbacks, [&](VAddr window) {
        return window < addr + size && addr < window + READBACK_WINDOW_SIZE;
    });
}

void BufferCache::MergeReadbackReturns() {
    for (const auto& [addr, size] : readback_returns) {
        gpu_modified_ranges.Add(addr, size);
        ReadbackHybrid::NoteGpuWritten(addr, size, true); // VR fork
    }
    readback_returns.clear();
}

namespace {
// bbport: BB_BUFFER_STATS=1 - how buffer bindings reach the GPU, by guest region (256 MiB),
// printed every 5 s: small read-only copies into the stream buffer, arena bindings, and the
// bytes those re-upload after CPU writes. Input for engine-level short paths.
struct BufferStats {
    struct Region {
        u64 stream_count{}, stream_bytes{}, arena_count{}, arena_bytes{}, upload_bytes{};
    };
    std::map<u64, Region> regions;
    /// Hot window: frame of the last binding per 64 KiB block, and the distribution of the
    /// frames between uses (1, 2, 3, 4, 5-8, 9-16, 17+): the ring's reuse distance.
    std::unordered_map<u64, u64> last_use;
    std::array<u64, 7> reuse{};
    std::chrono::steady_clock::time_point window = std::chrono::steady_clock::now();
    u64 calls = 0;
};
/// 256 MiB regions; 1 MiB inside the hot 0x104xxxxxxx window (the engine's frame data).
BufferStats& Stats();
void NoteHotUse(VAddr address, u32 size) {
    if ((address >> 28) != 0x104) {
        return;
    }
    auto& st = Stats();
    const u64 frame = BbStats::gpu_frames.load(std::memory_order_relaxed);
    for (u64 block = address >> 16; block <= (address + size - 1) >> 16; ++block) {
        auto [it, inserted] = st.last_use.try_emplace(block, frame);
        if (!inserted && it->second != frame) {
            const u64 d = frame - it->second;
            ++st.reuse[d <= 4 ? d - 1 : d <= 8 ? 4 : d <= 16 ? 5 : 6];
            it->second = frame;
        }
    }
}
u64 RegionKey(VAddr address) {
    return (address >> 28) == 0x104 ? (address >> 20) | (1ull << 40) : address >> 28;
}
bool BufferStatsEnabled() {
    static const bool enabled = [] {
        const char* value = std::getenv("BB_BUFFER_STATS");
        return value && value[0] == '1';
    }();
    return enabled;
}
BufferStats& Stats() {
    static BufferStats stats;
    return stats;
}
void PrintBufferStats() {
    auto& st = Stats();
    if ((++st.calls & 4095) != 0) {
        return;
    }
    const auto now = std::chrono::steady_clock::now();
    const double seconds = std::chrono::duration<double>(now - st.window).count();
    if (seconds < 5.0) {
        return;
    }
    std::printf("Buffer stats (%.1f s, per second): region, stream copies/bytes, arena bindings, "
                "re-uploaded bytes\n", seconds);
    std::printf("  hot window reuse distance in frames (1,2,3,4,5-8,9-16,17+): %llu %llu %llu %llu "
                "%llu %llu %llu\n",
                (unsigned long long)st.reuse[0], (unsigned long long)st.reuse[1],
                (unsigned long long)st.reuse[2], (unsigned long long)st.reuse[3],
                (unsigned long long)st.reuse[4], (unsigned long long)st.reuse[5],
                (unsigned long long)st.reuse[6]);
    st.reuse = {};
    for (const auto& [region, r] : st.regions) {
        const u64 base = (region >> 40) ? (region & ((1ull << 40) - 1)) << 20 : region << 28;
        if ((region >> 40) && r.stream_bytes + r.upload_bytes < 1e6 * seconds) {
            continue; // quiet MiB of the hot window
        }
        std::printf("  %#012llx: %8.0f copies %8.2f MB, %8.0f arena %8.2f MB bound, %8.2f MB uploads\n",
                    static_cast<unsigned long long>(base), r.stream_count / seconds,
                    r.stream_bytes / seconds / 1e6, r.arena_count / seconds,
                    r.arena_bytes / seconds / 1e6,
                    r.upload_bytes / seconds / 1e6);
    }
    st.regions.clear();
    st.window = now;
}
} // namespace

std::pair<const Buffer*, u64> BufferCache::ObtainBuffer(VAddr device_addr, u32 size,
                                                        bool is_written, bool is_texel_buffer) {
    const bool stats = BufferStatsEnabled();
    if (stats) {
        PrintBufferStats();
        NoteHotUse(device_addr, size);
    }
    // For read-only buffers use device local stream buffer to reduce renderpass breaks.
    if (!is_written && size <= STREAM_THRESHOLD && !IsRegionGpuModified(device_addr, size)) {
        if (stats) {
            auto& region = Stats().regions[RegionKey(device_addr)];
            ++region.stream_count;
            region.stream_bytes += size;
        }
        // bbport: the guest data is copied on a copy thread, started now; submission and
        // guest-visible fences wait for it (Scheduler::WaitHostCopies).
        if (!stream_buffer.mapped_data.empty() &&
            !BbToggle::Disabled(BbToggle::DeferredStreamCopies)) {
            if (const auto offset = stream_buffer.Reserve(size, instance.UniformMinAlignment())) {
                SmallGuestCopy({
                    .run = &RunGuestCopy,
                    .context = this,
                    .source = device_addr,
                    .destination = reinterpret_cast<u64>(stream_buffer.mapped_data.data() + *offset),
                    .size = size,
                    .extra = reinterpret_cast<u64>(static_cast<Buffer*>(&stream_buffer)),
                });
                return {&stream_buffer, *offset};
            }
        }
        const auto [data, offset] = stream_buffer.Map(size, instance.UniformMinAlignment());
        memory->CopySparseMemory(device_addr, data, size);
        stream_buffer.Commit();
        return {&stream_buffer, offset};
    }
    const u64 first_block = device_addr >> block_shift;
    const u64 last_block = (device_addr + size - 1) >> block_shift;
    const auto* arena = GetArena(first_block, last_block);
    EnsureResident(arena, first_block, last_block);
    const u64 uploaded_before = BbStats::buffer_upload_bytes.load(std::memory_order_relaxed);
    if (is_written) {
        ReadbackHybrid::PromoteForSmallWrite(device_addr, size); // VR fork: a small write's pages fast now
        ReadbackHybrid::NoteGpuWritten(device_addr, size, true); // VR fork: before its pages are protected
        ReadbackHybrid::NoteWriter(device_addr, size);           // VR fork: late outputs
    }
    SynchronizeMemory(arena, device_addr, size, is_written, is_texel_buffer);
    if (stats) {
        auto& region = Stats().regions[RegionKey(device_addr)];
        ++region.arena_count;
        region.arena_bytes += size;
        region.upload_bytes +=
            BbStats::buffer_upload_bytes.load(std::memory_order_relaxed) - uploaded_before;
    }
    if (is_written) {
        gpu_modified_ranges.Add(device_addr, size);
        StampGpuWrite(device_addr, size);
        NoteSmallGpuWrite(device_addr, size); // VR fork: small_writeback.cpp
    }
    return {arena, arena->Offset(device_addr)};
}

std::pair<const Buffer*, u64> BufferCache::ObtainBufferForImage(VAddr device_addr, u32 size) {
    if (IsRegionGpuModified(device_addr, size)) {
        return ObtainBuffer(device_addr, size, false);
    }
    const auto staging = staging_pool.Request(size, VideoCore::MemoryType::HostUncached,
                                              instance.StorageMinAlignment());
    if (!BbToggle::Disabled(BbToggle::DeferredUploads)) {
        // bbport: texture data is copied on the copy threads (streaming: 100+ MB per frame);
        // the upload reads the staging only after submission, which waits for the copies.
        constexpr u64 Chunk = 1_MB;
        for (u64 offset = 0; offset < staging.size; offset += Chunk) {
            const BbCopy::Item item{
                .run = &RunGuestCopy,
                .context = this,
                .source = device_addr + offset,
                .destination = reinterpret_cast<u64>(staging.mapped + offset),
                .size = std::min(Chunk, staging.size - offset),
                .extra = reinterpret_cast<u64>(staging.buffer),
            };
            if (staging.size < Chunk) {
                SmallGuestCopy(item);
            } else {
                BbCopy::Async([item] { item.run(item); });
            }
        }
        return {staging.buffer, staging.offset};
    }
    memory->CopySparseMemory(device_addr, staging.mapped, staging.size);
    staging.Flush();
    return {staging.buffer, staging.offset};
}

bool BufferCache::IsRegionCpuModified(VAddr addr, size_t size) {
    return memory_tracker->IsRegionCpuModified(addr, size);
}

bool BufferCache::IsRegionGpuModified(VAddr addr, size_t size) {
    return memory_tracker->IsRegionGpuModified(addr, size);
}

void BufferCache::ProcessFaultBuffer() {
    fault_manager->ProcessFaultBuffer();
}

void BufferCache::SynchronizeDmaBuffers() {
    for (const auto& range : resident_ranges) {
        const u64 page = range.start >> (arena_page_bits - block_shift);
        const VAddr device_addr = range.start << block_shift;
        const u64 size = (range.end - range.start) << block_shift;
        SynchronizeMemory(address_space[page], device_addr, size, false, false);
    }
}

const Buffer* BufferCache::GetArena(u64 first_block, u64 last_block) {
    const u64 first_page = first_block >> blocks_per_arena_page_shift;
    const u64 last_page = last_block >> blocks_per_arena_page_shift;
    ASSERT_MSG(last_page - first_page <= 1,
               "Buffer request cannot span more than two VA arena pages");

    const auto* first_arena = address_space[first_page];
    const auto* last_arena = address_space[last_page];
    if (first_arena == last_arena) {
        if (!first_arena) {
            const u64 base_block = Common::AlignDownPow2<u64>(first_block, blocks_per_arena_page);
            const u64 num_pages = last_page - first_page + 1;
            const auto* new_arena =
                &arenas.emplace_back(instance, base_block << block_shift,
                                     num_pages << arena_page_bits, MemoryType::Sparse);
            address_space[first_page] = new_arena;
            address_space[last_page] = new_arena;
        }
        return address_space[first_page];
    }

    LOG_WARNING(Render, "Migrating arena");

    const u64 first_addr = first_arena ? first_arena->cpu_addr : (first_page << arena_page_bits);
    const u64 first_size = first_arena ? first_arena->size_bytes : arena_page_size;
    const u64 last_size = last_arena ? last_arena->size_bytes : arena_page_size;

    const u64 base_block = first_addr >> block_shift;
    const u64 total_size = first_size + last_size;
    const u64 end_block = (first_addr + total_size) >> block_shift;
    auto* new_arena = &arenas.emplace_back(instance, first_addr, total_size, MemoryType::Sparse);
    auto* bind = BindsForArena(new_arena);
    resident_ranges.ForEachInRange(base_block, end_block, [&](const Backing& backing) {
        const u64 start = std::max(base_block, backing.start);
        const u64 end = std::min(end_block, backing.end);
        bind->binds.push_back(vk::SparseMemoryBind{
            .resourceOffset = (start - base_block) << block_shift,
            .size = (end - start) << block_shift,
            .memory = backing.memory,
            .memoryOffset = backing.offset + ((start - backing.start) << block_shift),
        });
    });

    u64 base_page = first_addr >> arena_page_bits;
    for (u32 page = 0; page < (first_size >> arena_page_bits); ++page) {
        address_space[base_page + page] = new_arena;
    }
    base_page = last_page;
    for (u32 page = 0; page < (last_size >> arena_page_bits); ++page) {
        address_space[base_page + page] = new_arena;
    }
    return new_arena;
}

void BufferCache::EnsureResident(const Buffer* arena, u64 first_block, u64 last_block) {
    u32 resident_blocks{};
    IntervalList bind_ranges;
    resident_ranges.ForEachGap(first_block, last_block + 1, [&](u64 start, u64 end) {
        resident_blocks += end - start;
        bind_ranges.Add({start, end});
    });

    if (bind_ranges.Empty()) {
        return;
    }
    BbStats::Timer timer{BbStats::t_resident};

    const vk::MemoryAllocateInfo alloc_info = {
        .allocationSize = resident_blocks << block_shift,
        .memoryTypeIndex = arena_memory_type_index,
    };
    const auto device_memory = Vulkan::Check(instance.GetDevice().allocateMemory(alloc_info));
    u8* mapped_base = nullptr;
    if (arena_host_visible) {
        const auto mapping = instance.GetDevice().mapMemory(device_memory, 0, vk::WholeSize);
        if (mapping.result == vk::Result::eSuccess) {
            mapped_base = static_cast<u8*>(mapping.value);
        }
    }

    boost::container::small_vector<vk::BufferCopy, 8> copies;
    const auto staging =
        staging_pool.Request(resident_blocks * sizeof(vk::DeviceAddress), MemoryType::HostUncached);

    u64 memory_offset{};
    ArenaBinds* binds = BindsForArena(arena);
    auto* bda_addrs = reinterpret_cast<vk::DeviceAddress*>(staging.mapped);
    u64 offset = staging.offset;
    for (const auto& range : bind_ranges) {
        Backing backing;
        backing.start = range.start;
        backing.end = range.end;
        backing.memory = device_memory;
        backing.offset = memory_offset;
        backing.base = mapped_base;
        resident_ranges.Add(backing);

        LOG_INFO(Render, "Making range start={}, end={} resident", backing.start, backing.end);

        const auto& bind = binds->binds.emplace_back(vk::SparseMemoryBind{
            .resourceOffset = (range.start << block_shift) - arena->cpu_addr,
            .size = (range.end - range.start) << block_shift,
            .memory = device_memory,
            .memoryOffset = memory_offset,
        });
        memory_offset += bind.size;

        for (u32 block = 0; block < bind.size; block += block_size) {
            *(bda_addrs++) = arena->BufferDeviceAddress() + bind.resourceOffset + block;
        }
        const u64 copy_size = (backing.end - backing.start) * sizeof(vk::DeviceAddress);
        copies.emplace_back(offset, backing.start * sizeof(vk::DeviceAddress), copy_size);
        offset += copy_size;
    }

    staging.Flush();
    runtime.CopyBuffer(staging.buffer, bda_pagetable_buffer.get(), copies);
}

bool BufferCache::SynchronizeMemory(const Buffer* arena, VAddr device_addr, u32 size,
                                    bool is_written, bool is_texel_buffer) {
    boost::container::small_vector<vk::BufferCopy, 4> copies;
    size_t total_size_bytes{};
    const Buffer* src_buffer{};
    memory_tracker->ForEachUploadRange(
        device_addr, size, is_written,
        [&](u64 addr, u64 size) {
            const auto add = [&](VAddr part, u64 part_size) {
                copies.emplace_back(total_size_bytes, part, part_size);
                total_size_bytes += part_size;
            };
            if (ReadbackHybrid::Enabled()) { // VR fork
                // The GPU's bytes not yet read back stay as the GPU has them; on a fast page the ones the game
                // changed since the shadow go up (readback_hybrid.h).
                ReadbackHybrid::ForEachNotGpuWritten(addr, size, add);
                for (const auto& [begin, end] : ReadbackHybrid::GpuWrittenParts(addr, size)) {
                    for (VAddr at = begin; at < end;) {
                        const VAddr page = at & ~VAddr(4095);
                        const u64 seg = std::min<VAddr>(end, page + 4096) - at;
                        u8 guest[4096];
                        memory->CopySparseMemory(at, guest, seg);
                        ReadbackHybrid::WithShadow(page, [&](u8* shadow) {
                            u8* s = shadow + (at - page);
                            for (u64 j = 0; j < seg;) {
                                if (guest[j] == s[j]) {
                                    ++j;
                                    continue;
                                }
                                u64 e = j;
                                while (e < seg && guest[e] != s[e]) {
                                    s[e] = guest[e];
                                    ++e;
                                }
                                add(at + j, e - j);
                                j = e;
                            }
                        });
                        at += seg;
                    }
                }
            } else {
                add(addr, size);
            }
        },
        [&] { src_buffer = UploadCopies(arena, copies, total_size_bytes); });

    if (src_buffer) {
        runtime.CopyBuffer(src_buffer, arena, copies);
    }
    if (is_texel_buffer && !is_written) {
        return SynchronizeMemoryFromImage(arena, device_addr, size);
    }
    return false;
}

const Buffer* BufferCache::UploadCopies(const Buffer* arena, std::span<vk::BufferCopy> copies,
                                        size_t total_size_bytes) {
    if (copies.empty()) {
        return nullptr;
    }
    BbStats::buffer_upload_bytes.fetch_add(total_size_bytes, std::memory_order_relaxed);
    const auto staging = staging_pool.Request(total_size_bytes, MemoryType::HostUncached);
    // bbport: the guest memory is copied into staging on the copy threads, started now in
    // groups of about 1 MiB; submission and guest-visible fences wait for them
    // (Scheduler::WaitHostCopies).
    if (!BbToggle::Disabled(BbToggle::DeferredUploads) && total_size_bytes < 1_MB) {
        // Small uploads join the calling thread's batch.
        for (auto& copy : copies) {
            SmallGuestCopy({
                .run = &RunGuestCopy,
                .context = this,
                .source = copy.dstOffset,
                .destination = reinterpret_cast<u64>(staging.mapped + copy.srcOffset),
                .size = copy.size,
                .extra = reinterpret_cast<u64>(staging.buffer),
            });
            copy.srcOffset += staging.offset;
            copy.dstOffset -= arena->cpu_addr;
        }
        return staging.buffer;
    }
    if (!BbToggle::Disabled(BbToggle::DeferredUploads)) {
        struct HostCopy {
            VAddr source;
            u8* destination;
            u64 size;
        };
        using Group = boost::container::small_vector<HostCopy, 8>;
        auto group = std::make_shared<Group>();
        u64 group_bytes = 0;
        const auto launch = [&] {
            BbCopy::Async([group, memory = memory, staging] {
                for (const auto& copy : *group) {
                    memory->CopySparseMemory(copy.source, copy.destination, copy.size);
                }
                staging.Flush();
            });
        };
        for (auto& copy : copies) {
            group->push_back({copy.dstOffset, staging.mapped + copy.srcOffset, copy.size});
            group_bytes += copy.size;
            copy.srcOffset += staging.offset;
            copy.dstOffset -= arena->cpu_addr;
            if (group_bytes >= 1_MB) {
                launch();
                group = std::make_shared<Group>();
                group_bytes = 0;
            }
        }
        if (!group->empty()) {
            launch();
        }
        return staging.buffer;
    }
    const auto copy = [&](std::size_t i) {
        memory->CopySparseMemory(copies[i].dstOffset, staging.mapped + copies[i].srcOffset,
                                 copies[i].size);
    };
    if (total_size_bytes >= 2_MB && copies.size() > 1) {
        BbCopy::ParallelFor(copies.size(), copy);
    } else {
        for (std::size_t i = 0; i < copies.size(); ++i) {
            copy(i);
        }
    }
    for (auto& copy : copies) {
        copy.srcOffset += staging.offset;
        copy.dstOffset -= arena->cpu_addr;
    }
    staging.Flush();
    return staging.buffer;
}

bool BufferCache::SynchronizeMemoryFromImage(const Buffer* arena, VAddr device_addr, u32 size) {
    if (auto type = texture_cache.IsMeta(device_addr)) {
        if (*type == TextureCache::MetaType::HTile) {
            static constexpr u32 ZmaskUncompressed = 0xf;
            runtime.FillBuffer(arena, arena->Offset(device_addr), size, ZmaskUncompressed);
            return true;
        } else {
            LOG_WARNING(Render_Vulkan, "Unhandled metadata type {}", magic_enum::enum_name(*type));
        }
    }
    // bbport: most texel buffers alias no image; remember misses until images change.
    const u64 generation = texture_cache.RegistryGeneration();
    auto& miss = image_miss_cache[((device_addr >> 6) ^ size * 0x9E3779B1u) % image_miss_cache.size()];
    if (miss.address == device_addr && miss.size == size && miss.generation == generation &&
        !BbToggle::Disabled(BbToggle::TextureBindingMemo)) {
        return false;
    }
    const ImageId image_id = texture_cache.FindImageFromRange(device_addr, size);
    if (!image_id) {
        miss = {device_addr, size, generation};
        return false;
    }
    // bbport: the lookups above only read what the texture binding helper leaves alone; the
    // copy below changes image state, so the helper finishes first.
    runtime.BeforeImageAccess();
    Image& image = texture_cache.GetImage(image_id);
    ASSERT_MSG(device_addr == image.info.guest_address,
               "Texel buffer aliases image subresources {:x} : {:x}", device_addr,
               image.info.guest_address);
    const u64 arena_offset = arena->Offset(device_addr);
    boost::container::small_vector<vk::BufferImageCopy, 8> buffer_copies;
    for (u32 mip = 0; mip < image.info.resources.levels; mip++) {
        const auto& mip_info = image.info.mips_layout[mip];
        const u32 width = std::max(image.info.size.width >> mip, 1u);
        const u32 height = std::max(image.info.size.height >> mip, 1u);
        const u32 depth = std::max(image.info.size.depth >> mip, 1u);
        if (arena_offset + mip_info.offset + mip_info.size > arena->size_bytes) {
            break;
        }
        buffer_copies.push_back(vk::BufferImageCopy{
            .bufferOffset = mip_info.offset,
            .bufferRowLength = mip_info.pitch,
            .bufferImageHeight = mip_info.height,
            .imageSubresource{
                .aspectMask = image.aspect_mask & ~vk::ImageAspectFlagBits::eStencil,
                .mipLevel = mip,
                .baseArrayLayer = 0,
                .layerCount = image.info.resources.layers,
            },
            .imageOffset = {0, 0, 0},
            .imageExtent = {width, height, depth},
        });
    }
    if (buffer_copies.empty()) {
        return false;
    }
    auto& tile_manager = texture_cache.GetTileManager();
    tile_manager.TileImage(image, buffer_copies, arena, arena_offset);
    return true;
}

void BufferCache::SubmitPendingArenaBinds(Vulkan::SubmitInfo& info) {
    if (pending_binds.empty()) {
        return;
    }

    std::vector<vk::SparseBufferMemoryBindInfo> buffer_binds;
    buffer_binds.reserve(pending_binds.size());

    for (const auto& binds : pending_binds) {
        buffer_binds.emplace_back(vk::SparseBufferMemoryBindInfo{
            .buffer = binds.arena->Handle(),
            .bindCount = static_cast<u32>(binds.binds.size()),
            .pBinds = binds.binds.data(),
        });
    }

    const u64 signal_tick = memory_semaphore.NextTick();
    const auto signal_sema = memory_semaphore.Handle();

    const vk::TimelineSemaphoreSubmitInfo timeline_si = {
        .signalSemaphoreValueCount = 1u,
        .pSignalSemaphoreValues = &signal_tick,
    };

    const vk::BindSparseInfo sparse_info = {
        .pNext = &timeline_si,
        .bufferBindCount = static_cast<u32>(buffer_binds.size()),
        .pBufferBinds = buffer_binds.data(),
        .signalSemaphoreCount = 1u,
        .pSignalSemaphores = &signal_sema,
    };

    info.AddWait(signal_sema, signal_tick);
    auto submit_result = instance.GetGraphicsQueue().bindSparse(sparse_info);
    if (submit_result == vk::Result::eErrorDeviceLost) {
        instance.LogDeviceLost();
    }
    ASSERT_MSG(submit_result != vk::Result::eErrorDeviceLost, "Device lost during submit");

    pending_binds.clear();
}

} // namespace VideoCore
