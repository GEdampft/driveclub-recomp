// SPDX-License-Identifier: GPL-2.0-or-later
// VR fork: guest memory the draw recording thread (stage B, vk_draw_pipe.h) reads directly, per packet still in
// flight: buffers the GPU command thread did not copy into the constant ring, vertex and index buffers, indirect
// arguments, the memory a task (fence, WriteData, DMA) touches, and anything for a flip or with the ring off. A guest
// write by the GPU command thread (the constant engine's DumpConstRam, ~80k per second) only waits for stage B when it
// overlaps one of these; before, every dump drained the pipe (stage A waited 35-40% of the time at 1080 per eye), and
// until 2026-10-22 every task and indirect draw counted as "anything" (12% waiting, ~5% FPS at 2560 per eye).

#pragma once

#include <algorithm>
#include <array>
#include <utility>
#include <vector>

#include "common/types.h"

namespace Vulkan {

class PipeReads {
public:
    /// Stage A, before a packet's buffers are collected.
    void BeginPacket() {
        num_boxes = 0;
        all = false;
    }

    /// Stage A: stage B reads [address, address + size) of guest memory for this packet.
    void Add(VAddr address, u64 size) {
        // One box per 64 GB region of the address space: a packet reading the GDS counter page (0x30...) and buffers
        // at 0x51... made one box over the constant dumps' region (0x40...) between them.
        const VAddr key = address >> RegionBits;
        for (u32 i = 0; i < num_boxes; ++i) {
            if (boxes[i].key == key) {
                boxes[i].begin = std::min(boxes[i].begin, address);
                boxes[i].end = std::max(boxes[i].end, address + size);
                return;
            }
        }
        if (num_boxes == boxes.size()) {
            all = true; // more regions than boxes: treated as anything
            return;
        }
        boxes[num_boxes++] = {key, address, address + size, 0};
    }

    /// Stage A: this packet may read any guest memory.
    void AddAll() {
        all = true;
    }

    /// Stage A, after the packet is committed: `position` is the pipe position stage B reaches once it ran it.
    void EndPacket(u64 position) {
        // A packet that may read anything is kept apart: merged into the ranges, every later packet overlapped it and
        // moved it on, so it never expired while draws kept coming and nearly every constant dump drained the pipe.
        if (std::exchange(all, false)) {
            all_position = position;
            all_pending = true;
            return;
        }
        for (u32 i = 0; i < num_boxes; ++i) {
            const Entry& box = boxes[i];
            if (box.begin >= box.end) {
                continue;
            }
            // Draws read the same buffers over and over: an overlapping entry of the region only grows and moves on.
            bool merged = false;
            for (auto& entry : entries) {
                if (entry.key == box.key && box.begin <= entry.end && entry.begin <= box.end) {
                    entry.begin = std::min(entry.begin, box.begin);
                    entry.end = std::max(entry.end, box.end);
                    entry.position = position;
                    merged = true;
                    break;
                }
            }
            if (!merged) {
                entries.push_back({box.key, box.begin, box.end, position});
            }
        }
        num_boxes = 0;
    }

    /// Stage A: whether a packet stage B has not run yet reads guest memory overlapping [address, address + size).
    template <typename Reached>
    bool Overlaps(VAddr address, u64 size, Reached&& reached) {
        if (all_pending) {
            if (!reached(all_position)) {
                return true;
            }
            all_pending = false;
        }
        std::erase_if(entries, [&](const Entry& entry) { return reached(entry.position); });
        const VAddr last = address + size;
        for (const auto& entry : entries) {
            if (address < entry.end && entry.begin < last) {
                return true;
            }
        }
        return false;
    }

private:
    static constexpr u32 RegionBits = 36;
    struct Entry {
        VAddr key;
        VAddr begin;
        VAddr end;
        u64 position;
    };
    std::vector<Entry> entries;
    std::array<Entry, 8> boxes{}; ///< this packet's, one per region
    u32 num_boxes = 0;
    bool all = false;         ///< this packet may read anything
    bool all_pending = false; ///< a packet that may read anything is still ahead of stage B
    u64 all_position = 0;     ///< the last such packet's
};

} // namespace Vulkan
