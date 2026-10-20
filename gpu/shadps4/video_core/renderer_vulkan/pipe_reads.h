// SPDX-License-Identifier: GPL-2.0-or-later
// VR fork: guest memory the draw recording thread (stage B, vk_draw_pipe.h) reads directly, per packet still in
// flight: buffers the GPU command thread did not copy into the constant ring, vertex and index buffers, and anything
// for packets whose reads are not known (tasks, special and indirect draws). A guest write by the GPU command thread
// (the constant engine's DumpConstRam, ~80k per second) only waits for stage B when it overlaps one of these; before,
// every dump drained the pipe (stage A waited 35-40% of the time at 1080 per eye).

#pragma once

#include <algorithm>
#include <vector>

#include "common/types.h"

namespace Vulkan {

class PipeReads {
public:
    /// Stage A, before a packet's buffers are collected.
    void BeginPacket() {
        begin = ~VAddr{0};
        end = 0;
    }

    /// Stage A: stage B reads [address, address + size) of guest memory for this packet.
    void Add(VAddr address, u64 size) {
        begin = std::min(begin, address);
        end = std::max(end, address + size);
    }

    /// Stage A: this packet may read any guest memory.
    void AddAll() {
        begin = 0;
        end = ~VAddr{0};
    }

    /// Stage A, after the packet is committed: `position` is the pipe position stage B reaches once it ran it.
    void EndPacket(u64 position) {
        if (begin >= end) {
            return;
        }
        // Draws read the same buffers over and over: an overlapping entry only grows and moves on.
        for (auto& entry : entries) {
            if (begin <= entry.end && entry.begin <= end) {
                entry.begin = std::min(entry.begin, begin);
                entry.end = std::max(entry.end, end);
                entry.position = position;
                return;
            }
        }
        entries.push_back({begin, end, position});
    }

    /// Stage A: whether a packet stage B has not run yet reads guest memory overlapping [address, address + size).
    template <typename Reached>
    bool Overlaps(VAddr address, u64 size, Reached&& reached) {
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
    struct Entry {
        VAddr begin;
        VAddr end;
        u64 position;
    };
    std::vector<Entry> entries;
    VAddr begin = ~VAddr{0};
    VAddr end = 0;
};

} // namespace Vulkan
