// SPDX-FileCopyrightText: Copyright 2025-2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "bbport_threads.h"
#include <algorithm>
#include <cmath>
#include <vector>
#include <chrono>
#include <cstdio>
#include <time.h>
#ifndef _WIN32
#include <sys/resource.h>
#endif
#include "common/assert.h"
#include "bbport_toggles.h"
#include "video_core/renderer_vulkan/vr/vr_openxr.h" // VR

extern "C" void BbDescribeAddress(unsigned long long rip, char* out, unsigned long size);
extern "C" void runtime_benchmark_flip(void); // src/runtime_benchmark.c (BB_BENCHMARK)
#include "video_core/renderer_vulkan/vk_frame_capture.h"
#include "common/debug.h"
#include "common/thread.h"
#include "core/debug_state.h"
#include "core/emulator_settings.h"
#include "core/libraries/kernel/time.h"
#include "core/libraries/videoout/driver.h"
#include "core/libraries/videoout/videoout_error.h"
#include "imgui/renderer/imgui_core.h"
#include "video_core/amdgpu/liverpool.h"
#include "video_core/renderer_vulkan/vk_presenter.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"

extern std::unique_ptr<Vulkan::Presenter> presenter;
extern std::unique_ptr<AmdGpu::Liverpool> liverpool;

namespace Vulkan {
extern std::atomic<u64> g_bb_compile_ns;
extern std::atomic<u32> g_bb_compiles;
} // namespace Vulkan

namespace Libraries::VideoOut {

constexpr static bool Is32BppPixelFormat(PixelFormat format) {
    switch (format) {
    case PixelFormat::A8R8G8B8Srgb:
    case PixelFormat::A8B8G8R8Srgb:
    case PixelFormat::A2R10G10B10:
    case PixelFormat::A2R10G10B10Srgb:
    case PixelFormat::A2R10G10B10Bt2020Pq:
        return true;
    default:
        return false;
    }
}

constexpr u32 PixelFormatBpp(PixelFormat pixel_format) {
    switch (pixel_format) {
    case PixelFormat::A16R16G16B16Float:
        return 8;
    default:
        return 4;
    }
}

VideoOutDriver::VideoOutDriver(u32 width, u32 height) {
    main_port.resolution.full_width = width;
    main_port.resolution.full_height = height;
    main_port.resolution.pane_width = width;
    main_port.resolution.pane_height = height;
    const char* separate = std::getenv("BB_PRESENT_THREAD");
    separate_swap = !(separate && separate[0] == '0');
    if (separate_swap) {
        swap_thread = std::jthread([&](std::stop_token token) { SwapThread(token); });
    }
    present_thread = std::jthread([&](std::stop_token token) { PresentThread(token); });
}

void VideoOutDriver::RunPresenter(std::function<void()> work, bool if_idle) {
    if (!separate_swap) {
        work();
        return;
    }
    {
        std::scoped_lock lock{swap_mutex};
        if (if_idle && (swap_busy || !swap_queue.empty())) {
            return; // a redraw of the last frame is pointless while frames are queued
        }
        swap_queue.push_back(std::move(work));
    }
    swap_cv.notify_one();
}

void VideoOutDriver::SwapThread(std::stop_token token) {
    Common::SetCurrentThreadName("bb:Present");
    while (true) {
        std::function<void()> work;
        {
            std::unique_lock lock{swap_mutex};
            swap_busy = false;
            if (!swap_cv.wait(lock, token, [this] { return !swap_queue.empty(); })) {
                return;
            }
            work = std::move(swap_queue.front());
            swap_queue.pop_front();
            swap_busy = true;
        }
        work();
    }
}

VideoOutDriver::~VideoOutDriver() = default;

int VideoOutDriver::Open(const ServiceThreadParams* params) {
    if (main_port.is_open) {
        return ORBIS_VIDEO_OUT_ERROR_RESOURCE_BUSY;
    }
    main_port.is_open = true;
    liverpool->SetVoPort(&main_port);
    return 1;
}

void VideoOutDriver::Close(s32 handle) {
    std::scoped_lock lock{mutex};

    // Mark as closed
    main_port.is_open = false;
    main_port.flip_rate = 0;
    main_port.prev_index = -1;

    // Clear port information
    std::memset(main_port.buffer_labels.data(), 0, sizeof(main_port.buffer_labels));
    std::memset(main_port.groups.data(), 0, sizeof(main_port.groups));
    std::memset(&main_port.vblank_status, 0, sizeof(main_port.vblank_status));
    main_port.flip_status = FlipStatus{};

    // Re-initialize buffers
    std::memset(main_port.buffer_slots.data(), 0, sizeof(main_port.buffer_slots));
    for (auto& buffer : main_port.buffer_slots) {
        buffer.group_index = -1;
    }

    // Clear events
    for (auto event : main_port.flip_events) {
        auto equeue = Kernel::GetEqueue(event);
        if (equeue != nullptr) {
            equeue->RemoveEvent(static_cast<u64>(OrbisVideoOutInternalEventId::Flip),
                                Kernel::OrbisKernelEvent::Filter::VideoOut);
        }
    }
    main_port.flip_events.clear();
    for (auto event : main_port.vblank_events) {
        auto equeue = Kernel::GetEqueue(event);
        if (equeue != nullptr) {
            equeue->RemoveEvent(static_cast<u64>(OrbisVideoOutInternalEventId::Vblank),
                                Kernel::OrbisKernelEvent::Filter::VideoOut);
        }
    }
    main_port.vblank_events.clear();
}

VideoOutPort* VideoOutDriver::GetPort(int handle) {
    if (handle != 1) [[unlikely]] {
        return nullptr;
    }
    return &main_port;
}

int VideoOutDriver::RegisterBuffers(VideoOutPort* port, s32 startIndex, void* const* addresses,
                                    s32 bufferNum, const BufferAttribute* attribute) {
    const s32 group_index = port->FindFreeGroup();
    if (group_index >= MaxDisplayBufferGroups) {
        return ORBIS_VIDEO_OUT_ERROR_NO_EMPTY_SLOT;
    }

    if (startIndex + bufferNum > MaxDisplayBuffers || startIndex > MaxDisplayBuffers ||
        bufferNum > MaxDisplayBuffers) {
        LOG_ERROR(Lib_VideoOut,
                  "Attempted to register too many buffers startIndex = {}, bufferNum = {}",
                  startIndex, bufferNum);
        return ORBIS_VIDEO_OUT_ERROR_INVALID_VALUE;
    }

    const s32 end_index = startIndex + bufferNum;
    if (bufferNum > 0 &&
        std::any_of(port->buffer_slots.begin() + startIndex, port->buffer_slots.begin() + end_index,
                    [](auto& buffer) { return buffer.group_index != -1; })) {
        return ORBIS_VIDEO_OUT_ERROR_SLOT_OCCUPIED;
    }

    if (attribute->reserved0 != 0 || attribute->reserved1 != 0) {
        LOG_ERROR(Lib_VideoOut, "Invalid reserved members");
        return ORBIS_VIDEO_OUT_ERROR_INVALID_VALUE;
    }
    if (attribute->aspect_ratio != 0) {
        LOG_ERROR(Lib_VideoOut, "Invalid aspect ratio = {}", attribute->aspect_ratio);
        return ORBIS_VIDEO_OUT_ERROR_INVALID_ASPECT_RATIO;
    }
    if (attribute->width > attribute->pitch_in_pixel) {
        LOG_ERROR(Lib_VideoOut, "Buffer width {} is larger than pitch {}", attribute->width,
                  attribute->pitch_in_pixel);
        return ORBIS_VIDEO_OUT_ERROR_INVALID_PITCH;
    }
    if (attribute->tiling_mode < TilingMode::Tile || attribute->tiling_mode > TilingMode::Linear) {
        LOG_ERROR(Lib_VideoOut, "Invalid tilingMode = {}",
                  static_cast<u32>(attribute->tiling_mode));
        return ORBIS_VIDEO_OUT_ERROR_INVALID_TILING_MODE;
    }

    LOG_INFO(Lib_VideoOut,
             "startIndex = {}, bufferNum = {}, pixelFormat = {}, aspectRatio = {}, "
             "tilingMode = {}, width = {}, height = {}, pitchInPixel = {}, option = {:#x}",
             startIndex, bufferNum, GetPixelFormatString(attribute->pixel_format),
             attribute->aspect_ratio, static_cast<u32>(attribute->tiling_mode), attribute->width,
             attribute->height, attribute->pitch_in_pixel, attribute->option);

    auto& group = port->groups[group_index];
    std::memcpy(&group.attrib, attribute, sizeof(BufferAttribute));
    group.is_occupied = true;

    for (u32 i = 0; i < bufferNum; i++) {
        const uintptr_t address = reinterpret_cast<uintptr_t>(addresses[i]);
        port->buffer_slots[startIndex + i] = VideoOutBuffer{
            .group_index = group_index,
            .address_left = address,
            .address_right = 0,
        };
        Vulkan::FrameCapture::AddDisplayBuffer(address);

        // Reset flip label also when registering buffer
        port->buffer_labels[startIndex + i] = 0;
        port->SignalVoLabel();

        presenter->RegisterVideoOutSurface(group, address);
        LOG_INFO(Lib_VideoOut, "buffers[{}] = {:#x}", i + startIndex, address);
    }

    return group_index;
}

int VideoOutDriver::UnregisterBuffers(VideoOutPort* port, s32 attributeIndex) {
    if (attributeIndex >= MaxDisplayBufferGroups || !port->groups[attributeIndex].is_occupied) {
        LOG_ERROR(Lib_VideoOut, "Invalid attribute index {}", attributeIndex);
        return ORBIS_VIDEO_OUT_ERROR_INVALID_VALUE;
    }

    auto& group = port->groups[attributeIndex];
    group.is_occupied = false;

    for (auto& buffer : port->buffer_slots) {
        if (buffer.group_index != attributeIndex) {
            continue;
        }
        buffer.group_index = -1;
    }

    return ORBIS_OK;
}

int VideoOutDriver::ChangeBufferAttribute(VideoOutPort* port, s32 attributeIndex,
                                          const BufferAttribute* attribute) {
    if (attributeIndex >= MaxDisplayBufferGroups || !port->groups[attributeIndex].is_occupied) {
        LOG_ERROR(Lib_VideoOut, "Invalid attribute index {}", attributeIndex);
        return ORBIS_VIDEO_OUT_ERROR_INVALID_VALUE;
    }

    if (attribute->reserved0 != 0 || attribute->reserved1 != 0) {
        LOG_ERROR(Lib_VideoOut, "Invalid reserved members");
        return ORBIS_VIDEO_OUT_ERROR_INVALID_VALUE;
    }
    if (attribute->aspect_ratio != 0) {
        LOG_ERROR(Lib_VideoOut, "Invalid aspect ratio = {}", attribute->aspect_ratio);
        return ORBIS_VIDEO_OUT_ERROR_INVALID_ASPECT_RATIO;
    }
    if (attribute->width > attribute->pitch_in_pixel) {
        LOG_ERROR(Lib_VideoOut, "Buffer width {} is larger than pitch {}", attribute->width,
                  attribute->pitch_in_pixel);
        return ORBIS_VIDEO_OUT_ERROR_INVALID_PITCH;
    }
    if (attribute->tiling_mode < TilingMode::Tile || attribute->tiling_mode > TilingMode::Linear) {
        LOG_ERROR(Lib_VideoOut, "Invalid tilingMode = {}",
                  static_cast<u32>(attribute->tiling_mode));
        return ORBIS_VIDEO_OUT_ERROR_INVALID_TILING_MODE;
    }

    LOG_INFO(Lib_VideoOut,
             "attributeIndex = {}, pixelFormat = {}, aspectRatio = {}, "
             "tilingMode = {}, width = {}, height = {}, pitchInPixel = {}, option = {:#x}",
             attributeIndex, GetPixelFormatString(attribute->pixel_format), attribute->aspect_ratio,
             static_cast<u32>(attribute->tiling_mode), attribute->width, attribute->height,
             attribute->pitch_in_pixel, attribute->option);

    std::unique_lock lock{port->port_mutex};
    std::memcpy(&port->groups[attributeIndex].attrib, attribute, sizeof(BufferAttribute));
    return 0;
}

void VideoOutDriver::Flip(const Request& req) {
    // Update HDR status before presenting, then present the frame (bbport: on the swap thread).
    RunPresenter([this, frame = req.frame, hdr = req.port->is_hdr] {
        presenter->SetHDR(hdr);
        presenter->Present(frame);
    });
    Vulkan::FrameCapture::OnFlip(req.index >= 0 ? req.port->buffer_slots[req.index].address_left
                                                : 0);
    runtime_benchmark_flip();

    // bbport: BB_FRAME_STATS=1 prints flip rate and frame time spread every 5 seconds.
    static const bool frame_stats = EmulatorSettingsImpl::Flag("BB_FRAME_STATS", false);
    if (frame_stats) {
        using Clock = std::chrono::steady_clock;
        static Clock::time_point window_start = Clock::now(), last = window_start;
        static u32 frames;
        static double worst_ms;
        const auto now = Clock::now();
        const double frame_ms = std::chrono::duration<double, std::milli>(now - last).count();
        worst_ms = std::max(worst_ms, frame_ms);
        static std::vector<double> intervals;
        intervals.push_back(frame_ms);
        last = now;
        // Stall diagnostics: what happened during a long frame (VR fork: BB_STALL_MS, default 40, for the 1% lows).
        static const double stall_ms = [] {
            const char* v = std::getenv("BB_STALL_MS");
            return v ? std::max(5.0, std::atof(v)) : 40.0;
        }();
        static u64 last_gpu_ns, last_images, last_image_bytes, last_buffer_bytes;
        static bool stall_primed; // VR fork: the counters were read once (last_gpu_ns stays 0 on Windows)
        static u64 last_t[6], last_minflt, last_sigf, last_pc, last_pp, last_rc, last_rp;
        const u64 pc = BbStats::protect_calls.load(), pp = BbStats::protect_pages.load(),
                  rc = BbStats::protect_revoke_calls.load(), rp = BbStats::protect_revoke_pages.load();
        const u64 minflt = BbStats::gpu_minor_faults.load(), sigf = BbStats::gpu_signal_faults.load();
        static u64 last_copy_cpu, last_copy_sys, last_copy_flt;
        const u64 copy_cpu = BbStats::t_copy_cpu.load(), copy_sys = BbStats::copy_sys_us.load(),
                  copy_flt = BbStats::copy_minflt.load();
        static u64 last_proc_flt, last_copy_ns, last_copy_bytes, last_rf, last_trf, last_twf;
        const u64 rf = BbStats::read_faults.load(), trf = BbStats::t_read_faults.load(),
                  twf = BbStats::t_write_faults.load();
        if (frame_ms > stall_ms && stall_primed) {
            std::printf("       fault handlers: %llu read faults %.1f ms, write faults %.1f ms\n",
                        static_cast<unsigned long long>(rf - last_rf), (trf - last_trf) / 1e6,
                        (twf - last_twf) / 1e6);
        }
        last_rf = rf;
        last_trf = trf;
        last_twf = twf;
        const u64 copy_ns = BbStats::t_copy.load(), copy_bytes = BbStats::copy_bytes.load();
        u64 proc_flt = 0;
#ifndef _WIN32
        if (rusage usage{}; getrusage(RUSAGE_SELF, &usage) == 0) {
            proc_flt = usage.ru_minflt;
        }
#endif
        const u64 t_now[6] = {BbStats::t_resident.load(), BbStats::t_protect.load(),
                              BbStats::t_image_create.load(), BbStats::t_refresh.load(),
                              BbStats::t_staging.load(), BbStats::t_host_wait.load()};
        static u64 last_draws, last_dispatches, last_subs, last_sys, last_user, last_invol, last_vol;
        const u64 draws = BbStats::draws.load(), dispatches = BbStats::dispatches.load(),
                  subs = BbStats::submissions.load(), sys_us = BbStats::gpu_sys_us.load(),
                  user_us = BbStats::gpu_user_us.load(), invol = BbStats::gpu_invol_switches.load(),
                  vol = BbStats::gpu_vol_switches.load();
        u64 gpu_ns = 0;
        if (const int clock = BbStats::gpu_thread_clock.load(); clock != -1) {
            timespec ts{};
            clock_gettime(static_cast<clockid_t>(clock), &ts);
            gpu_ns = u64(ts.tv_sec) * 1000000000ull + u64(ts.tv_nsec);
        }
        const u64 images = BbStats::images_registered.load();
        const u64 image_bytes = BbStats::image_upload_bytes.load();
        const u64 buffer_bytes = BbStats::buffer_upload_bytes.load();
        if (frame_ms > stall_ms && stall_primed) {
            static const Clock::time_point first = now; // VR fork: when, to match other log lines
            std::printf("Stall: %.1f ms frame at %.2f s; GPU thread on CPU %.1f ms; %llu images registered, "
                        "%.1f MB image uploads, %.1f MB buffer uploads\n",
                        frame_ms, std::chrono::duration<double>(now - first).count(), (gpu_ns - last_gpu_ns) / 1e6,
                        static_cast<unsigned long long>(images - last_images),
                        (image_bytes - last_image_bytes) / 1e6,
                        (buffer_bytes - last_buffer_bytes) / 1e6);
            std::printf("       %llu draws, %llu dispatches, %llu submissions; GPU thread user %.1f ms, "
                        "kernel %.1f ms, %llu preempted, %llu waits\n",
                        static_cast<unsigned long long>(draws - last_draws),
                        static_cast<unsigned long long>(dispatches - last_dispatches),
                        static_cast<unsigned long long>(subs - last_subs), (user_us - last_user) / 1e3,
                        (sys_us - last_sys) / 1e3, static_cast<unsigned long long>(invol - last_invol),
                        static_cast<unsigned long long>(vol - last_vol));
        }
        if (frame_ms > stall_ms && stall_primed) {
            std::printf("       ms in: resident %.1f, protect %.1f, image create %.1f, "
                        "image refresh %.1f, staging %.1f, waiting for host copies %.1f\n",
                        (t_now[0] - last_t[0]) / 1e6, (t_now[1] - last_t[1]) / 1e6,
                        (t_now[2] - last_t[2]) / 1e6, (t_now[3] - last_t[3]) / 1e6,
                        (t_now[4] - last_t[4]) / 1e6, (t_now[5] - last_t[5]) / 1e6);
            std::printf("       GPU thread page faults %llu (process %llu), protection faults %llu; "
                        "protect calls %llu (%llu pages), of which write-revoking %llu (%llu pages)\n",
                        static_cast<unsigned long long>(minflt - last_minflt),
                        static_cast<unsigned long long>(proc_flt - last_proc_flt),
                        static_cast<unsigned long long>(sigf - last_sigf),
                        static_cast<unsigned long long>(pc - last_pc),
                        static_cast<unsigned long long>(pp - last_pp),
                        static_cast<unsigned long long>(rc - last_rc),
                        static_cast<unsigned long long>(rp - last_rp));
        }
        last_pc = pc;
        last_pp = pp;
        last_rc = rc;
        last_rp = rp;
        last_minflt = minflt;
        if (frame_ms > stall_ms && stall_primed && copy_ns > last_copy_ns) {
            std::printf("       guest copies %.1f MB in %.1f thread-ms, %.1f ms on CPU (kernel %.1f ms, "
                        "%llu page faults in large copies) (%.2f GB/s per thread)\n",
                        (copy_bytes - last_copy_bytes) / 1e6, (copy_ns - last_copy_ns) / 1e6,
                        (copy_cpu - last_copy_cpu) / 1e6, (copy_sys - last_copy_sys) / 1e3,
                        static_cast<unsigned long long>(copy_flt - last_copy_flt),
                        double(copy_bytes - last_copy_bytes) / double(copy_ns - last_copy_ns));
        }
        last_copy_ns = copy_ns;
        last_copy_cpu = copy_cpu;
        last_copy_sys = copy_sys;
        last_copy_flt = copy_flt;
        last_copy_bytes = copy_bytes;
        last_proc_flt = proc_flt;
        last_sigf = sigf;
        std::copy(std::begin(t_now), std::end(t_now), std::begin(last_t));
        last_draws = draws;
        last_dispatches = dispatches;
        last_subs = subs;
        last_sys = sys_us;
        last_user = user_us;
        last_invol = invol;
        last_vol = vol;
        last_gpu_ns = gpu_ns;
        stall_primed = true;
        last_images = images;
        last_image_bytes = image_bytes;
        last_buffer_bytes = buffer_bytes;
        ++frames;
        const double window = std::chrono::duration<double>(now - window_start).count();
        static const double stats_seconds = [] { // VR fork: BB_FRAME_STATS_SECONDS (benchmarks: 1 s samples)
            const char* v = std::getenv("BB_FRAME_STATS_SECONDS");
            return v ? std::max(0.2, std::atof(v)) : 5.0;
        }();
        if (window >= stats_seconds) {
            const u32 compiles = Vulkan::g_bb_compiles.exchange(0);
            const u64 compile_ns = Vulkan::g_bb_compile_ns.exchange(0);
            const u64 direct = Vulkan::Scheduler::direct_recordings.exchange(0);
            const u64 faults = BbStats::tracker_faults.exchange(0);
            // GPU command thread CPU time per draw: comparable between builds even when the scene
            // (and so the frame rate) differs a little.
            static u64 window_draws = 0, window_gpu_us = 0;
            const u64 all_draws = BbStats::draws.load();
            const u64 gpu_us = BbStats::gpu_user_us.load() + BbStats::gpu_sys_us.load();
            const double us_per_draw = all_draws > window_draws
                                           ? double(gpu_us - window_gpu_us) / (all_draws - window_draws)
                                           : 0.0;
            const double draws_per_frame = frames ? double(all_draws - window_draws) / frames : 0.0;
            window_draws = all_draws;
            window_gpu_us = gpu_us;
            static u64 read_faults_prev = 0, read_ns_prev = 0;
            const u64 read_faults_now = BbStats::read_faults.load(), read_ns_now = BbStats::t_read_faults.load();
            std::printf("Frame stats: %.1f FPS, worst frame %.1f ms (vblank %u Hz); "
                        "%u shader/pipeline compiles, %.1f ms; %llu recorder syncs; "
                        "%.0f write faults/s, %.0f read faults/s (%.1f ms/s), %lld hot pages; GPU thread %.2f us/draw, "
                        "%.0f draws/frame, idle %.1f%%; blocked: recorder %.1f%%, host copies "
                        "%.1f%% (%.0f/frame), copy threads %.1f%%, GPU ticks %.1f%%; "
                        "reduced-size draws %.0f/frame of %.0f in the scene\n",
                        frames / window, worst_ms, EmulatorSettings.GetVblankFrequency(), compiles,
                        compile_ns / 1e6, static_cast<unsigned long long>(direct),
                        faults / window, double(read_faults_now - read_faults_prev) / window,
                        double(read_ns_now - read_ns_prev) / 1e6 / window,
                        static_cast<long long>(BbStats::hot_pages.load()),
                        us_per_draw, draws_per_frame,
                        BbStats::gpu_idle_ns.exchange(0) / (window * 1e7),
                        BbStats::sync_recording_ns.exchange(0) / (window * 1e7),
                        BbStats::host_copies_wait_ns.exchange(0) / (window * 1e7),
                        frames ? double(BbStats::host_copy_waits.exchange(0)) / frames : 0.0,
                        BbStats::copy_threads_wait_ns.exchange(0) / (window * 1e7),
                        BbStats::tick_wait_ns.exchange(0) / (window * 1e7),
                        frames ? double(BbStats::reduced_draws.exchange(0)) / frames : 0.0,
                        frames ? double(BbStats::scene_draws.exchange(0)) / frames : 0.0);
            std::printf("Phases (%% of wall time): filter %.1f, pipeline lookup %.1f, render state %.1f, "
                        "bind resources %.1f, begin rendering %.1f, emit and bind %.1f, record and flush %.1f\n",
                        BbStats::phase_ns[0].exchange(0) / (window * 1e7),
                        BbStats::phase_ns[1].exchange(0) / (window * 1e7),
                        BbStats::phase_ns[2].exchange(0) / (window * 1e7),
                        BbStats::phase_ns[3].exchange(0) / (window * 1e7),
                        BbStats::phase_ns[4].exchange(0) / (window * 1e7),
                        BbStats::phase_ns[5].exchange(0) / (window * 1e7),
                        BbStats::phase_ns[6].exchange(0) / (window * 1e7));
            std::printf("Tick waits (%% of wall time): other %.1f, buffer watch %.1f, fault buffer %.1f, "
                        "present pacing %.1f, finish %.1f, wait submitted %.1f, image download %.1f, GDS store %.1f\n",
                        BbStats::wait_site_ns[0].exchange(0) / (window * 1e7),
                        BbStats::wait_site_ns[1].exchange(0) / (window * 1e7),
                        BbStats::wait_site_ns[2].exchange(0) / (window * 1e7),
                        BbStats::wait_site_ns[3].exchange(0) / (window * 1e7),
                        BbStats::wait_site_ns[4].exchange(0) / (window * 1e7),
                        BbStats::wait_site_ns[5].exchange(0) / (window * 1e7),
                        BbStats::wait_site_ns[6].exchange(0) / (window * 1e7),
                        BbStats::wait_site_ns[7].exchange(0) / (window * 1e7));
            std::printf("Guest reads per second: offloaded %.0f, window split %.0f, unresolved %.0f, "
                        "locked caller %.0f, offload off %.0f; read directly from video memory %.0f, direct read "
                        "mismatches %.0f\n",
                        BbStats::readback_paths[0].exchange(0) / window,
                        BbStats::readback_paths[1].exchange(0) / window,
                        BbStats::readback_paths[2].exchange(0) / window,
                        BbStats::readback_paths[3].exchange(0) / window,
                        BbStats::readback_paths[4].exchange(0) / window,
                        BbStats::readback_paths[5].exchange(0) / window,
                        BbStats::readback_paths[6].exchange(0) / window);
            std::printf("Small scene (%llu color targets) draws per frame: %.0f seen (%.0f depth-only, never skipped), "
                        "%.0f skipped (of %.0f draws)\n",
                        static_cast<unsigned long long>(BbStats::mirror_targets.load()),
                        frames ? double(BbStats::mirror_seen.exchange(0)) / frames : 0.0,
                        frames ? double(BbStats::mirror_depth_only.exchange(0)) / frames : 0.0,
                        frames ? double(BbStats::mirror_skipped.exchange(0)) / frames : 0.0,
                        draws_per_frame);
            {
                std::vector<std::tuple<u64, u64, u64>> consumers;
                for (u32 i = 0; i < 16; ++i) {
                    if (const u64 n = BbStats::mirror_consumer_count[i].exchange(0)) {
                        consumers.emplace_back(n, BbStats::mirror_consumer_hash[i].load(),
                                               BbStats::mirror_consumer_stale[i].exchange(0));
                    }
                }
                std::sort(consumers.rbegin(), consumers.rend());
                std::string line = "Shaders sampling the mirror targets (draws, of them on a skipped frame):";
                for (size_t i = 0; i < consumers.size() && i < 6; ++i) {
                    line += fmt::format(" {:016x} {} ({});", std::get<1>(consumers[i]), std::get<0>(consumers[i]),
                                        std::get<2>(consumers[i]));
                }
                std::printf("%s\n", line.c_str());
                std::printf("Small scene color draws by index count per frame: <100 %.0f, <300 %.0f, <1000 %.0f, <3000 %.0f, "
                            "<10000 %.0f, more %.0f; with instances %.0f\n",
                            frames ? BbStats::mirror_size_hist[0].exchange(0) / double(frames) : 0.0,
                            frames ? BbStats::mirror_size_hist[1].exchange(0) / double(frames) : 0.0,
                            frames ? BbStats::mirror_size_hist[2].exchange(0) / double(frames) : 0.0,
                            frames ? BbStats::mirror_size_hist[3].exchange(0) / double(frames) : 0.0,
                            frames ? BbStats::mirror_size_hist[4].exchange(0) / double(frames) : 0.0,
                            frames ? BbStats::mirror_size_hist[5].exchange(0) / double(frames) : 0.0,
                            frames ? BbStats::mirror_size_hist[6].exchange(0) / double(frames) : 0.0);
                std::vector<std::pair<u64, u64>> frees;
                for (u32 i = 0; i < 16; ++i) {
                    if (const u64 n = BbStats::image_free_count[i].exchange(0)) {
                        frees.emplace_back(n, BbStats::image_free_rip[i].load());
                    }
                }
                std::sort(frees.rbegin(), frees.rend());
                std::string free_line;
                for (size_t i = 0; i < frees.size() && i < 6; ++i) {
                    char where[320];
                    BbDescribeAddress(frees[i].second, where, sizeof(where));
                    free_line += fmt::format(" {}x {};", frees[i].first, where);
                }
                std::printf("Images created per second: overlap recreate %.0f, expand %.0f, new %.0f, stencil %.0f; "
                            "page tracking changes: track %.0f, untrack %.0f; freed by:%s\n",
                            BbStats::image_creates[0].exchange(0) / window, BbStats::image_creates[1].exchange(0) / window,
                            BbStats::image_creates[2].exchange(0) / window, BbStats::image_creates[3].exchange(0) / window,
                            BbStats::image_tracking[0].exchange(0) / window, BbStats::image_tracking[1].exchange(0) / window,
                            free_line.c_str());
                std::printf("768x512 images per second: created overlap %.1f, expand %.1f, new %.1f (on a skipped frame %.1f); "
                            "freed %.1f (on a skipped frame %.1f)\n",
                            BbStats::mirror_img_created[0].exchange(0) / window,
                            BbStats::mirror_img_created[1].exchange(0) / window,
                            BbStats::mirror_img_created[2].exchange(0) / window,
                            BbStats::mirror_img_created_stale.exchange(0) / window,
                            BbStats::mirror_img_freed.exchange(0) / window,
                            BbStats::mirror_img_freed_stale.exchange(0) / window);
                std::printf("Mirror scene: feedback shaders learned %u, their draws kept %llu; refreshes from guest memory "
                            "blocked: %llu; compute passes writing a skipped mirror target dropped: %llu\n",
                            BbStats::mirror_feedback_count.load(),
                            static_cast<unsigned long long>(BbStats::mirror_feedback_kept.exchange(0)),
                            static_cast<unsigned long long>(BbStats::mirror_refresh_blocked.exchange(0)),
                            static_cast<unsigned long long>(BbStats::mirror_dispatch_skipped.exchange(0)));
            }
            {
                static u64 last_protect_calls = 0, last_protect_pages = 0;
                const u64 calls = BbStats::protect_calls.load(), pages = BbStats::protect_pages.load();
                std::printf("Device memory created per second: cache buffers %.0f (%.1f MB), images %.0f "
                            "(%.1f MB), temporary buffers %.0f (%.1f MB), recycled images %.0f; memory protection changes %.0f "
                            "(%.0f pages)\n",
                            BbStats::alloc_count[0].exchange(0) / window,
                            BbStats::alloc_bytes[0].exchange(0) / window / 1048576.0,
                            BbStats::alloc_count[1].exchange(0) / window,
                            BbStats::alloc_bytes[1].exchange(0) / window / 1048576.0,
                            BbStats::alloc_count[2].exchange(0) / window,
                            BbStats::alloc_bytes[2].exchange(0) / window / 1048576.0,
                            BbStats::alloc_count[3].exchange(0) / window,
                            (calls - last_protect_calls) / window, (pages - last_protect_pages) / window);
                last_protect_calls = calls;
                last_protect_pages = pages;
            }
            {
                std::vector<std::pair<u64, u64>> faults;
                for (u32 i = 0; i < 32; ++i) {
                    if (const u64 n = BbStats::gpu_fault_count[i].exchange(0)) {
                        faults.emplace_back(n, BbStats::gpu_fault_rip[i].load());
                    }
                }
                std::sort(faults.rbegin(), faults.rend());
                std::string line = "GPU thread read faults by instruction:";
                for (size_t i = 0; i < faults.size() && i < 6; ++i) {
                    char where[320];
                    BbDescribeAddress(faults[i].second & ~(1ull << 63), where, sizeof(where));
                    line += fmt::format(" {}x {}{};", faults[i].first,
                                        (faults[i].second >> 63) ? "WRITE " : "read ", where);
                }
                std::printf("%s\n", line.c_str());
            }
            {
                std::vector<std::pair<u64, u32>> ops;
                for (u32 i = 0; i < 256; ++i) {
                    if (const u64 ns = BbStats::opcode_ns[i].exchange(0)) {
                        ops.emplace_back(ns, i);
                    }
                }
                std::sort(ops.rbegin(), ops.rend());
                std::string line = "Packets (% of wall time, opcode):";
                for (size_t i = 0; i < ops.size() && i < 8; ++i) {
                    line += fmt::format(" {:02x}={:.1f}", ops[i].second, ops[i].first / (window * 1e7));
                }
                std::printf("%s\n", line.c_str());
            }
            read_faults_prev = read_faults_now;
            read_ns_prev = read_ns_now;
            // Frame pacing: spread of the guest flip intervals (judder that the mean hides).
            if (intervals.size() > 2) {
                std::vector<double> sorted = intervals;
                std::sort(sorted.begin(), sorted.end());
                const double median = sorted[sorted.size() / 2];
                double sum = 0, sq = 0;
                u32 spikes = 0;
                for (const double ms : intervals) {
                    sum += ms;
                    sq += ms * ms;
                    spikes += ms > 1.5 * median;
                }
                const double mean = sum / intervals.size();
                std::printf("Frame pacing: median %.2f ms, stddev %.2f ms, p99 %.2f ms, "
                            "%u frames over 1.5x median\n",
                            median, std::sqrt(std::max(0.0, sq / intervals.size() - mean * mean)),
                            sorted[std::min(sorted.size() - 1, sorted.size() * 99 / 100)], spikes);
            }
            intervals.clear();
            window_start = now;
            frames = 0;
            worst_ms = 0;
        }
    }

    // Update flip status.
    auto* port = req.port;
    {
        std::unique_lock lock{port->port_mutex};
        auto& flip_status = port->flip_status;
        flip_status.count++;
        flip_status.process_time = Libraries::Kernel::sceKernelGetProcessTime();
        flip_status.tsc = Libraries::Kernel::sceKernelReadTsc();
        flip_status.flip_arg = req.flip_arg;
        flip_status.current_buffer = req.index;
        if (req.eop) {
            --flip_status.gc_queue_num;
        }
        --flip_status.flip_pending_num;
    }

    // Trigger flip events for the port.
    for (auto event : port->flip_events) {
        auto equeue = Kernel::GetEqueue(event);
        if (equeue != nullptr) {
            equeue->TriggerEvent(
                static_cast<u64>(OrbisVideoOutInternalEventId::Flip),
                Kernel::OrbisKernelEvent::Filter::VideoOut,
                reinterpret_cast<void*>(static_cast<u64>(OrbisVideoOutInternalEventId::Flip) |
                                        (req.flip_arg << 16)));
        }
    }

    // Reset prev flip label
    if (port->prev_index != -1) {
        port->buffer_labels[port->prev_index] = 0;
        port->SignalVoLabel();
    }
    // save to prev buf index
    port->prev_index = req.index;
}

void VideoOutDriver::DrawBlankFrame() {
    RunPresenter([this] {
        const auto empty_frame = presenter->PrepareBlankFrame(true);
        presenter->Present(empty_frame, false, false);
    }, true);
}

void VideoOutDriver::DrawLastFrame() {
    RunPresenter([this] {
        const auto frame = presenter->PrepareLastFrame();
        if (frame != nullptr) {
            presenter->Present(frame, true);
        }
    }, true);
}

bool VideoOutDriver::SubmitFlip(VideoOutPort* port, s32 index, s64 flip_arg,
                                bool is_eop /*= false*/) {
    {
        std::unique_lock lock{port->port_mutex};
        if (index != -1 && port->flip_status.flip_pending_num > 16) {
            LOG_ERROR(Lib_VideoOut, "Flip queue is full");
            return false;
        }

        if (is_eop) {
            ++port->flip_status.gc_queue_num;
        }
        ++port->flip_status.flip_pending_num; // integral GPU and CPU pending flips counter
        port->flip_status.submit_tsc = Libraries::Kernel::sceKernelReadTsc();
    }

    if (!is_eop) {
        // Non EOP flips can arrive from any thread so ask GPU thread to perform them
        liverpool->SendCommand([=, this]() { SubmitFlipInternal(port, index, flip_arg, is_eop); });
    } else {
        SubmitFlipInternal(port, index, flip_arg, is_eop);
    }

    return true;
}

void VideoOutDriver::SubmitFlipInternal(VideoOutPort* port, s32 index, s64 flip_arg, bool is_eop) {
    Vulkan::Frame* frame;
    if (index == -1) {
        frame = presenter->PrepareBlankFrame(false);
    } else {
        const auto& buffer = port->buffer_slots[index];
        ASSERT_MSG(buffer.group_index >= 0, "Trying to flip an unregistered buffer!");
        const auto& group = port->groups[buffer.group_index];
        frame = presenter->PrepareFrame(group, buffer.address_left);
    }

    {
        std::scoped_lock lock{mutex};
        requests.push({
            .frame = frame,
            .port = port,
            .flip_arg = flip_arg,
            .index = index,
            .eop = is_eop,
        });
    }
    request_cv.notify_one();
}

void VideoOutDriver::PresentThread(std::stop_token token) {
    const std::chrono::nanoseconds vblank_period(1000000000 /
                                                 EmulatorSettings.GetVblankFrequency());

    Common::SetCurrentThreadName("shadPS4:PresentThread");
    BbThreads::MakeImportant();
    Common::SetCurrentThreadRealtime(vblank_period);

    Common::AccurateTimer timer{vblank_period};

    // bbport: frame limit (see EmulatorSettings::GetFrameLimit). A request waits in the queue
    // until its slot; slots advance by one period (no drift) but never lag behind by more.
    const u32 frame_limit = EmulatorSettings.GetFrameLimit();
    const auto frame_period = frame_limit ? std::chrono::nanoseconds(1000000000 / frame_limit)
                                          : std::chrono::nanoseconds(0);
    auto next_flip = std::chrono::steady_clock::now();
    // bbport: BB_PACE_CREDIT=<percent of a period, default 0>: how much earlier than one
    // period after a release the next release may come when frames run late. The slot after a
    // release at `now` is max(slot + period, now + period - credit). 200 is the old rule
    // (now - period: after a late frame the next one, and even a second, goes out at once:
    // a long frame followed by a short one); 0 keeps every release at least one period after
    // the one before (minimum spacing: a late frame only delays, it is never repaid).
    // Default 0 (minimum spacing): measured in DriveClub's race at equal FPS, frame time sd
    // 5.23 -> 3.66 ms, 1% low 29.1 -> 31.5 FPS, menus sd 5.07 -> 2.89 ms.
    const long credit_pct =
        std::clamp<long>(EmulatorSettingsImpl::Number("BB_PACE_CREDIT", 0), 0, 200);
    const auto credit = frame_period * credit_pct / 100;
    // Off the old rule the waits for a slot use the precise sleep: a release that wakes late
    // would otherwise come out short of its period.
    const bool precise_wait = credit_pct != 200;
    std::printf("VideoOut: vblank %u Hz, frame limit %u FPS, pacing credit %ld%%\n",
                EmulatorSettings.GetVblankFrequency(), frame_limit, credit_pct);
    // The slot of the next flip after one released at `now`.
    const auto advance = [&](std::chrono::steady_clock::time_point now) {
        next_flip = std::max(next_flip + Vulkan::Vr::FramePeriod(frame_period), now + frame_period - credit); // VR
    };
    const auto sleep_to = [&](std::chrono::steady_clock::time_point until) {
        if (precise_wait) {
            const auto now = std::chrono::steady_clock::now();
            if (until > now) {
                Common::AccurateSleep(
                    std::chrono::duration_cast<std::chrono::nanoseconds>(until - now), nullptr,
                    false);
            }
        } else {
            std::this_thread::sleep_until(until);
        }
    };

    const auto receive_request = [this] -> Request {
        std::scoped_lock lk{mutex};
        if (!requests.empty()) {
            const auto request = requests.front();
            requests.pop();
            return request;
        }
        return {};
    };

    // bbport: with a frame limit (uncapped presets) a queued flip is presented as soon as it
    // arrives and its slot allows, between vblanks, instead of on the next vblank tick.
    const bool immediate_flips = frame_limit != 0;

    while (!token.stop_requested()) {
        timer.Start();
        const auto tick_deadline = std::chrono::steady_clock::now() + vblank_period;

        if (DebugState.IsGuestThreadsPaused()) {
            DrawLastFrame();
            timer.End();
            continue;
        }

        // Check if it's time to take a request.
        auto& vblank_status = main_port.vblank_status;
        const auto now = std::chrono::steady_clock::now();
        const bool flip_slot = (!frame_limit || now >= next_flip) && Vulkan::Vr::FlipAllowed(); // VR
        // With a frame limit (the 60 FPS patches) the game's own 30 Hz flip rate is ignored.
        const u32 flip_rate = frame_limit ? 0u : u32(main_port.flip_rate);
        if (flip_slot && vblank_status.count % (flip_rate + 1) == 0) {
            const auto request = receive_request();
            if (request && frame_limit) {
                advance(now);
            }
            if (!request) {
                if (timer.GetTotalWait().count() < 0) { // Dont draw too fast
                    if (!main_port.is_open) {
                        DrawBlankFrame();
                    } else if (ImGui::Core::MustKeepDrawing()) {
                        DrawLastFrame();
                    }
                }
            } else {
                Flip(request);
                FRAME_END;
            }
        }

        {
            // Needs lock here as can be concurrently read by `sceVideoOutGetVblankStatus`
            std::scoped_lock lock{main_port.vo_mutex};

            // Trigger flip events for the port
            for (auto event : main_port.vblank_events) {
                auto equeue = Kernel::GetEqueue(event);
                if (equeue != nullptr) {
                    equeue->TriggerEvent(
                        static_cast<u64>(OrbisVideoOutInternalEventId::Vblank),
                        Kernel::OrbisKernelEvent::Filter::VideoOut,
                        reinterpret_cast<void*>(
                            static_cast<u64>(OrbisVideoOutInternalEventId::Vblank) |
                            (vblank_status.count << 16)));
                }
            }

            // Update vblank status
            vblank_status.count++;
            vblank_status.process_time = Libraries::Kernel::sceKernelGetProcessTime();
            vblank_status.tsc = Libraries::Kernel::sceKernelReadTsc();
            main_port.vblank_cv.notify_all();
        }

        if (!immediate_flips || flip_rate != 0) {
            timer.End();
            continue;
        }
        while (!token.stop_requested()) {
            {
                std::unique_lock lk{mutex};
                if (!request_cv.wait_until(lk, tick_deadline,
                                           [&] { return !requests.empty(); })) {
                    break; // next vblank
                }
            }
            if (std::chrono::steady_clock::now() < next_flip) {
                if (next_flip >= tick_deadline) {
                    break;
                }
                sleep_to(next_flip);
            }
            if (!Vulkan::Vr::FlipAllowed()) {
                break; // VR: the headset's next frame (the next vblank tick looks again)
            }
            const auto now = std::chrono::steady_clock::now();
            const auto request = receive_request();
            if (request) {
                advance(now);
                Flip(request);
                FRAME_END;
            }
        }
        sleep_to(tick_deadline);
    }
}

} // namespace Libraries::VideoOut
