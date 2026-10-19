// SPDX-License-Identifier: GPL-2.0-or-later
// VR (dcport-vr): OpenXR for the Vulkan renderer. Everything VR lives in renderer_vulkan/vr/; the renderer's own files
// only call in at a few points (marked "VR:"): instance and device extensions, after the device is made, and the
// presenter's frame.
//
// Stage 1: the game's finished frame on a screen floating in front of the player (a quad layer in the LOCAL space).
// On with BB_VR=1 (the launcher's VR row) when an OpenXR runtime answers; XR_RUNTIME_JSON picks the runtime (e.g.
// the simulator).

#pragma once

#include <chrono>

#include <string_view>
#include <vector>

#include "common/types.h"
#include "video_core/renderer_vulkan/vk_common.h"

namespace Vulkan::Vr {

/// An OpenXR instance and HMD system exist (made by the first call to InstanceExtensions).
bool Active();

/// The instance extensions the runtime needs (before vkCreateInstance). Makes the OpenXR instance on the first call.
const std::vector<const char*>& InstanceExtensions();

/// The device extensions the runtime needs for this physical device.
const std::vector<const char*>& DeviceExtensions(VkInstance instance, VkPhysicalDevice physical_device);

/// A fixed-capacity list (boost static_vector) takes no more than its capacity; others grow.
template <typename List>
bool HasRoom(const List& list) {
    if constexpr (requires { List::static_capacity; }) {
        return list.size() < List::static_capacity;
    } else {
        return true;
    }
}

/// Adds the runtime's extensions to a renderer's list (a std::vector or a static_vector), once each.
template <typename List>
void AppendInstanceExtensions(List& list) {
    for (const char* name : InstanceExtensions()) {
        bool listed = false;
        for (const char* e : list) {
            listed |= std::string_view{e} == name;
        }
        if (!listed && HasRoom(list)) {
            list.push_back(name);
        }
    }
}

template <typename List>
void AppendDeviceExtensions(VkInstance instance, VkPhysicalDevice physical_device, List& list) {
    for (const char* name : DeviceExtensions(instance, physical_device)) {
        bool listed = false;
        for (const char* e : list) {
            listed |= std::string_view{e} == name;
        }
        if (!listed && HasRoom(list)) {
            list.push_back(name);
        }
    }
}

/// After vkCreateDevice: the session on the renderer's queue (the only one; guarded by Scheduler::submit_mutex).
void OnDeviceCreated(VkInstance instance, VkPhysicalDevice physical_device, VkDevice device, u32 queue_family);

/// In the presenter's command buffer, with the finished frame in TRANSFER_SRC_OPTIMAL: waits for the headset's frame
/// and records the copy of the frame into this frame's layer image.
/// The distance between the eyes (the headset's IPD times BB_VR_WORLD_SCALE), in metres.
float StereoBaseline();

void RecordFrame(vk::CommandBuffer cmd, vk::Image frame, u32 width, u32 height, vk::Image window = {},
                 u32 window_width = 0, u32 window_height = 0);

/// The head's current position and yaw become the seat's origin (at the next frame).
void Recenter();

/// The present thread, each flip: its frame limit's period (`limit`, the launcher's: unpaced). While capped it is
/// the headset that paces the game: one flip per headset frame (FlipAllowed), so the game makes no frames the headset
/// never shows and the GPU keeps time for the headset's own work (2880 per eye: frames over a 90 Hz period 3.6% ->
/// 0.2%). Adaptive: capped once the game flips >= 112% of the headset's rate for 2 s, released below 95% (BB_VR_CAP=0
/// never caps). This flip takes the headset frame's token.
std::chrono::nanoseconds FramePeriod(std::chrono::nanoseconds limit);

/// The present thread: a flip may go now (uncapped, or the headset began a frame since the last flip).
bool FlipAllowed();

/// After the presenter's submit: hands the layer to the runtime (xrEndFrame).
void SubmitFrame(VkSemaphore semaphore, u64 tick);

/// Device creation: queues to make in the game's queue family (2 when VR runs: the headset's own queue).
u32 QueueCount(VkPhysicalDevice physical_device, u32 queue_family);


} // namespace Vulkan::Vr
