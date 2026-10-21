// SPDX-License-Identifier: GPL-2.0-or-later
// Experiment: adjustable VR HUD and manually toggled chase-camera offsets.
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#ifdef _WIN32
#include <windows.h>
#endif

namespace Vulkan::Vr::UserConfig {
struct Settings {
    float hud_width = 1.61f;
    float hud_height = 0.06f;
    float hud_distance = 1.10f;
    bool far_enabled = true;
    float far_distance_offset = 1.50f;
    float far_height_offset = 0.50f;
    float far_pitch_offset = 0.0f;
    std::string camera_toggle_button = "l3"; // l3, pad_up, pad_down, pad_left, pad_right, off
};
inline std::string Trim(std::string s) {
    const auto a = s.find_first_not_of(" \t\r\n");
    if (a == std::string::npos) return {};
    return s.substr(a, s.find_last_not_of(" \t\r\n") - a + 1);
}
inline void SetNumber(float& dst, const std::string& text, float lo, float hi) {
    char* end = nullptr;
    const float v = std::strtof(text.c_str(), &end);
    if (end != text.c_str() && Trim(end).empty() && std::isfinite(v))
        dst = std::clamp(v, lo, hi);
}
inline Settings Load() {
    Settings s;
#ifdef _WIN32
    wchar_t path[MAX_PATH]{};
    const DWORD n = GetModuleFileNameW(nullptr, path, MAX_PATH);
    if (!n || n >= MAX_PATH) return s;
    const auto exe_dir = std::filesystem::path(path).parent_path();
    auto ini = exe_dir.parent_path() / L"driveclub_vr.ini";
    if (!std::filesystem::exists(ini)) ini = exe_dir / L"driveclub_vr.ini";
#else
    auto ini = std::filesystem::path("driveclub_vr.ini");
#endif
    std::ifstream input(ini);
    if (!input) {
        std::puts("VR settings: no driveclub_vr.ini, using defaults");
        return s;
    }
    std::string line, section;
    while (std::getline(input, line)) {
        const auto comment = line.find_first_of(";#");
        if (comment != std::string::npos) line.erase(comment);
        line = Trim(line);
        if (line.empty()) continue;
        if (line.front() == '[' && line.back() == ']') {
            section = Trim(line.substr(1, line.size() - 2));
            continue;
        }
        const auto eq = line.find('=');
        if (eq == std::string::npos) continue;
        const std::string key = Trim(line.substr(0, eq));
        const std::string val = Trim(line.substr(eq + 1));
        if (section == "HUD") {
            if (key == "Width") SetNumber(s.hud_width, val, 0.5f, 5.f);
            else if (key == "Height") SetNumber(s.hud_height, val, -1.f, 1.f);
            else if (key == "Distance") SetNumber(s.hud_distance, val, 0.5f, 5.f);
        } else if (section == "ThirdPersonFar") {
            if (key == "Enabled") s.far_enabled = val == "1" || val == "true" || val == "on";
            else if (key == "DistanceOffset") SetNumber(s.far_distance_offset, val, -10.f, 15.f);
            else if (key == "HeightOffset") SetNumber(s.far_height_offset, val, -5.f, 10.f);
            else if (key == "PitchOffset") SetNumber(s.far_pitch_offset, val, -35.f, 35.f);
            else if (key == "ToggleButton") {
                if (val == "l3" || val == "pad_up" || val == "pad_down" ||
                    val == "pad_left" || val == "pad_right" || val == "off")
                    s.camera_toggle_button = val;
            }
        }
    }
    std::printf("VR settings: HUD %.2fm width, %.2fm up, %.2fm away; chase enabled %d, back %.2fm up %.2fm pitch %.2f deg\n",
                s.hud_width, s.hud_height, s.hud_distance,
                int(s.far_enabled), s.far_distance_offset, s.far_height_offset, s.far_pitch_offset);
    std::fflush(stdout);
    return s;
}
inline const Settings& Get() {
    static const Settings settings = Load();
    return settings;
}
} // namespace Vulkan::Vr::UserConfig
