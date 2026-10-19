// DriveClub (PC) - launcher.
//
// One native window: the two game folders, the game patches, a benchmark, Play. The look follows a dark card layout
// (rounded cards, pill switches, segmented choices, a footer with a red Play button).
// Play prepares the game once (a few minutes the first time, seconds afterwards), then starts it; this
// program closes. Settings live next to this program (launcher.ini, data\), nothing is installed.
//
// Only components that ship with Windows are used (user32, gdi32, GDI+, shell32, ole32), and the C++ runtime
// is linked statically.
//
//   DriveClub.exe              launcher window
//   DriveClub.exe --play       prepare if needed and start the game without a window

#include <windows.h>
#include <windowsx.h>
#include <objbase.h>
#include <shlobj.h>
#include <shellapi.h>
#include <objidl.h>
#include <gdiplus.h>

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

#include "datx.h"
#include "gpu_check.h"
#include "prepare.h"
#include "unlock.h"

using namespace Gdiplus;

namespace {

// ------------------------------------------------------------------------------------------
// look
// ------------------------------------------------------------------------------------------
const Color kBg(255, 16, 18, 22), kCard(255, 24, 27, 33), kCardLine(255, 38, 42, 51), kText(255, 232, 234, 237),
    kMuted(255, 139, 147, 161), kTrack(255, 33, 37, 45), kTrackHover(255, 45, 50, 60), kAccent(255, 224, 64, 38),
    kAccentHover(255, 240, 84, 58), kOff(255, 63, 69, 82), kWhite(255, 245, 246, 248), kOk(255, 90, 200, 120),
    kWarn(255, 240, 170, 60), kBad(255, 240, 96, 84);

constexpr float kMargin = 24.f, kGap = 18.f, kHeaderH = 92.f, kFooterH = 96.f, kPad = 22.f, kTitleH = 44.f;
constexpr float kWinW = 960.f;

// ------------------------------------------------------------------------------------------
// model
// ------------------------------------------------------------------------------------------
// Picker: one box that opens a menu (benchmark track and car). Action: the benchmark's Run button and its result.
enum class Kind { Segmented, Toggle, Picker, Action };

struct Ctl {
  const wchar_t* label;
  const wchar_t* hint;
  Kind kind;
  std::vector<std::wstring> items;     // segmented choices
  std::vector<std::string> patches;    // patch name per choice ("" = none); a toggle uses patches[0]
  int def = 0;                         // default choice (toggle: 1 = on)
  int sel = 0;
  RectF row{}, ctl{};
};

struct Card {
  const wchar_t* title;
  std::vector<int> ctls;
  int column;
  const wchar_t* note = nullptr;
  RectF rect{};
};

std::vector<Ctl> g_c;
std::vector<Card> g_cards;

struct FolderRow {
  const wchar_t* label;
  const wchar_t* hint;
  std::wstring path;
  bool ok = false;
  RectF row{}, browse{};
};
FolderRow g_folder[2];  // 0 = update, 1 = base game

std::wstring g_home, g_data, g_runtime;
HWND g_hwnd = nullptr;
HWND g_edit = nullptr;  // the in-place box for the player name
float g_dpi = 1.f;  // the display scale; g_scale is what is drawn at (smaller when the window is too short)
float g_scale = 1.f, g_content_h = 0, g_ox = 0, g_oy = 0;
constexpr float kMaxW = 1700.f;
float g_win_w = kWinW;       // logical window width: the layout reflows when the window is widened
float g_win_h = 0;           // logical window height; 0 or less than the content: as tall as the content needs
float g_client_h = 0;        // client height in pixels
float g_scroll = 0;          // pixels the content is scrolled up when the window is shorter than it
float g_natural_h = 0;       // the height the content needs
float g_row_extra = 0;       // extra space given to every row when the window is taller than the content
int g_hover_ctl = -1, g_hover_seg = -1, g_hover_btn = -1, g_hover_browse = -1, g_press_btn = -1;
std::wstring g_status;
std::wstring g_gpu_problem;  // from CheckGpu() at start; Play stays off while it is set, unless the check is skipped
bool g_skip_gpu_check = false;  // "Skip the graphics card check" in the launcher (skip_gpu_check= in launcher.ini)
bool GpuBlocked() { return !g_gpu_problem.empty() && !g_skip_gpu_check; }
Color g_status_color = kMuted;
RectF g_btn[3];  // reset, save, play
const wchar_t* kBtnText[3] = {L"Reset to defaults", L"Save", L"Play"};
std::atomic<bool> g_busy{false};
std::atomic<int> g_progress{0};
std::wstring g_progress_text;
bool g_changed = false;

// ------------------------------------------------------------------------------------------
// text helpers
// ------------------------------------------------------------------------------------------
std::string Narrow(const std::wstring& w) {
  if (w.empty()) return "";
  int n = WideCharToMultiByte(CP_UTF8, 0, w.data(), int(w.size()), nullptr, 0, nullptr, nullptr);
  std::string s(n, '\0');
  WideCharToMultiByte(CP_UTF8, 0, w.data(), int(w.size()), s.data(), n, nullptr, nullptr);
  return s;
}

std::wstring Widen(const std::string& s) {
  if (s.empty()) return L"";
  int n = MultiByteToWideChar(CP_UTF8, 0, s.data(), int(s.size()), nullptr, 0);
  std::wstring w(n, L'\0');
  MultiByteToWideChar(CP_UTF8, 0, s.data(), int(s.size()), w.data(), n);
  return w;
}

std::wstring Dir(const std::wstring& path) {
  size_t i = path.find_last_of(L"\\/");
  return i == std::wstring::npos ? L"." : path.substr(0, i);
}

bool Exists(const std::wstring& p) { return GetFileAttributesW(p.c_str()) != INVALID_FILE_ATTRIBUTES; }

void MakeDirs(const std::wstring& path) {
  for (size_t i = 3; i < path.size(); ++i) {
    if (path[i] == L'\\') CreateDirectoryW(path.substr(0, i).c_str(), nullptr);
  }
  CreateDirectoryW(path.c_str(), nullptr);
}

std::vector<std::string> ReadLines(const std::wstring& path) {
  std::vector<std::string> lines;
  std::ifstream in(path.c_str(), std::ios::binary);
  for (std::string line; std::getline(in, line);) {
    while (!line.empty() && (line.back() == '\r' || line.back() == ' ')) line.pop_back();
    lines.push_back(line);
  }
  return lines;
}

// ------------------------------------------------------------------------------------------
// settings: launcher.ini (folders) and data\user\patches_enabled.txt (what the game applies)
// ------------------------------------------------------------------------------------------
std::wstring IniPath() { return g_home + L"\\launcher.ini"; }
std::wstring SelectionPath() { return g_data + L"\\user\\patches_enabled.txt"; }

// True when the first 64 KiB of the file contain the text.
bool FileHas(const std::wstring& path, const char* text) {
  std::ifstream in(path.c_str(), std::ios::binary);
  std::string data(65536, '\0');
  in.read(data.data(), std::streamsize(data.size()));
  data.resize(size_t(in.gcount()));
  return data.find(text) != std::string::npos;
}

// DriveClub's param.sfo: the US release (CUSA00093) or the European one (CUSA00003), whose 1.28 dump
// prepares and plays the same way.
bool IsDriveClub(const std::wstring& sfo) { return FileHas(sfo, "CUSA00093") || FileHas(sfo, "CUSA00003"); }

// The base game dump has an eboot.bin and an sce_sys\param.sfo too, so the update folder is told apart by what
// the base game has and it does not (game000.dat) and by its version (01.28 in param.sfo).
void CheckFolders() {
  // A dump that already is version 1.28 in one folder (game000.dat next to the 1.28 eboot.bin) serves as
  // both: the update field may stay empty or name the same folder.
  const std::wstring& b0 = g_folder[1].path;
  const bool merged = !b0.empty() && Exists(b0 + L"\\game000.dat") && Exists(b0 + L"\\eboot.bin") &&
                      Exists(b0 + L"\\sce_sys\\param.sfo") &&
                      IsDriveClub(b0 + L"\\sce_sys\\param.sfo") &&
                      FolderIsVersion128(std::filesystem::path(b0));
  if (merged && g_folder[0].path.empty()) g_folder[0].path = b0;
  const std::wstring& u = g_folder[0].path;
  if (merged && u == b0) {
    g_folder[0].ok = true;
    g_folder[1].ok = true;
    return;
  }
  g_folder[0].ok = !u.empty() && Exists(u + L"\\eboot.bin") && Exists(u + L"\\sce_sys\\param.sfo") &&
                   !Exists(u + L"\\game000.dat") && IsDriveClub(u + L"\\sce_sys\\param.sfo") &&
                   FolderIsVersion128(std::filesystem::path(u));
  const std::wstring& b = g_folder[1].path;
  g_folder[1].ok = !b.empty() && Exists(b + L"\\game000.dat") && IsDriveClub(b + L"\\sce_sys\\param.sfo");
}

// Why a chosen folder is not accepted, in words a player can act on ("" when it is fine or not chosen yet).
std::wstring FolderProblem(int which) {
  const std::wstring& p = g_folder[which].path;
  if (p.empty() || g_folder[which].ok) return L"";
  if (!Exists(p)) return L"This folder does not exist (any more). Press Browse and choose it again.";
  const bool eboot = Exists(p + L"\\eboot.bin"), base = Exists(p + L"\\game000.dat");
  const bool sfo = Exists(p + L"\\sce_sys\\param.sfo");
  if (which == 0) {
    if (base) return L"This folder holds the base game. If your dump is already version 1.28 in this one folder, choose the same folder for both fields. Otherwise choose the v1.28 update folder here.";
    if (!eboot || !sfo) return L"Not the update: no eboot.bin and sce_sys here. If it is a parent folder, open it and choose the folder inside.";
    if (!IsDriveClub(p + L"\\sce_sys\\param.sfo")) return L"This is another game, not DriveClub (CUSA00093 or CUSA00003). Choose the DriveClub v1.28 update.";
    return L"This update is not version 1.28 (its sce_sys\\param.sfo has another version). Choose the v1.28 update dump.";
  }
  if (eboot && !base) return L"This is the update, not the base game. Choose the base game folder here (it contains game000.dat).";
  if (base && !IsDriveClub(p + L"\\sce_sys\\param.sfo")) return L"This is another game, not DriveClub (CUSA00093 or CUSA00003). Choose the DriveClub base game folder.";
  return L"Not the base game: no game000.dat here. If it is a parent folder, open it and choose the folder inside.";
}

int lang_saved = 1;   // console language number, from launcher.ini (1 = English, US)
// The game has a text file for each of these (gui/lams), in console language-number order.
const wchar_t* const kLangName[21] = {L"Japanese", L"English (US)", L"French", L"Spanish", L"German", L"Italian", L"Dutch",
                                      L"Portuguese", L"Russian", L"Korean", L"Chinese (Traditional)",
                                      L"Chinese (Simplified)", L"Finnish", L"Swedish", L"Danish", L"Norwegian",
                                      L"Polish", L"Portuguese (Brazil)", L"English (UK)", L"Turkish",
                                      L"Spanish (Mexico)"};
int g_lang_id = 1;
int up_saved = 0;     // "Upscaling" choice, from launcher.ini
int aa_saved = 0;     // "Anti-aliasing" choice, from launcher.ini
int fgen_saved = 0;   // "Frame generation" choice, from launcher.ini
int mirror_saved = 0;  // "Mirror and reflection updates" choice, from launcher.ini
int aniso_saved = 0;   // "Anisotropic filtering" choice, from launcher.ini
std::wstring g_name = L"Player";  // in-game player name, from "name=" in launcher.ini
int g_inres = -1, g_wide = -1, g_outres = -1, g_preset = -1, g_mirror = -1, g_aniso = -1, g_fps = -1, g_fgen = -1, g_aa = -1, g_up = -1, g_lang = -1;
bool unlocked_saved = false;  // "Unlocked" frame rate choice, from launcher.ini  // indices of the render resolution and widescreen choices

// ------------------------------------------------------------------------------------------
// benchmark: the game's AI drives one event and the runtime measures the frame times (BB_BENCHMARK, see
// src/runtime_benchmark.c). The ids are the game's own: "Track Name" in newdata/tracks.csv and "Vehicle Name" in
// newdata/vehicles.csv, all from the base game (no add-on content needed). The ones marked "not run" are taken
// from the game data; the others were run to a result. Circuits only: a point-to-point route can end before the
// measurement does.
// ------------------------------------------------------------------------------------------
struct BenchChoice {
  const char* id;
  const wchar_t* name;
  const wchar_t* group;  // the submenu: location or car class
};
const BenchChoice kBenchTracks[] = {
    {"canada_road_circuit_02_n", L"Fraser Valley", L"Canada"},
    {"canada_road_circuit_01_n", L"Oliver's Landing", L"Canada"},
    {"canada_dlc_road_circuit_03_n", L"Sinclair Point", L"Canada"},
    {"canada_race_track_v1", L"Maplewood Motorsport Park", L"Canada"},
    {"chile_road_circuit_01_n", L"Chungará Lake", L"Chile"},
    {"chile_road_circuit_02_n", L"Salar de Surire", L"Chile"},
    {"chile_race_track_v1", L"Autodromo Frontera", L"Chile"},
    {"india_road_circuit_01_n", L"Glenmorgan", L"India"},
    {"india_road_circuit_02_n", L"Bandipur", L"India"},
    {"india_dlc_road_circuit_03_n", L"Yedappalli", L"India"},
    {"india_race_track_v1", L"Tamil Nadu Circuit", L"India"},  // not run
    {"norway_road_circuit_01_n", L"Sentraltind", L"Norway"},
    {"norway_road_circuit_02_n", L"Skjolden", L"Norway"},
    {"norway_dlc_road_circuit_03_n", L"Atlanterhavsvegen", L"Norway"},  // not run
    {"norway_race_track_01_v1", L"Sognefjord Raceway", L"Norway"},
    {"uk_road_circuit_01_n", L"Loch Duich", L"Scotland"},  // not run
    {"uk_road_circuit_02_n", L"Kinloch", L"Scotland"},
    {"uk_race_track_v1", L"Black Hills Circuit", L"Scotland"},
    {"japan_road_circuit_01_n", L"Takahagi Hills", L"Japan"},
    {"japan_road_circuit_02_n", L"Lake Shoji", L"Japan"},
    {"japan_dlc_road_circuit_03_n", L"Kobago", L"Japan"},
    {"japan_race_track_01_v1", L"Asigiri Highland Raceway", L"Japan"},  // not run
};
const BenchChoice kBenchCars[] = {
    {"mc_p1", L"McLaren P1", L"Hyper"},
    {"pa_hu", L"Pagani Huayra", L"Hyper"},
    {"ko_ag", L"Koenigsegg Agera R", L"Hyper"},
    {"he_ve_gt", L"Hennessey Venom GT", L"Hyper"},
    {"pa_zo", L"Pagani Zonda R", L"Hyper"},
    {"fe_458", L"Ferrari 458 Italia", L"Super"},
    {"fe_f12", L"Ferrari F12berlinetta", L"Super"},
    {"au_r8", L"Audi R8 V10 plus", L"Super"},
    {"mc_mp_12", L"McLaren 12C", L"Super"},
    {"as_ma_177", L"Aston Martin One-77", L"Super"},
    {"me_sl_am", L"Mercedes-Benz SLS AMG Black Series", L"Super"},  // not run
    {"bm_m4", L"BMW M4", L"Performance"},
    {"me_gt_am", L"Mercedes-AMG GT S", L"Performance"},
    {"au_rs6", L"Audi RS 6 Avant", L"Performance"},  // not run
    {"ma_mc_st", L"Maserati GranTurismo MC Stradale", L"Performance"},
    {"lo_ex", L"Lotus Exige S", L"Performance"},  // not run
    {"au_rs5", L"Audi RS 5 Coupé", L"Sports"},
    {"bm_m2", L"BMW M235i", L"Sports"},
    {"ren_dez", L"Renault DeZir", L"Sports"},
    {"vw_golf_gti", L"Volkswagen Golf GTI", L"Hot hatch"},
    {"mi_jcw", L"MINI John Cooper Works GP", L"Hot hatch"},
    {"ren_cl", L"Renault Clio RS", L"Hot hatch"},  // not run
};
constexpr int kBenchTrackCount = int(sizeof(kBenchTracks) / sizeof(kBenchTracks[0]));
constexpr int kBenchCarCount = int(sizeof(kBenchCars) / sizeof(kBenchCars[0]));
constexpr int kBenchDefaultCar = 5;  // Ferrari 458 Italia
const int kBenchTod[] = {8, 12, 18, 22};
const char* const kBenchWeather[] = {"Clear", "Cloudy", "Overcast", "Stormy"};
const char* const kBenchRain[] = {"", "Light", "Heavy", "Wet"};
const int kBenchSeconds[] = {30, 60, 120, 300};
int g_bench_track = 0, g_bench_car = kBenchDefaultCar;
int g_btrack = -1, g_bcar = -1, g_bmode = -1, g_btod = -1, g_bweather = -1, g_brain = -1, g_blen = -1, g_brun = -1;
bool IsBenchRow(const Ctl& c) { return c.patches[0].rfind("@bench", 0) == 0; }

// The last result, from data\user\benchmark\result.txt.
struct BenchResult {
  bool ok = false;
  std::wstring fps, lows, about;  // "87.3", "1% low 71.2  0.1% low 60.4", "Fraser Valley, Ferrari 458 Italia, ..."
};
BenchResult g_bench_result;
bool g_bench_running = false;
bool g_bench_ran_hover = false;  // the pointer is over the result: the panel with every setting the game ran with is shown
std::wstring g_bench_ran_short;               // one line: what the game actually ran with (render -> output, frame limit, ...)
std::vector<std::wstring> g_bench_ran_lines;  // every setting the game reported (result.txt line "ran ...")

// The game options the launcher sets for every start that were already set when it started: the player's own
// overrides (BB_FPS_LIMIT=40, ...). Only these are left alone; the launcher's own values are replaced each time.
std::vector<std::wstring> g_user_env;
void SnapshotUserEnv() {
  for (const wchar_t* name : {L"BB_FPS_LIMIT", L"BB_LANGUAGE", L"BB_AA_SCALE", L"BB_AA", L"BB_FRAME_GEN", L"BB_FRAME_GEN_MULTIPLIER"}) {
    if (GetEnvironmentVariableW(name, nullptr, 0) != 0) g_user_env.push_back(name);
  }
}
bool UserEnv(const wchar_t* name) { return std::find(g_user_env.begin(), g_user_env.end(), name) != g_user_env.end(); }

void BuildModel() {
  g_folder[0].label = L"Update folder (version 1.28)";
  g_folder[0].hint = L"The DriveClub v1.28 update dump: the folder that contains eboot.bin. If your dump is already 1.28, leave this empty and choose it below.";
  g_folder[1].label = L"Base game folder";
  g_folder[1].hint = L"The DriveClub base game dump: the folder that contains game000.dat.";
  auto seg = [](const wchar_t* label, const wchar_t* hint, std::vector<std::wstring> items,
                std::vector<std::string> patches, int def) {
    Ctl c{label, hint, Kind::Segmented, std::move(items), std::move(patches), def, def};
    g_c.push_back(std::move(c));
    return int(g_c.size()) - 1;
  };
  auto toggle = [](const wchar_t* label, const wchar_t* hint, const char* patch, bool def) {
    Ctl c{label, hint, Kind::Toggle, {}, {patch}, def ? 1 : 0, def ? 1 : 0};
    g_c.push_back(std::move(c));
    return int(g_c.size()) - 1;
  };
  g_preset = seg(L"Performance preset",
                 L"Picks the render and output sizes for your PC. Lower the preset if the game is slow; "
                 L"you can still change each size below.",
                 {L"Custom", L"Handheld", L"Low", L"Medium", L"High", L"Ultra"}, {"@preset", "", "", "", "", ""}, 0);
  const int fps = (g_fps = int(g_c.size()), seg(L"Frame rate",
                      L"60 FPS (deltatime) scales the game by elapsed time, so speed stays right if the frame rate dips. "
                      L"Unlocked uses the same timing with no frame cap: the game runs as fast as your PC allows and the speed stays right. "
                      L"Fixed tick uses the engine's own 60 Hz mode (works with every feature, speed follows the frame rate).",
                      {L"30 FPS", L"60 FPS", L"60 FPS (fixed tick)", L"Unlocked"},
                      {"", "60 FPS with deltatime", "60 FPS with fixed tickrate", "@unlocked"}, 1));
  const int outres = (g_outres = int(g_c.size()), seg(L"Output resolution", L"The size of the game's finished picture (with the menus and HUD). It is then scaled to fill your screen, so the window and the signal to your display always stay at your screen's own size.",
                         {L"Default", L"720p", L"1440p", L"1800p", L"4K", L"5K", L"8K"},
                         {"", "720p [1280x720] Output resolution", "1440p [2560x1440] Output resolution",
                          "1800p [3200x1800] Output resolution", "4K [3840x2160] Output resolution",
                          "5K [5120x2880] Output resolution", "8K [7680x4320] Output resolution"},
                         0));
  const int inres = (g_inres = int(g_c.size()), seg(L"Render resolution",
                        L"The size the game draws the 3D scene at, before it is scaled to the output resolution: lower is faster, higher is sharper. The game's dynamic resolution can lower it further under load. Sizes above 1080p use more memory; the needed memory patch is added for you.",
                        {L"Default", L"720p", L"1440p", L"1800p", L"4K", L"8K"},
                        {"", "720p [1280x720] Internal resolution", "1440p [2560x1440] Internal resolution [DEPTH BUG]",
                         "1800p [3200x1800] Internal resolution [DEPTH BUG]", "4K [3840x2160] Internal resolution [DEPTH BUG]",
                         "8K [7680x4320] Internal resolution [DEPTH BUG]"},
                        00));
  const int wide = (g_wide = int(g_c.size()), seg(L"Widescreen (render and output)",
                       L"Ultrawide sizes. These change both the render and the output size. The needed memory patch is added for you.",
                       {L"Off", L"21:9 2560x1080", L"32:9 3840x1080", L"21:9 3440x1440", L"32:9 5120x1440"},
                       {"", "1080p 21:9 [2560x1080] Internal + Output resolution [DEPTH BUG]",
                        "1080p 32:9 [3840x1080] Internal + Output resolution [DEPTH BUG]",
                        "1440p 21:9 [3440x1440] Internal + Output resolution [DEPTH BUG]",
                        "1440p 32:9 [5120x1440] Internal + Output resolution [DEPTH BUG]"},
                       00));
  const int fov = toggle(L"Widescreen field-of-view fix", L"For wide and ultrawide resolutions only.",
                         "FOV fix for wide resolutions", false);
  const int blur = toggle(L"Disable motion blur and depth of field",
                          L"They share one shader, so both are turned off together. Leave this off when you use FSR or DLAA anti-aliasing: they need the game's blur pass.",
                          "Disable motion blur and depth of field", false);
  const int reset = toggle(L"No reset-to-track timer", L"Disables the reset to track timer.", "Reset Disable", false);
  const int barrier = toggle(L"Disable track barriers", L"Removes the invisible barriers at the track edges.",
                             "Barrier Disable", false);
  const int pcol = toggle(L"Photo mode: free camera through cars",
                          L"Disables car collision with the free camera, so you can look inside any car.",
                          "Disable Photomode Free Camera Collision", false);
  const int pproc = toggle(L"Photo mode: skip processing", L"Disables photo mode processing.",
                           "Disable Photomode Processing", false);
  g_lang = seg(L"Language",
               L"The language of the game's menus and text. Click to switch to the next one: all 21 languages of the game are available. Japanese, Korean and Chinese may show missing characters.",
               {L"English (US)"}, {"@lang"}, 0);
  const int pname = seg(L"Player name", L"The name the game shows for you. Click it to type a new one.", {g_name}, {"@name"}, 0);
  g_mirror = seg(L"Mirror and reflection updates",
                 L"The game draws a second small scene (the mirrors, rain and reflections) every frame, about a quarter of all draw calls. "
                 L"Static draws it once and keeps that picture: much faster and steady, but reflections and rain on the glass stop updating.",
                 {L"Live", L"Static (faster)"}, {"@mirror", ""}, 0);
  g_aa = seg(L"Anti-aliasing",
             L"Replaces the game's own anti-aliasing with AMD FSR 3.1 (any graphics card) or NVIDIA DLAA (GeForce RTX) for a sharper, steadier picture. The game's own is the default. It does not work together with the patch \"Disable motion blur and depth of field\": the game keeps its own anti-aliasing then.",
             {L"Game", L"FSR 3.1", L"DLAA (RTX)"}, {"@aa", "", ""}, 0);
  g_aniso = seg(L"Anisotropic filtering",
                L"Keeps textures on roads and terrain sharp at a shallow angle. Automatic is 16x (4x on integrated graphics). Off uses only what the game asks for. It costs little on any graphics card.",
                {L"Automatic", L"Off", L"2x", L"4x", L"8x", L"16x"}, {"@aniso", "", "", "", "", ""}, 0);
  g_fgen = seg(L"Frame generation (experimental)",
               L"Adds a made-up frame between each two real ones (FSR 3.1 works on any graphics card; DLSS needs a GeForce RTX 40 series for 2x, RTX 50 series for 3x and 4x), so the picture looks two to four times as smooth. It adds a little delay and can smear fast motion. Best when the game already runs at 45 FPS or more; turn on the Unlocked frame rate as well.",
               {L"Off", L"FSR 3.1 (2x)", L"DLSS (2x)", L"DLSS (3x)", L"DLSS (4x)"}, {"@fgen", "", "", "", ""}, 0);
  g_fgen = -1;  // frame generation is not offered in this release
  const int skipgpu = toggle(L"Skip the hardware check",
                             L"The launcher checks that your processor and graphics card can run the game (Vulkan 1.3). If it refuses hardware that does work, turn this on to start anyway.",
                             "@skipgpu", false);
  const int showfps = toggle(L"Show FPS counter", L"Draws the frame rate in the top right corner while playing.", "@fps", false);
  const int full = toggle(L"Fullscreen", L"Plays borderless fullscreen. Turn off for a window.", "@full", true);
  const int unlock = toggle(L"Unlock DLC content",
                            L"Unlocks the DLC content: add-on cars, bikes, tours and livery decals. Cars you earn by playing still unlock through normal progression. Built from your own game files; the game folders are not modified.",
                            "@unlock", false);
  auto picker = [](const wchar_t* label, const wchar_t* hint, const char* tag) {
    Ctl c{label, hint, Kind::Picker, {L""}, {tag}, 0, 0};
    g_c.push_back(std::move(c));
    return int(g_c.size()) - 1;
  };
  g_btrack = picker(L"Track", L"Where the benchmark drives. Click to choose a circuit.", "@bench_track");
  g_bcar = picker(L"Car", L"The car the game's AI drives. Click to choose a car.", "@bench_car");
  g_bmode = seg(L"Cars on track",
                L"Alone is a time trial and gives the steadiest numbers. Race adds seven AI cars, which loads the processor more.",
                {L"Alone", L"Race (8 cars)"}, {"@bench_mode", ""}, 0);
  g_btod = seg(L"Time of day", L"Morning 8:00, noon 12:00, evening 18:00, night 22:00.",
               {L"Morning", L"Noon", L"Evening", L"Night"}, {"@bench_tod", "", "", ""}, 1);
  g_bweather = seg(L"Weather", L"The sky and the clouds.", {L"Clear", L"Cloudy", L"Overcast", L"Stormy"},
                   {"@bench_weather", "", "", ""}, 0);
  g_brain = seg(L"Rain", L"Rain and wet roads make the scene heavier to draw. Wet road is a wet track without rain.",
                {L"Dry", L"Light", L"Heavy", L"Wet road"}, {"@bench_rain", "", "", ""}, 0);
  g_blen = seg(L"Length", L"How long the frame times are measured, after 5 seconds of warm-up. Loading takes about half a minute more.",
               {L"30 s", L"1 min", L"2 min", L"5 min"}, {"@bench_length", "", "", ""}, 1);
  g_brun = int(g_c.size());
  g_c.push_back(Ctl{L"Last result",
                    L"Starts the event with all the settings on this page. The game's AI drives and the game closes by itself when it is done. "
                    L"The 1% and 0.1% lows are the average frame rate of the slowest 1% and 0.1% of frames.",
                    Kind::Action, {}, {"@bench_run"}, 0, 0});
  // The cards flow into as many columns as the window width allows (see LayoutPass).
  g_cards.push_back({L"FRAME RATE AND RESOLUTION", {g_preset, fps, outres, inres, wide, fov}, 0});
  g_cards.push_back({L"ANTI-ALIASING", {g_aa, g_aniso}, 0});
  g_cards.push_back({L"GAMEPLAY", {pname, g_lang, g_mirror, full, unlock, showfps}, 0});
  g_cards.push_back({L"PATCHES", {blur, reset, barrier, pcol, pproc, skipgpu}, 0,
                     L"Community patches for DriveClub version 1.28."});
  g_cards.push_back({L"BENCHMARK", {g_btrack, g_bcar, g_bmode, g_btod, g_bweather, g_brain, g_blen, g_brun}, 0});
}

// The rows show the chosen track and car by name.
void ShowBenchChoices() {
  if (g_btrack >= 0) g_c[g_btrack].items[0] = std::wstring(kBenchTracks[g_bench_track].name) + L"  (" + kBenchTracks[g_bench_track].group + L")";
  if (g_bcar >= 0) g_c[g_bcar].items[0] = kBenchCars[g_bench_car].name;
}

int FindBench(const BenchChoice* list, int count, const std::string& id, int fallback) {
  for (int i = 0; i < count; ++i) {
    if (id == list[i].id) return i;
  }
  return fallback;
}

std::wstring BenchName(const BenchChoice* list, int count, const std::string& id) {
  const int i = FindBench(list, count, id, -1);
  return i >= 0 ? std::wstring(list[i].name) : Widen(id);
}

bool fps_counter = false, fast_render = false, full_screen = true, unlock_dlc = false;  // the settings that are not patches

// Output / render choices per preset: indices into the "Output resolution" and "Render resolution" rows,
// and the "Mirror and reflection updates" choice (slower presets update the mirrors less often).
const int kPresetOut[6] = {0, 1, 0, 2, 4, 4};     // Default 720p 1440p 1800p 4K 5K 8K
const int kPresetIn[6] = {0, 1, 1, 0, 2, 4};      // Default(1080p) 720p 1440p 1800p 4K 8K
const int kPresetMirror[6] = {0, 1, 0, 0, 0, 0};  // Live, Static (Handheld takes the speed)

// The preset row has no patch of its own: it shows the preset whose sizes are selected, else Custom.
void DerivePreset() {
  if (g_preset < 0 || g_outres < 0 || g_inres < 0) return;
  g_c[g_preset].sel = 0;
  if (g_wide >= 0 && g_c[g_wide].sel != 0) return;
  for (int p = 1; p <= 5; ++p) {
    if (g_c[g_outres].sel == kPresetOut[p] && g_c[g_inres].sel == kPresetIn[p]) g_c[g_preset].sel = p;
  }
}

void LoadSettings() {
  for (const auto& line : ReadLines(IniPath())) {
    if (line.rfind("update=", 0) == 0) g_folder[0].path = Widen(line.substr(7));
    if (line.rfind("base=", 0) == 0) g_folder[1].path = Widen(line.substr(5));
    if (line.rfind("fps=", 0) == 0) fps_counter = line.substr(4) == "1";
    if (line.rfind("fast=", 0) == 0) fast_render = line.substr(5) == "1";
    if (line.rfind("name=", 0) == 0) g_name = Widen(line.substr(5));
    if (line.rfind("mirror=", 0) == 0) mirror_saved = atoi(line.substr(7).c_str());
    if (line.rfind("aniso=", 0) == 0) aniso_saved = atoi(line.substr(6).c_str());
    if (line.rfind("language=", 0) == 0) lang_saved = atoi(line.substr(9).c_str());
    if (line.rfind("up=", 0) == 0) up_saved = atoi(line.substr(3).c_str());
    if (line.rfind("aa=", 0) == 0) aa_saved = atoi(line.substr(3).c_str());
    if (line.rfind("fgen=", 0) == 0) fgen_saved = atoi(line.substr(5).c_str());
    if (line.rfind("unlocked=", 0) == 0) unlocked_saved = line.substr(9) == "1";
    if (line.rfind("skip_gpu_check=", 0) == 0) g_skip_gpu_check = line.substr(15) == "1";
    if (line.rfind("full=", 0) == 0) full_screen = line.substr(5) == "1";
    if (line.rfind("dlc_unlock=", 0) == 0) unlock_dlc = line.substr(11) == "1";  // the "unlock=" of earlier versions is not read: off for everyone
    if (line.rfind("bench_track=", 0) == 0) g_bench_track = FindBench(kBenchTracks, kBenchTrackCount, line.substr(12), 0);
    if (line.rfind("bench_car=", 0) == 0) g_bench_car = FindBench(kBenchCars, kBenchCarCount, line.substr(10), kBenchDefaultCar);
    const struct { const char* key; int row; } bench_rows[] = {{"bench_mode=", g_bmode}, {"bench_tod=", g_btod},
                                                                {"bench_weather=", g_bweather}, {"bench_rain=", g_brain},
                                                                {"bench_length=", g_blen}};
    for (const auto& b : bench_rows) {
      const size_t n = std::strlen(b.key);
      if (b.row < 0 || line.compare(0, n, b.key) != 0) continue;
      const int v = atoi(line.substr(n).c_str());
      if (v >= 0 && v < int(g_c[b.row].items.size())) g_c[b.row].sel = v;
    }
  }
  ShowBenchChoices();
  for (auto& c : g_c) if (c.patches[0] == "@name") c.items[0] = g_name;
  CheckFolders();
  const auto lines = ReadLines(SelectionPath());
  const bool has_selection = !lines.empty() || Exists(SelectionPath());
  for (auto& c : g_c) {
    if (c.kind == Kind::Toggle && c.patches[0] == "@fps") {
      c.sel = fps_counter ? 1 : 0;
      continue;
    }
    if (c.kind == Kind::Toggle && c.patches[0] == "@fast") {
      c.sel = fast_render ? 1 : 0;
      continue;
    }
    if (c.kind == Kind::Toggle && c.patches[0] == "@skipgpu") {
      c.sel = g_skip_gpu_check ? 1 : 0;
      continue;
    }
    if (c.kind == Kind::Toggle && c.patches[0] == "@full") {
      c.sel = full_screen ? 1 : 0;
      continue;
    }
    if (c.kind == Kind::Toggle && c.patches[0] == "@unlock") {
      c.sel = unlock_dlc ? 1 : 0;
      continue;
    }
    if (!has_selection || IsBenchRow(c)) continue;  // defaults; the benchmark rows are in launcher.ini
    c.sel = 0;
    for (size_t i = 0; i < c.patches.size(); ++i) {
      if (c.patches[i].empty()) continue;
      if (std::find(lines.begin(), lines.end(), c.patches[i]) != lines.end()) c.sel = c.kind == Kind::Toggle ? 1 : int(i);
    }
  }
  // After the loop above, which resets every row that has no patch of its own.
  if (g_mirror >= 0 && mirror_saved == 1) g_c[g_mirror].sel = mirror_saved;
  if (g_aniso >= 0 && aniso_saved >= 1 && aniso_saved <= 5) g_c[g_aniso].sel = aniso_saved;
  if (g_lang >= 0) {
    g_lang_id = lang_saved >= 0 && lang_saved <= 20 ? lang_saved : 1;
    g_c[g_lang].items[0] = kLangName[g_lang_id];
  }
  if (g_up >= 0 && up_saved >= 1 && up_saved <= 3) g_c[g_up].sel = up_saved;
  if (g_aa >= 0 && aa_saved >= 1 && aa_saved <= 2) g_c[g_aa].sel = aa_saved;
  if (g_fgen >= 0 && fgen_saved >= 1 && fgen_saved <= 4) g_c[g_fgen].sel = fgen_saved;
  if (g_fps >= 0 && unlocked_saved && g_c[g_fps].sel == 1) g_c[g_fps].sel = 3;  // Unlocked is deltatime without the cap
  DerivePreset();
}

// Resolutions above 1080p need more memory: the game's own "Memory for up to ..." patch plus a larger
// direct memory budget. 0 = default, 1 = up to 1440p, 2 = up to 4K.
bool full_screen_now() {
  for (const auto& c : g_c) {
    if (c.kind == Kind::Toggle && c.patches[0] == "@full") return c.sel != 0;
  }
  return true;
}

// Upscaling: with FSR 3.1 or DLAA chosen, the render size follows from the output size.
bool UpscaleActive() {
  return g_up >= 0 && g_aa >= 0 && g_c[g_up].sel > 0 && g_c[g_aa].sel > 0;
}

void UpscaleSize(int& rw, int& rh, const char*& name) {
  static const int kOut[7][2] = {{1920, 1080}, {1280, 720}, {2560, 1440}, {3200, 1800}, {3840, 2160}, {5120, 2880}, {7680, 4320}};
  const int o = g_outres >= 0 ? g_c[g_outres].sel : 0;
  const int q = g_c[g_up].sel;  // 1 Quality 3/2, 2 Balanced 17/10, 3 Performance 2/1
  const int num = q == 1 ? 3 : q == 2 ? 17 : 2, den = q == 1 ? 2 : q == 2 ? 10 : 1;
  rw = ((kOut[o][0] * den / num) + 4) / 8 * 8;
  rh = ((kOut[o][1] * den / num) + 4) / 8 * 8;
  name = q == 1 ? "Quality" : q == 2 ? "Balanced" : "Performance";
}

// The output size counts too: 4K output with the Default render size crashed at boot on the default
// memory budget, since the presented images are output-sized.
int OutputMemoryTier() {
  if (g_outres < 0) return 0;
  const int o = g_c[g_outres].sel;  // output: Default 720p 1440p 1800p 4K 5K 8K
  return o >= 5 ? 3 : o >= 3 ? 2 : o == 2 ? 1 : 0;
}

int RenderMemoryTier() {
  if (UpscaleActive()) {
    int rw, rh;
    const char* name;
    UpscaleSize(rw, rh, name);
    return rh <= 1080 ? 0 : rh <= 1440 ? 1 : rh <= 2160 ? 2 : 3;
  }
  if (g_inres < 0 || g_wide < 0) return 0;
  const int r = g_c[g_inres].sel, w = g_c[g_wide].sel;  // render: Default 720p 1440p 1800p 4K; wide: Off + 3 sizes
  if (r >= 5) return 3;
  if (w > 0 || r >= 3) return 2;
  return r == 2 ? 1 : 0;
}

int MemoryTier() { return (std::max)(RenderMemoryTier(), OutputMemoryTier()); }

// Output / render choices per preset: indices into the "Output resolution" and "Render resolution" rows.
void ApplyPreset(int preset) {
  if (preset <= 0 || preset > 5 || g_outres < 0 || g_inres < 0) return;
  g_c[g_outres].sel = kPresetOut[preset];
  g_c[g_inres].sel = kPresetIn[preset];
  if (g_wide >= 0) g_c[g_wide].sel = 0;
  if (g_mirror >= 0) g_c[g_mirror].sel = kPresetMirror[preset];
}

void SaveSettings() {
  MakeDirs(g_data + L"\\user");
  {
    std::ofstream ini(IniPath().c_str(), std::ios::binary | std::ios::trunc);
    bool fps_on = false, fast_on = false, full_on = true, unlock_on = false;
    for (const auto& c : g_c) {
      if (c.kind == Kind::Toggle && c.patches[0] == "@fps") fps_on = c.sel != 0;
      if (c.kind == Kind::Toggle && c.patches[0] == "@fast") fast_on = c.sel != 0;
      if (c.kind == Kind::Toggle && c.patches[0] == "@full") full_on = c.sel != 0;
      if (c.kind == Kind::Toggle && c.patches[0] == "@unlock") unlock_on = c.sel != 0;
    }
    ini << "update=" << Narrow(g_folder[0].path) << "\r\nbase=" << Narrow(g_folder[1].path) << "\r\nfps=" << (fps_on ? 1 : 0)
        << "\r\nfast=" << (fast_on ? 1 : 0) << "\r\nunlocked=" << ((g_fps >= 0 && g_c[g_fps].sel == 3) ? 1 : 0) << "\r\nskip_gpu_check=" << (g_skip_gpu_check ? 1 : 0) << "\r\nmirror=" << (g_mirror >= 0 ? g_c[g_mirror].sel : 0) << "\r\nlanguage=" << g_lang_id << "\r\naniso=" << (g_aniso >= 0 ? g_c[g_aniso].sel : 0) << "\r\nup=" << (g_up >= 0 ? g_c[g_up].sel : 0) << "\r\naa=" << (g_aa >= 0 ? g_c[g_aa].sel : 0) << "\r\nfgen=" << (g_fgen >= 0 ? g_c[g_fgen].sel : 0) <<"\r\nname=" << Narrow(g_name) <<"\r\nfull=" << (full_on ? 1 : 0) << "\r\ndlc_unlock=" << (unlock_on ? 1 : 0) << "\r\n";
    ini << "bench_track=" << kBenchTracks[g_bench_track].id << "\r\nbench_car=" << kBenchCars[g_bench_car].id
        << "\r\nbench_mode=" << g_c[g_bmode].sel << "\r\nbench_tod=" << g_c[g_btod].sel << "\r\nbench_weather=" << g_c[g_bweather].sel
        << "\r\nbench_rain=" << g_c[g_brain].sel << "\r\nbench_length=" << g_c[g_blen].sel << "\r\n";
    // The renderer optimizations that are switched off, as a bit mask. The draw thread is bit 37.
    const std::wstring toggles = g_data + L"\\user\\toggles.txt";
    std::ofstream tog(toggles.c_str(), std::ios::binary | std::ios::trunc);
    tog << "137439506432";  // the renderer's experimental two-thread draw path stays off
    const std::wstring game_ini = g_data + L"\\user\\driveclub.ini";
    std::ofstream game(game_ini.c_str(), std::ios::binary | std::ios::trunc);
    game << "show_fps=" << (fps_on ? 1 : 0) << "\r\n";
  }
  std::ofstream out(SelectionPath().c_str(), std::ios::binary | std::ios::trunc);
  for (const auto& c : g_c) {
    if (c.kind == Kind::Toggle) {
      if (c.sel && c.patches[0][0] != '@') out << c.patches[0] << "\n";
    } else if (c.sel > 0 && c.patches[c.sel] == "@unlocked") {
      out << "60 FPS with deltatime\n";  // the same timing patch; only the frame cap differs
    } else if (c.sel > 0 && !c.patches[c.sel].empty() && c.patches[c.sel][0] != '@') {
      out << c.patches[c.sel] << "\n";
    }
  }
  if (UpscaleActive()) {
    int rw, rh;
    const char* name;
    UpscaleSize(rw, rh, name);
    out << "Upscale render " << rw << "x" << rh << "\n";
  }
  if (MemoryTier() == 1) out << "Memory for up to 1440p\n";
  if (MemoryTier() == 2) out << "Memory for up to 4K\n";
  if (MemoryTier() == 3) out << "Memory for up to 8K\n";
  g_changed = false;
}

void ResetDefaults() {
  for (auto& c : g_c) {
    c.sel = c.def;
    if (c.kind == Kind::Toggle && c.patches[0] == "@skipgpu") g_skip_gpu_check = c.sel != 0;
  }
  g_bench_track = 0;
  g_bench_car = kBenchDefaultCar;
  ShowBenchChoices();
  g_changed = true;
}

// ------------------------------------------------------------------------------------------
// layout
// ------------------------------------------------------------------------------------------
float RowHeight(const Ctl& c) { return (c.kind == Kind::Toggle ? 46.f : c.kind == Kind::Action ? 208.f : 70.f) + g_row_extra; }
float FolderRowHeight() { return 74.f; }

int g_cols = 2;  // columns of cards: more when the window is wide

void LayoutPass(float bottom_extra) {
  g_cols = (std::clamp)(int((g_win_w - 2 * kMargin + kGap) / (380.f + kGap)), 2, 4);
  const float colw = (g_win_w - 2 * kMargin - float(g_cols - 1) * kGap) / float(g_cols);
  // game folders: a full-width card on top
  const float folders_h = kTitleH + 2 * FolderRowHeight() + kPad * 0.4f + 8.f;
  const RectF folders_rect(kMargin, kHeaderH, g_win_w - 2 * kMargin, folders_h);
  float fy = folders_rect.Y + kTitleH;
  for (auto& f : g_folder) {
    f.row = RectF(folders_rect.X + kPad, fy, folders_rect.Width - 2 * kPad, FolderRowHeight());
    f.browse = RectF(f.row.X + f.row.Width - 104.f, fy + 24.f, 104.f, 36.f);
    fy += FolderRowHeight();
  }
  float y[4];
  for (float& v : y) v = kHeaderH + folders_h + kGap;
  // Card heights; the tallest cards are placed first, each into the shortest column.
  std::vector<float> heights(g_cards.size());
  std::vector<size_t> order(g_cards.size());
  for (size_t k = 0; k < g_cards.size(); ++k) {
    float h = kTitleH + kPad * 0.4f;
    for (int i : g_cards[k].ctls) h += RowHeight(g_c[i]);
    h += kPad * 0.6f;
    if (g_cards[k].note) h += 46.f;
    heights[k] = h;
    order[k] = k;
  }
  std::sort(order.begin(), order.end(), [&](size_t a, size_t b) { return heights[a] > heights[b]; });
  float fill[4] = {0, 0, 0, 0};
  for (size_t k : order) {
    int col = 0;
    for (int c = 1; c < g_cols; ++c) {
      if (fill[c] < fill[col] - 1.f) col = c;
    }
    g_cards[k].column = col;
    fill[col] += heights[k] + kGap;
  }
  for (size_t k = 0; k < g_cards.size(); ++k) {
    auto& card = g_cards[k];
    const float h = heights[k];
    card.rect = RectF(kMargin + card.column * (colw + kGap), y[card.column], colw, h);
    float cy = y[card.column] + kTitleH + kPad * 0.4f;
    for (int i : card.ctls) {
      Ctl& c = g_c[i];
      const float rh = RowHeight(c);
      c.row = RectF(card.rect.X + kPad, cy, colw - 2 * kPad, rh);
      c.ctl = c.kind == Kind::Toggle   ? RectF(c.row.X + c.row.Width - 46.f, cy + 10.f, 46.f, 26.f)
              : c.kind == Kind::Action ? RectF(c.row.X, cy + 8.f, c.row.Width, 44.f)
                                       : RectF(c.row.X, cy + 26.f, c.row.Width, 34.f);
      cy += rh;
    }
    y[card.column] += h + kGap;
  }
  float bottom_y = y[0];
  for (int col = 1; col < g_cols; ++col) bottom_y = (std::max)(bottom_y, y[col]);
  const float bottom = bottom_y - kGap + bottom_extra;
  for (int col = 0; col < g_cols; ++col) {
    for (int k = int(g_cards.size()) - 1; k >= 0; --k) {
      if (g_cards[k].column != col) continue;
      g_cards[k].rect.Height += bottom - (y[col] - kGap);
      break;
    }
  }
  g_content_h = bottom + kGap;
  const float bar_y = g_content_h + 2.f;
  const float bw[3] = {170.f, 110.f, 130.f};
  float x = g_win_w - kMargin;
  for (int i = 2; i >= 0; --i) {
    x -= bw[i];
    g_btn[i] = RectF(x, bar_y + 14.f, bw[i], 44.f);
    x -= 12.f;
  }
}

// Lays the window out for its current size. Wider: the columns widen. Taller: the rows spread out a little and
// the cards take the rest.
void Layout() {
  g_row_extra = 0;
  LayoutPass(0.f);
  g_natural_h = g_content_h + kFooterH;
  const float extra = g_win_h - g_natural_h;
  if (extra > 1.f) {
    int rows = 1;
    for (int col = 0; col < g_cols; ++col) {
      int n = 0;
      for (const auto& card : g_cards) if (card.column == col) n += int(card.ctls.size());
      rows = (std::max)(rows, n);
    }
    g_row_extra = (std::min)(extra / float(rows), 14.f);
    LayoutPass(0.f);
    const float left = g_win_h - (g_content_h + kFooterH);
    if (left > 1.f) LayoutPass(left);
  }
}

float TotalHeight() { return g_content_h + kFooterH; }

// A window shorter than the content scrolls it (mouse wheel); the footer stays at the bottom.
float MaxScroll() { return g_client_h > 0 ? (std::max)(0.f, TotalHeight() * g_scale - g_client_h) : 0.f; }
float ContentOy() { return -g_scroll; }
float FooterOy() { return g_client_h > 0 ? (std::min)(0.f, g_client_h - TotalHeight() * g_scale) : 0.f; }
float FooterTop() { return FooterOy() + g_content_h * g_scale; }

// ------------------------------------------------------------------------------------------
// painting
// ------------------------------------------------------------------------------------------
GraphicsPath* Round(const RectF& r, float rad) {
  auto* p = new GraphicsPath();
  const float d = rad * 2;
  p->AddArc(r.X, r.Y, d, d, 180, 90);
  p->AddArc(r.X + r.Width - d, r.Y, d, d, 270, 90);
  p->AddArc(r.X + r.Width - d, r.Y + r.Height - d, d, d, 0, 90);
  p->AddArc(r.X, r.Y + r.Height - d, d, d, 90, 90);
  p->CloseFigure();
  return p;
}

void FillRound(Graphics& g, const RectF& r, float rad, const Color& c) {
  SolidBrush br(c);
  GraphicsPath* p = Round(r, rad);
  g.FillPath(&br, p);
  delete p;
}

void StrokeRound(Graphics& g, const RectF& r, float rad, const Color& c, float w = 1.f) {
  Pen pen(c, w);
  GraphicsPath* p = Round(r, rad);
  g.DrawPath(&pen, p);
  delete p;
}

void Text(Graphics& g, const std::wstring& s, const Font& f, const RectF& r, const Color& c,
          StringAlignment h = StringAlignmentNear, StringAlignment v = StringAlignmentCenter, bool wrap = false,
          bool ellipsis = true) {
  StringFormat fmt;
  fmt.SetAlignment(h);
  fmt.SetLineAlignment(v);
  if (!wrap) fmt.SetFormatFlags(StringFormatFlagsNoWrap);
  fmt.SetTrimming(ellipsis ? StringTrimmingEllipsisCharacter : StringTrimmingNone);
  SolidBrush br(c);
  g.DrawString(s.c_str(), -1, &f, r, &fmt, &br);
}

RectF Scaled(const RectF& r) { return RectF(r.X * g_scale + g_ox, r.Y * g_scale + g_oy, r.Width * g_scale, r.Height * g_scale); }

bool FoldersOk() { return g_folder[0].ok && g_folder[1].ok; }

void Paint(HDC hdc, int w, int h) {
  Graphics g(hdc);
  g.SetSmoothingMode(SmoothingModeAntiAlias);
  g.SetTextRenderingHint(TextRenderingHintClearTypeGridFit);
  g.Clear(kBg);
  FontFamily fam(L"Segoe UI");
  const float S = g_scale;
  g_oy = ContentOy();
  Font fTitle(&fam, 22 * S, FontStyleBold, UnitPixel), fSub(&fam, 13 * S, FontStyleRegular, UnitPixel),
      fCard(&fam, 12 * S, FontStyleBold, UnitPixel), fLabel(&fam, 14 * S, FontStyleRegular, UnitPixel),
      fSegBold(&fam, 13 * S, FontStyleBold, UnitPixel), fHint(&fam, 12.5f * S, FontStyleRegular, UnitPixel),
      fBtn(&fam, 14 * S, FontStyleBold, UnitPixel), fSeg(&fam, 13 * S, FontStyleRegular, UnitPixel);

  // header
  FillRound(g, Scaled(RectF(kMargin, 26, 8, 40)), 3 * S, kAccent);
  Text(g, L"DRIVECLUB", fTitle, Scaled(RectF(kMargin + 22, 20, 520, 32)), kText);
  Text(g, L"PC  -  game folders and patches", fSub, Scaled(RectF(kMargin + 22, 50, 520, 22)), kMuted);
  {
    const bool folder_problem = !FolderProblem(0).empty() || !FolderProblem(1).empty();
    const wchar_t* chip = GpuBlocked() ? (g_gpu_problem.rfind(L"Your processor", 0) == 0 ? L"Processor not supported" : L"Graphics card not supported")
                          : FoldersOk() ? L"Game found" : folder_problem ? L"Check the game folders" : L"Choose the game folders";
    StringFormat measure(StringFormat::GenericTypographic());
    measure.SetFormatFlags(StringFormatFlagsNoWrap | StringFormatFlagsMeasureTrailingSpaces);
    RectF box;
    g.MeasureString(chip, -1, &fSeg, PointF(0, 0), &measure, &box);
    const float cw = box.Width / S + 52.f;
    RectF cr = Scaled(RectF(g_win_w - kMargin - cw, 32, cw, 30));
    const bool chip_ok = FoldersOk() && !GpuBlocked();
    FillRound(g, cr, 15 * S, chip_ok ? Color(255, 28, 52, 38) : Color(255, 60, 40, 28));
    SolidBrush dot(chip_ok ? kOk : kWarn);
    g.FillEllipse(&dot, cr.X + 13 * S, cr.Y + 11 * S, 8 * S, 8 * S);
    Text(g, chip, fSeg, RectF(cr.X + 30 * S, cr.Y, cr.Width - 30 * S, cr.Height), chip_ok ? kOk : kWarn,
         StringAlignmentNear, StringAlignmentCenter, false, false);
  }

  // game folders card
  {
    const RectF cr = Scaled(RectF(kMargin, kHeaderH, g_win_w - 2 * kMargin, kTitleH + 2 * FolderRowHeight() + kPad * 0.4f + 8.f));
    FillRound(g, cr, 12 * S, kCard);
    StrokeRound(g, cr, 12 * S, kCardLine);
    Text(g, L"GAME FILES", fCard, RectF(cr.X + kPad * S, cr.Y + 6 * S, cr.Width - 2 * kPad * S, (kTitleH - 6) * S), kMuted);
    for (int i = 0; i < 2; ++i) {
      const FolderRow& f = g_folder[i];
      const RectF row = Scaled(f.row), br = Scaled(f.browse);
      Text(g, f.label, fLabel, RectF(row.X, row.Y + 4 * S, row.Width - 130 * S, 22 * S), kText);
      SolidBrush dot(f.ok ? kOk : (f.path.empty() ? kOff : kWarn));
      g.FillEllipse(&dot, row.X, row.Y + 38 * S, 8 * S, 8 * S);
      const std::wstring shown = f.path.empty() ? std::wstring(L"Not chosen yet") : f.path;
      Text(g, shown, fHint, RectF(row.X + 16 * S, row.Y + 28 * S, row.Width - 130 * S, 28 * S),
           f.path.empty() ? kMuted : kText);
      if (const std::wstring problem = FolderProblem(i); !problem.empty()) {
        SolidBrush bad(kBad);
        g.FillEllipse(&bad, row.X, row.Y + 38 * S, 8 * S, 8 * S);
        Text(g, problem, fHint, RectF(row.X + 16 * S, row.Y + 50 * S, row.Width - 130 * S, 22 * S), kBad);
      }
      const bool over = g_hover_browse == i;
      FillRound(g, br, 10 * S, over ? Color(255, 66, 73, 88) : Color(255, 50, 56, 68));
      Text(g, L"Browse...", fBtn, br, kWhite, StringAlignmentCenter);
    }
  }

  // cards
  for (auto& card : g_cards) {
    RectF cr = Scaled(card.rect);
    FillRound(g, cr, 12 * S, kCard);
    StrokeRound(g, cr, 12 * S, kCardLine);
    Text(g, card.title, fCard, RectF(cr.X + kPad * S, cr.Y + 6 * S, cr.Width - 2 * kPad * S, (kTitleH - 6) * S), kMuted);
    if (card.note) {
      Text(g, card.note, fHint, RectF(cr.X + kPad * S, cr.Y + cr.Height - 54 * S, cr.Width - 2 * kPad * S, 44 * S), kMuted,
           StringAlignmentNear, StringAlignmentFar, true);
    }
    for (int i : card.ctls) {
      Ctl& c = g_c[i];
      const RectF row = Scaled(c.row), ct = Scaled(c.ctl);
      const bool hot = g_hover_ctl == i;
      if (c.kind == Kind::Toggle) {
        Text(g, c.label, fLabel, RectF(row.X, row.Y + 2 * S, row.Width - 60 * S, 40 * S), kText);
        FillRound(g, ct, ct.Height / 2, c.sel ? (hot ? kAccentHover : kAccent) : (hot ? Color(255, 80, 87, 102) : kOff));
        SolidBrush knob(kWhite);
        const float d = ct.Height - 6 * S;
        g.FillEllipse(&knob, c.sel ? ct.X + ct.Width - d - 3 * S : ct.X + 3 * S, ct.Y + 3 * S, d, d);
      } else if (c.kind == Kind::Picker) {
        // A box with the chosen name and a down arrow: a click opens the menu of choices.
        Text(g, c.label, fLabel, RectF(row.X, row.Y + 2 * S, row.Width, 22 * S), kText);
        FillRound(g, ct, 9 * S, hot ? kTrackHover : kTrack);
        Text(g, c.items[0], fSegBold, RectF(ct.X + 12 * S, ct.Y, ct.Width - 44 * S, ct.Height), kText);
        SolidBrush arrow(kMuted);
        const float ax = ct.X + ct.Width - 22 * S, ay = ct.Y + ct.Height / 2 - 2 * S;
        const PointF tri[3] = {PointF(ax - 5 * S, ay), PointF(ax + 5 * S, ay), PointF(ax, ay + 6 * S)};
        g.FillPolygon(&arrow, tri, 3);
      } else if (c.kind == Kind::Action) {
        // The benchmark: Run, then the last result.
        Text(g, c.label, fLabel, RectF(row.X, row.Y + 64 * S, row.Width, 22 * S), kText);
        const RectF res(row.X, row.Y + 88 * S, row.Width, row.Height - 88 * S);
        if (g_bench_running) {
          Text(g, L"Running. The game closes by itself when it is done.", fHint, res, kText,
               StringAlignmentNear, StringAlignmentNear, true);
        } else if (g_bench_result.ok) {
          Font fBig(&fam, 24 * S, FontStyleBold, UnitPixel);
          const std::wstring big = g_bench_result.fps + L" FPS";
          Text(g, big, fBig, RectF(res.X, res.Y, res.Width, 32 * S), kWhite);
          Text(g, g_bench_result.lows, fSeg, RectF(res.X, res.Y + 32 * S, res.Width, 20 * S), kText);
          Text(g, g_bench_result.about, fHint, RectF(res.X, res.Y + 56 * S, res.Width, 38 * S), kMuted,
               StringAlignmentNear, StringAlignmentNear, true);
          if (!g_bench_ran_short.empty())
            Text(g, g_bench_ran_short, fHint, RectF(res.X, res.Y + 96 * S, res.Width, res.Height - 96 * S),
                 g_bench_ran_hover ? kText : kMuted, StringAlignmentNear, StringAlignmentNear, true);
        } else {
          Text(g, L"No result yet.", fHint, res, kMuted, StringAlignmentNear,
               StringAlignmentNear, true);
        }
        const bool disabled = g_busy || !FoldersOk() || GpuBlocked();
        FillRound(g, ct, 10 * S, disabled ? kOff : hot && g_hover_seg == 0 ? kAccentHover : kAccent);
        Text(g, L"Run benchmark", fBtn, ct, disabled ? kMuted : kWhite, StringAlignmentCenter);
      } else {
        Text(g, c.label, fLabel, RectF(row.X, row.Y + 2 * S, row.Width, 22 * S), kText);
        FillRound(g, ct, 9 * S, kTrack);
        const float sw = ct.Width / float(c.items.size());
        for (size_t k = 0; k < c.items.size(); ++k) {
          RectF seg(ct.X + sw * k + 3 * S, ct.Y + 3 * S, sw - 6 * S, ct.Height - 6 * S);
          const bool sel = int(k) == c.sel;
          const bool over = hot && g_hover_seg == int(k);
          if (sel) FillRound(g, seg, 7 * S, over ? kAccentHover : kAccent);
          else if (over) FillRound(g, seg, 7 * S, kTrackHover);
          float px = 13.f;
          StringFormat mf;
          mf.SetFormatFlags(StringFormatFlagsNoWrap);
          for (; px > 9.f; px -= 0.5f) {
            Font ft(&fam, px * S, sel ? FontStyleBold : FontStyleRegular, UnitPixel);
            RectF bx;
            g.MeasureString(c.items[k].c_str(), -1, &ft, PointF(0, 0), &mf, &bx);
            if (bx.Width <= seg.Width - 8 * S) break;
          }
          Font fit(&fam, px * S, sel ? FontStyleBold : FontStyleRegular, UnitPixel);
          Text(g, c.items[k], fit, seg, sel ? kWhite : kMuted, StringAlignmentCenter);
        }
      }
    }
  }

  // footer: hint / status / progress, buttons
  g_oy = FooterOy();
  const float fy = g_oy + g_content_h * S;
  SolidBrush footer_brush(Color(255, 12, 13, 16));
  g.FillRectangle(&footer_brush, 0.f, fy, float(w), float(h) - fy);
  std::wstring msg = g_status;
  Color mc = g_status_color;
  if (g_busy) {
    msg = g_progress_text;
    mc = kText;
  } else if (g_hover_ctl >= 0) {
    msg = g_c[g_hover_ctl].hint;
    mc = kMuted;
  } else if (g_hover_browse >= 0) {
    msg = g_folder[g_hover_browse].hint;
    mc = kMuted;
  } else if (g_hover_btn == 0) {
    msg = L"Put every patch back to its default.";
    mc = kMuted;
  } else if (g_hover_btn == 2) {
    msg = FoldersOk() ? L"Save and start the game." : L"Choose both game folders first.";
    mc = kMuted;
  }
  {
    const RectF fr(g_ox + kMargin * S, fy + 8 * S, (g_btn[0].X - kMargin - 12) * S, (kFooterH - 16) * S);
    StringFormat mfmt;
    mfmt.SetTrimming(StringTrimmingNone);
    float px = 12.5f;
    for (; px > 9.f; px -= 0.5f) {
      Font ft(&fam, px * S, FontStyleRegular, UnitPixel);
      RectF bounds;
      g.MeasureString(msg.c_str(), -1, &ft, RectF(0, 0, fr.Width, 0), &mfmt, &bounds);
      if (bounds.Height <= fr.Height) break;
    }
    Font fit(&fam, px * S, FontStyleRegular, UnitPixel);
    Text(g, msg, fit, fr, mc, StringAlignmentNear, StringAlignmentCenter, true, false);
  }
  if (g_busy) {
    const RectF bar = Scaled(RectF(kMargin, g_content_h + kFooterH - 18.f, g_btn[0].X - kMargin - 12, 6.f));
    FillRound(g, bar, 3 * S, kTrack);
    FillRound(g, RectF(bar.X, bar.Y, bar.Width * float(g_progress.load()) / 100.f, bar.Height), 3 * S, kAccent);
  }
  for (int i = 0; i < 3; ++i) {
    RectF br = Scaled(g_btn[i]);
    const bool over = g_hover_btn == i && !g_busy, down = g_press_btn == i && over;
    const bool disabled = (i == 2 && (!FoldersOk() || GpuBlocked())) || g_busy;
    Color fill = i == 2 ? (disabled ? kOff : (down ? Color(255, 200, 52, 30) : over ? kAccentHover : kAccent))
                        : (i == 1 ? (down ? Color(255, 56, 62, 74) : over ? Color(255, 66, 73, 88) : Color(255, 50, 56, 68))
                                  : (over ? kTrackHover : kTrack));
    FillRound(g, br, 10 * S, fill);
    Text(g, kBtnText[i], fBtn, br, disabled ? kMuted : kWhite, StringAlignmentCenter);
  }
  g_oy = ContentOy();
  // The benchmark result under the pointer: every setting the game ran with, in a panel over the page.
  if (g_bench_ran_hover && g_brun >= 0 && !g_bench_running && g_bench_result.ok && !g_bench_ran_lines.empty()) {
    const RectF row = Scaled(g_c[g_brun].row);
    const float line_h = 19 * S, pad = 12 * S;
    const float ph = pad * 2 + line_h * float(g_bench_ran_lines.size()), pw = row.Width;
    float py = row.Y + 88 * S - ph - 8 * S;  // above the result, where the page has room
    if (py < 8 * S) py = row.Y + row.Height + 4 * S;
    if (py + ph > float(h) - 8 * S) py = (std::max)(8 * S, float(h) - 8 * S - ph);
    const RectF panel(row.X, py, pw, ph);
    FillRound(g, panel, 10 * S, Color(250, 30, 34, 42));
    for (size_t i = 0; i < g_bench_ran_lines.size(); ++i) {
      Text(g, g_bench_ran_lines[i], i == 0 ? fCard : fHint,
           RectF(panel.X + pad, panel.Y + pad + line_h * float(i), panel.Width - 2 * pad, line_h), i == 0 ? kWhite : kText);
    }
  }
}

// ------------------------------------------------------------------------------------------
// preparing and starting the game
// ------------------------------------------------------------------------------------------
std::wstring PatchXmlPath() {
  const std::wstring bundled = g_runtime + L"\\patches\\DriveClub.xml";
  return Exists(bundled) ? bundled : g_home + L"\\..\\..\\patches\\DriveClub.xml";  // development tree
}

std::wstring GameExe() {
  const std::wstring bundled = g_runtime + L"\\DriveClubGame.exe";
  return Exists(bundled) ? bundled : g_home + L"\\..\\DriveClubGame.exe";
}

void SetProgress(int percent, const std::wstring& text) {
  g_progress = percent;
  g_progress_text = text;
  if (g_hwnd) PostMessageW(g_hwnd, WM_APP + 1, 0, 0);
}

// Prepares the game from the two dumps (a few seconds the first time). Returns "" on success, else the error.
std::wstring RunPrepare() {
  const std::wstring stamp_path = g_data + L"\\prep\\stamp.txt";
  const std::wstring stamp_now = g_folder[0].path + L"|" + g_folder[1].path;
  std::wstring stamp_old;
  for (const auto& l : ReadLines(stamp_path)) stamp_old = Widen(l);
  const bool ready = Exists(g_data + L"\\prep\\boot-linked.bin") && Exists(g_data + L"\\prep\\image.meta");
  const bool patches_only = ready && stamp_old == stamp_now;
  const std::string error = PrepareGame(
      std::filesystem::path(g_folder[0].path), std::filesystem::path(g_folder[1].path), std::filesystem::path(g_data),
      std::filesystem::path(PatchXmlPath()), patches_only, [](int step, const std::string& text) {
        SetProgress(step * 100 / 6, Widen(text));
      });
  if (!error.empty()) return Widen(error);
  std::ofstream out(stamp_path.c_str(), std::ios::binary | std::ios::trunc);
  out << Narrow(stamp_now);
  return L"";
}

std::wstring Timestamp() {
  SYSTEMTIME t;
  GetLocalTime(&t);
  wchar_t b[40];
  wsprintfW(b, L"%04d%02d%02d_%02d%02d%02d", t.wYear, t.wMonth, t.wDay, t.wHour, t.wMinute, t.wSecond);
  return b;
}

// ------------------------------------------------------------------------------------------
// launcher log and checks before the game starts
// ------------------------------------------------------------------------------------------
// logs\launcher.log keeps what the launcher saw (the log of the game only exists once the game has started, so a
// failure while preparing would leave nothing to attach to a report). The checks refuse only what cannot work.
std::wstring Leaf(const std::wstring& path) {
  std::wstring p = path;
  while (!p.empty() && (p.back() == L'\\' || p.back() == L'/')) p.pop_back();
  const size_t slash = p.find_last_of(L"\\/");
  return slash == std::wstring::npos ? p : p.substr(slash + 1);
}

void LauncherLog(const std::string& text) {
  if (g_home.empty()) return;
  MakeDirs(g_home + L"\\logs");
  FILE* f = _wfopen((g_home + L"\\logs\\launcher.log").c_str(), L"ab");
  if (!f) return;
  SYSTEMTIME t;
  GetLocalTime(&t);
  std::fprintf(f, "%04d-%02d-%02d %02d:%02d:%02d  %s\r\n", t.wYear, t.wMonth, t.wDay, t.wHour, t.wMinute, t.wSecond, text.c_str());
  std::fclose(f);
}

// Keeps the newest logs and crash dumps; older ones are deleted so the folder does not grow without end.
void PruneLogs(const wchar_t* pattern, size_t keep) {
  std::vector<std::pair<unsigned long long, std::wstring>> found;
  WIN32_FIND_DATAW data;
  const std::wstring dir = g_home + L"\\logs\\";
  HANDLE h = FindFirstFileW((dir + pattern).c_str(), &data);
  if (h == INVALID_HANDLE_VALUE) return;
  do {
    found.push_back({(static_cast<unsigned long long>(data.ftLastWriteTime.dwHighDateTime) << 32) | data.ftLastWriteTime.dwLowDateTime,
                     dir + data.cFileName});
  } while (FindNextFileW(h, &data));
  FindClose(h);
  std::sort(found.begin(), found.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
  for (size_t i = keep; i < found.size(); ++i) DeleteFileW(found[i].second.c_str());
}

void StartLauncherLog() {
  PruneLogs(L"driveclub_*.log", 25);
  PruneLogs(L"crash_*.dmp", 5);
  // One file for all starts; it starts over once it is big.
  WIN32_FILE_ATTRIBUTE_DATA info;
  const std::wstring path = g_home + L"\\logs\\launcher.log";
  if (GetFileAttributesExW(path.c_str(), GetFileExInfoStandard, &info) && info.nFileSizeLow > 256 * 1024) DeleteFileW(path.c_str());
  std::string version = "unknown";
  {
    std::ifstream in((g_home + L"\\version.txt").c_str());
    std::string line;
    if (std::getline(in, line)) {
      while (!line.empty() && (line.back() == '\r' || line.back() == ' ')) line.pop_back();
      if (!line.empty()) version = line;
    }
  }
  LauncherLog("launcher started, release " + version + ", in folder " + Narrow(Leaf(g_home)));
  LauncherLog("system: " + Narrow(DescribeSystem()));
  LauncherLog("processor: " + Narrow(DescribeCpu()));
  const std::wstring home_lower = g_home;
  if (home_lower.find(L"\\Temp\\") != std::wstring::npos || home_lower.find(L"\\Rar$") != std::wstring::npos)
    LauncherLog("warning: the launcher runs from a temporary folder (unzip the download first)");
  if (home_lower.find(L"Program Files") != std::wstring::npos)
    LauncherLog("warning: the folder is under Program Files (settings and logs cannot be written there)");
  if (home_lower.find(L"OneDrive") != std::wstring::npos)
    LauncherLog("warning: the folder is synced by OneDrive (files can be locked while they sync)");
  for (wchar_t ch : g_home) {
    if (ch > 127) {
      LauncherLog("note: the folder path has non-ASCII characters");
      break;
    }
  }
}

// What a game folder looks like, for the log: the facts that decide whether the dump is the right one.
std::string FolderFacts(const char* label, const std::wstring& path) {
  std::string s = std::string(label) + ": " + (path.empty() ? "not chosen" : Narrow(Leaf(path)));
  if (path.empty() || !Exists(path)) return s + (path.empty() ? "" : " (folder missing)");
  WIN32_FILE_ATTRIBUTE_DATA info;
  if (GetFileAttributesExW((path + L"\\eboot.bin").c_str(), GetFileExInfoStandard, &info)) {
    unsigned char head[4] = {0, 0, 0, 0};
    FILE* f = _wfopen((path + L"\\eboot.bin").c_str(), L"rb");
    if (f) {
      if (std::fread(head, 1, 4, f) != 4) std::memset(head, 0, 4);
      std::fclose(f);
    }
    char text[80];
    std::snprintf(text, sizeof text, "; eboot.bin %llu bytes, starts %02X %02X %02X %02X",
                  (unsigned long long)info.nFileSizeLow + ((unsigned long long)info.nFileSizeHigh << 32), head[0], head[1], head[2], head[3]);
    s += text;
  } else {
    s += "; no eboot.bin";
  }
  const std::wstring sfo = path + L"\\sce_sys\\param.sfo";
  if (Exists(sfo)) {
    s += "; TITLE_ID " + SfoField(std::filesystem::path(sfo), "TITLE_ID") + ", APP_VER " + SfoField(std::filesystem::path(sfo), "APP_VER") +
         ", VERSION " + SfoField(std::filesystem::path(sfo), "VERSION");
  } else {
    s += "; no param.sfo";
  }
  s += Exists(path + L"\\game000.dat") ? "; game000.dat present" : "; no game000.dat";
  s += Exists(path + L"\\game.ndx") ? ", game.ndx present" : ", no game.ndx";
  return s;
}

// "" when the game can start. Otherwise a sentence for the player.
std::string g_preflight_note;  // something the player should know about, but not a reason to refuse

std::wstring Preflight() {
  g_preflight_note.clear();
  LauncherLog(FolderFacts("update folder", g_folder[0].path));
  LauncherLog(FolderFacts("base folder", g_folder[1].path));
  // The program files the release came with (antivirus programs sometimes remove some of them).
  {
    std::ifstream in((g_runtime + L"\\files.txt").c_str());
    std::string line;
    std::string missing;
    int count = 0;
    while (std::getline(in, line)) {
      while (!line.empty() && (line.back() == '\r' || line.back() == ' ')) line.pop_back();
      if (line.empty()) continue;
      std::wstring name(line.begin(), line.end());
      // The DLSS files are optional (players on other graphics cards may delete them): the game then uses FSR.
      if (line.rfind("nvngx_", 0) == 0 || line.rfind("bbport_dlss", 0) == 0) continue;
      if (!Exists(g_runtime + L"\\" + name)) {
        if (++count <= 6) missing += (missing.empty() ? "" : ", ") + line;
      }
    }
    if (count) {
      LauncherLog("missing program files: " + missing + (count > 6 ? " and more" : ""));
      return L"Some program files are missing (" + std::wstring(missing.begin(), missing.end()) +
             (count > 6 ? L" and more" : L"") + L"). Unzip the download again; an antivirus program may have removed them.";
    }
  }
  // The folders for settings and logs must be writable.
  {
    MakeDirs(g_data + L"\\user");
    const std::wstring probe = g_data + L"\\user\\write_test.tmp";
    FILE* f = _wfopen(probe.c_str(), L"wb");
    if (!f) {
      LauncherLog("the data folder cannot be written");
      return L"The folder of the game cannot be written to. Move it somewhere you can write to (not Program Files).";
    }
    std::fclose(f);
    DeleteFileW(probe.c_str());
  }
  // The archive of the game is spread over many volume files.
  {
    unsigned used = 0;
    std::vector<unsigned> missing, truncated;
    const std::string problem = DatxCheckVolumes(g_folder[0].path, g_folder[1].path, used, missing, &truncated);
    if (problem.empty() && !truncated.empty()) {
      std::string list;
      for (size_t i = 0; i < truncated.size() && i < 8; ++i) {
        char name[24];
        std::snprintf(name, sizeof name, "game%03u.dat", truncated[i]);
        list += (list.empty() ? "" : ", ") + std::string(name);
      }
      LauncherLog("archive check: " + std::to_string(truncated.size()) + " volumes are shorter than the archive says (" + list + ")");
      return L"These game files are incomplete: " + std::wstring(list.begin(), list.end()) +
             (truncated.size() > 8 ? L" and more" : L"") +
             L". The copy of the game was cut short. Copy the game files again, then start the game.";
    }
    if (!problem.empty()) {
      LauncherLog("archive check: " + problem);
    } else if (!missing.empty()) {
      std::string list;
      for (size_t i = 0; i < missing.size() && i < 8; ++i) {
        char name[24];
        std::snprintf(name, sizeof name, "game%03u.dat", missing[i]);
        list += (list.empty() ? "" : ", ") + std::string(name);
      }
      // Not a reason to refuse: a dump without some volumes may still run, until the game reads from one.
      g_preflight_note = std::to_string(missing.size()) + " of " + std::to_string(used) + " archive volumes are missing (" + list +
                         "); the game can crash when it needs them";
      LauncherLog("archive check: " + g_preflight_note);
    } else {
      LauncherLog("archive check: all " + std::to_string(used) + " volumes present");
    }
  }
  return L"";
}

std::wstring g_game_log;  // the log file of the game started last

// Starts the game. With `process` the caller gets the game's process handle (and closes it).
std::wstring StartGame(HANDLE* process = nullptr) {
  const std::wstring exe = GameExe();
  if (!Exists(exe)) return L"The game program (DriveClubGame.exe) was not found.";
  MakeDirs(g_home + L"\\logs");
  SECURITY_ATTRIBUTES sa{sizeof(sa), nullptr, TRUE};
  g_game_log = g_home + L"\\logs\\driveclub_" + Timestamp() + L".log";
  HANDLE log = CreateFileW(g_game_log.c_str(), GENERIC_WRITE, FILE_SHARE_READ,
                           &sa, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
  MakeDirs(g_home + L"\\mods");
  const std::wstring prep = g_data + L"\\prep", user = g_data + L"\\user";
  // Add-on content: changed copies of three data lists, built from the player's own archive. Files in the mods
  // folder come first, so a player's own versions win.
  std::wstring unlock_arg;
  bool unlock_on = false;
  for (const Ctl& c : g_c) {
    if (c.kind == Kind::Toggle && c.patches[0] == "@unlock") unlock_on = c.sel != 0;
  }
  if (unlock_on) {
    const std::wstring unlock_dir = g_data + L"\\unlock";
    const std::wstring problem = BuildUnlock(g_folder[0].path, g_folder[1].path, unlock_dir);
    LauncherLog(problem.empty() ? std::string("add-on content unlock: ok") : "add-on content unlock: " + Narrow(problem));
    if (problem.empty()) unlock_arg = L" --mods \"" + unlock_dir + L"\"";
    else MessageBoxW(nullptr, (L"The add-on content could not be unlocked (" + problem + L"). The game starts without it.").c_str(), L"DriveClub", MB_OK | MB_ICONWARNING);
  }
  std::wstring cmd = L"\"" + exe + L"\" \"" + prep + L"\\boot-linked.bin\" --content-profile \"" + prep +
                     L"\\content.bin\" --patches \"" + prep + L"\\patches.bin\" --app0 \"" + g_folder[0].path +
                     L"\" --app0-base \"" + g_folder[1].path + L"\" --user \"" + user + L"\" --mods \"" + g_home + L"\\mods\"" +
                     unlock_arg + L" --stub-unresolved";
  SetEnvironmentVariableW(L"BB_GPU_USER_DIR", user.c_str());
  SetEnvironmentVariableW(L"BB_PREFLIGHT_NOTE", g_preflight_note.empty() ? nullptr : std::wstring(g_preflight_note.begin(), g_preflight_note.end()).c_str());
  SetEnvironmentVariableW(L"BB_CRASH_DIR", (g_home + L"\\logs").c_str());
  // The switchable-graphics layer of AMD laptops gives a black window on some of them.
  if (GetEnvironmentVariableW(L"VK_LOADER_LAYERS_DISABLE", nullptr, 0) == 0) SetEnvironmentVariableW(L"VK_LOADER_LAYERS_DISABLE", L"VK_LAYER_AMD_switchable_graphics");
  {
    // With the deltatime patch the game times itself by the wall clock, so each frame must be shown as soon as
    // it is finished; waiting for the next 60 Hz tick alternates 16 and 33 ms frames and shakes the camera.
    bool delta = false;
    for (const Ctl& c : g_c) {
      if (c.kind == Kind::Segmented && c.sel > 0 && c.sel < int(c.patches.size()) && c.patches[c.sel] == "60 FPS with deltatime") delta = true;
    }
    // Unlocked: every frame is shown at once (1 ms spacing is no cap in practice); the deltatime patch keeps the game speed right.
    const bool unlocked = g_fps >= 0 && g_c[g_fps].sel == 3;
    SetEnvironmentVariableW(L"BB_FPS_LIMIT", unlocked ? L"1000" : delta ? L"60" : nullptr);
  }
  // A value set before the launcher started is the player's override and is kept. Checking the environment
  // itself is not enough: the launcher sets these for every start, so a second start (Play, or Run benchmark
  // again) would keep the first start's values and ignore every choice changed since.
  if (g_lang >= 0 && !UserEnv(L"BB_LANGUAGE")) {
    SetEnvironmentVariableW(L"BB_LANGUAGE", std::to_wstring(g_lang_id).c_str());
  }
  if (g_up >= 0 && !UserEnv(L"BB_AA_SCALE")) {
    const int u = UpscaleActive() ? g_c[g_up].sel : 0;
    SetEnvironmentVariableW(L"BB_AA_SCALE", u == 0 ? L"native" : u == 1 ? L"quality" : u == 2 ? L"balanced" : L"performance");
  }
  if (g_aa >= 0 && !UserEnv(L"BB_AA")) {
    const int a = g_c[g_aa].sel;
    SetEnvironmentVariableW(L"BB_AA", a == 0 ? L"off" : a == 1 ? L"fsr3" : L"dlss");
  }
  if (g_fgen < 0 && !UserEnv(L"BB_FRAME_GEN")) SetEnvironmentVariableW(L"BB_FRAME_GEN", L"off");
  if (g_fgen >= 0 && !UserEnv(L"BB_FRAME_GEN")) {
    const int f = g_c[g_fgen].sel;
    SetEnvironmentVariableW(L"BB_FRAME_GEN", f == 0 ? L"off" : f == 1 ? L"fsr3" : L"dlss");
    if (!UserEnv(L"BB_FRAME_GEN_MULTIPLIER"))
      SetEnvironmentVariableW(L"BB_FRAME_GEN_MULTIPLIER", f == 2 ? L"2" : f == 3 ? L"3" : f == 4 ? L"4" : nullptr);
  }
  if (g_aniso >= 0 && g_c[g_aniso].sel >= 1) {
    static const wchar_t* const kAniso[] = {L"", L"0", L"2", L"4", L"8", L"16"};
    SetEnvironmentVariableW(L"BB_ANISO", kAniso[g_c[g_aniso].sel]);
  }
  if (g_mirror >= 0) {
    SetEnvironmentVariableW(L"BB_MIRROR_RATE", g_c[g_mirror].sel == 1 ? L"0" : L"1");
    // Diagnostics only: DRIVECLUB_MIRROR_RATE=<n> overrides the choice above (0: the scene is drawn once and then kept).
    wchar_t forced[16];
    if (GetEnvironmentVariableW(L"DRIVECLUB_MIRROR_RATE", forced, 16) > 0) SetEnvironmentVariableW(L"BB_MIRROR_RATE", forced);
  }
  {
    // One line of the launcher's choices for the log (without the player name and folders).
    std::wstring version;
    {
      std::ifstream in((g_home + L"\\version.txt").c_str());
      std::string line;
      if (std::getline(in, line)) {
        while (!line.empty() && (line.back() == '\r' || line.back() == ' ')) line.pop_back();
        version.assign(line.begin(), line.end());
      }
    }
    std::wstring summary = L"release " + (version.empty() ? std::wstring(L"unknown") : version) + L":";
    for (const auto& c : g_c) {
      if (c.patches.empty() || c.patches[0] == "@name" || c.patches[0] == "@lang" || IsBenchRow(c) || c.sel < 0) continue;
      summary += L" ";
      summary += c.label;
      summary += L"=";
      summary += c.kind == Kind::Toggle ? (c.sel ? L"on" : L"off") : (size_t(c.sel) < c.items.size() ? c.items[size_t(c.sel)] : L"?");
      summary += L";";
    }
    SetEnvironmentVariableW(L"BB_SETTINGS_SUMMARY", summary.c_str());
  }
  SetEnvironmentVariableW(L"BB_USER_NAME", g_name.c_str());
  SetEnvironmentVariableW(L"BB_FULLSCREEN", full_screen_now() ? L"1" : L"0");
  SetEnvironmentVariableW(L"BB_TOGGLE_FILE", (user + L"\\toggles.txt").c_str());
  const int tier = MemoryTier();
  SetEnvironmentVariableW(L"BB_DMEM_MB", tier == 3 ? L"17056" : tier == 2 ? L"10056" : tier == 1 ? L"8056" : L"5056");
  SetEnvironmentVariableW(L"BB_CONFIG", (user + L"\\driveclub.ini").c_str());
  STARTUPINFOW si{sizeof(si)};
  si.dwFlags = STARTF_USESTDHANDLES;
  si.hStdOutput = log;
  si.hStdError = log;
  si.hStdInput = GetStdHandle(STD_INPUT_HANDLE);
  PROCESS_INFORMATION pi{};
  std::vector<wchar_t> buf(cmd.begin(), cmd.end());
  buf.push_back(L'\0');
  const std::wstring cwd = Dir(exe);
  const BOOL ok = CreateProcessW(exe.c_str(), buf.data(), nullptr, nullptr, TRUE, 0, nullptr, cwd.c_str(), &si, &pi);
  if (log != INVALID_HANDLE_VALUE) CloseHandle(log);
  if (!ok) {
    LauncherLog("CreateProcess failed, error " + std::to_string(GetLastError()));
    return L"Could not start the game.";
  }
  CloseHandle(pi.hThread);
  if (process) *process = pi.hProcess;
  else CloseHandle(pi.hProcess);
  return L"";
}

void PlayWorker() {
  SetProgress(0, L"Preparing the game...");
  LauncherLog("Play pressed");
  std::wstring error = Preflight();
  if (error.empty()) error = RunPrepare();
  LauncherLog(error.empty() ? "preparation: ok" : "preparation: " + Narrow(error));
  if (error.empty()) {
    SetProgress(100, L"Starting DriveClub...");
    error = StartGame();
    LauncherLog(error.empty() ? "the game was started" : "start: " + Narrow(error));
  }
  g_busy = false;
  if (error.empty()) {
    if (g_hwnd) PostMessageW(g_hwnd, WM_APP + 2, 0, 0);  // close the launcher
  } else {
    g_status = error;
    g_status_color = kWarn;
    if (g_hwnd) PostMessageW(g_hwnd, WM_APP + 1, 0, 0);
  }
}

void Play() {
  if (g_busy || !FoldersOk()) return;
  if (GpuBlocked()) {
    g_status = g_gpu_problem;
    g_status_color = kBad;
    return;
  }
  SaveSettings();
  g_busy = true;
  g_status.clear();
  std::thread(PlayWorker).detach();
}

// ------------------------------------------------------------------------------------------
// benchmark: runs the game with BB_BENCHMARK, waits for it and shows data\user\benchmark\result.txt
// ------------------------------------------------------------------------------------------
std::wstring BenchResultPath() { return g_data + L"\\user\\benchmark\\result.txt"; }

// The BB_BENCHMARK value for the choices on the page (keys: src/runtime_benchmark.c).
std::string BenchSpec() {
  char spec[400];
  std::snprintf(spec, sizeof spec, "track=%s,car=%s,mode=%s,tod=%d,weather=%s,rain=%s,laps=20,warmup=5,seconds=%d,timeout=420",
                kBenchTracks[g_bench_track].id, kBenchCars[g_bench_car].id, g_c[g_bmode].sel == 1 ? "Race" : "TimeTrial",
                kBenchTod[g_c[g_btod].sel], kBenchWeather[g_c[g_bweather].sel], kBenchRain[g_c[g_brain].sel],
                kBenchSeconds[g_c[g_blen].sel]);
  return spec;
}

// result.txt is one line of key=value pairs (runtime_benchmark.c, finish()).
void LoadBenchResult() {
  g_bench_result = BenchResult{};
  const auto lines = ReadLines(BenchResultPath());
  if (lines.empty()) return;
  auto value = [&](const char* key) -> std::string {
    const std::string& l = lines[0];
    const std::string k = std::string(key) + "=";
    for (size_t p = l.find(k); p != std::string::npos; p = l.find(k, p + 1)) {
      if (p != 0 && l[p - 1] != ' ') continue;
      const size_t b = p + k.size(), e = l.find(' ', b);
      return l.substr(b, e == std::string::npos ? std::string::npos : e - b);
    }
    return "";
  };
  auto one = [](const std::string& v) {
    char b[32];
    std::snprintf(b, sizeof b, "%.1f", std::atof(v.c_str()));
    return Widen(b);
  };
  if (value("avg_fps").empty()) return;
  g_bench_result.ok = true;
  g_bench_result.fps = one(value("avg_fps"));
  g_bench_result.lows = L"1% low " + one(value("low1_fps")) + L"    0.1% low " + one(value("low01_fps"));
  const std::string weather = value("weather");
  std::string rain = value("rain");  // the game's preset name with '_' for each space
  std::replace(rain.begin(), rain.end(), '_', ' ');
  std::wstring about = BenchName(kBenchTracks, kBenchTrackCount, value("track")) + L", " +
                       BenchName(kBenchCars, kBenchCarCount, value("car")) + (value("mode") == "Race" ? L", race" : L"") + L", " +
                       Widen(value("tod")) + L":00 " + Widen(weather) + (rain.empty() || rain == "dry" ? L"" : L", rain " + Widen(rain)) +
                       L", " + Widen(std::to_string(int(std::atof(value("seconds").c_str()) + 0.5))) + L" s measured";
  WIN32_FILE_ATTRIBUTE_DATA info;
  if (GetFileAttributesExW(BenchResultPath().c_str(), GetFileExInfoStandard, &info)) {
    FILETIME local;
    SYSTEMTIME t;
    FileTimeToLocalFileTime(&info.ftLastWriteTime, &local);
    FileTimeToSystemTime(&local, &t);
    wchar_t when[40];
    wsprintfW(when, L", %04d-%02d-%02d %02d:%02d", t.wYear, t.wMonth, t.wDay, t.wHour, t.wMinute);
    about += when;
  }
  g_bench_result.about = about;

  // The second line: what the game itself ran with ("ran key=value ...", runtime_benchmark.c ran_with()).
  g_bench_ran_short.clear();
  g_bench_ran_lines.clear();
  if (lines.size() < 2 || lines[1].rfind("ran ", 0) != 0) return;
  auto ran = [&](const char* key) -> std::string {
    const std::string& l = lines[1];
    const std::string k = std::string(" ") + key + "=";
    const size_t p = l.find(k);
    if (p == std::string::npos) return "";
    const size_t b = p + k.size(), e = l.find(' ', b);
    return l.substr(b, e == std::string::npos ? std::string::npos : e - b);
  };
  const std::string fps = ran("fps_limit"), aa = ran("aa"), scale = ran("aa_scale"), fgen = ran("frame_gen");
  const int limit = std::atoi(fps.c_str());
  const std::wstring cap = fps == "none" || limit >= 1000 ? L"no frame cap" : Widen(fps) + L" FPS cap";
  const std::wstring aa_name = aa == "dlss" ? L"DLAA" : aa == "fsr3" ? L"FSR 3.1" : L"game AA";
  const std::wstring up_name = scale.empty() || scale == "native" ? L"" : L" (" + Widen(scale) + L" upscaling)";
  const std::wstring fg_name = fgen.empty() || fgen == "off" ? L"off" : Widen(fgen) + L" " + Widen(ran("frame_gen_x")) + L"x";
  const bool live = ran("mirror_rate") != "0";
  auto onoff = [&](const char* key) { return ran(key) == "1" ? std::wstring(L"on") : std::wstring(L"off"); };
  std::wstring aniso = Widen(ran("aniso"));
  if (aniso != L"auto") aniso += L"x";
  g_bench_ran_short = L"Ran at " + Widen(ran("render")) + L" \x2192 " + Widen(ran("output")) + L", " + cap + L", " + aa_name +
                      L". Point here for every setting.";
  g_bench_ran_lines = {
      L"What the game ran with (read by the game itself)",
      L"Render resolution (the 3D scene): " + Widen(ran("render")),
      L"Output resolution (the finished picture): " + Widen(ran("output")),
      L"The window is always your screen's size; dynamic resolution can lower the render size further under load.",
      L"Frame limit: " + cap + L", vblank " + Widen(ran("vblank_hz")) + L" Hz, present mode " + Widen(ran("present")),
      L"Anti-aliasing: " + aa_name + up_name + L", frame generation " + fg_name,
      L"Texture filtering: " + aniso + L", mirrors and reflections: " + (live ? L"live" : L"static"),
      L"Your car full detail: " + onoff("car_detail") + L", other cars full detail: " + onoff("opponent_detail"),
      L"More scenery detail: " + onoff("scenery_detail") + L", high-quality reflections: " + onoff("hq_reflections"),
      L"Dynamic resolution: " + onoff("dynamic_res") + L", add-on content: " + (ran("dlc") == "1" ? L"unlocked" : L"not unlocked"),
  };
}

// The game's own reason when it stopped the benchmark ("Benchmark: FAILED, ..." in its log), or "".
// The game's output can interleave with other lines, so the text is searched for anywhere in a line.
std::wstring BenchFailure() {
  std::string why;
  for (const auto& l : ReadLines(g_game_log)) {
    const size_t p = l.find("Benchmark: FAILED, ");
    if (p == std::string::npos) continue;
    const size_t b = p + 19, e = l.find(" (", b);  // without the note for developers in brackets
    why = l.substr(b, e == std::string::npos ? std::string::npos : e - b);
  }
  return Widen(why);
}

// When result.txt was last written (0 = not there).
unsigned long long ResultTime() {
  WIN32_FILE_ATTRIBUTE_DATA info;
  if (!GetFileAttributesExW(BenchResultPath().c_str(), GetFileExInfoStandard, &info)) return 0;
  return (static_cast<unsigned long long>(info.ftLastWriteTime.dwHighDateTime) << 32) | info.ftLastWriteTime.dwLowDateTime;
}

void BenchWorker() {
  SetProgress(0, L"Preparing the game...");
  LauncherLog("Run benchmark pressed: " + BenchSpec());
  std::wstring error = Preflight();
  if (error.empty()) error = RunPrepare();
  HANDLE game = nullptr;
  unsigned long long before = 0;
  if (error.empty()) {
    SetProgress(100, L"Benchmark running. The game closes by itself when it is done.");
    before = ResultTime();  // the last result stays on show if this run fails
    const std::wstring spec = Widen(BenchSpec());
    SetEnvironmentVariableW(L"BB_BENCHMARK", spec.c_str());
    SetEnvironmentVariableW(L"BB_BENCHMARK_OUT", BenchResultPath().c_str());
    error = StartGame(&game);
    SetEnvironmentVariableW(L"BB_BENCHMARK", nullptr);  // a later Play is an ordinary game
    SetEnvironmentVariableW(L"BB_BENCHMARK_OUT", nullptr);
  }
  DWORD code = 1;
  if (error.empty() && game) {
    WaitForSingleObject(game, INFINITE);
    GetExitCodeProcess(game, &code);
    CloseHandle(game);
    if (code == 0 && ResultTime() > before) {
      const auto lines = ReadLines(BenchResultPath());
      LauncherLog("benchmark result: " + (lines.empty() ? std::string("(empty)") : lines[0]));
    } else {
      std::wstring why = BenchFailure();
      if (why.empty() && code == 3) why = L"the race did not start, or the game stopped drawing";
      error = why.empty() ? L"The benchmark stopped before the end: the game was closed or crashed. The log is in the logs folder."
                          : L"The benchmark failed: " + why + L". The log is in the logs folder.";
      LauncherLog("benchmark: exit code " + std::to_string(code) + ", " + Narrow(error));
    }
  } else {
    LauncherLog("benchmark: " + Narrow(error));
  }
  g_bench_running = false;
  g_busy = false;
  if (!error.empty()) {
    g_status = error;
    g_status_color = kWarn;
  } else {
    g_status = L"Benchmark done.";
    g_status_color = kOk;
  }
  if (g_hwnd) PostMessageW(g_hwnd, WM_APP + 3, 0, 0);  // show the result
}

void RunBenchmark() {
  if (g_busy || !FoldersOk()) return;
  if (GpuBlocked()) {
    g_status = g_gpu_problem;
    g_status_color = kBad;
    return;
  }
  SaveSettings();
  g_busy = true;
  g_bench_running = true;
  g_status.clear();
  std::thread(BenchWorker).detach();
}

// The menu of tracks or cars, grouped by location or class. Returns the chosen index or -1.
int BenchMenu(HWND h, const RectF& at, const BenchChoice* list, int count, int current) {
  HMENU menu = CreatePopupMenu();
  std::vector<std::pair<std::wstring, HMENU>> groups;
  for (int i = 0; i < count; ++i) {
    auto it = std::find_if(groups.begin(), groups.end(), [&](const auto& g) { return g.first == list[i].group; });
    if (it == groups.end()) {
      groups.push_back({list[i].group, CreatePopupMenu()});
      it = groups.end() - 1;
    }
    AppendMenuW(it->second, MF_STRING | (i == current ? MF_CHECKED : 0), UINT_PTR(1 + i), list[i].name);
  }
  for (const auto& g : groups) {
    const bool has_current = current >= 0 && g.first == list[current].group;
    AppendMenuW(menu, MF_POPUP | (has_current ? MF_CHECKED : 0), UINT_PTR(g.second), g.first.c_str());
  }
  POINT pt{LONG(at.X), LONG(at.Y + at.Height)};
  ClientToScreen(h, &pt);
  const int pick = int(TrackPopupMenu(menu, TPM_RETURNCMD | TPM_LEFTALIGN | TPM_TOPALIGN | TPM_NONOTIFY, pt.x, pt.y, 0, h, nullptr));
  DestroyMenu(menu);  // and its submenus
  return pick - 1;
}

// ------------------------------------------------------------------------------------------
// folder picker
// ------------------------------------------------------------------------------------------
void BrowseFolder(int which) {
  BROWSEINFOW bi{};
  bi.hwndOwner = g_hwnd;
  bi.lpszTitle = which == 0 ? L"Choose the DriveClub v1.28 update folder (it contains eboot.bin)"
                            : L"Choose the DriveClub base game folder (it contains game000.dat)";
  bi.ulFlags = BIF_RETURNONLYFSDIRS | BIF_NEWDIALOGSTYLE | BIF_USENEWUI;
  PIDLIST_ABSOLUTE pidl = SHBrowseForFolderW(&bi);
  if (!pidl) return;
  wchar_t path[MAX_PATH];
  if (SHGetPathFromIDListW(pidl, path)) {
    g_folder[which].path = path;
    CheckFolders();
    if (!g_folder[which].ok) {
      g_status = FolderProblem(which);
      g_status_color = kBad;
    } else {
      g_status.clear();
    }
    g_changed = true;
  }
  CoTaskMemFree(pidl);
}

// ------------------------------------------------------------------------------------------
// window
// ------------------------------------------------------------------------------------------
void CommitName() {
  if (!g_edit) return;
  wchar_t buf[32] = {};
  GetWindowTextW(g_edit, buf, 31);
  DestroyWindow(g_edit);
  g_edit = nullptr;
  std::wstring n;
  for (wchar_t ch : std::wstring(buf)) if (ch >= 32 && ch < 127) n += ch;
  if (!n.empty() && n != g_name) { g_name = n; g_changed = true; }
  for (auto& c : g_c) if (c.patches[0] == "@name") c.items[0] = g_name;
}

void StartNameEdit(HWND parent, const Ctl& c) {
  const RectF r = Scaled(c.ctl);
  g_edit = CreateWindowExW(0, L"EDIT", g_name.c_str(), WS_CHILD | WS_VISIBLE | ES_AUTOHSCROLL, int(r.X + 6), int(r.Y + 6),
                           int(r.Width - 12), int(r.Height - 12), parent, nullptr, GetModuleHandleW(nullptr), nullptr);
  SendMessageW(g_edit, EM_LIMITTEXT, 20, 0);
  SendMessageW(g_edit, WM_SETFONT, (WPARAM)GetStockObject(DEFAULT_GUI_FONT), TRUE);
  SendMessageW(g_edit, EM_SETSEL, 0, -1);
  SetFocus(g_edit);
}

int HitControl(float x, float y, int* seg) {
  if (y >= FooterTop()) { *seg = -1; return -1; }
  *seg = -1;
  for (size_t i = 0; i < g_c.size(); ++i) {
    Ctl& c = g_c[i];
    RectF r = Scaled(c.row);
    if (x < r.X || x > r.X + r.Width || y < r.Y || y > r.Y + r.Height) continue;
    if (c.kind == Kind::Action) {
      const RectF ct = Scaled(c.ctl);
      if (x >= ct.X && x <= ct.X + ct.Width && y >= ct.Y && y <= ct.Y + ct.Height) *seg = 0;  // over the Run button
    }
    if (c.kind == Kind::Segmented) {
      RectF ct = Scaled(c.ctl);
      if (x >= ct.X && x <= ct.X + ct.Width && y >= ct.Y && y <= ct.Y + ct.Height) {
        *seg = std::clamp(int((x - ct.X) / (ct.Width / float(c.items.size()))), 0, int(c.items.size()) - 1);
      }
    }
    return int(i);
  }
  return -1;
}

int HitButton(float x, float y) {
  const float content_oy = g_oy;
  g_oy = FooterOy();
  struct Restore { float v; ~Restore() { g_oy = v; } } restore{content_oy};
  for (int i = 0; i < 3; ++i) {
    RectF r = Scaled(g_btn[i]);
    if (x >= r.X && x <= r.X + r.Width && y >= r.Y && y <= r.Y + r.Height) return i;
  }
  return -1;
}

int HitBrowse(float x, float y) {
  if (y >= FooterTop()) return -1;
  for (int i = 0; i < 2; ++i) {
    RectF r = Scaled(g_folder[i].browse);
    if (x >= r.X && x <= r.X + r.Width && y >= r.Y && y <= r.Y + r.Height) return i;
  }
  return -1;
}

LRESULT CALLBACK WndProc(HWND h, UINT m, WPARAM w, LPARAM l) {
  switch (m) {
    case WM_SIZE: {
      // The layout keeps its scale and reflows: wider windows get wider columns, taller ones more room.
      if (w == SIZE_MINIMIZED || g_scale <= 0.f) return 0;
      g_client_h = float(HIWORD(l));
      {
        // Everything shrinks to fit a window that is too short (more columns open up as it does),
        // down to 60%; below that the content scrolls.
        const float cw = float(LOWORD(l));
        float s = g_dpi;
        for (int i = 0; i < 12; ++i) {
          g_scale = s;
          g_win_w = (std::max)(kWinW, (std::min)(kMaxW, cw / s));
          g_win_h = g_client_h / s;
          Layout();
          const float need = TotalHeight() * s;
          if (need <= g_client_h + 0.5f || s <= g_dpi * 0.6f + 0.001f) break;
          s = (std::max)(g_dpi * 0.6f, s * g_client_h / need);
        }
      }
      g_scroll = (std::min)(g_scroll, MaxScroll());
      g_oy = ContentOy();
      InvalidateRect(h, nullptr, FALSE);
      return 0;
    }
    case WM_GETMINMAXINFO: {
      RECT wr, cr;
      GetWindowRect(h, &wr);
      GetClientRect(h, &cr);
      const int ncw = (wr.right - wr.left) - cr.right, nch = (wr.bottom - wr.top) - cr.bottom;
      auto* mm = reinterpret_cast<MINMAXINFO*>(l);
      mm->ptMinTrackSize.x = LONG(kWinW * g_dpi * 0.6f) + ncw;
      mm->ptMinTrackSize.y = LONG((kHeaderH + kFooterH + 160.f) * g_dpi * 0.6f) + nch;  // shorter: the content scrolls
      mm->ptMaxTrackSize.x = LONG(kMaxW * g_dpi) + ncw;
      return 0;
    }
    case WM_DPICHANGED: {
      // Moved to a screen with another scale: take the size Windows suggests and draw at that scale.
      g_dpi = g_scale = float(HIWORD(w)) / 96.f;
      const RECT* r = reinterpret_cast<const RECT*>(l);
      SetWindowPos(h, nullptr, r->left, r->top, r->right - r->left, r->bottom - r->top, SWP_NOZORDER | SWP_NOACTIVATE);
      InvalidateRect(h, nullptr, FALSE);
      return 0;
    }
    case WM_PAINT: {
      PAINTSTRUCT ps;
      HDC dc = BeginPaint(h, &ps);
      RECT rc;
      GetClientRect(h, &rc);
      HDC mem = CreateCompatibleDC(dc);
      HBITMAP bmp = CreateCompatibleBitmap(dc, rc.right, rc.bottom);
      HGDIOBJ old = SelectObject(mem, bmp);
      Paint(mem, rc.right, rc.bottom);
      BitBlt(dc, 0, 0, rc.right, rc.bottom, mem, 0, 0, SRCCOPY);
      SelectObject(mem, old);
      DeleteObject(bmp);
      DeleteDC(mem);
      EndPaint(h, &ps);
      return 0;
    }
    case WM_ERASEBKGND:
      return 1;
    case WM_MOUSEWHEEL: {
      const float before = g_scroll;
      g_scroll -= float(GET_WHEEL_DELTA_WPARAM(w)) / WHEEL_DELTA * 60.f * g_scale;
      g_scroll = (std::max)(0.f, (std::min)(g_scroll, MaxScroll()));
      if (g_scroll != before) {
        CommitName();  // the name box is a child window placed at the old position
        g_oy = ContentOy();
        InvalidateRect(h, nullptr, FALSE);
      }
      return 0;
    }
    case WM_MOUSEMOVE: {
      const float x = float(GET_X_LPARAM(l)), y = float(GET_Y_LPARAM(l));
      int seg;
      const int c = HitControl(x, y, &seg), b = HitButton(x, y), br = HitBrowse(x, y);
      // Over the benchmark's result (its row below the Run button): the panel of settings the game ran with.
      bool ran_hover = false;
      if (c >= 0 && c == g_brun && seg != 0) {
        const RectF row = Scaled(g_c[c].row);
        ran_hover = y >= row.Y + 88 * g_scale;
      }
      if (ran_hover != g_bench_ran_hover) {
        g_bench_ran_hover = ran_hover;
        InvalidateRect(h, nullptr, FALSE);
      }
      if (c != g_hover_ctl || seg != g_hover_seg || b != g_hover_btn || br != g_hover_browse) {
        g_hover_ctl = c;
        g_hover_seg = seg;
        g_hover_btn = b;
        g_hover_browse = br;
        InvalidateRect(h, nullptr, FALSE);
      }
      TRACKMOUSEEVENT t{sizeof(t), TME_LEAVE, h, 0};
      TrackMouseEvent(&t);
      return 0;
    }
    case WM_MOUSELEAVE:
      g_hover_ctl = g_hover_seg = g_hover_btn = g_hover_browse = -1;
      g_bench_ran_hover = false;
      InvalidateRect(h, nullptr, FALSE);
      return 0;
    case WM_LBUTTONDOWN: {
      const float x = float(GET_X_LPARAM(l)), y = float(GET_Y_LPARAM(l));
      if (g_busy) return 0;
      CommitName();
      const int br = HitBrowse(x, y);
      if (br >= 0) {
        BrowseFolder(br);
        InvalidateRect(h, nullptr, FALSE);
        return 0;
      }
      g_press_btn = HitButton(x, y);
      int seg;
      const int c = HitControl(x, y, &seg);
      if (c >= 0) {
        Ctl& ctl = g_c[c];
        if (ctl.patches[0] == "@name") {
          StartNameEdit(h, ctl);
          return 0;
        }
        if (ctl.kind == Kind::Picker) {
          const bool track = c == g_btrack;
          const int pick = BenchMenu(h, Scaled(ctl.ctl), track ? kBenchTracks : kBenchCars, track ? kBenchTrackCount : kBenchCarCount,
                                     track ? g_bench_track : g_bench_car);
          if (pick >= 0) {
            (track ? g_bench_track : g_bench_car) = pick;
            ShowBenchChoices();
            g_changed = true;
          }
          InvalidateRect(h, nullptr, FALSE);
          return 0;
        }
        if (ctl.kind == Kind::Action) {
          if (seg == 0) RunBenchmark();
          InvalidateRect(h, nullptr, FALSE);
          return 0;
        }
        if (ctl.patches[0] == "@lang") {
          g_lang_id = (g_lang_id + 1) % 21;
          ctl.items[0] = kLangName[g_lang_id];
          g_changed = true;
          InvalidateRect(h, nullptr, FALSE);
          return 0;
        }
        if (ctl.kind == Kind::Toggle) {
          ctl.sel = !ctl.sel;
          if (ctl.patches[0] == "@skipgpu") {
            g_skip_gpu_check = ctl.sel != 0;
            if (!GpuBlocked() && g_status == g_gpu_problem) g_status.clear();
            else if (GpuBlocked()) { g_status = g_gpu_problem; g_status_color = kBad; }
          }
          g_changed = true;
        } else if (seg >= 0) {
          ctl.sel = seg;
          if (c == g_preset) ApplyPreset(seg);
          else if (c == g_outres || c == g_inres || c == g_wide) DerivePreset();
          g_changed = true;
        }
      }
      SetCapture(h);
      InvalidateRect(h, nullptr, FALSE);
      return 0;
    }
    case WM_LBUTTONUP: {
      ReleaseCapture();
      const float x = float(GET_X_LPARAM(l)), y = float(GET_Y_LPARAM(l));
      const int b = HitButton(x, y);
      const int pressed = g_press_btn;
      g_press_btn = -1;
      if (!g_busy && b >= 0 && b == pressed) {
        if (b == 0) ResetDefaults();
        else if (b == 1) {
          SaveSettings();
          g_status = L"Saved.";
          g_status_color = kOk;
        } else if (b == 2) Play();
      }
      InvalidateRect(h, nullptr, FALSE);
      return 0;
    }
    case WM_COMMAND:
      if (HIWORD(w) == EN_KILLFOCUS && g_edit) {
        CommitName();
        InvalidateRect(h, nullptr, FALSE);
      }
      return 0;
    case WM_APP + 1:
      InvalidateRect(h, nullptr, FALSE);
      return 0;
    case WM_APP + 2:
      DestroyWindow(h);
      return 0;
    case WM_APP + 3:  // the benchmark has ended: show its result in front of the other windows
      LoadBenchResult();
      if (IsIconic(h)) ShowWindow(h, SW_RESTORE);
      SetForegroundWindow(h);
      InvalidateRect(h, nullptr, FALSE);
      return 0;
    case WM_DESTROY:
      PostQuitMessage(0);
      return 0;
  }
  return DefWindowProcW(h, m, w, l);
}

}  // namespace

int WINAPI wWinMain(HINSTANCE inst, HINSTANCE, PWSTR, int show) {
  if (wcsstr(GetCommandLineW(), L"--gpu-check")) {
    int argc = 0;
    wchar_t** argv = CommandLineToArgvW(GetCommandLineW(), &argc);
    for (int i = 1; argv && i + 1 < argc; ++i) {
      if (wcscmp(argv[i], L"--gpu-check") == 0) return GpuCheckChildMain(argv[i + 1]);
    }
    return 0;
  }
  SetProcessDPIAware();
  CoInitialize(nullptr);
  SnapshotUserEnv();
  wchar_t selfbuf[MAX_PATH * 2];
  GetModuleFileNameW(nullptr, selfbuf, MAX_PATH * 2);
  g_home = Dir(selfbuf);
  g_data = g_home + L"\\data";
  g_runtime = g_home + L"\\runtime";
  BuildModel();
  MakeDirs(g_data + L"\\user");
  LoadSettings();
  LoadBenchResult();
  StartLauncherLog();
  if (!g_skip_gpu_check) {
    g_gpu_problem = CheckCpu();
    if (g_gpu_problem.empty()) g_gpu_problem = CheckGpu();
  }
  LauncherLog(g_skip_gpu_check ? std::string("hardware check: skipped by the player") : g_gpu_problem.empty() ? std::string("hardware check: passed") : "hardware check: " + Narrow(g_gpu_problem));
  if (GpuBlocked()) {
    g_status = g_gpu_problem;
    g_status_color = kBad;
  }

  const bool play_now = wcsstr(GetCommandLineW(), L"--play") != nullptr;
  if (play_now) {
    if (!FoldersOk()) {
      MessageBoxW(nullptr, L"Open DriveClub.exe once and choose the game folders.", L"DriveClub", MB_OK | MB_ICONERROR);
      return 1;
    }
    if (!g_skip_gpu_check) {
      if (const std::wstring gpu = CheckGpu(); !gpu.empty()) {
        MessageBoxW(nullptr, gpu.c_str(), L"DriveClub", MB_OK | MB_ICONERROR);
        return 1;
      }
    }
    SaveSettings();
    const std::wstring error = [] {
      SetProgress(0, L"");
      std::wstring e = RunPrepare();
      return e.empty() ? StartGame() : e;
    }();
    if (!error.empty()) MessageBoxW(nullptr, error.c_str(), L"DriveClub", MB_OK | MB_ICONERROR);
    return error.empty() ? 0 : 1;
  }

  GdiplusStartupInput gsi;
  ULONG_PTR token = 0;
  GdiplusStartup(&token, &gsi, nullptr);

  // --snapshot FILE: render the window to a PNG and exit (checks the layout without a screen).
  {
    int argc = 0;
    wchar_t** argv = CommandLineToArgvW(GetCommandLineW(), &argc);
    for (int i = 1; argv && i + 1 < argc; ++i) {
      if (wcscmp(argv[i], L"--snapshot") != 0) continue;
      HDC screen = GetDC(nullptr);
      g_dpi = g_scale = float(GetDeviceCaps(screen, LOGPIXELSX)) / 96.f;
      ReleaseDC(nullptr, screen);
      if (const char* sw = std::getenv("LAUNCHER_SNAPSHOT_W")) g_win_w = float(std::atoi(sw));
      Layout();
      const int w = int(g_win_w * g_scale), h = int(TotalHeight() * g_scale);
      {
      Bitmap bmp(w, h, PixelFormat32bppARGB);
      {
        Graphics gr(&bmp);
        HDC hdc = gr.GetHDC();
        Paint(hdc, w, h);
        gr.ReleaseHDC(hdc);
      }
      UINT count = 0, bytes = 0;
      GetImageEncodersSize(&count, &bytes);
      std::vector<char> raw(bytes);
      auto* encoders = reinterpret_cast<ImageCodecInfo*>(raw.data());
      GetImageEncoders(count, bytes, encoders);
      for (UINT k = 0; k < count; ++k) {
        if (wcscmp(encoders[k].MimeType, L"image/png") == 0) bmp.Save(argv[i + 1], &encoders[k].Clsid, nullptr);
      }
      }
      GdiplusShutdown(token);
      return 0;
    }
  }

  WNDCLASSW wc{};
  wc.lpfnWndProc = WndProc;
  wc.hInstance = inst;
  wc.lpszClassName = L"driveclub_launcher";
  wc.hCursor = LoadCursorW(nullptr, MAKEINTRESOURCEW(32512));
  wc.hIcon = LoadIconW(inst, MAKEINTRESOURCEW(1));
  RegisterClassW(&wc);

  HDC screen = GetDC(nullptr);
  g_dpi = g_scale = float(GetDeviceCaps(screen, LOGPIXELSX)) / 96.f;
  ReleaseDC(nullptr, screen);
  // The first window: the narrowest that shows the whole layout on this screen; wider windows hold
  // more columns and so are shorter. If none fits, the widest and the content scrolls.
  float start_w = kWinW;
  {
    RECT work{};
    if (SystemParametersInfoW(SPI_GETWORKAREA, 0, &work, 0)) {
      const float avail_w = float(work.right - work.left), avail_h = float(work.bottom - work.top);
      const float candidates[3] = {kWinW, 1320.f, kMaxW};
      for (float cw : candidates) {
        if (cw * g_scale > avail_w) break;
        g_win_w = cw;
        g_win_h = 0;
        Layout();
        start_w = cw;
        if (TotalHeight() * g_scale + 48.f * g_scale <= avail_h) break;
      }
    }
  }
  g_win_w = start_w;
  Layout();
  const DWORD style = WS_OVERLAPPED | WS_CAPTION | WS_SYSMENU | WS_MINIMIZEBOX | WS_THICKFRAME;  // every edge and corner resizes
  RECT rc{0, 0, LONG(start_w * g_scale), LONG(TotalHeight() * g_scale)};
  AdjustWindowRect(&rc, style, FALSE);
  {
    // At most 85% of the work area high and 95% wide; the layout then shrinks to fit (see WM_SIZE).
    RECT work{};
    if (SystemParametersInfoW(SPI_GETWORKAREA, 0, &work, 0)) {
      const LONG max_h = LONG((work.bottom - work.top) * 0.85), max_w = LONG((work.right - work.left) * 0.95);
      if (rc.bottom - rc.top > max_h) rc.bottom = rc.top + max_h;
      if (rc.right - rc.left > max_w) rc.right = rc.left + max_w;
    }
  }
  // Centred in the work area (the screen without the taskbar), never above its top edge.
  RECT area{0, 0, GetSystemMetrics(SM_CXSCREEN), GetSystemMetrics(SM_CYSCREEN)};
  SystemParametersInfoW(SPI_GETWORKAREA, 0, &area, 0);
  const int win_x = (std::max)(int(area.left), int(area.left) + (int(area.right - area.left) - int(rc.right - rc.left)) / 2);
  const int win_y = (std::max)(int(area.top), int(area.top) + (int(area.bottom - area.top) - int(rc.bottom - rc.top)) / 2);
  g_hwnd = CreateWindowExW(0, wc.lpszClassName, L"DriveClub - PC", style, win_x, win_y, rc.right - rc.left,
                           rc.bottom - rc.top, nullptr, nullptr, inst, nullptr);
  ShowWindow(g_hwnd, show);
  MSG msg;
  while (GetMessageW(&msg, nullptr, 0, 0)) {
    TranslateMessage(&msg);
    DispatchMessageW(&msg);
  }
  GdiplusShutdown(token);
  CoUninitialize();
  return int(msg.wParam);
}
