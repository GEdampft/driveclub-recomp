#!/usr/bin/env bash
# VR fork: builds the runtime and the launcher and lays out a playable folder, release/vr-dev/ (package.sh without
# the DLSS libraries and the zip), with the OpenXR loader beside the game program.
# usage: packaging/stage-vr.sh [path to openxr_loader.dll]
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH="/c/msys64/clang64/bin:$PATH"
LOADER="${1:-${OPENXR_LOADER:-/f/rpsc3/source/rpcs3/bin/openxr_loader.dll}}"
OUT="release/vr-dev"

bash build.sh
mkdir -p out/launcher
windres -I launcher launcher/launcher.rc -O coff -o out/launcher/launcher.res.o
clang++ -O2 -std=c++20 -include gpu/libcxx_compat.h -municode -mwindows -static -o out/launcher/DriveClub.exe   launcher/driveclub_launcher.cpp launcher/prepare.cpp launcher/gpu_check.cpp launcher/datx.cpp launcher/unlock.cpp out/launcher/launcher.res.o   -lgdiplus -lgdi32 -luser32 -lshell32 -lole32 -ldwmapi

# The player's data (launcher.ini, data/, logs/, mods/) stays; the program files are replaced.
rm -rf "$OUT/runtime"
mkdir -p "$OUT/runtime/patches"
cp out/launcher/DriveClub.exe "$OUT/"
cp out/DriveClubGame.exe "$OUT/runtime/"
cp patches/DriveClub.xml "$OUT/runtime/patches/"
cp "$LOADER" "$OUT/runtime/openxr_loader.dll"
echo "vr-dev" > "$OUT/version.txt"
ldd out/DriveClubGame.exe | awk '{print $3}' | { grep -iE '/msys64/|^/clang64/' || true; } | sort -u | while read -r dll; do
  cp "$dll" "$OUT/runtime/"
done
for round in 1 2; do
  for staged in "$OUT"/runtime/*.dll; do
    ldd "$staged" 2>/dev/null | awk '{print $3}' | { grep -iE '/msys64/|^/clang64/' || true; } | sort -u | while read -r dll; do
      [ -f "$OUT/runtime/$(basename "$dll")" ] || cp "$dll" "$OUT/runtime/"
    done
  done
done
for extra in SDL3.dll; do
  [ -f "out/$extra" ] && cp "out/$extra" "$OUT/runtime/" || true
done
(cd "$OUT/runtime" && find . -type f ! -name files.txt | sed 's|^\./||' | sort > files.txt)
echo "Staged $PWD/$OUT/DriveClub.exe"
