#!/usr/bin/env bash
# VR fork: the player release, release/driveclub-recomp-<version>-win64/ and a zip next to it: stage-vr.sh's folder
# (runtime, launcher, OpenXR loader, DLSS bridge and library) with the readme, licences and a leak check.
# usage: packaging/package-vr.sh <version>   (for example 0.2.4-vr1.0)
set -euo pipefail
cd "$(dirname "$0")/.."
VERSION="${1:?version}"
NAME="driveclub-recomp-$VERSION-win64"
OUT="release/$NAME"
rm -rf "$OUT" "release/$NAME.zip"
STAGE_OUT="$OUT" bash packaging/stage-vr.sh
echo "$VERSION" > "$OUT/version.txt"
cp README.md LICENSE "$OUT/"
mkdir -p "$OUT/licenses"
cp packaging/windows/licenses/* packaging/dlss/NVIDIA-DLSS-LICENSE.txt packaging/dlss/dlss_bridge-LICENSE.txt "$OUT/licenses/"
# Nothing of the game, the console firmware or the player may end up in the package.
leaks="$(find "$OUT" -type f \( -iname 'eboot.bin' -o -iname '*.sprx' -o -iname '*.prx' -o -iname '*.pkg' -o -iname '*.sfo' \
  -o -iname '*.psarc' -o -iname 'game*.dat' -o -iname 'game.ndx' -o -iname '*.log' -o -iname '*.spv' -o -iname '*.dmp' \
  -o -iname 'launcher.ini' -o -path '*/data/*' -o -path '*/logs/*' -o -path '*/sce_sys/*' \) -print)"
if [ -n "$leaks" ]; then
  echo "Package leak check failed, these files must not ship:" >&2
  echo "$leaks" >&2
  exit 1
fi
echo "files: $(find "$OUT" -type f | wc -l), size: $(du -sh "$OUT" | cut -f1)"
(cd release && python -c "
import os, sys, zipfile
name = sys.argv[1]
with zipfile.ZipFile(name + '.zip', 'w', zipfile.ZIP_DEFLATED, compresslevel=9) as z:
    for root, _, files in os.walk(name):
        for f in sorted(files):
            p = os.path.join(root, f)
            z.write(p, os.path.relpath(p, '.'))
" "$NAME")
echo "Packaged $PWD/release/$NAME.zip"
