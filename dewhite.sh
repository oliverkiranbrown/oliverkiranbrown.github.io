#!/usr/bin/env bash
#
# dewhite.sh — list images in a folder, let you pick which ones to process,
# then replace white/near-white pixels with a target colour, overwriting
# each file in place under its original name.
#
# Requires ImageMagick (`convert`, or `magick` on IM7).
#   macOS:  brew install imagemagick
#   Debian: sudo apt install imagemagick
#
# Usage:
#   ./dewhite.sh [directory]        # defaults to current directory
#
# Edit TARGET_COLOR / FUZZ below to match your CSS background colour.

set -euo pipefail

# --- config -----------------------------------------------------------
TARGET_COLOR="#fdf6e3"   # <- set this to your CSS --background-color value
FUZZ="8%"                # tolerance: how far from pure white still counts
                          # as "white" (raise if edges/anti-aliasing are left
                          # un-recoloured, lower if it eats into real content)
DIR="${1:-.}"

# --- pick the right ImageMagick binary --------------------------------
if command -v magick >/dev/null 2>&1; then
    IM="magick"
elif command -v convert >/dev/null 2>&1; then
    IM="convert"
else
    echo "ImageMagick not found. Install it first (brew install imagemagick / apt install imagemagick)." >&2
    exit 1
fi

# --- find images (top level of DIR only) ------------------------------
# (avoiding `mapfile` since macOS ships bash 3.2, which doesn't have it)
images=()
while IFS= read -r f; do
    images+=("$f")
done < <(find "$DIR" -maxdepth 1 -type f \( \
        -iname '*.png' -o -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.webp' \
    \) | sort)

if [ "${#images[@]}" -eq 0 ]; then
    echo "No images found in $DIR"
    exit 1
fi

echo "Found ${#images[@]} image(s) in $DIR:"
for i in "${!images[@]}"; do
    printf "  [%2d] %s\n" "$((i + 1))" "$(basename "${images[$i]}")"
done

echo
read -rp "Which to de-white? (e.g. '1 3 4', or 'all'): " selection

selected=()
if [[ "$selection" == "all" ]]; then
    selected=("${images[@]}")
else
    for n in ${selection//,/ }; do
        idx=$((n - 1))
        if [ "$idx" -ge 0 ] && [ "$idx" -lt "${#images[@]}" ]; then
            selected+=("${images[$idx]}")
        else
            echo "  (skipping invalid selection: $n)"
        fi
    done
fi

if [ "${#selected[@]}" -eq 0 ]; then
    echo "Nothing selected — exiting."
    exit 0
fi

echo
echo "Replacing white (fuzz=$FUZZ) with $TARGET_COLOR in:"
for img in "${selected[@]}"; do
    echo "  -> $(basename "$img")"
    "$IM" "$img" -fuzz "$FUZZ" -fill "$TARGET_COLOR" -opaque white "$img"
done

echo "Done."