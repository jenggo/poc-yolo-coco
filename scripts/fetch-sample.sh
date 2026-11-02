#!/usr/bin/env bash
# Download a free-license traffic clip into samples/traffic.webm.
# Safe to re-run: an existing plausible file is left untouched.
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p samples

OUT="samples/traffic.webm"
URL="https://upload.wikimedia.org/wikipedia/commons/c/cb/Road_traffic_in_Ljubljana.webm"
MIN_BYTES=5000000

if [ -s "$OUT" ] && [ "$(stat -c%s "$OUT")" -ge "$MIN_BYTES" ]; then
    echo "sample already present: $OUT ($(stat -c%s "$OUT") bytes)"
    exit 0
fi

echo "downloading $URL"
curl -fL --retry 3 --retry-delay 2 -A "cctvpro-poc/0.1" -o "$OUT.part" "$URL"
mv "$OUT.part" "$OUT"

SIZE=$(stat -c%s "$OUT")
if [ "$SIZE" -lt "$MIN_BYTES" ]; then
    rm -f "$OUT"
    echo "downloaded file is too small ($SIZE bytes); removed it" >&2
    exit 1
fi
echo "sample ready: $OUT ($SIZE bytes)"
