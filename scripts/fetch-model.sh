#!/usr/bin/env bash
# Download the pre-exported YOLOv8n ONNX weights into models/.
# Safe to re-run: an existing plausible file is left untouched.
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p models

OUT="models/yolov8n.onnx"
URL="https://huggingface.co/cabelo/yolov8/resolve/main/yolov8n.onnx"
FALLBACK="https://huggingface.co/salim4n/yolov8n-detect-onnx/resolve/main/yolov8n-onnx-web/yolov8n.onnx"
MIN_BYTES=1000000

if [ -s "$OUT" ] && [ "$(stat -c%s "$OUT")" -ge "$MIN_BYTES" ]; then
    echo "model already present: $OUT ($(stat -c%s "$OUT") bytes)"
    exit 0
fi

download() {
    local url="$1"
    echo "downloading $url"
    curl -fL --retry 3 --retry-delay 2 -o "$OUT.part" "$url"
    mv "$OUT.part" "$OUT"
}

if ! download "$URL"; then
    rm -f "$OUT.part"
    echo "primary mirror failed, trying fallback" >&2
    download "$FALLBACK"
fi

SIZE=$(stat -c%s "$OUT")
if [ "$SIZE" -lt "$MIN_BYTES" ]; then
    rm -f "$OUT"
    echo "downloaded file is too small ($SIZE bytes); removed it" >&2
    exit 1
fi
echo "model ready: $OUT ($SIZE bytes)"
