#!/usr/bin/env bash
# Final acceptance check: run the sample clip and confirm live detections with
# track IDs, classes, and speeds at a smooth frame rate.
set -uo pipefail

cd "$(dirname "$0")/.."
LOG=/tmp/acceptance.log
BIN=./target/release/cctvpro

pkill -f "$BIN" 2>/dev/null
sleep 1

RUST_LOG=info,cctvpro=debug "$BIN" samples/traffic.webm --config config.toml >"$LOG" 2>&1 &
APP=$!

sleep 22
DISPLAY=:0 spectacle -b -n -f -o /tmp/acceptance.png >/dev/null 2>&1
sleep 1

kill "$APP" 2>/dev/null
wait "$APP" 2>/dev/null

CLEAN=/tmp/acceptance.clean
sed -E 's/\x1b\[[0-9;]*m//g' "$LOG" >"$CLEAN"

echo "=== speeds reported ==="
grep "speed available" "$CLEAN" | tail -5
echo "=== display rate ==="
grep "display fps" "$CLEAN" | tail -4
echo "=== distinct classes drawn ==="
grep -oE '"[a-z]+"' "$CLEAN" | sort | uniq -c | sort -rn | head -6
echo "=== screenshot ==="
ls -la /tmp/acceptance.png
