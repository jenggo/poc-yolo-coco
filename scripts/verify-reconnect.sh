#!/usr/bin/env bash
# Verify that a mid-stream drop is detected and recovered, and that the frame
# index keeps increasing across the reconnect.
set -uo pipefail

cd "$(dirname "$0")/.."
LOG=/tmp/reconnect-test.log
PORT=8123
BIN=./target/release/cctvpro

serve() { (cd samples && python3 -m http.server "$PORT" --bind 127.0.0.1 >/dev/null 2>&1 &); }
stop_server() { pkill -f "http.server $PORT" 2>/dev/null; sleep 1; }

pkill -f "http.server $PORT" 2>/dev/null
pkill -f "$BIN" 2>/dev/null
sleep 1

serve; sleep 2
RUST_LOG=info,cctvpro=debug "$BIN" "http://127.0.0.1:$PORT/traffic.webm" \
    --config config.toml >"$LOG" 2>&1 &
APP=$!

sleep 6
stop_server
echo "server stopped"
sleep 4
serve
echo "server restarted"
sleep 12

kill "$APP" 2>/dev/null
wait "$APP" 2>/dev/null
stop_server

echo "--- log ---"
sed -E 's/\x1b\[[0-9;]*m//g' "$LOG" | grep -E "stream opened|reconnect|reconnected|display" | tail -12
echo "--- frame indices around the reconnect ---"
sed -E 's/\x1b\[[0-9;]*m//g' "$LOG" | grep -oE "frame=[0-9]+" | tail -5
