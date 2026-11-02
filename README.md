# cctvpro

Vehicle detection, tracking and speed estimation from a single video source, in
real time, on a laptop CPU.

Point it at a local clip, an HLS playlist, or an RTSP camera and it draws a box
around every vehicle with its class, a persistent track number, and an estimated
speed in km/h. Vehicles over the configured limit are drawn in red.

This is a proof of concept: no recording, no zone analytics, no multi-camera
orchestration.

## How it works

Three stages run on separate threads, connected by bounded queues that drop the
oldest frame when a consumer falls behind:

```
decode thread ──▶ mailbox (4) ──▶ inference thread ──▶ mailbox (1) ──▶ window
```

- **Decode** opens the source through ffmpeg, converts to RGB24, and paces local
  files to their frame rate. Network streams are paced by the server, so they are
  left alone. If a network stream drops, the decoder retries with backoff.
- **Inference** runs YOLOv8n through ONNX Runtime every Nth frame, then runs the
  tracker and the speed estimator on every frame. Skipped frames still advance
  the tracks, using each track's Kalman prediction.
- **Render** owns the window, never blocks on the pipeline, and redraws the last
  frame when nothing new has arrived, so key handling stays responsive.

Dropping the oldest frame is deliberate. An unbounded queue would grow latency
without limit and the window would show increasingly stale video.

## Requirements

- Linux with a display (X11 or Wayland)
- Rust 1.92 or newer
- ffmpeg 9.x development libraries, `pkg-config`, and `clang`

On Debian or Ubuntu:

```sh
sudo apt install build-essential pkg-config clang \
    libavcodec-dev libavformat-dev libavutil-dev libswscale-dev
```

ONNX Runtime is not installed system-wide. The `ort` crate downloads the 1.31
shared libraries on the first build (about 50 MB) and links them statically into
the binary.

## Quick start

```sh
./scripts/fetch-model.sh     # models/yolov8n.onnx, about 13 MB
./scripts/fetch-sample.sh    # samples/traffic.webm, about 30 MB

cargo build --release
cargo run --release -- samples/traffic.webm --config config.toml
```

Press Escape or `q` to quit.

The scripts are idempotent, so re-running them is harmless.

### Running against a stream

Every source goes through the same ffmpeg path, so the only thing that changes is
the argument:

```sh
# HLS playlist
cargo run --release -- "https://example.com/live/stream.m3u8" --config config.toml

# RTSP camera
cargo run --release -- "rtsp://user:pass@192.168.1.50:554/Streaming/Channels/101" \
    --config config.toml
```

Network sources reconnect automatically after a drop. Retries wait 1, 2, 4, and 8
seconds, then 10 seconds each time, forever. The frame counter keeps running
across reconnects, so speeds stay correct.

### Optional: exporting your own model

The bundled weights come pre-exported, so Python is not needed to run this
project. If you would rather export your own YOLOv8n, this is the equivalent
command, run once on a machine that has ultralytics installed:

```sh
yolo export model=yolov8n.pt format=onnx imgsz=640 opset=12
```

Then point `detector.model` at the resulting file. Startup checks the output
shape and refuses to run with an unexpected one, naming the shape it found.

## Command line

```
cctvpro <SOURCE> [--config <PATH>] [--model <PATH>] [--skip <N>] [--provider <cpu|cuda>]
```

| Flag | Meaning |
| --- | --- |
| `<SOURCE>` | Local file, `.m3u8` URL, or `rtsp://` URL. Required. |
| `--config` | TOML configuration file. Defaults to `config.toml`. |
| `--model` | Overrides `detector.model`. |
| `--skip` | Overrides `detector.skip`. |
| `--provider` | Overrides `detector.provider`. |

Flags always win over the file.

Set `RUST_LOG` to change log verbosity, for example `RUST_LOG=cctvpro=debug` to
log every inference time and every processed frame.

## Configuration

`config.toml` has three sections. Every key is optional: an omitted key uses its
default, which is the value shown below.

### `[detector]`

| Key | Default | Meaning |
| --- | --- | --- |
| `model` | `"models/yolov8n.onnx"` | Path to the ONNX weights. |
| `confidence` | `0.35` | Minimum class score. Lower finds more, at the cost of false positives. |
| `iou_threshold` | `0.45` | Above this overlap, two same-class boxes collapse into one. |
| `intra_threads` | `0` | ONNX Runtime worker threads. `0` lets the runtime decide. |
| `provider` | `"cpu"` | `cpu` or `cuda`. CUDA needs a build with `--features cuda`. |
| `skip` | `2` | Run inference every Nth frame. The tracker predicts in between. |

`skip` is the main speed/accuracy dial. `skip = 1` runs inference on every frame
and gives the most accurate boxes at the highest CPU cost. Raising it keeps the
display smooth on a slow machine, because the tracker carries the boxes between
detections.

### `[tracker]`

| Key | Default | Meaning |
| --- | --- | --- |
| `iou_threshold` | `0.3` | Minimum overlap to associate a detection with a track. |
| `min_hits` | `3` | Matches before a track is drawn. Suppresses one-frame flickers. |
| `max_age` | `10` | Inference frames a track survives without a match before it is dropped. |

Raising `max_age` keeps a track alive through longer occlusions, at the risk of
carrying a stale box. Lowering it frees the number sooner.

### `[calibration]`

| Key | Default | Meaning |
| --- | --- | --- |
| `method` | `"line"` | `line` or `homography`. |
| `speed_limit_kmh` | `50.0` | Vehicles above this are drawn in red. |
| `line_axis` | from `orientation` | `horizontal` lines measure up/down motion; `vertical` lines measure left/right motion. |
| `line1` | from `orientation` | First line. Fraction of the frame extent if `0.0..=1.0`, else absolute pixels. |
| `line2` | from `orientation` | Second line, same units as `line1`. |
| `distance_m` | `20.0` | Real-world gap between the two lines. Line method only. |
| `pixels` | `[]` | Four pixel corners of the road patch. Homography only. |
| `world_m` | `[]` | The same four corners in metres. Homography only. |
| `smoothing` | `0.4` | Speed smoothing, 0 to 1. Homography only. |
| `show_geometry` | `true` | Draw the calibration lines or polygon over the video. Set `false` once tuned. |

Validation rejects a zero distance, identical line positions, a wrong point
count, and an unknown method, naming the offending key.

## Choosing a calibration method

Speed needs a scale: pixels alone cannot tell you how fast something moves. Both
methods supply one.

**Line crossing** is the default because it needs three numbers. Draw two
parallel lines across the lane, measure the real gap between them, and tell the
config how far apart they are. When a vehicle's bottom edge crosses the first
line the video time is recorded; when it crosses the second, the speed follows
from distance over elapsed time.

Which way the lines run is set by `line_axis`, or inherited from `orientation`:

- `orientation = "horizontal"` — the road crosses the frame left to right, so the
  lines run vertically and **left-right** motion is measured.
- `orientation = "vertical"` — the road recedes toward a horizon, so the lines run
  horizontally and **up-down** motion is measured.

A line only sees motion that crosses it. Left-right traffic never crosses a
horizontal line, which is why the axis matters.

To pick the lines, run the program and read the yellow geometry on screen. They
are drawn where your config says they are. Move `line1` and `line2` until each
line sits on a visible feature across the road, then measure that real gap in
metres and set `distance_m`. Positions in `0.0..=1.0` are fractions of the frame,
so one config survives a resolution change on a fixed camera.

The lines should be roughly perpendicular to the traffic flow. A vehicle driving
diagonally across them travels further than the straight-line distance, so its
speed reads low.

**A line-crossing speed arrives late.** A track needs *two* crossing timestamps
before there is any elapsed time to divide by, so a vehicle that has crossed only
the first line has no speed yet. On a receding road that means waiting until the
vehicle reaches the second line, which may be most of the way down the frame.
That is arithmetic, not a bug: one timestamp cannot produce a speed. If you want a
speed on every vehicle as soon as it is tracked, use homography.

**Homography** handles angled views. Give four corners of a road patch in the
image, then the same four corners in metres, clockwise from the bottom-left. The
estimator maps each vehicle's bottom-centre onto the road plane and measures
distance travelled there, so perspective is corrected.

`config-homography.toml` is a working example:

```sh
cargo run --release -- samples/traffic.webm --config config-homography.toml
```

Both methods derive time only from `frame_index / fps`. Wall-clock time is never
used, so a slow machine reports the same speeds as a fast one for the same
frames.

**The shipped values are a starting point, not a measurement.** `distance_m` is
the one number that scales every speed linearly: if it is wrong by a factor of
two, every reported speed is wrong by the same factor.

## On screen

| Element | Meaning |
| --- | --- |
| `FPS` | Display refresh rate. |
| `INF` | Milliseconds the last inference took. |
| `DROP` | Frames discarded because a stage fell behind. |
| Yellow lines or quad | The active calibration geometry, if `show_geometry` is on. |
| Box colour | Class colour, or red when over the limit. |
| `car #7 42 KMH` | Class, persistent track number, and speed. Speed appears only after a vehicle has crossed the calibration geometry. |

Track numbers are stable while a vehicle stays visible. They are not a
re-identification system: a vehicle that leaves the frame and returns gets a new
number.

Detected classes are car, motorcycle, bus, truck, and bicycle. Everything else
the model finds, including people, is discarded.

## Tests

```sh
cargo test
```

Covers letterbox geometry and its round trip, IoU, per-class NMS, output layout
handling, Kalman prediction and update, track association and lifecycle, the
line-crossing speed maths, homography solving and projection, mailbox
drop-oldest behaviour, backoff scheduling, and frame-rate fallbacks.

`scripts/verify-reconnect.sh` exercises the reconnect path end to end: it serves
the sample clip over HTTP, kills the server mid-stream, restarts it, and checks
that the decoder recovers and the frame counter keeps increasing.

## Project layout

| File | Responsibility |
| --- | --- |
| `src/main.rs` | CLI, logging, window loop, drawing the overlay |
| `src/config.rs` | TOML schema, defaults, validation |
| `src/decode.rs` | ffmpeg decoding, frame rate resolution, reconnect |
| `src/detector.rs` | Letterbox, ONNX Runtime session, postprocessing, NMS |
| `src/tracker.rs` | Kalman filter, IoU association, track lifecycle |
| `src/speed.rs` | Line crossing, homography, km/h computation |
| `src/pipeline.rs` | Thread orchestration, frame pacing, frame dropping |
| `src/threads.rs` | Bounded drop-oldest mailbox, shutdown flag |
| `src/render.rs` | Framebuffer, bitmap font, window presentation |
| `src/geometry.rs` | Bounding boxes and IoU |
