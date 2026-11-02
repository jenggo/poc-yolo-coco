mod config;
mod decode;
mod detector;
mod geometry;
mod pipeline;
mod render;
mod speed;
mod threads;
mod tracker;

use std::path::PathBuf;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use clap::Parser;
use ffmpeg_next as ffmpeg;
use tracing::{info, warn};
use tracing_subscriber::EnvFilter;

use crate::config::{CalibrationMethod, Config, LineAxis, ResolvedCalibration};
use crate::pipeline::{Pipeline, ProcessedFrame, VehicleView};
use crate::render::{Canvas, Display, RED, YELLOW, class_color};

#[derive(Parser, Debug)]
#[command(
    name = "cctvpro",
    about = "Vehicle detection, tracking and speed estimation proof of concept"
)]
struct Cli {
    /// Video source: a local file, an HLS (.m3u8) URL, or an rtsp:// URL
    source: String,

    /// Path to the TOML configuration file
    #[arg(long, default_value = "config.toml")]
    config: PathBuf,

    /// Override the detector model path
    #[arg(long)]
    model: Option<PathBuf>,

    /// Override the inference skip interval
    #[arg(long)]
    skip: Option<usize>,

    /// Override the execution provider (cpu or cuda)
    #[arg(long)]
    provider: Option<String>,

    /// Camera viewpoint: horizontal (road crosses the frame) or vertical
    /// (road recedes toward a horizon). Sets sensible default line positions.
    #[arg(long, value_name = "DIRECTION")]
    orientation: Option<String>,

    /// Real-world distance between the two calibration lines, in metres.
    #[arg(long, value_name = "METRES")]
    distance_m: Option<f32>,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    init_tracing();

    ffmpeg::init().context("initializing ffmpeg")?;

    let mut config = Config::load(&cli.config)?;
    config.apply_overrides(
        cli.model.as_deref(),
        cli.skip,
        cli.provider.as_deref(),
        cli.orientation.as_deref(),
        cli.distance_m,
    );
    config.validate()?;

    run(&cli.source, &config)
}

fn init_tracing() {
    // ONNX Runtime logs through `tracing` and is extremely verbose at info level.
    let filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new("info,ort=warn,ort_sys=warn"));
    tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_target(false)
        .init();
}

fn run(source: &str, config: &Config) -> Result<()> {
    let pipeline = Pipeline::start(config, source)?;
    let info = pipeline.info;

    let mut display = Display::new("cctvpro", info.width, info.height)?;
    let mut canvas = Canvas::new(info.width, info.height);
    let mut meter = FrameMeter::new();
    let mut rendered: Option<ProcessedFrame> = None;
    let calibration = &pipeline.calibration;
    let method = calibration.method;

    while !display.should_quit() {
        if let Some(frame) = pipeline.try_next() {
            compose(&mut canvas, &frame, calibration, method);
            meter.tick();
            canvas.status_overlay(&meter.status_lines(&pipeline, frame.inference_ms), 2);
            rendered = Some(frame);
        } else if rendered.is_none() {
            // Nothing to show yet: wait briefly rather than spinning.
            std::thread::sleep(Duration::from_millis(2));
            continue;
        }

        display.present(&canvas)?;

        if rendered.is_some() && pipeline.is_finished() {
            info!("source finished");
            break;
        }
    }

    if !display.is_open() {
        warn!("window closed by the user");
    }
    pipeline.shutdown();
    Ok(())
}

/// Compose one frame: video, tracked vehicles, and calibration geometry.
fn compose(
    canvas: &mut Canvas,
    frame: &ProcessedFrame,
    calibration: &ResolvedCalibration,
    method: CalibrationMethod,
) {
    if canvas
        .fill_from_rgb(frame.width, frame.height, &frame.rgb)
        .is_err()
    {
        return;
    }

    if calibration.show_geometry {
        draw_calibration(canvas, calibration, method);
    }
    for vehicle in &frame.vehicles {
        draw_vehicle(canvas, vehicle);
    }
}

fn draw_calibration(
    canvas: &mut Canvas,
    calibration: &ResolvedCalibration,
    method: CalibrationMethod,
) {
    let width = canvas.width() as i32;
    match method {
        CalibrationMethod::Line => {
            let height = canvas.height() as i32;
            for position in [calibration.line1, calibration.line2] {
                let position = position.round() as i32;
                match calibration.line_axis {
                    LineAxis::Horizontal => canvas.hline(0, width, position, YELLOW),
                    LineAxis::Vertical => canvas.vline(position, 0, height, YELLOW),
                }
            }
        }
        CalibrationMethod::Homography => {
            let points = &calibration.pixels;
            if points.len() != 4 {
                return;
            }
            for i in 0..4 {
                let a = points[i];
                let b = points[(i + 1) % 4];
                canvas.line(
                    a[0].round() as i32,
                    a[1].round() as i32,
                    b[0].round() as i32,
                    b[1].round() as i32,
                    YELLOW,
                );
            }
        }
    }
}

fn draw_vehicle(canvas: &mut Canvas, vehicle: &VehicleView) {
    let over_limit = vehicle.speed.is_some_and(|s| s.over_limit);
    let color = if over_limit {
        RED
    } else {
        class_color(vehicle.class_id)
    };

    let bbox = vehicle.bbox;
    let x = bbox.x1.round() as i32;
    let y = bbox.y1.round() as i32;
    let w = bbox.width().round() as i32;
    let h = bbox.height().round() as i32;
    canvas.rect(x, y, w, h, color, 2);

    let label = match vehicle.speed {
        Some(speed) => format!("{} #{} {:.0} KMH", vehicle.class_name, vehicle.id, speed.kmh),
        None => format!("{} #{}", vehicle.class_name, vehicle.id),
    };
    canvas.label(x, (y - 20).max(0), &label, color, 2);
}

/// Display frame rate plus a periodic log line, both driven by the wall clock.
struct FrameMeter {
    frames: u32,
    total: u64,
    window_start: Instant,
    last_log: Instant,
    fps: f32,
}

impl FrameMeter {
    const REPORT_INTERVAL: Duration = Duration::from_millis(500);
    const LOG_INTERVAL: Duration = Duration::from_secs(2);

    fn new() -> Self {
        let now = Instant::now();
        Self {
            frames: 0,
            total: 0,
            window_start: now,
            last_log: now,
            fps: 0.0,
        }
    }

    fn tick(&mut self) {
        self.frames += 1;
        self.total += 1;

        let elapsed = self.window_start.elapsed();
        if elapsed >= Self::REPORT_INTERVAL {
            self.fps = self.frames as f32 / elapsed.as_secs_f32();
            self.frames = 0;
            self.window_start = Instant::now();
        }
    }

    fn status_lines(&mut self, pipeline: &Pipeline, inference_ms: f32) -> Vec<String> {
        if self.last_log.elapsed() >= Self::LOG_INTERVAL {
            info!(
                fps = self.fps,
                frames = self.total,
                inference_ms,
                dropped = pipeline.dropped(),
                "display"
            );
            self.last_log = Instant::now();
        }

        vec![
            format!("FPS {:.1}", self.fps),
            format!("INF {:.0} MS", inference_ms),
            format!("DROP {}", pipeline.dropped() + pipeline.decode_dropped()),
        ]
    }
}
