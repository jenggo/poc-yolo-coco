use std::sync::Arc;
use std::sync::mpsc;
use std::thread::{self, JoinHandle};
use std::time::Duration;

use anyhow::{Context, Result, anyhow};
use tracing::{debug, error, info, warn};

use crate::config::{Config, Orientation, ResolvedCalibration};
use crate::decode::{VideoSource, is_network_source};
use crate::detector::Detector;
use crate::geometry::BBox;
use crate::speed::{SpeedEstimator, SpeedReading};
use crate::threads::{Mailbox, Shutdown};
use crate::tracker::{Tracker, TrackerSettings};

/// Properties of the opened stream, needed before the window can be created.
#[derive(Debug, Clone, Copy)]
pub struct StreamInfo {
    pub width: usize,
    pub height: usize,
    pub fps: f32,
}

/// Everything the renderer needs to draw one tracked vehicle.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VehicleView {
    pub id: u64,
    pub class_id: u32,
    pub class_name: &'static str,
    pub bbox: BBox,
    pub speed: Option<SpeedReading>,
}

/// A decoded frame with its tracked vehicles, ready to draw.
pub struct ProcessedFrame {
    pub width: usize,
    pub height: usize,
    pub rgb: Vec<u8>,
    pub vehicles: Vec<VehicleView>,
    pub inference_ms: f32,
}

/// The running three-stage pipeline: decode, inference, and the renderer in the
/// caller's thread.
pub struct Pipeline {
    pub info: StreamInfo,
    /// Calibration resolved against `info`'s frame size.
    pub calibration: ResolvedCalibration,
    processed: Arc<Mailbox<ProcessedFrame>>,
    shutdown: Shutdown,
    decode_handle: Option<JoinHandle<()>>,
    infer_handle: Option<JoinHandle<()>>,
    decode_frames: Arc<Mailbox<crate::decode::Frame>>,
}

impl Pipeline {
    /// Start the decode and inference threads.
    ///
    /// The detector is loaded before any thread starts so a bad model path fails
    /// fast, and the source is opened inside the decode thread because ffmpeg
    /// contexts are not `Send`.
    pub fn start(config: &Config, source: &str) -> Result<Self> {
        let detector = Detector::load(&config.detector)?;

        let shutdown = Shutdown::new();
        let (info_tx, info_rx) = mpsc::channel::<Result<StreamInfo>>();

        let decode_frames = Arc::new(Mailbox::new(4));
        let processed = Arc::new(Mailbox::new(1));

        let decode_handle = {
            let shutdown = shutdown.clone();
            let decode_frames = Arc::clone(&decode_frames);
            let source = source.to_string();
            thread::Builder::new()
                .name("decode".to_string())
                .spawn(move || decode_loop(&source, &decode_frames, &shutdown, &info_tx))
                .context("spawning the decode thread")?
        };

        let info = match info_rx.recv() {
            Ok(Ok(info)) => info,
            Ok(Err(err)) => {
                shutdown.trigger();
                decode_frames.close();
                let _ = decode_handle.join();
                return Err(err);
            }
            Err(_) => {
                shutdown.trigger();
                let _ = decode_handle.join();
                return Err(anyhow!("the decode thread stopped before opening {source}"));
            }
        };

        info!(
            width = info.width,
            height = info.height,
            fps = info.fps,
            "stream opened"
        );

        // The frame size is known now, so fractional line positions and the
        // calibration-vs-resolution check can both be applied.
        let calibration = match config.resolve_calibration(info.width, info.height) {
            Ok(calibration) => calibration,
            Err(err) => {
                shutdown.trigger();
                decode_frames.close();
                let _ = decode_handle.join();
                return Err(err.context(format!(
                    "the calibration does not fit this {}x{} source",
                    info.width, info.height
                )));
            }
        };

        if let Some(warning) = &config.calibration.orientation
            && let Ok(orientation) = Orientation::parse(warning)
            && let Some(message) = orientation.perspective_warning()
        {
            warn!(message);
        }

        let infer_handle = {
            let shutdown = shutdown.clone();
            let decode_frames = Arc::clone(&decode_frames);
            let processed = Arc::clone(&processed);
            let settings = TrackerSettings {
                iou_threshold: config.tracker.iou_threshold,
                min_hits: config.tracker.min_hits,
                max_age: config.tracker.max_age,
            };
            let skip = config.detector.skip as u64;
            let calibration = calibration.clone();
            let fps = info.fps;

            thread::Builder::new()
                .name("infer".to_string())
                .spawn(move || {
                    infer_loop(
                        detector,
                        settings,
                        &calibration,
                        skip,
                        fps,
                        &decode_frames,
                        &processed,
                        &shutdown,
                    )
                })
                .context("spawning the inference thread")?
        };

        Ok(Self {
            info,
            calibration,
            processed,
            shutdown,
            decode_handle: Some(decode_handle),
            infer_handle: Some(infer_handle),
            decode_frames,
        })
    }

    /// Take the newest processed frame without blocking.
    pub fn try_next(&self) -> Option<ProcessedFrame> {
        self.processed.try_pop()
    }

    /// Frames the renderer missed because inference could not keep up.
    pub fn dropped(&self) -> u64 {
        self.processed.dropped()
    }

    /// Frames dropped between decode and inference.
    pub fn decode_dropped(&self) -> u64 {
        self.decode_frames.dropped()
    }

    pub fn is_finished(&self) -> bool {
        self.processed.is_closed() && self.processed.is_empty()
    }

    /// Signal every stage to stop and join the worker threads.
    pub fn shutdown(mut self) {
        self.stop();
    }

    fn stop(&mut self) {
        self.shutdown.trigger();
        self.decode_frames.close();
        self.processed.close();

        if let Some(handle) = self.decode_handle.take()
            && handle.join().is_err()
        {
            error!("the decode thread panicked");
        }
        if let Some(handle) = self.infer_handle.take()
            && handle.join().is_err()
        {
            error!("the inference thread panicked");
        }
    }
}

impl Drop for Pipeline {
    fn drop(&mut self) {
        self.stop();
    }
}

/// Emit decoded frames at the source frame rate for local files.
///
/// A network stream arrives already paced, but a file would otherwise be read as
/// fast as the disk allows. Combined with drop-oldest that turns playback into a
/// fast-forward: content advances many times faster than the display refreshes.
/// Pacing here makes a file play at real speed and lets drop-oldest do its actual
/// job, which is shedding frames the display cannot keep up with.
struct Pacer {
    start: std::time::Instant,
    /// Frame index of the first emitted frame.
    origin: u64,
    fps: f32,
    enabled: bool,
}

impl Pacer {
    fn new(fps: f32, origin: u64, enabled: bool) -> Self {
        Self {
            start: std::time::Instant::now(),
            origin,
            fps,
            enabled,
        }
    }

    /// Sleep until this frame is due. Returns `false` when shutdown was requested.
    fn wait(&self, index: u64, shutdown: &Shutdown) -> bool {
        if !self.enabled || self.fps <= 0.0 {
            return !shutdown.is_triggered();
        }

        let due = Duration::from_secs_f64((index.saturating_sub(self.origin)) as f64 / self.fps as f64);
        loop {
            if shutdown.is_triggered() {
                return false;
            }
            let elapsed = self.start.elapsed();
            if elapsed >= due {
                return true;
            }
            let remaining = due - elapsed;
            std::thread::sleep(remaining.min(Duration::from_millis(5)));
        }
    }
}

fn decode_loop(
    source: &str,
    frames: &Mailbox<crate::decode::Frame>,
    shutdown: &Shutdown,
    info_tx: &mpsc::Sender<Result<StreamInfo>>,
) {
    let mut video = match VideoSource::open(source) {
        Ok(video) => video,
        Err(err) => {
            let _ = info_tx.send(Err(err));
            return;
        }
    };

    let info = StreamInfo {
        width: video.width(),
        height: video.height(),
        fps: video.fps(),
    };
    if info_tx.send(Ok(info)).is_err() {
        return;
    }

    // Network sources are already paced by the remote server; files are not.
    let pacer = Pacer::new(info.fps, 0, !is_network_source(source));

    loop {
        if shutdown.is_triggered() {
            break;
        }
        match video.next_frame(shutdown) {
            Ok(Some(frame)) => {
                if !pacer.wait(frame.index, shutdown) {
                    break;
                }
                if !frames.push_drop_oldest(frame) {
                    break;
                }
            }
            Ok(None) => {
                info!("source finished");
                break;
            }
            Err(err) => {
                error!(error = %err, "decode failed");
                break;
            }
        }
    }

    frames.close();
}

#[allow(clippy::too_many_arguments)]
fn infer_loop(
    mut detector: Detector,
    settings: TrackerSettings,
    calibration: &ResolvedCalibration,
    skip: u64,
    fps: f32,
    frames: &Mailbox<crate::decode::Frame>,
    processed: &Mailbox<ProcessedFrame>,
    shutdown: &Shutdown,
) {
    let mut tracker = Tracker::new(settings);
    let mut speed = match SpeedEstimator::new(calibration) {
        Ok(speed) => speed,
        Err(err) => {
            error!(error = %err, "speed estimator could not start");
            return;
        }
    };

    // The tracker advances by the *actual* video time between processed frames.
    // Decode-to-inference frames can be dropped, so a fixed 1/fps step would make
    // the Kalman prediction lag behind reality and break association.
    let fallback_dt = if fps > 0.0 { 1.0 / fps } else { 0.04 };
    let mut previous_index: Option<u64> = None;

    while let Some(frame) = frames.pop_blocking() {
        if shutdown.is_triggered() {
            break;
        }

        let dt = match previous_index {
            Some(previous) if frame.index > previous => (frame.index - previous) as f32 / fps,
            Some(_) => fallback_dt,
            None => fallback_dt,
        };
        previous_index = Some(frame.index);

        let detections = if frame.index % skip == 0 {
            match detector.detect(&frame) {
                Ok(detections) => Some(detections),
                Err(err) => {
                    error!(error = %err, "inference failed");
                    None
                }
            }
        } else {
            None
        };

        let tracked = tracker.step(detections.as_deref(), dt);
        speed.update(&tracked, frame.index, frame.fps);
        speed.retain(&tracker.live_ids());

        let vehicles: Vec<VehicleView> = tracked
            .iter()
            .map(|track| VehicleView {
                id: track.id,
                class_id: track.class_id,
                class_name: track.class_name,
                bbox: track.bbox,
                speed: speed.reading(track.id),
            })
            .collect();

        debug!(
            frame = frame.index,
            vehicles = vehicles.len(),
            ids = ?vehicles.iter().map(|v| v.id).collect::<Vec<_>>(),
            inference_ms = detector.last_inference_ms(),
            "processed frame"
        );

        let processed_frame = ProcessedFrame {
            width: frame.width,
            height: frame.height,
            rgb: frame.rgb,
            vehicles,
            inference_ms: detector.last_inference_ms(),
        };

        if !processed.push_drop_oldest(processed_frame) {
            break;
        }
    }

    processed.close();
}
