use std::collections::HashMap;

use anyhow::{Result, bail};
use tracing::{debug, info};

use crate::config::{CalibrationMethod, LineAxis, ResolvedCalibration};
use crate::tracker::TrackedObject;

/// Metres per second to kilometres per hour.
const MPS_TO_KMH: f32 = 3.6;

/// A computed speed and whether it exceeds the configured limit.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SpeedReading {
    pub kmh: f32,
    pub over_limit: bool,
}

/// Per-track state for the line-crossing method.
#[derive(Debug, Clone, Default)]
struct LineState {
    first_crossing_at: Option<f32>,
    second_crossing_at: Option<f32>,
    last_position: Option<f32>,
}

/// Per-track state for the homography method.
#[derive(Debug, Clone, Default)]
struct HomographyState {
    samples: Vec<(f32, [f32; 2])>,
    smoothed: Option<f32>,
}

/// Seconds of video time the homography displacement is measured over.
const HOMOGRAPHY_SPAN: f32 = 0.5;
/// Safety cap on retained samples, so a very high frame rate cannot grow the
/// history without bound.
const HOMOGRAPHY_MAX_SAMPLES: usize = 64;
/// Fraction of the span that must be filled before a speed is reported. Below
/// this, box jitter divided by a short interval is pure noise.
const HOMOGRAPHY_MIN_FILL: f32 = 0.9;

/// Turns track motion and video time into kilometres per hour.
///
/// Time always comes from `frame_index / fps`, never from the wall clock, so a
/// slow machine reports the same speeds as a fast one for the same frames.
pub struct SpeedEstimator {
    calibration: ResolvedCalibration,
    method: CalibrationMethod,
    homography: Option<[[f32; 3]; 3]>,
    lines: HashMap<u64, LineState>,
    homographies: HashMap<u64, HomographyState>,
    readings: HashMap<u64, SpeedReading>,
}

impl SpeedEstimator {
    pub fn new(calibration: &ResolvedCalibration) -> Result<Self> {
        let method = calibration.method;
        let homography = match method {
            CalibrationMethod::Line => None,
            CalibrationMethod::Homography => Some(solve_homography(
                &calibration.pixels,
                &calibration.world_m,
            )?),
        };

        Ok(Self {
            calibration: calibration.clone(),
            method,
            homography,
            lines: HashMap::new(),
            homographies: HashMap::new(),
            readings: HashMap::new(),
        })
    }

    /// Update the estimate for every reported track on this frame.
    pub fn update(&mut self, tracks: &[TrackedObject], frame_index: u64, fps: f32) {
        let now = video_time(frame_index, fps);
        for track in tracks {
            let reading = match self.method {
                CalibrationMethod::Line => self.update_line(track, now),
                CalibrationMethod::Homography => self.update_homography(track, now),
            };
            if let Some(kmh) = reading {
                let over_limit = kmh > self.calibration.speed_limit_kmh;
                match self.readings.get(&track.id) {
                    None => info!(track = track.id, kmh, over_limit, "speed available"),
                    Some(previous) if (previous.kmh - kmh).abs() > 0.5 => {
                        debug!(track = track.id, kmh, over_limit, "speed updated");
                    }
                    Some(_) => {}
                }
                self.readings.insert(
                    track.id,
                    SpeedReading {
                        kmh,
                        over_limit,
                    },
                );
            }
        }
    }

    pub fn reading(&self, track_id: u64) -> Option<SpeedReading> {
        self.readings.get(&track_id).copied()
    }

    /// Drop state for tracks that no longer exist.
    pub fn retain(&mut self, live_ids: &[u64]) {
        self.readings.retain(|id, _| live_ids.contains(id));
        self.lines.retain(|id, _| live_ids.contains(id));
        self.homographies.retain(|id, _| live_ids.contains(id));
    }

    fn update_line(&mut self, track: &TrackedObject, now: f32) -> Option<f32> {
        // Measure along the axis the lines run across. Horizontal lines are
        // crossed by vertical motion (y); vertical lines by horizontal motion (x).
        let (x, y) = track.bbox.bottom_center();
        let position = match self.calibration.line_axis {
            LineAxis::Horizontal => y,
            LineAxis::Vertical => x,
        };
        let line1 = self.calibration.line1;
        let line2 = self.calibration.line2;

        let state = self.lines.entry(track.id).or_default();
        let previous = state.last_position.replace(position);

        if let Some(previous) = previous {
            let crossed = |line: f32| (previous - line).signum() != (position - line).signum();
            if crossed(line1) && state.first_crossing_at.is_none() {
                state.first_crossing_at = Some(now);
            } else if crossed(line2) && state.second_crossing_at.is_none() {
                state.second_crossing_at = Some(now);
            }
        }

        let (Some(first), Some(second)) = (state.first_crossing_at, state.second_crossing_at) else {
            return None;
        };
        let elapsed = (second - first).abs();
        if elapsed <= 0.0 {
            return None;
        }
        Some(self.calibration.distance_m / elapsed * MPS_TO_KMH)
    }

    fn update_homography(&mut self, track: &TrackedObject, now: f32) -> Option<f32> {
        let homography = self.homography?;
        let (px, py) = track.bbox.bottom_center();
        let road = project(&homography, px, py);

        let state = self.homographies.entry(track.id).or_default();
        state.samples.push((now, road));

        // Keep only samples inside the smoothing span.
        let cutoff = now - HOMOGRAPHY_SPAN;
        while state.samples.len() > 2 && state.samples[0].0 < cutoff {
            state.samples.remove(0);
        }
        while state.samples.len() > HOMOGRAPHY_MAX_SAMPLES {
            state.samples.remove(0);
        }

        let (first_time, first_pos) = *state.samples.first()?;
        let elapsed = now - first_time;

        // Report only once the window is nearly full, so the measurement spans
        // enough frames for box jitter to average out.
        if elapsed < HOMOGRAPHY_SPAN * HOMOGRAPHY_MIN_FILL {
            return None;
        }

        let dx = road[0] - first_pos[0];
        let dy = road[1] - first_pos[1];
        let speed = (dx * dx + dy * dy).sqrt() / elapsed * MPS_TO_KMH;
        if !speed.is_finite() {
            return None;
        }

        let smoothing = self.calibration.smoothing.clamp(0.0, 1.0);
        let smoothed = match state.smoothed {
            Some(previous) => previous + smoothing * (speed - previous),
            None => speed,
        };
        state.smoothed = Some(smoothed);
        Some(smoothed)
    }
}

/// Video time in seconds, derived only from the frame index and frame rate.
pub fn video_time(frame_index: u64, fps: f32) -> f32 {
    if fps <= 0.0 {
        return 0.0;
    }
    frame_index as f32 / fps
}

/// Solve the 3x3 homography from four pixel/world correspondences by DLT.
///
/// The two linear equations per correspondence are stacked into an 8x8 system in
/// the eight free parameters (`h33` is fixed at 1) and solved by Gaussian
/// elimination with partial pivoting.
pub fn solve_homography(pixels: &[[f32; 2]], world: &[[f32; 2]]) -> Result<[[f32; 3]; 3]> {
    if pixels.len() != 4 || world.len() != 4 {
        bail!(
            "homography needs exactly 4 pixel points and 4 world points, got {} and {}",
            pixels.len(),
            world.len()
        );
    }

    let mut a = [[0.0f32; 9]; 8];
    for i in 0..4 {
        let [x, y] = pixels[i];
        let [u, v] = world[i];
        a[2 * i] = [x, y, 1.0, 0.0, 0.0, 0.0, -u * x, -u * y, u];
        a[2 * i + 1] = [0.0, 0.0, 0.0, x, y, 1.0, -v * x, -v * y, v];
    }

    // Gauss-Jordan on the augmented 8x9 system.
    for col in 0..8 {
        let mut pivot = col;
        for row in col + 1..8 {
            if a[row][col].abs() > a[pivot][col].abs() {
                pivot = row;
            }
        }
        if a[pivot][col].abs() < 1e-9 {
            bail!("homography points are degenerate: no unique solution exists");
        }
        a.swap(col, pivot);

        let scale = a[col][col];
        for k in 0..9 {
            a[col][k] /= scale;
        }

        for row in 0..8 {
            if row == col {
                continue;
            }
            let factor = a[row][col];
            if factor == 0.0 {
                continue;
            }
            for k in 0..9 {
                a[row][k] -= factor * a[col][k];
            }
        }
    }

    Ok([
        [a[0][8], a[1][8], a[2][8]],
        [a[3][8], a[4][8], a[5][8]],
        [a[6][8], a[7][8], 1.0],
    ])
}

/// Apply a homography to a point, returning the normalised result.
pub fn project(h: &[[f32; 3]; 3], x: f32, y: f32) -> [f32; 2] {
    let w = h[2][0] * x + h[2][1] * y + h[2][2];
    if w.abs() < 1e-9 {
        return [f32::NAN, f32::NAN];
    }
    [
        (h[0][0] * x + h[0][1] * y + h[0][2]) / w,
        (h[1][0] * x + h[1][1] * y + h[1][2]) / w,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::BBox;

    fn line_calibration() -> ResolvedCalibration {
        ResolvedCalibration {
            method: CalibrationMethod::Line,
            speed_limit_kmh: 50.0,
            line_axis: LineAxis::Horizontal,
            line1: 400.0,
            line2: 800.0,
            distance_m: 12.0,
            pixels: Vec::new(),
            world_m: Vec::new(),
            smoothing: 0.4,
            show_geometry: true,
        }
    }

    fn track_at(id: u64, bottom_y: f32) -> TrackedObject {
        TrackedObject {
            id,
            class_id: 2,
            class_name: "car",
            bbox: BBox::new(100.0, bottom_y - 40.0, 140.0, bottom_y),
        }
    }

    fn track_at_x(id: u64, center_x: f32) -> TrackedObject {
        TrackedObject {
            id,
            class_id: 2,
            class_name: "car",
            bbox: BBox::new(center_x - 20.0, 200.0, center_x + 20.0, 240.0),
        }
    }

    #[test]
    fn vertical_lines_measure_left_to_right_traffic() {
        let mut calibration = line_calibration();
        calibration.line_axis = LineAxis::Vertical;
        calibration.line1 = 400.0;
        calibration.line2 = 800.0;
        let mut estimator = SpeedEstimator::new(&calibration).unwrap();

        // Crossing x=400 at frame 25 and x=800 at frame 75, as before, but now
        // the motion is horizontal and y never changes.
        for frame in 0..=26u64 {
            let x = if frame <= 25 { 395.0 } else { 405.0 };
            estimator.update(&[track_at_x(7, x)], frame, 25.0);
        }
        assert!(estimator.reading(7).is_none(), "no speed before the second crossing");

        for frame in 27..=76u64 {
            let x = if frame <= 75 { 795.0 } else { 805.0 };
            estimator.update(&[track_at_x(7, x)], frame, 25.0);
        }

        let reading = estimator.reading(7).expect("speed for horizontal motion");
        assert!((reading.kmh - 21.6).abs() < 0.5, "expected 21.6 km/h, got {}", reading.kmh);
    }

    #[test]
    fn vertical_lines_ignore_vertical_motion() {
        let mut calibration = line_calibration();
        calibration.line_axis = LineAxis::Vertical;
        calibration.line1 = 400.0;
        calibration.line2 = 800.0;
        let mut estimator = SpeedEstimator::new(&calibration).unwrap();

        // A car driving down the frame never changes x, so it crosses nothing.
        for frame in 0..=76u64 {
            let y = 300.0 + frame as f32 * 5.0;
            estimator.update(&[track_at(8, y)], frame, 25.0);
        }
        assert!(estimator.reading(8).is_none());
    }

    #[test]
    fn horizontal_lines_ignore_left_to_right_motion() {
        let mut estimator = SpeedEstimator::new(&line_calibration()).unwrap();
        for frame in 0..=76u64 {
            let x = 150.0 + frame as f32 * 10.0;
            estimator.update(&[track_at_x(9, x)], frame, 25.0);
        }
        assert!(estimator.reading(9).is_none());
    }

    #[test]
    fn line_crossings_at_frames_25_and_75_give_21_6_kmh() {
        let mut estimator = SpeedEstimator::new(&line_calibration()).unwrap();

        // Frame 25 sits just above line 1, frame 26 has crossed it.
        for frame in 0..=26u64 {
            let y = if frame <= 25 { 395.0 } else { 405.0 };
            estimator.update(&[track_at(1, y)], frame, 25.0);
        }
        assert!(estimator.reading(1).is_none(), "no speed before the second crossing");

        for frame in 27..=76u64 {
            let y = if frame <= 75 { 795.0 } else { 805.0 };
            estimator.update(&[track_at(1, y)], frame, 25.0);
        }

        // 12 m in (75 - 25) / 25 s = 2 s is 6 m/s, which is 21.6 km/h.
        let reading = estimator.reading(1).expect("speed after the second crossing");
        assert!(
            (reading.kmh - 21.6).abs() < 0.5,
            "expected 21.6 km/h, got {}",
            reading.kmh
        );
        assert!(!reading.over_limit, "21.6 km/h is under the 50 km/h limit");
    }

    #[test]
    fn crossing_works_in_the_upward_direction_too() {
        let mut estimator = SpeedEstimator::new(&line_calibration()).unwrap();
        for frame in 0..=26u64 {
            let y = if frame <= 25 { 805.0 } else { 795.0 };
            estimator.update(&[track_at(2, y)], frame, 25.0);
        }
        for frame in 27..=76u64 {
            let y = if frame <= 75 { 405.0 } else { 395.0 };
            estimator.update(&[track_at(2, y)], frame, 25.0);
        }
        let reading = estimator.reading(2).expect("speed after the second crossing");
        assert!((reading.kmh - 21.6).abs() < 0.5, "got {}", reading.kmh);
    }

    #[test]
    fn speed_over_the_limit_is_flagged() {
        let mut calibration = line_calibration();
        calibration.speed_limit_kmh = 10.0;
        let mut estimator = SpeedEstimator::new(&calibration).unwrap();
        for frame in 0..=76u64 {
            let y = match frame {
                0..=25 => 395.0,
                26..=75 => 795.0,
                _ => 805.0,
            };
            estimator.update(&[track_at(3, y)], frame, 25.0);
        }
        let reading = estimator.reading(3).unwrap();
        assert!(reading.over_limit, "{} km/h exceeds the 10 km/h limit", reading.kmh);
    }

    #[test]
    fn wall_clock_does_not_enter_the_speed() {
        // Two identical frame sequences produce identical speeds.
        let run = || {
            let mut estimator = SpeedEstimator::new(&line_calibration()).unwrap();
            for frame in 0..=76u64 {
                let y = match frame {
                    0..=25 => 395.0,
                    26..=75 => 795.0,
                    _ => 805.0,
                };
                estimator.update(&[track_at(9, y)], frame, 25.0);
                std::thread::sleep(std::time::Duration::from_micros(1));
            }
            estimator.reading(9).unwrap().kmh
        };
        assert_eq!(run(), run());
    }

    fn homography_calibration() -> ResolvedCalibration {
        ResolvedCalibration {
            method: CalibrationMethod::Homography,
            speed_limit_kmh: 50.0,
            line_axis: LineAxis::Horizontal,
            line1: 400.0,
            line2: 800.0,
            distance_m: 12.0,
            pixels: vec![
                [100.0, 800.0],
                [700.0, 800.0],
                [800.0, 400.0],
                [0.0, 400.0],
            ],
            world_m: vec![[0.0, 0.0], [12.0, 0.0], [12.0, 40.0], [0.0, 40.0]],
            smoothing: 0.5,
            show_geometry: true,
        }
    }

    #[test]
    fn homography_maps_the_configured_points_onto_their_world_points() {
        let calibration = homography_calibration();
        let h = solve_homography(&calibration.pixels, &calibration.world_m).unwrap();
        for (pixel, world) in calibration.pixels.iter().zip(&calibration.world_m) {
            let projected = project(&h, pixel[0], pixel[1]);
            assert!(
                (projected[0] - world[0]).abs() < 1e-2 && (projected[1] - world[1]).abs() < 1e-2,
                "pixel {pixel:?} projected to {projected:?}, expected {world:?}"
            );
        }
    }

    #[test]
    fn homography_reports_speed_from_road_displacement() {
        let mut estimator = SpeedEstimator::new(&homography_calibration()).unwrap();
        // 12 m of road width spans 600 px at the bottom of the frame, so 1 px is
        // 0.02 m. Move 5 px per frame at 25 fps = 0.1 m per 0.04 s = 9 km/h.
        for frame in 0..=20u64 {
            let x = 200.0 + frame as f32 * 5.0;
            let track = TrackedObject {
                id: 4,
                class_id: 2,
                class_name: "car",
                bbox: BBox::new(x - 20.0, 760.0, x + 20.0, 800.0),
            };
            estimator.update(&[track], frame, 25.0);
        }
        let reading = estimator.reading(4).expect("homography speed");
        assert!(
            (reading.kmh - 9.0).abs() < 2.5,
            "expected about 9 km/h, got {}",
            reading.kmh
        );
    }

    #[test]
    fn degenerate_homography_points_are_rejected() {
        let pixels = vec![[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]];
        let world = vec![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
        let err = solve_homography(&pixels, &world).unwrap_err().to_string();
        assert!(err.contains("degenerate"), "{err}");
    }

    #[test]
    fn video_time_uses_the_frame_rate() {
        assert!((video_time(50, 25.0) - 2.0).abs() < 1e-6);
        assert_eq!(video_time(50, 0.0), 0.0);
    }
}
