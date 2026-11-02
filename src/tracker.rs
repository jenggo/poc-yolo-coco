use crate::detector::Detection;
use crate::geometry::{BBox, iou};

/// State vector size: `cx, cy, w, h, vcx, vcy, vw, vh`.
const DIM: usize = 8;
/// Measurement size: `cx, cy, w, h`.
const MEAS: usize = 4;

/// Constant-velocity Kalman filter over a box centre, its size, and their rates.
///
/// The transition is `F = I + dt * [[0, I], [0, 0]]`, so position integrates
/// velocity over one time step and velocity is held constant. Matrices are fixed
/// size and stack allocated: an 8x8 filter does not justify a linear algebra
/// dependency.
#[derive(Debug, Clone)]
pub struct KalmanBoxFilter {
    state: [f32; DIM],
    covariance: [[f32; DIM]; DIM],
    process_noise: [f32; DIM],
    measurement_noise: [f32; MEAS],
}

impl KalmanBoxFilter {
    /// Create a filter from an initial measurement. Velocity starts at zero.
    pub fn new(bbox: BBox, dt: f32) -> Self {
        let (cx, cy) = bbox.center();
        let (w, h) = (bbox.width(), bbox.height());
        let mut state = [0.0; DIM];
        state[0] = cx;
        state[1] = cy;
        state[2] = w;
        state[3] = h;

        // Position and size are trusted from the measurement, velocities are not.
        let mut covariance = [[0.0; DIM]; DIM];
        for i in 0..MEAS {
            covariance[i][i] = 10.0;
        }
        for i in MEAS..DIM {
            covariance[i][i] = 1000.0;
        }

        let mut filter = Self {
            state,
            covariance,
            // SORT-style noise: small on the measurement, larger on the rates.
            process_noise: [1.0, 1.0, 1.0, 1.0, 0.01, 0.01, 0.0001, 0.0001],
            measurement_noise: [1.0, 1.0, 1.0, 1.0],
        };
        filter.scale_covariance(dt.max(0.0));
        filter
    }

    /// Current best estimate as a box.
    pub fn bbox(&self) -> BBox {
        BBox::from_center(self.state[0], self.state[1], self.state[2], self.state[3])
    }

    /// Full state vector, used by the tests to check the learned velocity.
    #[cfg(test)]
    pub fn state(&self) -> &[f32; DIM] {
        &self.state
    }

    /// Advance the estimate by `dt` seconds using the motion model alone.
    pub fn predict(&mut self, dt: f32) {
        if dt <= 0.0 {
            return;
        }

        // x = F x
        for i in 0..MEAS {
            self.state[i] += self.state[MEAS + i] * dt;
        }

        // P = F P F^T + Q
        let mut fp = [[0.0f32; DIM]; DIM];
        for row in 0..DIM {
            for col in 0..DIM {
                fp[row][col] = self.covariance[row][col];
                if row < MEAS {
                    fp[row][col] += dt * self.covariance[MEAS + row][col];
                }
            }
        }
        for row in 0..DIM {
            for col in 0..DIM {
                let mut value = fp[row][col];
                if col < MEAS {
                    value += dt * fp[row][MEAS + col];
                }
                self.covariance[row][col] = value;
            }
        }
        for i in 0..DIM {
            self.covariance[i][i] += self.process_noise[i];
        }
    }

    /// Correct the estimate with an observed box.
    pub fn update(&mut self, bbox: BBox) {
        let (cx, cy) = bbox.center();
        let measurement = [cx, cy, bbox.width(), bbox.height()];

        // Innovation y = z - H x
        let mut innovation = [0.0f32; MEAS];
        for i in 0..MEAS {
            innovation[i] = measurement[i] - self.state[i];
        }

        // S = H P H^T + R, which is the top-left 4x4 block plus the measurement noise.
        let mut s = [[0.0f32; MEAS]; MEAS];
        for row in 0..MEAS {
            for col in 0..MEAS {
                s[row][col] = self.covariance[row][col];
            }
            s[row][row] += self.measurement_noise[row];
        }
        let Some(s_inv) = invert4(&s) else {
            return;
        };

        // K = P H^T S^-1, taking the first four columns of P.
        let mut gain = [[0.0f32; MEAS]; DIM];
        for row in 0..DIM {
            for col in 0..MEAS {
                let mut acc = 0.0;
                for k in 0..MEAS {
                    acc += self.covariance[row][k] * s_inv[k][col];
                }
                gain[row][col] = acc;
            }
        }

        // x = x + K y
        for row in 0..DIM {
            let mut delta = 0.0;
            for col in 0..MEAS {
                delta += gain[row][col] * innovation[col];
            }
            self.state[row] += delta;
        }

        // P = (I - K H) P
        let mut next = [[0.0f32; DIM]; DIM];
        for row in 0..DIM {
            for col in 0..DIM {
                let mut acc = self.covariance[row][col];
                for k in 0..MEAS {
                    acc -= gain[row][k] * self.covariance[k][col];
                }
                next[row][col] = acc;
            }
        }
        self.covariance = next;
    }

    fn scale_covariance(&mut self, dt: f32) {
        let factor = if dt > 0.0 { 1.0 + dt } else { 1.0 };
        for row in 0..DIM {
            for col in 0..DIM {
                self.covariance[row][col] *= factor;
            }
        }
    }
}

/// Gauss-Jordan inverse of a 4x4 matrix, or `None` when it is singular.
fn invert4(m: &[[f32; MEAS]; MEAS]) -> Option<[[f32; MEAS]; MEAS]> {
    let mut a = *m;
    let mut inv = [[0.0f32; MEAS]; MEAS];
    for (i, row) in inv.iter_mut().enumerate() {
        row[i] = 1.0;
    }

    for col in 0..MEAS {
        // Partial pivoting keeps the filter stable when a size is near zero.
        let mut pivot = col;
        for row in col + 1..MEAS {
            if a[row][col].abs() > a[pivot][col].abs() {
                pivot = row;
            }
        }
        if a[pivot][col].abs() < 1e-9 {
            return None;
        }
        a.swap(col, pivot);
        inv.swap(col, pivot);

        let scale = a[col][col];
        for k in 0..MEAS {
            a[col][k] /= scale;
            inv[col][k] /= scale;
        }

        for row in 0..MEAS {
            if row == col {
                continue;
            }
            let factor = a[row][col];
            if factor == 0.0 {
                continue;
            }
            for k in 0..MEAS {
                a[row][k] -= factor * a[col][k];
                inv[row][k] -= factor * inv[col][k];
            }
        }
    }
    Some(inv)
}

/// One tracked vehicle: a persistent identifier plus its filter.
#[derive(Debug, Clone)]
pub struct Track {
    pub id: u64,
    pub class_id: u32,
    pub class_name: &'static str,
    filter: KalmanBoxFilter,
    hits: u32,
    misses: u32,
    confirmed: bool,
}

impl Track {
    pub fn bbox(&self) -> BBox {
        self.filter.bbox()
    }

    #[cfg(test)]
    pub fn is_confirmed(&self) -> bool {
        self.confirmed
    }

    #[cfg(test)]
    pub fn hits(&self) -> u32 {
        self.hits
    }

    #[cfg(test)]
    pub fn misses(&self) -> u32 {
        self.misses
    }
}

/// Tunables for association and track lifetime.
#[derive(Debug, Clone, Copy)]
pub struct TrackerSettings {
    pub iou_threshold: f32,
    pub min_hits: u32,
    pub max_age: u32,
}

/// A track reported to the renderer and the speed estimator.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TrackedObject {
    pub id: u64,
    pub class_id: u32,
    pub class_name: &'static str,
    pub bbox: BBox,
}

/// Greedy IoU multi-object tracker with a Kalman filter per track.
///
/// Association sorts candidate (track, detection) pairs by IoU descending and
/// matches greedily, requiring the same vehicle class. Unmatched detections start
/// new tracks; unmatched tracks coast on their prediction until `max_age`
/// inference frames have passed without a match.
pub struct Tracker {
    settings: TrackerSettings,
    tracks: Vec<Track>,
    next_id: u64,
}

impl Tracker {
    pub fn new(settings: TrackerSettings) -> Self {
        Self {
            settings,
            tracks: Vec::new(),
            next_id: 1,
        }
    }

    /// Advance one frame. `detections` is `None` when inference was skipped, in
    /// which case every track coasts on its prediction.
    ///
    /// Returns the confirmed tracks in a stable order (by identifier).
    pub fn step(&mut self, detections: Option<&[Detection]>, dt: f32) -> Vec<TrackedObject> {
        for track in &mut self.tracks {
            track.filter.predict(dt);
        }

        if let Some(detections) = detections {
            let unmatched = self.associate(detections);

            for track in &mut self.tracks {
                if track.misses > self.settings.max_age {
                    track.confirmed = false;
                }
            }

            for index in unmatched {
                self.tracks.push(new_track(
                    self.next_id,
                    detections[index],
                    dt,
                    self.settings.min_hits,
                ));
                self.next_id += 1;
            }

            self.tracks.retain(|t| t.misses <= self.settings.max_age);
        }

        let mut reported: Vec<TrackedObject> = self
            .tracks
            .iter()
            .filter(|t| t.confirmed)
            .map(|t| TrackedObject {
                id: t.id,
                class_id: t.class_id,
                class_name: t.class_name,
                bbox: t.bbox(),
            })
            .collect();
        reported.sort_by_key(|t| t.id);
        reported
    }

    /// Identifiers of tracks that still exist, used to prune per-track state.
    pub fn live_ids(&self) -> Vec<u64> {
        self.tracks.iter().map(|t| t.id).collect()
    }

    /// Match detections to tracks, returning the indices of detections that found
    /// no track.
    fn associate(&mut self, detections: &[Detection]) -> Vec<usize> {
        let mut pairs: Vec<(f32, usize, usize)> = Vec::new();
        for (track_index, track) in self.tracks.iter().enumerate() {
            let predicted = track.bbox();
            for (detection_index, detection) in detections.iter().enumerate() {
                if detection.class_id != track.class_id {
                    continue;
                }
                let score = iou(&predicted, &detection.bbox);
                if score >= self.settings.iou_threshold {
                    pairs.push((score, track_index, detection_index));
                }
            }
        }

        pairs.sort_by(|a, b| b.0.total_cmp(&a.0));

        let mut track_taken = vec![false; self.tracks.len()];
        let mut detection_taken = vec![false; detections.len()];

        for (_, track_index, detection_index) in pairs {
            if track_taken[track_index] || detection_taken[detection_index] {
                continue;
            }
            track_taken[track_index] = true;
            detection_taken[detection_index] = true;

            let track = &mut self.tracks[track_index];
            track.filter.update(detections[detection_index].bbox);
            track.hits += 1;
            track.misses = 0;
            if track.hits >= self.settings.min_hits {
                track.confirmed = true;
            }
        }

        for (index, track) in self.tracks.iter_mut().enumerate() {
            if !track_taken[index] {
                track.misses += 1;
            }
        }

        detection_taken
            .iter()
            .enumerate()
            .filter_map(|(index, taken)| (!taken).then_some(index))
            .collect()
    }
}

fn new_track(id: u64, detection: Detection, dt: f32, min_hits: u32) -> Track {
    Track {
        id,
        class_id: detection.class_id,
        class_name: detection.class_name,
        filter: KalmanBoxFilter::new(detection.bbox, dt),
        hits: 1,
        misses: 0,
        confirmed: min_hits <= 1,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::detector::Detection;

    const DT: f32 = 0.04; // 25 fps

    fn settings() -> TrackerSettings {
        TrackerSettings {
            iou_threshold: 0.3,
            min_hits: 3,
            max_age: 5,
        }
    }

    fn detection(class_id: u32, bbox: BBox) -> Detection {
        Detection {
            class_id,
            class_name: "car",
            score: 0.9,
            bbox,
        }
    }

    fn moving(x: f32, y: f32) -> BBox {
        BBox::new(x, y, x + 40.0, y + 30.0)
    }

    #[test]
    fn identity_survives_consecutive_frames() {
        let mut tracker = Tracker::new(settings());
        let mut ids = Vec::new();
        for frame in 0..6 {
            let x = 100.0 + frame as f32 * 4.0;
            let reported = tracker.step(Some(&[detection(2, moving(x, 200.0))]), DT);
            if let Some(first) = reported.first() {
                ids.push(first.id);
            }
        }
        assert!(!ids.is_empty());
        assert!(ids.iter().all(|id| *id == ids[0]), "{ids:?}");
    }

    #[test]
    fn tentative_tracks_are_not_reported() {
        let mut tracker = Tracker::new(settings());
        assert!(tracker.step(Some(&[detection(2, moving(100.0, 200.0))]), DT).is_empty());
        assert!(tracker.step(Some(&[detection(2, moving(104.0, 200.0))]), DT).is_empty());
        let reported = tracker.step(Some(&[detection(2, moving(108.0, 200.0))]), DT);
        assert_eq!(reported.len(), 1);
        assert_eq!(reported[0].class_name, "car");
    }

    #[test]
    fn new_vehicles_get_distinct_identifiers() {
        let mut tracker = Tracker::new(settings());
        for frame in 0..4 {
            let x = frame as f32 * 4.0;
            tracker.step(
                Some(&[
                    detection(2, moving(100.0 + x, 200.0)),
                    detection(2, moving(600.0 + x, 400.0)),
                ]),
                DT,
            );
        }
        let mut ids = tracker.live_ids();
        ids.sort_unstable();
        assert_eq!(ids.len(), 2);
        assert_ne!(ids[0], ids[1]);
    }

    #[test]
    fn class_mismatch_prevents_association() {
        let mut tracker = Tracker::new(settings());
        tracker.step(Some(&[detection(2, moving(100.0, 200.0))]), DT);
        let first_id = tracker.live_ids()[0];

        // Same place, different class: a second track must appear.
        tracker.step(Some(&[detection(7, moving(100.0, 200.0))]), DT);
        let ids = tracker.live_ids();
        assert_eq!(ids.len(), 2);
        assert!(ids.contains(&first_id));
    }

    #[test]
    fn prediction_advances_tracks_on_skipped_frames() {
        let mut tracker = Tracker::new(TrackerSettings {
            min_hits: 1,
            ..settings()
        });
        // Build up velocity by matching a moving box.
        for frame in 0..6 {
            let x = 100.0 + frame as f32 * 10.0;
            tracker.step(Some(&[detection(2, moving(x, 200.0))]), DT);
        }
        let before = tracker.step(None, DT);
        assert_eq!(before.len(), 1);
        let after = tracker.step(None, DT);
        assert_eq!(after.len(), 1);
        assert!(
            after[0].bbox.x1 > before[0].bbox.x1,
            "prediction should move the box right: {:?} -> {:?}",
            before[0].bbox,
            after[0].bbox
        );
    }

    #[test]
    fn skipped_frames_do_not_count_as_misses() {
        let mut tracker = Tracker::new(TrackerSettings {
            min_hits: 1,
            max_age: 2,
            ..settings()
        });
        tracker.step(Some(&[detection(2, moving(100.0, 200.0))]), DT);
        for _ in 0..10 {
            tracker.step(None, DT);
        }
        assert_eq!(tracker.live_ids().len(), 1);
    }

    #[test]
    fn track_is_dropped_after_misses() {
        let mut tracker = Tracker::new(TrackerSettings {
            min_hits: 1,
            max_age: 2,
            ..settings()
        });
        tracker.step(Some(&[detection(2, moving(100.0, 200.0))]), DT);
        assert_eq!(tracker.live_ids().len(), 1);

        for _ in 0..3 {
            tracker.step(Some(&[]), DT);
        }
        assert!(tracker.live_ids().is_empty());
    }

    #[test]
    fn kalman_update_pulls_the_box_towards_the_detection() {
        let mut filter = KalmanBoxFilter::new(moving(100.0, 200.0), DT);
        filter.predict(DT);
        let predicted = filter.bbox();
        let observed = BBox::new(140.0, 200.0, 180.0, 230.0);
        filter.update(observed);
        let corrected = filter.bbox();
        let distance = |a: BBox, b: BBox| (a.center().0 - b.center().0).abs();
        assert!(
            distance(corrected, observed) < distance(predicted, observed),
            "corrected {corrected:?} should be closer to {observed:?} than {predicted:?}"
        );
    }

    #[test]
    fn kalman_learns_velocity_from_repeated_measurements() {
        let mut filter = KalmanBoxFilter::new(moving(100.0, 200.0), DT);
        for step in 1..20 {
            filter.predict(DT);
            filter.update(moving(100.0 + step as f32 * 10.0, 200.0));
        }
        // 10 px per 0.04 s is 250 px/s.
        assert!(
            (filter.state()[4] - 250.0).abs() < 60.0,
            "velocity estimate {} is far from 250 px/s",
            filter.state()[4]
        );
    }

    #[test]
    fn singular_covariance_is_ignored_rather_than_panicking() {
        let mut filter = KalmanBoxFilter::new(BBox::new(0.0, 0.0, 0.0, 0.0), DT);
        filter.predict(DT);
        filter.update(BBox::new(0.0, 0.0, 0.0, 0.0));
        assert!(filter.bbox().width().abs() < 1e-3);
    }

    #[test]
    fn invert4_round_trips_the_identity() {
        let mut identity = [[0.0f32; MEAS]; MEAS];
        for (i, row) in identity.iter_mut().enumerate() {
            row[i] = 1.0;
        }
        let inverse = invert4(&identity).unwrap();
        assert_eq!(inverse, identity);
    }
}
