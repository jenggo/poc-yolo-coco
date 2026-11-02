use std::path::Path;

use anyhow::{Context, Result, bail};
use serde::Deserialize;

/// Runtime settings loaded from `config.toml`.
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
pub struct Config {
    pub detector: DetectorConfig,
    pub tracker: TrackerConfig,
    pub calibration: CalibrationConfig,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct DetectorConfig {
    /// Path to the YOLOv8n ONNX weights.
    pub model: String,
    /// Minimum class score a candidate box must reach to be kept.
    pub confidence: f32,
    /// IoU above which two same-class boxes are considered duplicates.
    pub iou_threshold: f32,
    /// ONNX Runtime intra-op thread count. 0 means "let the runtime decide".
    pub intra_threads: usize,
    /// Execution provider: `cpu` or `cuda`.
    pub provider: String,
    /// Run inference on every Nth frame.
    pub skip: usize,
}

impl Default for DetectorConfig {
    fn default() -> Self {
        Self {
            model: "models/yolov8n.onnx".to_string(),
            confidence: 0.35,
            iou_threshold: 0.45,
            intra_threads: 0,
            provider: "cpu".to_string(),
            skip: 2,
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct TrackerConfig {
    /// Minimum IoU between a detection and a predicted track box to associate them.
    pub iou_threshold: f32,
    /// Matches required before a track is reported.
    pub min_hits: u32,
    /// Consecutive misses after which a track is dropped.
    pub max_age: u32,
}

impl Default for TrackerConfig {
    fn default() -> Self {
        Self {
            iou_threshold: 0.3,
            min_hits: 3,
            max_age: 10,
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct CalibrationConfig {
    /// `line` or `homography`.
    pub method: String,
    /// Camera viewpoint preset, used when the line positions are not given.
    pub orientation: Option<String>,
    /// Speed above this value is drawn in red.
    pub speed_limit_kmh: f32,
    /// Orientation of the calibration lines, which decides the coordinate they
    /// measure. `horizontal` lines run left to right and measure `y`; `vertical`
    /// lines run top to bottom and measure `x`.
    pub line_axis: Option<String>,
    /// First virtual line, as a position along the measured axis. A value in
    /// `0.0..=1.0` is a fraction of the frame extent; anything larger is an
    /// absolute pixel coordinate. `None` uses the orientation preset.
    pub line1: Option<f32>,
    /// Second virtual line, same units as `line1`.
    pub line2: Option<f32>,
    /// Real-world distance between the two lines, in meters (line method).
    pub distance_m: f32,
    /// Four pixel points, clockwise, top-left first (homography method).
    pub pixels: Vec<[f32; 2]>,
    /// Four road-plane points in meters, matching `pixels` (homography method).
    pub world_m: Vec<[f32; 2]>,
    /// EMA weight for new samples in the homography method, 0..1.
    pub smoothing: f32,
    /// Draw the calibration lines or polygon over the video.
    pub show_geometry: bool,
}

impl Default for CalibrationConfig {
    fn default() -> Self {
        Self {
            method: "line".to_string(),
            orientation: None,
            speed_limit_kmh: 50.0,
            line_axis: None,
            line1: None,
            line2: None,
            distance_m: 20.0,
            pixels: Vec::new(),
            world_m: Vec::new(),
            smoothing: 0.4,
            show_geometry: true,
        }
    }
}

/// Which way the calibration lines run, and therefore which coordinate a vehicle
/// crossing them is measured along.
///
/// A line only detects motion that crosses it, so the lines must always run
/// roughly perpendicular to the direction of travel in image space.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LineAxis {
    /// Lines run left to right; the measured coordinate is `y`.
    Horizontal,
    /// Lines run top to bottom; the measured coordinate is `x`.
    Vertical,
}

impl LineAxis {
    pub const ACCEPTED: [&'static str; 2] = ["horizontal", "vertical"];

    pub fn parse(value: &str) -> Result<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "horizontal" => Ok(Self::Horizontal),
            "vertical" => Ok(Self::Vertical),
            other => bail!(
                "calibration.line_axis must be one of {}, got \"{other}\"",
                Self::ACCEPTED.join(", ")
            ),
        }
    }

    /// Frame extent the line positions are measured against.
    pub fn extent(self, width: usize, height: usize) -> f32 {
        match self {
            Self::Horizontal => height as f32,
            Self::Vertical => width as f32,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::Horizontal => "horizontal",
            Self::Vertical => "vertical",
        }
    }
}

/// How the road crosses the frame, which decides where the default calibration
/// lines go.
///
/// The lines are horizontal in both cases, because a line has to be roughly
/// perpendicular to the direction of travel in image space. The difference is
/// where in the frame the traffic actually is, and whether perspective correction
/// matters.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Orientation {
    /// The road crosses the frame left to right; vehicles travel horizontally.
    Horizontal,
    /// The road recedes toward a horizon; vehicles travel up or down the frame.
    Vertical,
}

impl Orientation {
    pub const ACCEPTED: [&'static str; 2] = ["horizontal", "vertical"];

    pub fn parse(value: &str) -> Result<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "horizontal" => Ok(Self::Horizontal),
            "vertical" => Ok(Self::Vertical),
            other => bail!(
                "orientation must be one of {}, got \"{other}\"",
                Self::ACCEPTED.join(", ")
            ),
        }
    }

    /// Default line positions as fractions of the frame height.
    ///
    /// Both bracket the band where vehicle bottom edges actually sit. On a
    /// receding road that band is pushed toward the horizon, so the lines move up.
    pub fn line_fractions(self) -> (f32, f32) {
        match self {
            Self::Horizontal => (0.45, 0.80),
            Self::Vertical => (0.58, 0.85),
        }
    }

    /// Perspective error grows as the road recedes, so the plain line method is
    /// only a rough estimate on a vertical view.
    pub fn perspective_warning(self) -> Option<&'static str> {
        match self {
            Self::Horizontal => None,
            Self::Vertical => Some(
                "orientation = vertical: a receding road makes one pixel near the horizon worth \
                 many metres and one pixel near the camera worth centimetres, so line-crossing \
                 speeds are rough. Use method = \"homography\" for correct speeds on this view.",
            ),
        }
    }
}

/// Calibration with every value already resolved to absolute pixels for one
/// specific source resolution.
#[derive(Debug, Clone, PartialEq)]
pub struct ResolvedCalibration {
    pub method: CalibrationMethod,
    pub speed_limit_kmh: f32,
    pub line_axis: LineAxis,
    pub line1: f32,
    pub line2: f32,
    pub distance_m: f32,
    pub pixels: Vec<[f32; 2]>,
    pub world_m: Vec<[f32; 2]>,
    pub smoothing: f32,
    pub show_geometry: bool,
}

/// The calibration method selected in `config.toml`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CalibrationMethod {
    /// Two horizontal lines with a known real-world distance.
    Line,
    /// Four pixel/world correspondences projected through a homography.
    Homography,
}

impl CalibrationMethod {
    pub const ACCEPTED: [&'static str; 2] = ["line", "homography"];

    pub fn parse(value: &str) -> Result<Self> {
        match value.to_ascii_lowercase().as_str() {
            "line" => Ok(Self::Line),
            "homography" => Ok(Self::Homography),
            other => bail!(
                "calibration.method must be one of {}, got \"{other}\"",
                Self::ACCEPTED.join(", ")
            ),
        }
    }
}

impl Config {
    /// Read `path` and validate it. A missing file is an error that names the path.
    pub fn load(path: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("reading the configuration file {}", path.display()))?;
        let config: Config = toml::from_str(&text)
            .with_context(|| format!("parsing the configuration file {}", path.display()))?;
        config.validate()?;
        Ok(config)
    }

    /// Apply command line overrides, which always win over the file.
    pub fn apply_overrides(
        &mut self,
        model: Option<&Path>,
        skip: Option<usize>,
        provider: Option<&str>,
        orientation: Option<&str>,
        distance_m: Option<f32>,
    ) {
        if let Some(model) = model {
            self.detector.model = model.display().to_string();
        }
        if let Some(skip) = skip {
            self.detector.skip = skip;
        }
        if let Some(provider) = provider {
            self.detector.provider = provider.to_string();
        }
        if let Some(orientation) = orientation {
            self.calibration.orientation = Some(orientation.to_string());
        }
        if let Some(distance_m) = distance_m {
            self.calibration.distance_m = distance_m;
        }
    }

    /// Validate the calibration geometry against the frame size of the source
    /// that was actually opened.
    ///
    /// A configuration tuned for one camera is almost always wrong for another,
    /// and a line drawn outside the frame is invisible and silently produces no
    /// speeds. This check turns that into a startup error that names the source
    /// resolution and the offending value.
    /// Turn the calibration into absolute pixels for one specific frame size.
    ///
    /// Line positions may be fractions of the frame height, so the same
    /// configuration survives a resolution change on a fixed camera. Explicit
    /// absolute pixels are still honoured and validated against the frame.
    pub fn resolve_calibration(
        &self,
        width: usize,
        height: usize,
    ) -> Result<ResolvedCalibration> {
        let c = &self.calibration;
        if width == 0 || height == 0 {
            bail!("the source reports a {width}x{height} frame");
        }
        let (w, h) = (width as f32, height as f32);
        let method = CalibrationMethod::parse(&c.method)?;

        let orientation = match &c.orientation {
            Some(value) => Orientation::parse(value)?,
            None => Orientation::Horizontal,
        };
        let (default1, default2) = orientation.line_fractions();

        // Default to the axis that reads a preset viewpoint correctly: a road
        // running left to right is measured along x, a receding road along y.
        let line_axis = match &c.line_axis {
            Some(value) => LineAxis::parse(value)?,
            None => match orientation {
                Orientation::Horizontal => LineAxis::Vertical,
                Orientation::Vertical => LineAxis::Horizontal,
            },
        };
        let extent = line_axis.extent(width, height);

        let resolve_line = |name: &str, value: Option<f32>, fallback: f32| -> Result<f32> {
            let resolved = match value {
                // A fraction of the extent along the measured axis.
                Some(v) if (0.0..=1.0).contains(&v) => v * extent,
                Some(v) => v,
                None => fallback * extent,
            };
            if resolved <= 0.0 || resolved >= extent {
                bail!(
                    "calibration.{name} resolves to {resolved:.0}, outside the {width}x{height} frame \
                     along the {} axis (valid values are 1 to {max}); this configuration does not \
                     match the source",
                    line_axis.name(),
                    max = extent - 1.0
                );
            }
            Ok(resolved)
        };

        let line1 = resolve_line("line1", c.line1, default1)?;
        let line2 = resolve_line("line2", c.line2, default2)?;

        if method == CalibrationMethod::Line {
            if line1 == line2 {
                bail!(
                    "calibration.line1 and calibration.line2 both resolve to {line1:.0}; \
                     they must differ"
                );
            }
            if c.distance_m <= 0.0 {
                bail!(
                    "calibration.distance_m must be positive for the line method, got {}",
                    c.distance_m
                );
            }
        }

        if method == CalibrationMethod::Homography {
            for (i, point) in c.pixels.iter().enumerate() {
                let [x, y] = *point;
                if !(0.0..w).contains(&x) || !(0.0..h).contains(&y) {
                    bail!(
                        "calibration.pixels[{i}] = [{x}, {y}] lies outside the {width}x{height} \
                         frame; this configuration does not match the source"
                    );
                }
            }
        }

        Ok(ResolvedCalibration {
            method,
            speed_limit_kmh: c.speed_limit_kmh,
            line_axis,
            line1,
            line2,
            distance_m: c.distance_m,
            pixels: c.pixels.clone(),
            world_m: c.world_m.clone(),
            smoothing: c.smoothing,
            show_geometry: c.show_geometry,
        })
    }

    pub fn validate(&self) -> Result<()> {
        let d = &self.detector;
        if d.model.trim().is_empty() {
            bail!("detector.model must not be empty");
        }
        if !(0.0..=1.0).contains(&d.confidence) {
            bail!(
                "detector.confidence must be between 0.0 and 1.0, got {}",
                d.confidence
            );
        }
        if !(0.0..=1.0).contains(&d.iou_threshold) {
            bail!(
                "detector.iou_threshold must be between 0.0 and 1.0, got {}",
                d.iou_threshold
            );
        }
        if d.skip == 0 {
            bail!("detector.skip must be at least 1");
        }
        match d.provider.to_ascii_lowercase().as_str() {
            "cpu" | "cuda" => {}
            other => bail!(
                "detector.provider must be one of cpu, cuda, got \"{other}\""
            ),
        }

        let t = &self.tracker;
        if !(0.0..=1.0).contains(&t.iou_threshold) {
            bail!(
                "tracker.iou_threshold must be between 0.0 and 1.0, got {}",
                t.iou_threshold
            );
        }
        if t.min_hits == 0 {
            bail!("tracker.min_hits must be at least 1");
        }

        let c = &self.calibration;
        if c.speed_limit_kmh <= 0.0 {
            bail!(
                "calibration.speed_limit_kmh must be positive, got {}",
                c.speed_limit_kmh
            );
        }
        if let Some(orientation) = &c.orientation {
            Orientation::parse(orientation)?;
        }
        if let Some(axis) = &c.line_axis {
            LineAxis::parse(axis)?;
        }
        for (name, value) in [("line1", c.line1), ("line2", c.line2)] {
            if let Some(v) = value {
                if v <= 0.0 {
                    bail!("calibration.{name} must be positive or a fraction of the frame, got {v}");
                }
            }
        }

        match CalibrationMethod::parse(&c.method)? {
            CalibrationMethod::Line => {
                if let (Some(a), Some(b)) = (c.line1, c.line2)
                    && a == b
                {
                    bail!("calibration.line1 and calibration.line2 must differ, both are {a}");
                }
                if c.distance_m <= 0.0 {
                    bail!(
                        "calibration.distance_m must be positive for the line method, got {}",
                        c.distance_m
                    );
                }
            }
            CalibrationMethod::Homography => {
                if c.pixels.len() != 4 {
                    bail!(
                        "calibration.pixels must contain exactly 4 points for the homography method, got {}",
                        c.pixels.len()
                    );
                }
                if c.world_m.len() != 4 {
                    bail!(
                        "calibration.world_m must contain exactly 4 points for the homography method, got {}",
                        c.world_m.len()
                    );
                }
                if !(0.0..=1.0).contains(&c.smoothing) {
                    bail!(
                        "calibration.smoothing must be between 0.0 and 1.0, got {}",
                        c.smoothing
                    );
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_validate() {
        Config {
            detector: DetectorConfig::default(),
            tracker: TrackerConfig::default(),
            calibration: CalibrationConfig::default(),
        }
        .validate()
        .unwrap();
    }

    #[test]
    fn omitted_keys_take_defaults() {
        let config: Config = toml::from_str("[detector]\nconfidence = 0.5\n").unwrap();
        assert_eq!(config.detector.confidence, 0.5);
        assert_eq!(config.detector.skip, DetectorConfig::default().skip);
        assert_eq!(config.calibration.method, "line");
    }

    #[test]
    fn identical_lines_are_rejected() {
        let mut config = Config::default();
        config.calibration.line1 = Some(400.0);
        config.calibration.line2 = Some(400.0);
        let err = config.validate().unwrap_err().to_string();
        assert!(err.contains("line2"), "{err}");
    }

    #[test]
    fn homography_needs_eight_points() {
        let mut config = Config {
            detector: DetectorConfig::default(),
            tracker: TrackerConfig::default(),
            calibration: CalibrationConfig::default(),
        };
        config.calibration.method = "homography".to_string();
        config.calibration.pixels = vec![[0.0, 0.0]; 3];
        let err = config.validate().unwrap_err().to_string();
        assert!(err.contains("4 points"), "{err}");
    }

    #[test]
    fn unknown_method_is_rejected() {
        let mut config = Config {
            detector: DetectorConfig::default(),
            tracker: TrackerConfig::default(),
            calibration: CalibrationConfig::default(),
        };
        config.calibration.method = "radar".to_string();
        let err = config.validate().unwrap_err().to_string();
        assert!(err.contains("homography"), "{err}");
    }

    #[test]
    fn missing_file_names_the_path() {
        let err = Config::load(Path::new("/nonexistent/cctvpro.toml"))
            .unwrap_err()
            .to_string();
        assert!(err.contains("/nonexistent/cctvpro.toml"), "{err}");
    }

    #[test]
    fn absolute_lines_outside_the_frame_are_rejected() {
        let mut config = Config::default();
        // Absolute 1080p rows must be refused on a 720p frame, not silently ignored.
        config.calibration.line_axis = Some("horizontal".to_string());
        config.calibration.line1 = Some(640.0);
        config.calibration.line2 = Some(900.0);
        let err = config.resolve_calibration(1280, 720).unwrap_err().to_string();
        assert!(err.contains("line2"), "{err}");
        assert!(err.contains("1280x720"), "{err}");
        assert!(config.resolve_calibration(1920, 1080).is_ok());
    }

    #[test]
    fn fractional_lines_resolve_against_any_frame_size() {
        let mut config = Config::default();
        config.calibration.line_axis = Some("horizontal".to_string());
        config.calibration.line1 = Some(0.5);
        config.calibration.line2 = Some(0.8);

        let small = config.resolve_calibration(1280, 720).unwrap();
        assert!((small.line1 - 360.0).abs() < 1e-3, "{}", small.line1);
        assert!((small.line2 - 576.0).abs() < 1e-3, "{}", small.line2);

        let large = config.resolve_calibration(1920, 1080).unwrap();
        assert!((large.line1 - 540.0).abs() < 1e-3, "{}", large.line1);
        assert!((large.line2 - 864.0).abs() < 1e-3, "{}", large.line2);
    }

    #[test]
    fn orientation_supplies_line_positions_when_none_are_given() {
        let mut config = Config::default();
        config.calibration.orientation = Some("vertical".to_string());
        let resolved = config.resolve_calibration(1280, 720).unwrap();
        let (f1, f2) = Orientation::Vertical.line_fractions();
        // A receding road is measured along y, so the extent is the height.
        assert_eq!(resolved.line_axis, LineAxis::Horizontal);
        assert!((resolved.line1 - f1 * 720.0).abs() < 1e-3);
        assert!((resolved.line2 - f2 * 720.0).abs() < 1e-3);
    }

    #[test]
    fn explicit_lines_override_the_orientation_preset() {
        let mut config = Config::default();
        config.calibration.orientation = Some("vertical".to_string());
        config.calibration.line1 = Some(0.30);
        config.calibration.line2 = Some(0.60);
        let resolved = config.resolve_calibration(1280, 720).unwrap();
        assert!((resolved.line1 - 216.0).abs() < 1e-3);
        assert!((resolved.line2 - 432.0).abs() < 1e-3);
    }

    #[test]
    fn vertical_line_axis_measures_against_the_width() {
        let mut config = Config::default();
        config.calibration.line_axis = Some("vertical".to_string());
        config.calibration.line1 = Some(0.25);
        config.calibration.line2 = Some(0.75);
        let resolved = config.resolve_calibration(1280, 720).unwrap();
        assert_eq!(resolved.line_axis, LineAxis::Vertical);
        assert!((resolved.line1 - 320.0).abs() < 1e-3, "{}", resolved.line1);
        assert!((resolved.line2 - 960.0).abs() < 1e-3, "{}", resolved.line2);
    }

    #[test]
    fn vertical_axis_rejects_positions_past_the_width() {
        let mut config = Config::default();
        config.calibration.line_axis = Some("vertical".to_string());
        config.calibration.line1 = Some(0.5);
        // Inside a 1080-high frame but past the 1280-wide one.
        config.calibration.line2 = Some(1500.0);
        let err = config.resolve_calibration(1280, 1080).unwrap_err().to_string();
        assert!(err.contains("vertical"), "{err}");
        assert!(err.contains("line2"), "{err}");
    }

    #[test]
    fn unknown_line_axis_is_rejected() {
        let mut config = Config::default();
        config.calibration.line_axis = Some("diagonal".to_string());
        let err = config.validate().unwrap_err().to_string();
        assert!(err.contains("line_axis"), "{err}");
    }

    #[test]
    fn orientation_picks_the_measuring_axis() {
        let mut config = Config::default();
        config.calibration.orientation = Some("horizontal".to_string());
        assert_eq!(
            config.resolve_calibration(1280, 720).unwrap().line_axis,
            LineAxis::Vertical
        );

        config.calibration.orientation = Some("vertical".to_string());
        assert_eq!(
            config.resolve_calibration(1280, 720).unwrap().line_axis,
            LineAxis::Horizontal
        );
    }

    #[test]
    fn unknown_orientation_is_rejected() {
        let mut config = Config::default();
        config.calibration.orientation = Some("diagonal".to_string());
        let err = config.validate().unwrap_err().to_string();
        assert!(err.contains("horizontal"), "{err}");
        assert!(err.contains("vertical"), "{err}");
    }

    #[test]
    fn homography_points_outside_the_frame_are_rejected() {
        let mut config = Config::default();
        config.calibration.method = "homography".to_string();
        config.calibration.pixels = vec![
            [100.0, 700.0],
            [600.0, 700.0],
            [700.0, 400.0],
            [3000.0, 400.0],
        ];
        config.calibration.world_m = vec![[0.0, 0.0]; 4];
        let err = config.resolve_calibration(1280, 720).unwrap_err().to_string();
        assert!(err.contains("pixels[3]"), "{err}");
        assert!(err.contains("1280x720"), "{err}");
    }
}
