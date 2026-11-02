use anyhow::{Context, Result, anyhow, bail};
use ndarray::Array4;
use ort::session::Session;
use ort::session::builder::GraphOptimizationLevel;
use ort::value::TensorRef;
use tracing::{debug, info};

use crate::config::DetectorConfig;
use crate::decode::Frame;
use crate::geometry::{BBox, iou};

/// Model input edge length.
pub const INPUT_SIZE: usize = 640;
/// YOLOv8 letterbox fill value, in 0..255 units.
pub const PAD_VALUE: f32 = 114.0;
/// Channels per candidate: `cx, cy, w, h` then 80 class scores.
const CHANNELS: usize = 84;
/// Candidate boxes YOLOv8 emits per frame.
const CANDIDATES: usize = 8400;

/// COCO class indices kept as vehicles, with their display names.
pub const VEHICLE_CLASSES: [(u32, &str); 5] = [
    (1, "bicycle"),
    (2, "car"),
    (3, "motorcycle"),
    (5, "bus"),
    (7, "truck"),
];

/// Geometry of the resize-and-pad step, invertible so boxes map back to source pixels.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Letterbox {
    pub scale: f32,
    pub pad_x: f32,
    pub pad_y: f32,
    pub new_w: f32,
    pub new_h: f32,
}

/// Fit `src` into a `dst`x`dst` square without distorting the aspect ratio.
pub fn letterbox_params(src_w: u32, src_h: u32, dst: u32) -> Letterbox {
    assert!(src_w > 0 && src_h > 0 && dst > 0, "letterbox needs non-zero sizes");
    let dst = dst as f32;
    let (src_w, src_h) = (src_w as f32, src_h as f32);
    let scale = (dst / src_w).min(dst / src_h);
    let new_w = src_w * scale;
    let new_h = src_h * scale;
    Letterbox {
        scale,
        pad_x: (dst - new_w) / 2.0,
        pad_y: (dst - new_h) / 2.0,
        new_w,
        new_h,
    }
}

impl Letterbox {
    /// Map a point from model space back to source pixels.
    pub fn unproject(&self, x: f32, y: f32) -> (f32, f32) {
        ((x - self.pad_x) / self.scale, (y - self.pad_y) / self.scale)
    }

    /// Map a box from model space back to source pixels, clipped to the frame.
    pub fn unproject_box(&self, cx: f32, cy: f32, w: f32, h: f32, src_w: f32, src_h: f32) -> Option<BBox> {
        let (x1, y1) = self.unproject(cx - w / 2.0, cy - h / 2.0);
        let (x2, y2) = self.unproject(cx + w / 2.0, cy + h / 2.0);
        BBox::new(x1, y1, x2, y2).clipped(src_w, src_h)
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Detection {
    pub class_id: u32,
    pub class_name: &'static str,
    pub score: f32,
    pub bbox: BBox,
}

/// Greedy per-class non-maximum suppression, highest score first.
pub fn nms(mut detections: Vec<Detection>, iou_threshold: f32) -> Vec<Detection> {
    detections.sort_by(|a, b| b.score.total_cmp(&a.score));
    let mut kept: Vec<Detection> = Vec::with_capacity(detections.len());
    for candidate in detections {
        let suppressed = kept
            .iter()
            .any(|k| k.class_id == candidate.class_id && iou(&k.bbox, &candidate.bbox) > iou_threshold);
        if !suppressed {
            kept.push(candidate);
        }
    }
    kept
}

/// Which axis of the model output carries the 84 channels.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutputLayout {
    /// `[1, 84, 8400]`
    ChannelsFirst,
    /// `[1, 8400, 84]`
    ChannelsLast,
}

impl OutputLayout {
    fn from_dims(dims: &[i64]) -> Result<Self> {
        if dims.len() != 3 {
            bail!("expected a 3-dimensional model output, got shape {dims:?}");
        }
        if dims[0] != 1 {
            bail!("expected a batch size of 1 in the model output, got shape {dims:?}");
        }
        match (dims[1], dims[2]) {
            (c, n) if c == CHANNELS as i64 && n == CANDIDATES as i64 => Ok(Self::ChannelsFirst),
            (n, c) if n == CANDIDATES as i64 && c == CHANNELS as i64 => Ok(Self::ChannelsLast),
            _ => bail!(
                "unsupported model output shape {dims:?}; expected \
                 [1, {CHANNELS}, {CANDIDATES}] or [1, {CANDIDATES}, {CHANNELS}]"
            ),
        }
    }
}

/// Reusable letterbox-to-tensor conversion. The destination index maps are only
/// rebuilt when the source frame size changes, so steady-state frames cost one
/// buffer fill plus the sampling copy.
pub struct Preprocessor {
    letterbox: Letterbox,
    src_w: usize,
    src_h: usize,
    /// Source column per destination column, or -1 for padding.
    src_x: Vec<i32>,
    /// Source row per destination row, or -1 for padding.
    src_y: Vec<i32>,
    tensor: Array4<f32>,
}

impl Preprocessor {
    pub fn new() -> Self {
        Self {
            letterbox: letterbox_params(1, 1, INPUT_SIZE as u32),
            src_w: 0,
            src_h: 0,
            src_x: vec![-1; INPUT_SIZE],
            src_y: vec![-1; INPUT_SIZE],
            tensor: Array4::from_elem((1, 3, INPUT_SIZE, INPUT_SIZE), PAD_VALUE / 255.0),
        }
    }

    pub fn letterbox(&self) -> &Letterbox {
        &self.letterbox
    }

    /// Letterbox `frame` into a `[1, 3, 640, 640]` channel-first tensor.
    pub fn run(&mut self, frame: &Frame) -> &Array4<f32> {
        if self.src_w != frame.width || self.src_h != frame.height {
            self.rebuild_maps(frame.width, frame.height);
        }

        self.tensor.fill(PAD_VALUE / 255.0);

        let stride = frame.width * 3;
        let rgb = &frame.rgb;
        for (dy, &sy) in self.src_y.iter().enumerate() {
            if sy < 0 {
                continue;
            }
            let row = sy as usize * stride;
            for (dx, &sx) in self.src_x.iter().enumerate() {
                if sx < 0 {
                    continue;
                }
                let px = row + sx as usize * 3;
                self.tensor[[0, 0, dy, dx]] = rgb[px] as f32 / 255.0;
                self.tensor[[0, 1, dy, dx]] = rgb[px + 1] as f32 / 255.0;
                self.tensor[[0, 2, dy, dx]] = rgb[px + 2] as f32 / 255.0;
            }
        }
        &self.tensor
    }

    fn rebuild_maps(&mut self, src_w: usize, src_h: usize) {
        let letterbox = letterbox_params(src_w as u32, src_h as u32, INPUT_SIZE as u32);

        for (dst, slot) in self.src_x.iter_mut().enumerate() {
            let sx = ((dst as f32 - letterbox.pad_x) / letterbox.scale).floor();
            *slot = if sx >= 0.0 && (sx as usize) < src_w {
                sx as i32
            } else {
                -1
            };
        }
        for (dst, slot) in self.src_y.iter_mut().enumerate() {
            let sy = ((dst as f32 - letterbox.pad_y) / letterbox.scale).floor();
            *slot = if sy >= 0.0 && (sy as usize) < src_h {
                sy as i32
            } else {
                -1
            };
        }

        self.letterbox = letterbox;
        self.src_w = src_w;
        self.src_h = src_h;
    }
}

impl Default for Preprocessor {
    fn default() -> Self {
        Self::new()
    }
}

pub struct Detector {
    session: Session,
    input_name: String,
    layout: OutputLayout,
    preprocessor: Preprocessor,
    last_inference_ms: f32,
    confidence: f32,
    iou_threshold: f32,
}

impl Detector {
    pub fn load(config: &DetectorConfig) -> Result<Self> {
        let environment = build_environment(&config.provider)?;

        let mut builder = Session::builder(&environment)?;
        if config.intra_threads > 0 {
            builder = builder
                .with_intra_threads(config.intra_threads)
                .map_err(|e| anyhow!("setting detector.intra_threads: {e}"))?;
        }
        let session = builder
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| anyhow!("setting the graph optimization level: {e}"))?
            .commit_from_file(&config.model)
            .with_context(|| {
                format!(
                    "loading the ONNX model {} (fetch it with scripts/fetch-model.sh)",
                    config.model
                )
            })?;

        let input_name = session
            .inputs()
            .first()
            .map(|i| i.name().to_string())
            .ok_or_else(|| anyhow!("model {} declares no inputs", config.model))?;

        let layout = declared_layout(&session, &config.model)?;

        info!(
            model = config.model,
            provider = config.provider,
            ?layout,
            "loaded detector"
        );

        Ok(Self {
            session,
            input_name,
            layout,
            preprocessor: Preprocessor::new(),
            last_inference_ms: 0.0,
            confidence: config.confidence,
            iou_threshold: config.iou_threshold,
        })
    }

    /// Elapsed milliseconds of the most recent inference call.
    pub fn last_inference_ms(&self) -> f32 {
        self.last_inference_ms
    }

    pub fn detect(&mut self, frame: &Frame) -> Result<Vec<Detection>> {
        let started = std::time::Instant::now();

        let detections = {
            let input = self.preprocessor.run(frame);
            let tensor = TensorRef::from_array_view(input.view())?;
            let outputs = self
                .session
                .run(ort::inputs![self.input_name.as_str() => tensor])?;
            let (shape, data) = outputs[0].try_extract_tensor::<f32>()?;
            let dims: Vec<i64> = shape.iter().copied().collect();

            // Dynamic shapes cannot be checked until the first run.
            if OutputLayout::from_dims(&dims)? != self.layout {
                bail!(
                    "model output shape {dims:?} does not match the declared layout {:?}",
                    self.layout
                );
            }

            postprocess(
                data,
                self.layout,
                self.preprocessor.letterbox(),
                frame.width as f32,
                frame.height as f32,
                self.confidence,
                self.iou_threshold,
            )?
        };

        self.last_inference_ms = started.elapsed().as_secs_f32() * 1000.0;
        debug!(
            frame = frame.index,
            detections = detections.len(),
            inference_ms = self.last_inference_ms,
            "inference"
        );
        Ok(detections)
    }
}

/// Build the process-wide ONNX Runtime environment for the configured provider.
pub fn build_environment(provider: &str) -> Result<ort::environment::Environment> {
    let environment = match provider.to_ascii_lowercase().as_str() {
        "cpu" => ort::init().with_name("cctvpro").build()?,
        "cuda" => {
            #[cfg(feature = "cuda")]
            {
                ort::init()
                    .with_name("cctvpro")
                    .with_execution_providers([ort::ep::CUDA::default().build()])
                    .build()?
            }
            #[cfg(not(feature = "cuda"))]
            {
                bail!(
                    "detector.provider = \"cuda\" needs a build with CUDA support; \
                     rebuild with `cargo build --release --features cuda`"
                )
            }
        }
        other => bail!("detector.provider must be one of cpu, cuda, got \"{other}\""),
    };
    environment.set_log_level(ort::logging::LogLevel::Warning);
    Ok(environment)
}

/// Read the output shape the model declares, failing startup when it is static and
/// unsupported.
fn declared_layout(session: &Session, model: &str) -> Result<OutputLayout> {
    let output = session
        .outputs()
        .first()
        .ok_or_else(|| anyhow!("model {model} declares no outputs"))?;
    let ort::value::ValueType::Tensor { shape, .. } = output.dtype() else {
        bail!("model {model} does not output a tensor");
    };
    let dims: Vec<i64> = shape.iter().copied().collect();
    if dims.iter().any(|d| *d < 0) {
        // Dynamic output: fall back to the layout YOLOv8 exports and let the
        // first inference confirm it.
        debug!(?dims, "model output shape is dynamic; assuming channels-first");
        return Ok(OutputLayout::ChannelsFirst);
    }
    OutputLayout::from_dims(&dims)
}

/// Confidence filter, vehicle-class filter, box decoding, letterbox inversion and
/// per-class NMS.
pub fn postprocess(
    data: &[f32],
    layout: OutputLayout,
    letterbox: &Letterbox,
    src_w: f32,
    src_h: f32,
    confidence: f32,
    iou_threshold: f32,
) -> Result<Vec<Detection>> {
    let expected = CHANNELS * CANDIDATES;
    if data.len() != expected {
        bail!(
            "model output holds {} values, expected {expected}",
            data.len()
        );
    }

    let detections = match layout {
        OutputLayout::ChannelsFirst => {
            decode(data, |candidate, channel| channel * CANDIDATES + candidate, letterbox, src_w, src_h, confidence)
        }
        OutputLayout::ChannelsLast => {
            decode(data, |candidate, channel| candidate * CHANNELS + channel, letterbox, src_w, src_h, confidence)
        }
    };

    Ok(nms(detections, iou_threshold))
}

/// Decode candidates through a layout-specific index function so the hot loop
/// stays monomorphic.
fn decode(
    data: &[f32],
    index: impl Fn(usize, usize) -> usize,
    letterbox: &Letterbox,
    src_w: f32,
    src_h: f32,
    confidence: f32,
) -> Vec<Detection> {
    let mut detections = Vec::new();
    for candidate in 0..CANDIDATES {
        let mut best = (0u32, 0.0f32, "car");
        for (class_id, name) in VEHICLE_CLASSES {
            // Channels 0..4 hold the box, so class scores start at channel 4.
            let score = data[index(candidate, 4 + class_id as usize)];
            if score > best.1 {
                best = (class_id, score, name);
            }
        }
        let (class_id, score, class_name) = best;
        if score < confidence {
            continue;
        }

        let cx = data[index(candidate, 0)];
        let cy = data[index(candidate, 1)];
        let w = data[index(candidate, 2)];
        let h = data[index(candidate, 3)];

        let Some(bbox) = letterbox.unproject_box(cx, cy, w, h, src_w, src_h) else {
            continue;
        };

        detections.push(Detection {
            class_id,
            class_name,
            score,
            bbox,
        });
    }
    detections
}

#[cfg(test)]
mod tests {
    use super::*;

    fn det(class_id: u32, score: f32, x1: f32, y1: f32, x2: f32, y2: f32) -> Detection {
        Detection {
            class_id,
            class_name: "car",
            score,
            bbox: BBox::new(x1, y1, x2, y2),
        }
    }

    fn frame(width: usize, height: usize, value: u8) -> Frame {
        Frame {
            index: 0,
            width,
            height,
            rgb: vec![value; width * height * 3],
            fps: 25.0,
        }
    }

    /// Write one candidate into a channels-first buffer.
    fn write_candidate_first(data: &mut [f32], candidate: usize, cx: f32, cy: f32, w: f32, h: f32, class_id: u32, score: f32) {
        data[candidate] = cx;
        data[CANDIDATES + candidate] = cy;
        data[2 * CANDIDATES + candidate] = w;
        data[3 * CANDIDATES + candidate] = h;
        data[(4 + class_id as usize) * CANDIDATES + candidate] = score;
    }

    #[test]
    fn letterbox_square() {
        let lb = letterbox_params(640, 640, 640);
        assert!((lb.scale - 1.0).abs() < 1e-6);
        assert!(lb.pad_x.abs() < 1e-6 && lb.pad_y.abs() < 1e-6);
    }

    #[test]
    fn letterbox_wide_centers_vertically() {
        let lb = letterbox_params(1920, 1080, 640);
        assert!((lb.scale - 640.0 / 1920.0).abs() < 1e-6);
        assert!(lb.pad_x.abs() < 1e-4);
        assert!((lb.pad_y - (640.0 - 1080.0 * 640.0 / 1920.0) / 2.0).abs() < 1e-4);
        assert!(lb.new_w > lb.new_h);
    }

    #[test]
    fn letterbox_tall_centers_horizontally() {
        let lb = letterbox_params(1080, 1920, 640);
        assert!((lb.scale - 640.0 / 1920.0).abs() < 1e-6);
        assert!(lb.pad_y.abs() < 1e-4);
        assert!(lb.pad_x > 0.0);
        assert!(lb.new_h > lb.new_w);
    }

    #[test]
    fn letterbox_round_trip_maps_corners_back() {
        let lb = letterbox_params(1920, 1080, 640);
        for (sx, sy) in [(0.0, 0.0), (1920.0, 1080.0), (960.0, 540.0), (100.0, 900.0)] {
            let (bx, by) = lb.unproject(sx * lb.scale + lb.pad_x, sy * lb.scale + lb.pad_y);
            assert!((bx - sx).abs() < 1e-3, "{bx} != {sx}");
            assert!((by - sy).abs() < 1e-3, "{by} != {sy}");
        }
    }

    #[test]
    fn preprocess_shape_padding_and_range() {
        let mut preprocessor = Preprocessor::new();
        let tensor = preprocessor.run(&frame(4, 2, 128)).clone();
        assert_eq!(tensor.shape(), &[1, 3, INPUT_SIZE, INPUT_SIZE]);
        assert!(tensor.iter().all(|v| (0.0..=1.0).contains(v)));
        assert!((tensor[[0, 0, 0, INPUT_SIZE / 2]] - PAD_VALUE / 255.0).abs() < 1e-6);
        assert!((tensor[[0, 0, INPUT_SIZE / 2, INPUT_SIZE / 2]] - 128.0 / 255.0).abs() < 1e-6);
    }

    #[test]
    fn preprocess_rebuilds_maps_when_frame_size_changes() {
        let mut preprocessor = Preprocessor::new();
        preprocessor.run(&frame(1920, 1080, 10));
        let wide = *preprocessor.letterbox();
        preprocessor.run(&frame(640, 640, 10));
        let square = *preprocessor.letterbox();
        assert!(wide.pad_y > 0.0);
        assert!(square.pad_y.abs() < 1e-6);
    }

    #[test]
    fn iou_disjoint_identical_and_half_overlap() {
        let a = det(2, 0.9, 0.0, 0.0, 10.0, 10.0);
        assert_eq!(iou(&a.bbox, &det(2, 0.9, 20.0, 20.0, 30.0, 30.0).bbox), 0.0);
        assert!((iou(&a.bbox, &det(2, 0.9, 0.0, 0.0, 10.0, 10.0).bbox) - 1.0).abs() < 1e-6);
        let half = det(2, 0.9, 5.0, 0.0, 15.0, 10.0);
        assert!((iou(&a.bbox, &half.bbox) - 50.0 / 150.0).abs() < 1e-6);
    }

    #[test]
    fn nms_collapses_same_class_and_keeps_other_classes() {
        let a = det(2, 0.9, 0.0, 0.0, 10.0, 10.0);
        let b = det(2, 0.8, 1.0, 1.0, 11.0, 11.0);
        let kept = nms(vec![a, b], 0.3);
        assert_eq!(kept.len(), 1);
        assert!((kept[0].score - 0.9).abs() < 1e-6);

        let truck = det(7, 0.7, 1.0, 1.0, 11.0, 11.0);
        assert_eq!(nms(vec![a, truck], 0.3).len(), 2);
    }

    #[test]
    fn postprocess_maps_boxes_back_to_source_coordinates() {
        let letterbox = letterbox_params(1280, 720, 640);
        let mut data = vec![0.0f32; CHANNELS * CANDIDATES];
        write_candidate_first(&mut data, 0, 320.0, 320.0, 64.0, 64.0, 2, 0.9);

        let dets = postprocess(&data, OutputLayout::ChannelsFirst, &letterbox, 1280.0, 720.0, 0.25, 0.45)
            .unwrap();
        assert_eq!(dets.len(), 1);
        let d = dets[0];
        assert_eq!(d.class_id, 2);
        let (cx, cy) = d.bbox.center();
        assert!((cx - 640.0).abs() < 1.0, "{cx}");
        assert!((cy - 360.0).abs() < 1.0, "{cy}");
        assert!(d.bbox.x1 >= 0.0 && d.bbox.x2 <= 1280.0 && d.bbox.y1 >= 0.0 && d.bbox.y2 <= 720.0);
    }

    #[test]
    fn postprocess_handles_transposed_layout() {
        let letterbox = letterbox_params(640, 640, 640);
        let mut data = vec![0.0f32; CHANNELS * CANDIDATES];
        // Candidate 0, channels last.
        data[0] = 320.0;
        data[1] = 320.0;
        data[2] = 64.0;
        data[3] = 64.0;
        data[4 + 5] = 0.8; // bus

        let dets = postprocess(&data, OutputLayout::ChannelsLast, &letterbox, 640.0, 640.0, 0.25, 0.45)
            .unwrap();
        assert_eq!(dets.len(), 1);
        assert_eq!(dets[0].class_id, 5);
        assert_eq!(dets[0].class_name, "bus");
    }

    #[test]
    fn postprocess_drops_non_vehicle_and_low_confidence() {
        let letterbox = letterbox_params(640, 640, 640);
        let mut data = vec![0.0f32; CHANNELS * CANDIDATES];
        // Candidate 0: person (class 0) with a high score.
        write_candidate_first(&mut data, 0, 320.0, 320.0, 64.0, 64.0, 0, 0.95);
        // Candidate 1: car below the threshold.
        write_candidate_first(&mut data, 1, 100.0, 100.0, 40.0, 40.0, 2, 0.10);

        let dets = postprocess(&data, OutputLayout::ChannelsFirst, &letterbox, 640.0, 640.0, 0.25, 0.45)
            .unwrap();
        assert!(dets.is_empty(), "{dets:?}");
    }

    #[test]
    fn postprocess_keeps_all_vehicle_classes() {
        let letterbox = letterbox_params(640, 640, 640);
        let mut data = vec![0.0f32; CHANNELS * CANDIDATES];
        for (i, (class_id, _)) in VEHICLE_CLASSES.iter().enumerate() {
            let x = 60.0 + i as f32 * 120.0;
            write_candidate_first(&mut data, i, x, 320.0, 50.0, 50.0, *class_id, 0.9);
        }

        let dets = postprocess(&data, OutputLayout::ChannelsFirst, &letterbox, 640.0, 640.0, 0.25, 0.45)
            .unwrap();
        assert_eq!(dets.len(), VEHICLE_CLASSES.len());
    }

    #[test]
    fn unsupported_output_shape_is_rejected_with_the_observed_shape() {
        let err = OutputLayout::from_dims(&[1, 25200, 6]).unwrap_err().to_string();
        assert!(err.contains("[1, 25200, 6]"), "{err}");
    }

    #[test]
    fn batch_size_other_than_one_is_rejected() {
        let err = OutputLayout::from_dims(&[2, 84, 8400]).unwrap_err().to_string();
        assert!(err.contains("batch size"), "{err}");
    }
}
