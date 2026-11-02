use std::time::Duration;

use anyhow::{Context, Result, anyhow, bail};
use ffmpeg::format::Pixel;
use ffmpeg::media::Type;
use ffmpeg::software::scaling::{Context as ScalingContext, flag::Flags};
use ffmpeg::util::frame::video::Video;
use ffmpeg_next as ffmpeg;
use tracing::{info, warn};

use crate::threads::Shutdown;

/// One decoded video frame in tightly packed RGB24 plus its metadata.
#[derive(Clone)]
pub struct Frame {
    pub index: u64,
    pub width: usize,
    pub height: usize,
    pub rgb: Vec<u8>,
    pub fps: f32,
}

/// Opens one ffmpeg input and yields RGB frames. Local files, HLS playlists and
/// RTSP streams all go through the same path; there is no per-format branching.
pub struct Decoder {
    input: ffmpeg::format::context::Input,
    decoder: ffmpeg::decoder::Video,
    scaler: ScalingContext,
    stream_index: usize,
    fps: f32,
    eof_sent: bool,
}

impl Decoder {
    pub fn open(source: &str) -> Result<Self> {
        let input = ffmpeg::format::input(&source)
            .with_context(|| format!("opening the video source {source}"))?;

        let stream = input
            .streams()
            .best(Type::Video)
            .ok_or_else(|| anyhow!("no video stream found in {source}"))?;
        let stream_index = stream.index();
        let fps = resolve_fps(stream.avg_frame_rate(), stream.rate());

        let decoder = ffmpeg::codec::context::Context::from_parameters(stream.parameters())
            .with_context(|| format!("creating a decoder for {source}"))?
            .decoder()
            .video()
            .with_context(|| format!("opening the video decoder for {source}"))?;

        let scaler = ScalingContext::get(
            decoder.format(),
            decoder.width(),
            decoder.height(),
            Pixel::RGB24,
            decoder.width(),
            decoder.height(),
            Flags::BILINEAR,
        )
        .with_context(|| format!("creating the RGB scaler for {source}"))?;

        Ok(Self {
            input,
            decoder,
            scaler,
            stream_index,
            fps,
            eof_sent: false,
        })
    }

    pub fn fps(&self) -> f32 {
        self.fps
    }

    pub fn width(&self) -> usize {
        self.decoder.width() as usize
    }

    pub fn height(&self) -> usize {
        self.decoder.height() as usize
    }

    /// Return the next decoded frame, or `None` at end of stream.
    pub fn next_frame(&mut self) -> Result<Option<Frame>> {
        loop {
            let mut decoded = Video::empty();
            match self.decoder.receive_frame(&mut decoded) {
                Ok(()) => {
                    let mut rgb = Video::empty();
                    self.scaler
                        .run(&decoded, &mut rgb)
                        .context("converting a frame to RGB24")?;
                    return Ok(Some(self.pack(&rgb)?));
                }
                Err(ffmpeg::Error::Other { errno }) if errno == ffmpeg::util::error::EAGAIN => {}
                Err(ffmpeg::Error::Eof) => return Ok(None),
                Err(err) => bail!("decoding a frame: {err}"),
            }

            if self.eof_sent {
                return Ok(None);
            }

            match self.next_video_packet()? {
                Some(packet) => self.decoder.send_packet(&packet).context("feeding a packet")?,
                None => {
                    self.decoder.send_eof().context("flushing the decoder")?;
                    self.eof_sent = true;
                }
            }
        }
    }

    fn next_video_packet(&mut self) -> Result<Option<ffmpeg::Packet>> {
        let index = self.stream_index;
        for (stream, packet) in self.input.packets() {
            if stream.index() == index {
                return Ok(Some(packet));
            }
        }
        Ok(None)
    }

    /// Copy the scaler output row by row so stride padding is not leaked into the
    /// tightly packed buffer.
    fn pack(&self, rgb: &Video) -> Result<Frame> {
        let width = rgb.width() as usize;
        let height = rgb.height() as usize;
        let stride = rgb.stride(0);
        let row_len = width * 3;

        if stride < row_len {
            bail!("decoded frame stride {stride} is smaller than the row length {row_len}");
        }

        let mut out = Vec::with_capacity(row_len * height);
        let data = rgb.data(0);
        for row in 0..height {
            let start = row * stride;
            out.extend_from_slice(&data[start..start + row_len]);
        }

        Ok(Frame {
            index: 0,
            width,
            height,
            rgb: out,
            fps: self.fps,
        })
    }
}

/// True for sources that can drop and be reopened: RTSP, RTMP and HTTP(S).
pub fn is_network_source(source: &str) -> bool {
    let lower = source.trim().to_ascii_lowercase();
    ["rtsp://", "rtsps://", "rtmp://", "http://", "https://"]
        .iter()
        .any(|scheme| lower.starts_with(scheme))
}

/// Delay before the next reconnect attempt: 1, 2, 4, 8, then capped at 10 s.
pub fn backoff_delay(attempt: u32) -> Duration {
    let seconds = if attempt >= 4 { 10 } else { 1u64 << attempt };
    Duration::from_secs(seconds)
}

/// A video source that reopens itself after a network drop.
///
/// The frame index is carried across reconnects so downstream timing (and
/// therefore speed) never resets.
pub struct VideoSource {
    source: String,
    decoder: Decoder,
    next_index: u64,
    network: bool,
    attempt: u32,
}

impl VideoSource {
    pub fn open(source: &str) -> Result<Self> {
        let decoder = Decoder::open(source)?;
        Ok(Self {
            source: source.to_string(),
            decoder,
            next_index: 0,
            network: is_network_source(source),
            attempt: 0,
        })
    }

    pub fn fps(&self) -> f32 {
        self.decoder.fps()
    }

    pub fn width(&self) -> usize {
        self.decoder.width()
    }

    pub fn height(&self) -> usize {
        self.decoder.height()
    }

    /// Next frame, reconnecting a network source after a failure.
    ///
    /// Returns `Ok(None)` at the end of a local file, or once `shutdown` is set.
    pub fn next_frame(&mut self, shutdown: &Shutdown) -> Result<Option<Frame>> {
        loop {
            if shutdown.is_triggered() {
                return Ok(None);
            }

            match self.decoder.next_frame() {
                Ok(Some(mut frame)) => {
                    frame.index = self.next_index;
                    self.next_index += 1;
                    self.attempt = 0;
                    return Ok(Some(frame));
                }
                Ok(None) => {
                    if !self.network {
                        return Ok(None);
                    }
                    // A network stream that ends is treated like a drop.
                    self.reconnect(shutdown)?;
                }
                Err(err) => {
                    if !self.network {
                        return Err(err);
                    }
                    warn!(source = self.source, error = %err, "stream read failed");
                    self.reconnect(shutdown)?;
                }
            }
        }
    }

    /// Sleep out the backoff, then reopen the source.
    fn reconnect(&mut self, shutdown: &Shutdown) -> Result<()> {
        let delay = backoff_delay(self.attempt);
        warn!(
            source = self.source,
            attempt = self.attempt + 1,
            delay_secs = delay.as_secs(),
            "reconnecting"
        );

        if !sleep_interruptible(delay, shutdown) {
            return Ok(());
        }

        match Decoder::open(&self.source) {
            Ok(decoder) => {
                info!(
                    source = self.source,
                    width = decoder.width(),
                    height = decoder.height(),
                    "reconnected"
                );
                self.decoder = decoder;
                self.attempt = 0;
            }
            Err(err) => {
                self.attempt = self.attempt.saturating_add(1);
                warn!(source = self.source, error = %err, "reconnect failed");
            }
        }
        Ok(())
    }
}

/// Sleep in short slices so a shutdown request is noticed promptly.
/// Returns `false` when shutdown was requested.
fn sleep_interruptible(total: Duration, shutdown: &Shutdown) -> bool {
    const SLICE: Duration = Duration::from_millis(50);
    let mut remaining = total;
    while remaining > Duration::ZERO {
        if shutdown.is_triggered() {
            return false;
        }
        let step = remaining.min(SLICE);
        std::thread::sleep(step);
        remaining -= step;
    }
    !shutdown.is_triggered()
}

/// `avg_frame_rate`, then `r_frame_rate`, then 25 fps.
fn resolve_fps(avg: ffmpeg::Rational, base: ffmpeg::Rational) -> f32 {
    for candidate in [avg, base] {
        if candidate.numerator() > 0 && candidate.denominator() > 0 {
            let fps = f64::from(candidate) as f32;
            if fps.is_finite() && fps > 0.0 {
                return fps;
            }
        }
    }
    25.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn average_rate_wins() {
        let fps = resolve_fps(ffmpeg::Rational::new(30, 1), ffmpeg::Rational::new(60, 1));
        assert!((fps - 30.0).abs() < 1e-3, "{fps}");
    }

    #[test]
    fn zero_average_falls_back_to_the_base_rate() {
        let fps = resolve_fps(ffmpeg::Rational::new(0, 0), ffmpeg::Rational::new(24, 1));
        assert!((fps - 24.0).abs() < 1e-3, "{fps}");
    }

    #[test]
    fn unusable_rates_fall_back_to_25() {
        let fps = resolve_fps(ffmpeg::Rational::new(0, 0), ffmpeg::Rational::new(0, 0));
        assert_eq!(fps, 25.0);
    }

    #[test]
    fn network_sources_are_recognised() {
        assert!(is_network_source("rtsp://cam/stream"));
        assert!(is_network_source("https://example.com/live.m3u8"));
        assert!(is_network_source("HTTP://example.com/live.m3u8"));
        assert!(!is_network_source("samples/traffic.webm"));
        assert!(!is_network_source("/tmp/clip.mp4"));
    }

    #[test]
    fn backoff_grows_then_caps_at_ten_seconds() {
        let delays: Vec<u64> = (0..7).map(|a| backoff_delay(a).as_secs()).collect();
        assert_eq!(delays, vec![1, 2, 4, 8, 10, 10, 10]);
    }

    #[test]
    fn interruptible_sleep_returns_immediately_on_shutdown() {
        let shutdown = Shutdown::new();
        shutdown.trigger();
        let started = std::time::Instant::now();
        assert!(!sleep_interruptible(Duration::from_secs(5), &shutdown));
        assert!(started.elapsed() < Duration::from_millis(200));
    }
}
