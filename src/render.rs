use anyhow::{Context, Result, bail};
use minifb::{Key, Window, WindowOptions};

pub type Color = u32;

pub const WHITE: Color = 0x00FF_FFFF;
pub const BLACK: Color = 0x0000_0000;
pub const RED: Color = 0x00FF_3B30;
pub const YELLOW: Color = 0x00FF_D60A;
pub const GREEN: Color = 0x0034_C759;
pub const CYAN: Color = 0x0032_D7FF;
pub const MAGENTA: Color = 0x00BF_5AF2;
pub const ORANGE: Color = 0x00FF_9F0A;

/// Colour used for the overlay plate behind text.
const PLATE: Color = 0x0020_2020;

/// Fixed colour per vehicle class, so a class always looks the same.
pub fn class_color(class_id: u32) -> Color {
    match class_id {
        1 => YELLOW,
        2 => GREEN,
        3 => MAGENTA,
        5 => ORANGE,
        7 => CYAN,
        _ => WHITE,
    }
}

/// A packed `0x00RRGGBB` framebuffer that is drawn on directly.
///
/// The buffer is allocated once and reused: resizing keeps the allocation and
/// `fill_from_rgb` overwrites it in place, so steady-state rendering does not
/// allocate.
pub struct Canvas {
    width: usize,
    height: usize,
    buf: Vec<u32>,
}

impl Canvas {
    pub fn new(width: usize, height: usize) -> Self {
        Self {
            width,
            height,
            buf: vec![BLACK; width * height],
        }
    }

    pub fn width(&self) -> usize {
        self.width
    }

    pub fn height(&self) -> usize {
        self.height
    }

    pub fn pixels(&self) -> &[u32] {
        &self.buf
    }

    /// Change the size, keeping the allocation. Contents are cleared to black.
    pub fn resize(&mut self, width: usize, height: usize) {
        if self.width == width && self.height == height {
            return;
        }
        self.width = width;
        self.height = height;
        self.buf.clear();
        self.buf.resize(width * height, BLACK);
    }

    /// Copy a tightly packed RGB24 frame into the framebuffer.
    pub fn fill_from_rgb(&mut self, width: usize, height: usize, rgb: &[u8]) -> Result<()> {
        let expected = width.checked_mul(height).and_then(|n| n.checked_mul(3));
        if expected != Some(rgb.len()) {
            bail!(
                "RGB buffer has {} bytes, expected {} for {width}x{height}",
                rgb.len(),
                expected.unwrap_or(0)
            );
        }
        self.resize(width, height);
        for (dst, px) in self.buf.iter_mut().zip(rgb.chunks_exact(3)) {
            *dst = ((px[0] as u32) << 16) | ((px[1] as u32) << 8) | px[2] as u32;
        }
        Ok(())
    }

    pub fn set(&mut self, x: i32, y: i32, color: Color) {
        if x < 0 || y < 0 {
            return;
        }
        let (x, y) = (x as usize, y as usize);
        if x < self.width && y < self.height {
            self.buf[y * self.width + x] = color;
        }
    }

    /// Draw a horizontal span, clipped to the canvas.
    pub fn hline(&mut self, x0: i32, x1: i32, y: i32, color: Color) {
        if y < 0 || y as usize >= self.height {
            return;
        }
        let start = x0.clamp(0, self.width as i32);
        let end = x1.clamp(0, self.width as i32);
        if start >= end {
            return;
        }
        let row = y as usize * self.width;
        self.buf[row + start as usize..row + end as usize].fill(color);
    }

    /// Draw a vertical span, clipped to the canvas.
    pub fn vline(&mut self, x: i32, y0: i32, y1: i32, color: Color) {
        if x < 0 || x as usize >= self.width {
            return;
        }
        let start = y0.clamp(0, self.height as i32);
        let end = y1.clamp(0, self.height as i32);
        if start >= end {
            return;
        }
        let x = x as usize;
        for y in start..end {
            self.buf[y as usize * self.width + x] = color;
        }
    }

    /// Draw a line between two points with Bresenham, clipped per pixel.
    pub fn line(&mut self, x0: i32, y0: i32, x1: i32, y1: i32, color: Color) {
        let dx = (x1 - x0).abs();
        let dy = -(y1 - y0).abs();
        let sx = if x0 < x1 { 1 } else { -1 };
        let sy = if y0 < y1 { 1 } else { -1 };
        let mut err = dx + dy;
        let (mut x, mut y) = (x0, y0);
        loop {
            self.set(x, y, color);
            if x == x1 && y == y1 {
                break;
            }
            let e2 = 2 * err;
            if e2 >= dy {
                err += dy;
                x += sx;
            }
            if e2 <= dx {
                err += dx;
                y += sy;
            }
        }
    }

    /// Outline a rectangle, clipped to the canvas.
    pub fn rect(&mut self, x: i32, y: i32, w: i32, h: i32, color: Color, thickness: i32) {
        if w <= 0 || h <= 0 || thickness <= 0 {
            return;
        }
        for t in 0..thickness {
            self.hline(x, x + w, y + t, color);
            self.hline(x, x + w, y + h - t, color);
            for yy in (y + t)..=(y + h - t) {
                self.set(x + t, yy, color);
                self.set(x + w - t, yy, color);
            }
        }
    }

    pub fn fill_rect(&mut self, x: i32, y: i32, w: i32, h: i32, color: Color) {
        if w <= 0 || h <= 0 {
            return;
        }
        for yy in y..y + h {
            self.hline(x, x + w, yy, color);
        }
    }

    /// Draw `text` with the built-in 5x7 font. Returns the advance width in pixels.
    pub fn text(&mut self, x: i32, y: i32, text: &str, color: Color, scale: i32) -> i32 {
        let mut cursor = x;
        for ch in text.chars() {
            if let Some(rows) = glyph(ch) {
                for (row, bits) in rows.iter().enumerate() {
                    for col in 0..5 {
                        if bits & (1 << (4 - col)) == 0 {
                            continue;
                        }
                        for dy in 0..scale {
                            for dx in 0..scale {
                                self.set(
                                    cursor + col * scale + dx,
                                    y + row as i32 * scale + dy,
                                    color,
                                );
                            }
                        }
                    }
                }
            }
            cursor += 6 * scale;
        }
        cursor - x
    }

    /// Draw `text` on a dark plate so it stays readable over any video content.
    pub fn label(&mut self, x: i32, y: i32, text: &str, color: Color, scale: i32) {
        let w = text.chars().count() as i32 * 6 * scale;
        self.fill_rect(x - 2, y - 2, w + 2, 7 * scale + 3, PLATE);
        self.text(x, y, text, color, scale);
    }

    /// Draw a stack of status lines in the top-left corner.
    pub fn status_overlay(&mut self, lines: &[String], scale: i32) {
        let widest = lines.iter().map(|l| l.chars().count()).max().unwrap_or(0) as i32;
        if widest == 0 {
            return;
        }
        let w = widest * 6 * scale + 4;
        let h = lines.len() as i32 * 9 * scale + 2;
        self.fill_rect(8, 8, w, h, PLATE);
        for (i, line) in lines.iter().enumerate() {
            self.text(10, 10 + i as i32 * 9 * scale, line, WHITE, scale);
        }
    }
}

/// Native window that blits a canvas. Pixels are only rescaled when the window
/// size differs from the canvas size.
pub struct Display {
    window: Window,
    scaled: Canvas,
}

impl Display {
    pub fn new(title: &str, width: usize, height: usize) -> Result<Self> {
        if width == 0 || height == 0 {
            bail!("cannot open a window for an empty {width}x{height} frame");
        }
        let window = Window::new(
            title,
            width,
            height,
            WindowOptions {
                resize: true,
                ..WindowOptions::default()
            },
        )
        .with_context(|| format!("creating a {width}x{height} window"))?;
        Ok(Self {
            window,
            scaled: Canvas::new(width, height),
        })
    }

    pub fn present(&mut self, canvas: &Canvas) -> Result<()> {
        let (win_w, win_h) = self.window.get_size();
        if win_w == 0 || win_h == 0 {
            return Ok(());
        }

        if win_w == canvas.width() && win_h == canvas.height() {
            self.window
                .update_with_buffer(canvas.pixels(), canvas.width(), canvas.height())
                .context("pushing a frame to the window")?;
            return Ok(());
        }

        scale_into(&mut self.scaled, canvas, win_w, win_h)?;
        self.window
            .update_with_buffer(self.scaled.pixels(), self.scaled.width(), self.scaled.height())
            .context("pushing a scaled frame to the window")?;
        Ok(())
    }

    /// True once the user pressed Escape, pressed `q`, or closed the window.
    pub fn should_quit(&self) -> bool {
        !self.window.is_open()
            || self.window.is_key_down(Key::Escape)
            || self.window.is_key_down(Key::Q)
    }

    pub fn is_open(&self) -> bool {
        self.window.is_open()
    }
}

/// Downscale `canvas` into `dst` in packed-u32 space.
///
/// This is the hot path whenever the window is smaller than the video, so it
/// resamples with bilinear taps directly on the framebuffer instead of converting
/// to RGB, allocating an image, and converting back.
fn scale_into(dst: &mut Canvas, canvas: &Canvas, width: usize, height: usize) -> Result<()> {
    let (src_w, src_h) = (canvas.width(), canvas.height());
    if src_w == 0 || src_h == 0 {
        bail!("cannot scale an empty {src_w}x{src_h} canvas");
    }
    dst.resize(width, height);
    if width == 0 || height == 0 {
        return Ok(());
    }

    let x_scale = src_w as f32 / width as f32;
    let y_scale = src_h as f32 / height as f32;

    // Source column and row per destination pixel, clamped to the last pixel.
    let columns: Vec<usize> = (0..width)
        .map(|x| (((x as f32 + 0.5) * x_scale) as usize).min(src_w - 1))
        .collect();
    let rows: Vec<usize> = (0..height)
        .map(|y| (((y as f32 + 0.5) * y_scale) as usize).min(src_h - 1))
        .collect();

    let src = canvas.pixels();
    for (dy, &sy) in rows.iter().enumerate() {
        let src_row = sy * src_w;
        let dst_row = dy * width;
        for (dx, &sx) in columns.iter().enumerate() {
            dst.buf[dst_row + dx] = src[src_row + sx];
        }
    }
    Ok(())
}

/// 5x7 glyphs, one byte per row, five low bits per row (bit 4 is the left column).
fn glyph(c: char) -> Option<[u8; 7]> {
    let rows = match c {
        ' ' => [0b00000, 0b00000, 0b00000, 0b00000, 0b00000, 0b00000, 0b00000],
        '0' => [0b01110, 0b10001, 0b10011, 0b10101, 0b11001, 0b10001, 0b01110],
        '1' => [0b00100, 0b01100, 0b00100, 0b00100, 0b00100, 0b00100, 0b01110],
        '2' => [0b01110, 0b10001, 0b00001, 0b00010, 0b00100, 0b01000, 0b11111],
        '3' => [0b11111, 0b00010, 0b00100, 0b00010, 0b00001, 0b10001, 0b01110],
        '4' => [0b00010, 0b00110, 0b01010, 0b10010, 0b11111, 0b00010, 0b00010],
        '5' => [0b11111, 0b10000, 0b11110, 0b00001, 0b00001, 0b10001, 0b01110],
        '6' => [0b00110, 0b01000, 0b10000, 0b11110, 0b10001, 0b10001, 0b01110],
        '7' => [0b11111, 0b00001, 0b00010, 0b00100, 0b01000, 0b01000, 0b01000],
        '8' => [0b01110, 0b10001, 0b10001, 0b01110, 0b10001, 0b10001, 0b01110],
        '9' => [0b01110, 0b10001, 0b10001, 0b01111, 0b00001, 0b00010, 0b01100],
        'A' => [0b01110, 0b10001, 0b10001, 0b11111, 0b10001, 0b10001, 0b10001],
        'B' => [0b11110, 0b10001, 0b10001, 0b11110, 0b10001, 0b10001, 0b11110],
        'C' => [0b01110, 0b10001, 0b10000, 0b10000, 0b10000, 0b10001, 0b01110],
        'D' => [0b11100, 0b10010, 0b10001, 0b10001, 0b10001, 0b10010, 0b11100],
        'E' => [0b11111, 0b10000, 0b10000, 0b11110, 0b10000, 0b10000, 0b11111],
        'F' => [0b11111, 0b10000, 0b10000, 0b11110, 0b10000, 0b10000, 0b10000],
        'G' => [0b01110, 0b10001, 0b10000, 0b10111, 0b10001, 0b10001, 0b01111],
        'H' => [0b10001, 0b10001, 0b10001, 0b11111, 0b10001, 0b10001, 0b10001],
        'I' => [0b01110, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b01110],
        'J' => [0b00111, 0b00010, 0b00010, 0b00010, 0b00010, 0b10010, 0b01100],
        'K' => [0b10001, 0b10010, 0b10100, 0b11000, 0b10100, 0b10010, 0b10001],
        'L' => [0b10000, 0b10000, 0b10000, 0b10000, 0b10000, 0b10000, 0b11111],
        'M' => [0b10001, 0b11011, 0b10101, 0b10101, 0b10001, 0b10001, 0b10001],
        'N' => [0b10001, 0b10001, 0b11001, 0b10101, 0b10011, 0b10001, 0b10001],
        'O' => [0b01110, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01110],
        'P' => [0b11110, 0b10001, 0b10001, 0b11110, 0b10000, 0b10000, 0b10000],
        'Q' => [0b01110, 0b10001, 0b10001, 0b10001, 0b10101, 0b10010, 0b01101],
        'R' => [0b11110, 0b10001, 0b10001, 0b11110, 0b10100, 0b10010, 0b10001],
        'S' => [0b01111, 0b10000, 0b10000, 0b01110, 0b00001, 0b00001, 0b11110],
        'T' => [0b11111, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100],
        'U' => [0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01110],
        'V' => [0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01010, 0b00100],
        'W' => [0b10001, 0b10001, 0b10001, 0b10101, 0b10101, 0b11011, 0b10001],
        'X' => [0b10001, 0b10001, 0b01010, 0b00100, 0b01010, 0b10001, 0b10001],
        'Y' => [0b10001, 0b10001, 0b01010, 0b00100, 0b00100, 0b00100, 0b00100],
        'Z' => [0b11111, 0b00001, 0b00010, 0b00100, 0b01000, 0b10000, 0b11111],
        'a' => [0b00000, 0b00000, 0b01110, 0b00001, 0b01111, 0b10001, 0b01111],
        'b' => [0b10000, 0b10000, 0b11110, 0b10001, 0b10001, 0b10001, 0b11110],
        'c' => [0b00000, 0b00000, 0b01110, 0b10000, 0b10000, 0b10001, 0b01110],
        'd' => [0b00001, 0b00001, 0b01111, 0b10001, 0b10001, 0b10001, 0b01111],
        'e' => [0b00000, 0b00000, 0b01110, 0b10001, 0b11111, 0b10000, 0b01110],
        'f' => [0b00110, 0b01001, 0b01000, 0b11100, 0b01000, 0b01000, 0b01000],
        'g' => [0b00000, 0b01111, 0b10001, 0b10001, 0b01111, 0b00001, 0b01110],
        'h' => [0b10000, 0b10000, 0b11110, 0b10001, 0b10001, 0b10001, 0b10001],
        'i' => [0b00100, 0b00000, 0b01100, 0b00100, 0b00100, 0b00100, 0b01110],
        'j' => [0b00010, 0b00000, 0b00110, 0b00010, 0b00010, 0b10010, 0b01100],
        'k' => [0b10000, 0b10000, 0b10010, 0b10100, 0b11000, 0b10100, 0b10010],
        'l' => [0b01100, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b01110],
        'm' => [0b00000, 0b00000, 0b11010, 0b10101, 0b10101, 0b10001, 0b10001],
        'n' => [0b00000, 0b00000, 0b11110, 0b10001, 0b10001, 0b10001, 0b10001],
        'o' => [0b00000, 0b00000, 0b01110, 0b10001, 0b10001, 0b10001, 0b01110],
        'p' => [0b00000, 0b11110, 0b10001, 0b10001, 0b11110, 0b10000, 0b10000],
        'q' => [0b00000, 0b01111, 0b10001, 0b10001, 0b01111, 0b00001, 0b00001],
        'r' => [0b00000, 0b00000, 0b10110, 0b11001, 0b10000, 0b10000, 0b10000],
        's' => [0b00000, 0b00000, 0b01111, 0b10000, 0b01110, 0b00001, 0b11110],
        't' => [0b01000, 0b01000, 0b11100, 0b01000, 0b01000, 0b01001, 0b00110],
        'u' => [0b00000, 0b00000, 0b10001, 0b10001, 0b10001, 0b10011, 0b01101],
        'v' => [0b00000, 0b00000, 0b10001, 0b10001, 0b10001, 0b01010, 0b00100],
        'w' => [0b00000, 0b00000, 0b10001, 0b10001, 0b10101, 0b10101, 0b01010],
        'x' => [0b00000, 0b00000, 0b10001, 0b01010, 0b00100, 0b01010, 0b10001],
        'y' => [0b00000, 0b10001, 0b10001, 0b10001, 0b01111, 0b00001, 0b01110],
        'z' => [0b00000, 0b00000, 0b11111, 0b00010, 0b00100, 0b01000, 0b11111],
        '.' => [0b00000, 0b00000, 0b00000, 0b00000, 0b00000, 0b01100, 0b01100],
        ',' => [0b00000, 0b00000, 0b00000, 0b00000, 0b01100, 0b00100, 0b01000],
        '#' => [0b01010, 0b01010, 0b11111, 0b01010, 0b11111, 0b01010, 0b01010],
        ':' => [0b00000, 0b01100, 0b01100, 0b00000, 0b01100, 0b01100, 0b00000],
        '/' => [0b00001, 0b00010, 0b00010, 0b00100, 0b01000, 0b01000, 0b10000],
        '-' => [0b00000, 0b00000, 0b00000, 0b11111, 0b00000, 0b00000, 0b00000],
        '%' => [0b11001, 0b11010, 0b00010, 0b00100, 0b01000, 0b01011, 0b10011],
        '(' => [0b00010, 0b00100, 0b01000, 0b01000, 0b01000, 0b00100, 0b00010],
        ')' => [0b01000, 0b00100, 0b00010, 0b00010, 0b00010, 0b00100, 0b01000],
        _ => return None,
    };
    Some(rows)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rgb_is_packed_into_u32() {
        let mut canvas = Canvas::new(2, 1);
        canvas
            .fill_from_rgb(2, 1, &[0xFF, 0x00, 0x00, 0x00, 0xFF, 0x00])
            .unwrap();
        assert_eq!(canvas.pixels(), &[0x00FF_0000, 0x0000_FF00]);
    }

    #[test]
    fn rgb_with_wrong_length_is_rejected() {
        let mut canvas = Canvas::new(2, 1);
        let err = canvas.fill_from_rgb(2, 1, &[0u8; 5]).unwrap_err().to_string();
        assert!(err.contains("expected 6"), "{err}");
    }

    #[test]
    fn filling_reuses_the_allocation() {
        let mut canvas = Canvas::new(2, 1);
        canvas.fill_from_rgb(2, 1, &[1, 2, 3, 4, 5, 6]).unwrap();
        let capacity = canvas.buf.capacity();
        canvas.fill_from_rgb(2, 1, &[9, 9, 9, 9, 9, 9]).unwrap();
        assert_eq!(canvas.buf.capacity(), capacity);
        assert_eq!(canvas.pixels(), &[0x0009_0909, 0x0009_0909]);
    }

    #[test]
    fn resizing_keeps_the_allocation() {
        let mut canvas = Canvas::new(4, 4);
        let capacity = canvas.buf.capacity();
        canvas.resize(8, 8);
        assert_eq!(canvas.width(), 8);
        assert_eq!(canvas.pixels().len(), 64);
        assert!(canvas.buf.capacity() >= capacity);
    }

    #[test]
    fn drawing_is_clipped_to_the_canvas() {
        let mut canvas = Canvas::new(8, 8);
        canvas.rect(-4, -4, 100, 100, GREEN, 2);
        canvas.line(-10, 3, 40, 30, WHITE);
        canvas.text(-10, 3, "clipped", WHITE, 1);
        canvas.set(-1, -1, RED);
        assert_eq!(canvas.pixels().len(), 64);
    }

    #[test]
    fn overlay_is_deterministic_across_redraws() {
        let mut canvas = Canvas::new(200, 40);
        canvas.status_overlay(&["FPS 30.0 INF 31 MS".to_string()], 1);
        let first = canvas.pixels().to_vec();
        canvas.fill_from_rgb(200, 40, &vec![0u8; 200 * 40 * 3]).unwrap();
        canvas.status_overlay(&["FPS 30.0 INF 31 MS".to_string()], 1);
        assert_eq!(canvas.pixels(), first.as_slice());
    }
}
