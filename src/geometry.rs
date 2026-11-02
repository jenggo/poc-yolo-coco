/// An axis-aligned box in pixel coordinates, stored as two corners.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BBox {
    pub x1: f32,
    pub y1: f32,
    pub x2: f32,
    pub y2: f32,
}

impl BBox {
    pub const fn new(x1: f32, y1: f32, x2: f32, y2: f32) -> Self {
        Self { x1, y1, x2, y2 }
    }

    /// Build a box from a centre point and a size.
    pub fn from_center(cx: f32, cy: f32, w: f32, h: f32) -> Self {
        Self {
            x1: cx - w / 2.0,
            y1: cy - h / 2.0,
            x2: cx + w / 2.0,
            y2: cy + h / 2.0,
        }
    }

    pub fn width(&self) -> f32 {
        self.x2 - self.x1
    }

    pub fn height(&self) -> f32 {
        self.y2 - self.y1
    }

    pub fn area(&self) -> f32 {
        self.width().max(0.0) * self.height().max(0.0)
    }

    pub fn center(&self) -> (f32, f32) {
        ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)
    }

    /// The midpoint of the bottom edge: where a vehicle touches the road.
    pub fn bottom_center(&self) -> (f32, f32) {
        ((self.x1 + self.x2) / 2.0, self.y2)
    }

    /// True when the box has positive extent in both directions.
    pub fn is_valid(&self) -> bool {
        self.width() > 0.0 && self.height() > 0.0
    }

    /// Clip the box to `0..width` and `0..height`, or `None` if nothing remains.
    pub fn clipped(&self, width: f32, height: f32) -> Option<Self> {
        let box_ = Self {
            x1: self.x1.clamp(0.0, width),
            y1: self.y1.clamp(0.0, height),
            x2: self.x2.clamp(0.0, width),
            y2: self.y2.clamp(0.0, height),
        };
        box_.is_valid().then_some(box_)
    }
}

/// Intersection over union of two boxes, in `0.0..=1.0`.
pub fn iou(a: &BBox, b: &BBox) -> f32 {
    let x1 = a.x1.max(b.x1);
    let y1 = a.y1.max(b.y1);
    let x2 = a.x2.min(b.x2);
    let y2 = a.y2.min(b.y2);
    let inter = (x2 - x1).max(0.0) * (y2 - y1).max(0.0);
    let union = a.area() + b.area() - inter;
    if union <= 0.0 { 0.0 } else { inter / union }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn center_and_size_round_trip() {
        let b = BBox::from_center(100.0, 50.0, 20.0, 10.0);
        assert_eq!(b, BBox::new(90.0, 45.0, 110.0, 55.0));
        assert_eq!(b.center(), (100.0, 50.0));
        assert_eq!(b.bottom_center(), (100.0, 55.0));
        assert_eq!(b.area(), 200.0);
    }

    #[test]
    fn iou_of_disjoint_boxes_is_zero() {
        let a = BBox::new(0.0, 0.0, 10.0, 10.0);
        let b = BBox::new(20.0, 20.0, 30.0, 30.0);
        assert_eq!(iou(&a, &b), 0.0);
    }

    #[test]
    fn iou_of_identical_boxes_is_one() {
        let a = BBox::new(0.0, 0.0, 10.0, 10.0);
        assert!((iou(&a, &a) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn iou_of_half_overlap() {
        let a = BBox::new(0.0, 0.0, 10.0, 10.0);
        let b = BBox::new(5.0, 0.0, 15.0, 10.0);
        assert!((iou(&a, &b) - 50.0 / 150.0).abs() < 1e-6);
    }

    #[test]
    fn degenerate_boxes_have_zero_iou() {
        let a = BBox::new(0.0, 0.0, 0.0, 0.0);
        let b = BBox::new(0.0, 0.0, 0.0, 0.0);
        assert_eq!(iou(&a, &b), 0.0);
    }

    #[test]
    fn clipping_trims_and_rejects_empty_boxes() {
        let b = BBox::new(-10.0, -10.0, 50.0, 50.0).clipped(100.0, 100.0).unwrap();
        assert_eq!(b, BBox::new(0.0, 0.0, 50.0, 50.0));
        assert!(BBox::new(200.0, 200.0, 300.0, 300.0).clipped(100.0, 100.0).is_none());
    }
}
