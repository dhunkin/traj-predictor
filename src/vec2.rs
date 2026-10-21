//! vec2.rs — Vec2, Mat2, cholesky_sample.
//!
//! A tiny 2D vector/matrix algebra core, dependency-free, no `unsafe`.
//! Mat2 wraps a lower-triangular Cholesky factor of a 2x2 covariance
//! matrix `[xx, xy, yy]`; `sample` applies it to two standard normals.
//!
//! TODO(Stage2): (reserved)

use crate::prng::Xoshiro256ss;

/// 2D vector.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vec2 {
    pub x: f64,
    pub y: f64,
}

impl Vec2 {
    pub fn new(x: f64, y: f64) -> Self {
        Vec2 { x, y }
    }

    pub fn add(self, o: Vec2) -> Vec2 {
        Vec2 {
            x: self.x + o.x,
            y: self.y + o.y,
        }
    }

    pub fn sub(self, o: Vec2) -> Vec2 {
        Vec2 {
            x: self.x - o.x,
            y: self.y - o.y,
        }
    }

    pub fn scale(self, s: f64) -> Vec2 {
        Vec2 {
            x: self.x * s,
            y: self.y * s,
        }
    }

    pub fn dot(self, o: Vec2) -> f64 {
        self.x * o.x + self.y * o.y
    }

    pub fn len(self) -> f64 {
        (self.dot(self)).sqrt()
    }

    /// Rotate the vector by `angle` radians (counterclockwise).
    pub fn rotate(self, angle: f64) -> Vec2 {
        let (s, c) = angle.sin_cos();
        Vec2 {
            x: self.x * c - self.y * s,
            y: self.x * s + self.y * c,
        }
    }
}

/// Lower-triangular Cholesky factor of a 2x2 covariance matrix.
///
/// Built from `[xx, xy, yy]` (variances + covariance) as:
///   L11 = sqrt(xx)
///   L21 = xy / L11
///   L22 = sqrt(yy - L21^2)
///
/// Degenerate handling:
///   - xx == 0 && yy == 0 -> zero noise (identity-free zero factor).
///   - xx == 0 && yy > 0   -> swap axes so the non-zero variance lands on
///     the diagonal (documented choice; see below).
#[derive(Debug, Clone, Copy)]
pub struct Mat2 {
    pub l11: f64,
    pub l21: f64,
    pub l22: f64,
}

impl Mat2 {
    /// Build a Cholesky factor from `[xx, xy, yy]`.
    pub fn from_cov(cov: [f64; 3]) -> Self {
        let (xx, xy, yy) = (cov[0], cov[1], cov[2]);
        debug_assert!(xx >= 0.0, "xx must be a variance, got {xx}");
        debug_assert!(yy >= 0.0, "yy must be a variance, got {yy}");
        debug_assert!(
            xy * xy <= xx * yy + 1e-12,
            "covariance is not PSD: xy^2 = {}, xx*yy = {}",
            xy * xy,
            xx * yy
        );
        if xx == 0.0 && yy == 0.0 {
            // Degenerate: zero noise.
            return Mat2 {
                l11: 0.0,
                l21: 0.0,
                l22: 0.0,
            };
        }
        if xx == 0.0 {
            // xx == 0, yy > 0: swap axes. The variance now lives on the
            // y-axis; L22 = sqrt(yy), off-diagonal zero.
            return Mat2 {
                l11: 0.0,
                l21: 0.0,
                l22: yy.sqrt(),
            };
        }
        let l11 = xx.sqrt();
        let l21 = xy / l11;
        let l22 = (yy - l21 * l21).max(0.0).sqrt();
        Mat2 {
            l11,
            l21,
            l22,
        }
    }

    /// Apply the factor to a standard-normal pair: `L * [a, b]`.
    pub fn apply(self, a: f64, b: f64) -> Vec2 {
        Vec2 {
            x: self.l11 * a,
            y: self.l21 * a + self.l22 * b,
        }
    }

    /// Sample `L * [gauss(0,1), gauss(0,1)]` using the given RNG.
    pub fn sample(&self, rng: &mut Xoshiro256ss) -> Vec2 {
        let a = rng.gauss(0.0, 1.0);
        let b = rng.gauss(0.0, 1.0);
        self.apply(a, b)
    }
}
