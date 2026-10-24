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
/// Built from `[xx, xy, yy]` (variances + covariance) via the
/// correlation-parametrized decomposition:
///   sx = sqrt(xx), sy = sqrt(yy)
///   rho = clamp(xy / (sx * sy), -1, 1)
///   L11 = sx, L21 = rho * sy, L22 = sy * sqrt(1 - rho^2)
///
/// Degenerate handling: if either standard deviation is zero the factor is
/// diagonal (`L11 = sx`, `L22 = sy`, `L21 = 0`).
#[derive(Debug, Clone, Copy)]
pub struct Mat2 {
    pub l11: f64,
    pub l21: f64,
    pub l22: f64,
}

impl Mat2 {
    /// Build a Cholesky factor from `[xx, xy, yy]`.
    pub fn from_cov(cov: [f64; 3]) -> Self {
        let sx = cov[0].max(0.0).sqrt();
        let sy = cov[2].max(0.0).sqrt();
        if sx == 0.0 || sy == 0.0 {
            return Mat2 { l11: sx, l21: 0.0, l22: sy };
        }
        let rho = (cov[1] / (sx * sy)).clamp(-1.0, 1.0);
        Mat2 { l11: sx, l21: rho * sy, l22: sy * (1.0 - rho * rho).sqrt() }
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
