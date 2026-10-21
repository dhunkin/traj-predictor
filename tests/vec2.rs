//! Tests for vec2.rs — Vec2 ops, Mat2 Cholesky, and sampling.

use predictor::prng::Xoshiro256ss;
use predictor::vec2::{Mat2, Vec2};

#[test]
fn vec2_basic_ops() {
    let a = Vec2::new(3.0, 4.0);
    let b = Vec2::new(1.0, 2.0);
    assert_eq!(a.add(b), Vec2::new(4.0, 6.0));
    assert_eq!(a.sub(b), Vec2::new(2.0, 2.0));
    assert_eq!(a.scale(2.0), Vec2::new(6.0, 8.0));
    assert_eq!(a.dot(b), 11.0);
    assert!((a.len() - 5.0).abs() < 1e-12);
}

#[test]
fn vec2_rotate_by_90() {
    let a = Vec2::new(1.0, 0.0);
    let r = a.rotate(std::f64::consts::PI / 2.0);
    assert!((r.x - 0.0).abs() < 1e-12, "x: {}", r.x);
    assert!((r.y - 1.0).abs() < 1e-12, "y: {}", r.y);
}

#[test]
fn vec2_rotate_identity() {
    let a = Vec2::new(3.0, 4.0);
    let r = a.rotate(0.0);
    assert_eq!(r, a);
}

#[test]
fn mat2_cholesky_reconstructs_cov() {
    // L * L^T should reproduce [xx, xy, yy] on the diagonal-ish form.
    let cov = [4.0, 2.0, 5.0]; // xx=4, xy=2, yy=5
    let l = Mat2::from_cov(cov);
    assert!((l.l11 - 2.0).abs() < 1e-12); // sqrt(4)
    assert!((l.l21 - 1.0).abs() < 1e-12); // 2/2
    assert!((l.l22 - 2.0).abs() < 1e-12); // sqrt(5 - 1)

    // L * [1,0] = first column gives the covariance's x-part.
    let v = l.apply(1.0, 0.0);
    assert!((v.x - l.l11).abs() < 1e-12);
    assert!((v.y - l.l21).abs() < 1e-12);
}

#[test]
fn mat2_zero_cov_is_zero_noise() {
    let l = Mat2::from_cov([0.0, 0.0, 0.0]);
    assert_eq!((l.l11, l.l21, l.l22), (0.0, 0.0, 0.0));
    let mut rng = Xoshiro256ss::from_seed(1);
    let s = l.sample(&mut rng);
    assert_eq!((s.x, s.y), (0.0, 0.0));
}

#[test]
fn mat2_xx_zero_swaps_axes() {
    // xx == 0, yy > 0 -> axes swapped: l22 = sqrt(yy).
    let l = Mat2::from_cov([0.0, 0.0, 9.0]);
    assert!((l.l11 - 0.0).abs() < 1e-12);
    assert!((l.l21 - 0.0).abs() < 1e-12);
    assert!((l.l22 - 3.0).abs() < 1e-12);
}

#[test]
fn mat2_sample_is_centered() {
    // Over many draws, the mean of L * N(0, I) should be ~0.
    let l = Mat2::from_cov([4.0, 0.0, 9.0]);
    let mut rng = Xoshiro256ss::from_seed(55);
    let n = 100_000u64;
    let mut sx = 0.0f64;
    let mut sy = 0.0f64;
    for _ in 0..n {
        let s = l.sample(&mut rng);
        sx += s.x;
        sy += s.y;
    }
    assert!((sx / n as f64).abs() < 0.05, "mean x: {}", sx / n as f64);
    assert!((sy / n as f64).abs() < 0.05, "mean y: {}", sy / n as f64);
}
