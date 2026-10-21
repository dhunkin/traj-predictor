//! Tests for prng.rs — determinism by seed, and gaussian statistics.

use predictor::prng::Xoshiro256ss;

#[test]
fn determinism_by_seed() {
    // Same seed + same sequence of draws => bit-identical output.
    let mut a = Xoshiro256ss::from_seed(12345);
    let mut b = Xoshiro256ss::from_seed(12345);

    for _ in 0..1000 {
        let fa = a.next_f64();
        let fb = b.next_f64();
        assert_eq!(fa.to_bits(), fb.to_bits(), "f64 streams diverged");
    }
    for _ in 0..1000 {
        let ua = a.next_u64();
        let ub = b.next_u64();
        assert_eq!(ua, ub, "u64 streams diverged");
    }
    for _ in 0..500 {
        let ga = a.gauss(3.0, 2.0);
        let gb = b.gauss(3.0, 2.0);
        assert_eq!(ga.to_bits(), gb.to_bits(), "gaussian streams diverged");
    }
}

#[test]
fn different_seed_differs() {
    let mut a = Xoshiro256ss::from_seed(1);
    let mut b = Xoshiro256ss::from_seed(2);
    let mut diffs = 0;
    for _ in 0..1000 {
        if a.next_u64() != b.next_u64() {
            diffs += 1;
        }
    }
    assert!(diffs > 500, "different seeds should mostly differ");
}

#[test]
fn next_f64_in_unit_interval() {
    let mut rng = Xoshiro256ss::from_seed(42);
    for _ in 0..100_000 {
        let x = rng.next_f64();
        assert!((0.0..1.0).contains(&x), "f64 out of [0,1): {x}");
    }
}

#[test]
fn gaussian_mean_std() {
    // ~1e6 draws: |mean| < 0.01, |std - 1| < 0.01 for standard normal.
    let mut rng = Xoshiro256ss::from_seed(7);
    let n = 1_000_000u64;
    let mut sum = 0.0f64;
    let mut sum2 = 0.0f64;
    for _ in 0..n {
        let x = rng.gauss(0.0, 1.0);
        sum += x;
        sum2 += x * x;
    }
    let mean = sum / n as f64;
    let var = sum2 / n as f64 - mean * mean;
    let std = var.sqrt();
    assert!(mean.abs() < 0.01, "mean too far from 0: {mean}");
    assert!((std - 1.0).abs() < 0.01, "std too far from 1: {std}");
}

#[test]
fn gaussian_scale_and_shift() {
    // gauss(mean, std) should have the right mean and std.
    let mut rng = Xoshiro256ss::from_seed(99);
    let n = 200_000u64;
    let mut sum = 0.0f64;
    let mut sum2 = 0.0f64;
    for _ in 0..n {
        let x = rng.gauss(10.0, 3.0);
        sum += x;
        sum2 += x * x;
    }
    let mean = sum / n as f64;
    let var = sum2 / n as f64 - mean * mean;
    let std = var.sqrt();
    assert!((mean - 10.0).abs() < 0.05, "bad mean: {mean}");
    assert!((std - 3.0).abs() < 0.05, "bad std: {std}");
}
