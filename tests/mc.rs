//! Tests for mc.rs — Monte-Carlo prediction engine (§4.4).

use predictor::mc::predict;
use predictor::types::{Observation, Ooi, PredictConfig, Ttc};
use std::f64::consts::PI;

fn ttc() -> Ttc {
    Ttc {
        v_min: 2.0,
        v_max: 8.0,
        omega_max: 0.5,
    }
}

fn cfg(n_samples: u32, p_turn: f64, seed: u64, azimuth_bins: u32) -> PredictConfig {
    PredictConfig {
        horizon_s: 10.0,
        n_samples,
        dt_s: 1.0, // reserved, unused in Stage 1
        p_turn,
        azimuth_bins,
        speed_noise_mps: 0.0,
        seed,
    }
}

fn obs() -> Observation {
    Observation {
        t: 0.0,
        pos: [0.0, 0.0],
        v: 5.0,
        psi: 0.0,
        cov: [1.0, 0.0, 1.0],
    }
}

fn oois() -> Vec<Ooi> {
    vec![Ooi { pos: [10.0, 0.0] }]
}

#[test]
fn determinism_same_seed_bit_identical() {
    let o = obs();
    let t = ttc();
    let c = cfg(1000, 0.3, 42, 72);
    let a = predict(&o, &t, &oois(), &c);
    let b = predict(&o, &t, &oois(), &c);

    assert_eq!(a.point_cloud.len(), b.point_cloud.len());
    for (p, q) in a.point_cloud.iter().zip(b.point_cloud.iter()) {
        assert_eq!(*p, *q);
    }
    assert_eq!(a.bin_centers, b.bin_centers);
    assert_eq!(a.bin_probs, b.bin_probs);
    assert_eq!(a.sector50, b.sector50);
    assert_eq!(a.sector90, b.sector90);
    assert_eq!(a.final_heading_mean, b.final_heading_mean);
}

#[test]
fn bin_centers_count_and_probs_sum_to_one() {
    let o = obs();
    let t = ttc();
    let c = cfg(1000, 0.3, 1, 72);
    let p = predict(&o, &t, &oois(), &c);

    assert_eq!(p.bin_centers.len(), 72);
    assert_eq!(p.bin_probs.len(), 72);

    let sum: f64 = p.bin_probs.iter().sum::<f64>();
    assert!((sum - 1.0).abs() < 1e-9, "bin_probs sum = {}", sum);
}

#[test]
fn point_cloud_has_n_samples() {
    let o = obs();
    let t = ttc();
    let c = cfg(500, 0.3, 7, 72);
    let p = predict(&o, &t, &oois(), &c);
    assert_eq!(p.point_cloud.len(), 500);
}

#[test]
fn zero_turn_probability_concentrates_all_samples() {
    // p_turn = 0 => omega = 0 for all samples => psi_new = last.psi = 0
    // for every sample. All samples fall in bin 0, so both sectors are
    // exactly one bin width (5 deg).
    let o = obs();
    let t = ttc();
    let c = cfg(2000, 0.0, 1, 72);
    let p = predict(&o, &t, &oois(), &c);

    let bin_width = 2.0 * PI / 72.0;
    let s50 = p.sector50;
    let s90 = p.sector90;

    assert!((s50 - bin_width).abs() < 1e-9, "s50 = {}", s50);
    assert!((s90 - bin_width).abs() < 1e-9, "s90 = {}", s90);

    // Circular mean should be near 0 (last.psi).
    assert!(p.final_heading_mean.abs() < 1e-6);

    // Exactly one non-empty bin.
    let nonzero = p.bin_probs.iter().filter(|&&x| x > 0.0).count();
    assert_eq!(nonzero, 1);
}

#[test]
fn nonzero_turn_probability_spreads_sectors() {
    // p_turn = 1 => omega drawn every time => heading distribution is
    // spread over more bins => sector90 > 1 bin width.
    let o = obs();
    let t = ttc();
    let c = cfg(5000, 1.0, 123, 72);
    let p = predict(&o, &t, &oois(), &c);

    let bin_width = 2.0 * PI / 72.0;
    assert!(
        p.sector90 > bin_width * 2.0,
        "s90 = {} should be > 2 bins",
        p.sector90
    );
}

#[test]
fn initial_position_noise_samples_from_cov() {
    // With cov = [1, 0, 1], the initial position noise has std 1 in each axis.
    // p_turn = 0 => omega = 0 => psi_new = last.psi = 0 for all samples, so
    // each final position = pos0 + v*T*[1, 0] = pos0 + [50, 0] + noise.
    // The cloud is therefore centered near [50, 0] with std ~1 in x and y.
    let o = obs();
    let t = ttc();
    let c = cfg(2000, 0.0, 5, 72);
    let p = predict(&o, &t, &oois(), &c);

    let mean_x: f64 = p.point_cloud.iter().map(|p| p[0]).sum::<f64>() / 2000.0;
    let mean_y: f64 = p.point_cloud.iter().map(|p| p[1]).sum::<f64>() / 2000.0;

    // Noise is zero-mean, so means should be near the deterministic
    // propagation endpoint [50, 0].
    assert!((mean_x - 50.0).abs() < 0.2, "mean_x = {}", mean_x);
    assert!(mean_y.abs() < 0.2, "mean_y = {}", mean_y);
}

#[test]
fn different_seed_differs() {
    let o = obs();
    let t = ttc();
    let c1 = cfg(1000, 0.3, 1, 72);
    let c2 = cfg(1000, 0.3, 2, 72);
    let p1 = predict(&o, &t, &oois(), &c1);
    let p2 = predict(&o, &t, &oois(), &c2);

    // At least one point in the cloud differs.
    let mut any_diff = false;
    for (p, q) in p1.point_cloud.iter().zip(p2.point_cloud.iter()) {
        if *p != *q {
            any_diff = true;
        }
    }
    assert!(any_diff, "different seeds should yield different clouds");
}
