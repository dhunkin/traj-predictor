//! filter.rs — 2D Constant-Velocity Kalman Filter tests.
//!
//! Stage 2: verify that `filter_track` recovers velocity and heading from a
//! noisy straight track, and that the smoothed estimate is closer to the
//! truth than a naive finite-difference of the last two points.
//!
//! Deterministic: fixed seeds, tolerance-based assertions (no exact floats).

use predictor::filter::filter_track;
use predictor::prng::Xoshiro256ss;
use predictor::types::Observation;
use predictor::vec2::Mat2;
use std::f64::consts::PI;

/// Build a straight +X track at speed `v`, heading 0, sampled every 1s.
/// Truth position at time t is (v*t, 0). Each observation is the truth plus
/// Gaussian noise of covariance `obs_cov`, drawn from a seeded RNG.
fn straight_track(
    v: f64,
    n: usize,
    dt: f64,
    obs_cov: [f64; 3],
    seed: u64,
) -> Vec<Observation> {
    let mut rng = Xoshiro256ss::from_seed(seed);
    let mat = Mat2::from_cov(obs_cov);
    let mut track = Vec::with_capacity(n);
    for i in 0..n {
        let t = i as f64 * dt;
        let truth = [v * t, 0.0];
        let noise = mat.sample(&mut rng);
        track.push(Observation {
            t,
            pos: [truth[0] + noise.x, truth[1] + noise.y],
            v,
            psi: 0.0,
            cov: obs_cov,
        });
    }
    track
}

/// Finite-difference velocity estimate from the last two observations:
///   v_fd = |pos[k] - pos[k-1]| / dt
/// psi_fd = atan2(dy, dx).
fn finite_diff(track: &[Observation]) -> (f64, f64) {
    let a = &track[track.len() - 2];
    let b = &track[track.len() - 1];
    let dt = b.t - a.t;
    let dx = b.pos[0] - a.pos[0];
    let dy = b.pos[1] - a.pos[1];
    ((dx * dx + dy * dy).sqrt() / dt, dy.atan2(dx))
}

#[test]
fn perfect_strait_no_noise() {
    // Perfect straight track, zero noise, speed 1 m/s, heading 0.
    let track = straight_track(1.0, 20, 1.0, [0.0, 0.0, 0.0], 42);
    let est = filter_track(&track);

    assert!(
        (est.v - 1.0).abs() < 1e-3,
        "v_est = {}, expected ~1.0 (no noise)",
        est.v
    );
    assert!(
        est.psi.abs() < 1e-6,
        "psi_est = {}, expected ~0.0 (no noise)",
        est.psi
    );
    // Position should be at (v*T, 0) = (19, 0) for n=20, dt=1.
    assert!(
        (est.pos[0] - 19.0).abs() < 1e-6,
        "x_est = {}, expected ~19.0",
        est.pos[0]
    );
    assert!(
        (est.pos[1]).abs() < 1e-6,
        "y_est = {}, expected ~0.0",
        est.pos[1]
    );
}

#[test]
fn noisy_track_better_than_finite_diff() {
    // Noisy straight track. The smoothed velocity/heading should be closer to
    // truth than the finite-difference estimate from just the last two points.
    let v_truth = 5.0;
    let obs_cov = [1.0, 0.0, 1.0]; // sigma = 1 m/axis
    let track = straight_track(v_truth, 30, 1.0, obs_cov, 7);
    let est = filter_track(&track);
    let (v_fd, psi_fd) = finite_diff(&track);

    // Velocity: filter should be closer to truth than finite-difference.
    let v_err_filter = (est.v - v_truth).abs();
    let v_err_fd = (v_fd - v_truth).abs();
    assert!(
        v_err_filter < v_err_fd,
        "filter v_err = {}, fd v_err = {}, filter should be closer",
        v_err_filter,
        v_err_fd
    );

    // Heading: filter should be closer to 0 than finite-difference.
    let psi_err_filter = est.psi.abs();
    let psi_err_fd = psi_fd.abs();
    assert!(
        psi_err_filter < psi_err_fd,
        "filter psi_err = {}, fd psi_err = {}, filter should be closer",
        psi_err_filter,
        psi_err_fd
    );
}

#[test]
fn psi_near_pi_wraps_correctly() {
    // Track heading in +X but at ~180 deg: psi should be near 0 or ~pi, not
    // wildly off. Use a track moving in -X (psi ~ pi) and check the filter
    // recovers a heading whose cosine is near -1.
    let mut rng = Xoshiro256ss::from_seed(3);
    let mat = Mat2::from_cov([0.5, 0.0, 0.5]);
    let n = 25;
    let mut track = Vec::with_capacity(n);
    for i in 0..n {
        let t = i as f64;
        // Move in -X direction: pos = (-v*t, 0), heading ~ pi.
        let v = 2.0;
        let truth = [-v * t, 0.0];
        let noise = mat.sample(&mut rng);
        track.push(Observation {
            t,
            pos: [truth[0] + noise.x, truth[1] + noise.y],
            v,
            psi: PI,
            cov: [0.5, 0.0, 0.5],
        });
    }
    let est = filter_track(&track);
    let c = est.psi.cos();
    assert!(
        (c - (-1.0)).abs() < 0.2,
        "psi_est = {}, cos = {}, expected cos ~ -1 (moving in -X)",
        est.psi,
        c
    );
}

#[test]
fn single_observation_falls_back() {
    // A single-observation track returns that observation's position (velocity
    // is 0 because there's no second point to infer it from — correct KF
    // behaviour for a one-sample track).
    let track = straight_track(3.0, 1, 1.0, [0.0, 0.0, 0.0], 1);
    let est = filter_track(&track);
    assert!(
        (est.pos[0] - 0.0).abs() < 1e-6,
        "single-obs x_est = {}, expected ~0.0",
        est.pos[0]
    );
    assert!(
        (est.pos[1]).abs() < 1e-6,
        "single-obs y_est = {}, expected ~0.0",
        est.pos[1]
    );
}
