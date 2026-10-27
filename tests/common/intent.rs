//! Shared track generators for the intent-inference scenarios
//! (used by tests/intent.rs and examples/intent_probe.rs).

#![allow(dead_code)]

use predictor::prng::Xoshiro256ss;
use predictor::types::{Observation, Ooi, PredictConfig, Ttc};
use std::f64::consts::PI;

pub const DT: f64 = 1.0;
pub const SIGMA: f64 = 1.0;

pub fn ttc() -> Ttc {
    Ttc { v_min: 3.0, v_max: 12.0, a_lat: 9.0, a_long: 3.0, omega_cap: 1.0 }
}

pub fn cfg(seed: u64) -> PredictConfig {
    PredictConfig {
        horizon_s: 10.0,
        n_samples: 4000,
        dt_s: 1.0,
        p_turn: 0.3,
        azimuth_bins: 72,
        speed_noise_mps: 0.5,
        intent_hypothesis_prob: 0.5,
        seed,
    }
}

/// Sample `truth(t)` every DT seconds for `duration` s and add N(0, SIGMA^2)
/// position noise. `truth` returns the true position; heading/speed fields of
/// the observation are left at 0 (the filter must not use them).
pub fn observe(duration: f64, seed: u64, truth: impl Fn(f64) -> [f64; 2]) -> Vec<Observation> {
    let mut rng = Xoshiro256ss::from_seed(seed ^ 0xA5A5_5A5A);
    let n = (duration / DT).round() as usize;
    (0..=n)
        .map(|k| {
            let t = k as f64 * DT;
            let p = truth(t);
            Observation {
                t,
                pos: [p[0] + rng.gauss(0.0, SIGMA), p[1] + rng.gauss(0.0, SIGMA)],
                v: 0.0,
                psi: 0.0,
                cov: [SIGMA * SIGMA, 0.0, SIGMA * SIGMA],
            }
        })
        .collect()
}

/// Straight line from the origin along `psi` at speed `v`.
pub fn straight(v: f64, psi: f64) -> impl Fn(f64) -> [f64; 2] {
    move |t| [v * t * psi.cos(), v * t * psi.sin()]
}

/// Circle of radius `r` flown at turn rate `w` (counter-clockwise), whose
/// centre drifts from `c0` along `drift_dir` at `drift` m/s.
pub fn drifting_circle(r: f64, w: f64, c0: [f64; 2], drift: f64, drift_dir: f64) -> impl Fn(f64) -> [f64; 2] {
    move |t| {
        let th = w * t - PI / 2.0; // start at the bottom of the circle, heading +x
        [
            c0[0] + drift * t * drift_dir.cos() + r * th.cos(),
            c0[1] + drift * t * drift_dir.sin() + r * th.sin(),
        ]
    }
}

/// Truth that obeys the goal model: heading bends toward `goal` at gain `k`
/// (1/s), starting from heading `psi0` at the origin. Integrated at 0.05 s.
pub fn homing(v: f64, psi0: f64, goal: [f64; 2], k: f64) -> impl Fn(f64) -> [f64; 2] {
    move |t| {
        let (mut x, mut y, mut psi) = (0.0f64, 0.0f64, psi0);
        let h = 0.05;
        let n = (t / h).round() as usize;
        for _ in 0..n {
            let b = (goal[1] - y).atan2(goal[0] - x);
            let e = (b - psi + PI).rem_euclid(2.0 * PI) - PI;
            psi += k * e * h;
            x += v * h * psi.cos();
            y += v * h * psi.sin();
        }
        [x, y]
    }
}

pub fn ooi(x: f64, y: f64) -> Vec<Ooi> {
    vec![Ooi { pos: [x, y] }]
}
