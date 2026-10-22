//! simulator.rs — synthetic track generator.

use crate::prng::Xoshiro256ss;
use crate::types::{KinState, Observation, Ttc};
use crate::vec2::Mat2;

/// Generate a synthetic track for Stage 1: constant-velocity
/// straight-line truth starting at truth_waypoints[0], observed
/// with Gaussian noise of covariance obs_cov.
///
/// Only truth_waypoints[0] is used (pos, v, psi). The number of
/// observations is truth_waypoints.len(); consecutive legs in the
/// input are ignored in Stage 1. Per-observation noise is drawn
/// from Mat2::from_cov(obs_cov).sample(&mut rng).
///
/// TODO(Stage2): honour per-leg waypoints (piecewise straight or
/// curved) and consume turn-rate limits from ttc.
pub fn generate_track(
    truth_waypoints: &[KinState],
    ttc: &Ttc,
    dt_obs_s: f64,
    obs_cov: [f64; 3],
    seed: u64,
) -> Vec<Observation> {
    assert!(
        !truth_waypoints.is_empty(),
        "generate_track needs at least one waypoint"
    );
    assert!(dt_obs_s > 0.0, "dt_obs_s must be positive");

    let _ = ttc; // unused in Stage 1; kept for Stage 2 curves

    let start = &truth_waypoints[0];
    let n_legs = truth_waypoints.len() - 1;
    let mut rng = Xoshiro256ss::from_seed(seed);
    let mat = Mat2::from_cov(obs_cov);

    let mut observations = Vec::with_capacity(n_legs + 1);
    for i in 0..=n_legs {
        let t = i as f64 * dt_obs_s;
        let truth_pos = [
            start.pos[0] + start.v * t * start.psi.cos(),
            start.pos[1] + start.v * t * start.psi.sin(),
        ];
        let noise = mat.sample(&mut rng);
        observations.push(Observation {
            t,
            pos: [truth_pos[0] + noise.x, truth_pos[1] + noise.y],
            v: start.v,
            psi: start.psi,
            cov: obs_cov,
        });
    }
    observations
}
