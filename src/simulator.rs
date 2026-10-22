//! simulator.rs — synthetic track generator.
//!
//! Generates a straight-line truth track from `truth_waypoints` and returns a
//! vector of `Observation`s sampled at `dt_obs_s`. Truth motion is
//! constant-velocity (omega = 0 for all legs); the per-observation `cov` field
//! carries `obs_cov` (the observation-noise covariance applied by the
//! predictor). Truth integration uses `motion::propagate` so the track and the
//! Monte-Carlo predictor share one motion model.
//!
//! TODO(Stage2): curved truth paths.

use crate::motion::propagate;
use crate::prng::Xoshiro256ss;
use crate::types::{KinState, Observation, Ttc};

/// Generate a straight-line truth track.
///
/// # Arguments
/// * `truth_waypoints` — constant-velocity legs; leg `i` runs from waypoint `i`
///   to waypoint `i+1` for `t_obs` seconds.
/// * `ttc` — speed/turn-rate limits (passed to `propagate`).
/// * `dt_obs_s` — observation interval; also the per-leg duration.
/// * `obs_cov` — `[xx, xy, yy]` observation-noise covariance, copied into each
///   returned observation's `cov` field.
/// * `seed` — RNG seed for the truth integration.
///
/// Returns `n_obs + 1` observations where `n_obs = (waypoints.len() - 1)`.
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

    let start = truth_waypoints[0].clone();
    let t_obs = dt_obs_s;
    let total_s = (truth_waypoints.len() - 1) as f64 * t_obs;
    let n_obs = (total_s / t_obs) as u32;

    let mut rng = Xoshiro256ss::from_seed(seed);

    let mut observations = Vec::with_capacity(n_obs as usize + 1);
    let mut prev = start.clone();

    for i in 0..=n_obs {
        let t_obs_i = i as f64 * t_obs;
        let pos = if i == 0 {
            start.pos
        } else {
            // Straight-line leg: omega = 0, speed_noise = 0. `propagate` still
            // consumes one RNG draw per leg (gauss(0, 0)), so the RNG path is
            // exercised and the track stays deterministic for a given seed.
            let (pos_new, _psi, _v, _o) = propagate(
                prev.pos,
                prev.v,
                prev.psi,
                0.0,
                t_obs,
                0.0,
                &mut rng,
                ttc,
            );
            pos_new
        };

        observations.push(Observation {
            t: t_obs_i,
            pos,
            cov: obs_cov,
        });

        // Advance to the current state for the next leg (constant v, psi).
        prev = KinState {
            pos,
            v: prev.v,
            psi: prev.psi,
        };
    }

    observations
}
