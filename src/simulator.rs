//! simulator.rs — synthetic track generator.
//!
//! Straight-line truth generator (Stage 1) plus analytic curved-trajectory
//! generators (Spiral, Sine, Zigzag) for Stage 2. `generate_track` stays as
//! the legacy straight-line function; `generate_curved_track` is the new
//! analytic path used by the WASM `generate_*` methods and the CLI
//! `trajectory` directive.

use crate::prng::Xoshiro256ss;
use crate::types::{KinState, Observation, Ttc};
use crate::vec2::Mat2;

/// Curved (or straight) trajectory geometry, sampled analytically.
#[derive(Debug, Clone, Copy)]
pub enum Trajectory {
    /// Straight line at constant heading `psi0` (radians from +x).
    Straight { psi0: f64 },
    /// Archimedean spiral: radius grows linearly with angle.
    /// `r0` = starting radius, `r_growth` = d(r)/d(theta), `turns` = full turns.
    Spiral {
        r0: f64,
        r_growth: f64,
        turns: f64,
    },
    /// Sinusoidal path: amplitude perpendicular to `direction` (radians),
    /// `wavelength_m` along the baseline.
    Sine {
        wavelength_m: f64,
        amplitude_m: f64,
        direction: f64,
    },
    /// Zigzag: legs of length `segment_m`, each tilted `angle_rad` from +x
    /// (alternating +angle / -angle), offset by `baseline_m` from +x.
    Zigzag {
        segment_m: f64,
        angle_rad: f64,
        baseline_m: f64,
    },
}

/// Estimate the maximum path speed and lateral acceleration `a_lat = v^2 * kappa`
/// of an analytic trajectory over `[0, duration_s]`, sampled on a coarse grid.
///
/// Used by `generate_curved_track` to warn when the true path exceeds the
/// turn-table constraints (`v_max`, `a_lat`): a path that physically cannot be
/// traversed would make the filter cap its own particles and fall behind.
fn path_speed_and_a_lat(
    traj: &Trajectory,
    duration_s: f64,
    speed: f64,
) -> (f64, f64) {
    // theta grows linearly in t for the spiral, so sample uniformly across the
    // duration; the other trajectories are smooth enough for the same grid.
    let n = 128usize;
    let mut v_max = 0.0f64;
    let mut a_lat_max = 0.0f64;
    for i in 0..=n {
        let t = (i as f64 / n as f64) * duration_s;
        let (_pos, psi, v) = sample_point(traj, t, duration_s, speed);
        // Curvature kappa = |d(psi)/dt|; a_lat = v^2 * kappa.
        let dt = duration_s / n as f64;
        // Numerical derivative of psi over the grid cell.
        let (_pos_p, psi_p, _) = sample_point(traj, t + dt, duration_s, speed);
        let dpsi = (psi_p - psi).abs();
        let kappa = dpsi / dt.max(1e-9);
        let a_lat = v * v * kappa;
        v_max = v_max.max(v);
        a_lat_max = a_lat_max.max(a_lat);
    }
    (v_max, a_lat_max)
}

/// Generate a synthetic track for a curved (or straight) trajectory.
///
/// Each observation is sampled analytically at `t_k = k * dt_obs_s`
/// (`k = 0..=floor(duration / dt_obs`): position, heading, and speed are all
/// computed directly from the trajectory formula via `sample_point` — the
/// dense time grid and `propagate()` are never used. Per-observation Gaussian
/// noise of covariance `obs_cov` is drawn from
/// `Mat2::from_cov(obs_cov).sample(&mut rng)`.
///
/// `speed` is the forward speed used for `Straight` (arc length `s = speed*t`)
/// and for the Zigzag timing (`t_i = i * segment_m / speed`). For `Spiral`
/// the tangential speed is not constant; the time-uniform parametrization
/// (theta grows linearly in t) is used and `v` is derived analytically.
pub fn generate_curved_track(
    traj: &Trajectory,
    speed: f64,
    duration_s: f64,
    dt_obs_s: f64,
    obs_cov: [f64; 3],
    seed: u64,
    ttc: &Ttc,
) -> Vec<Observation> {
    assert!(dt_obs_s > 0.0, "dt_obs_s must be positive");

    // Validate the analytic path against the turn-table constraints: the path
    // speed must not exceed v_max and the lateral acceleration v^2 * kappa must
    // not exceed a_lat. Warn to stderr if either is violated (non-fatal).
    let (v_max_path, a_lat_max) = path_speed_and_a_lat(traj, duration_s, speed);
    if v_max_path > ttc.v_max || a_lat_max > ttc.a_lat {
        eprintln!(
            "WARNING: Trajectory exceeds physical limits (a_lat = {:.3}, v = {:.3})",
            a_lat_max, v_max_path
        );
    }

    let mut rng = Xoshiro256ss::from_seed(seed);
    let mat = Mat2::from_cov(obs_cov);

    // Observation times: t_k = k * dt_obs_s, k = 0..=floor(duration / dt_obs).
    let n_obs = (duration_s / dt_obs_s).floor() as usize;

    let mut observations = Vec::with_capacity(n_obs + 1);
    for k in 0..=n_obs {
        let t = k as f64 * dt_obs_s;
        // Sample analytically: position, heading, and speed are all computed
        // directly from the trajectory formula at t_k.
        let (pos, psi, v) = sample_point(traj, t, duration_s, speed);

        let noise = mat.sample(&mut rng);
        observations.push(Observation {
            t,
            pos: [pos[0] + noise.x, pos[1] + noise.y],
            v,
            psi,
            cov: obs_cov,
        });
    }
    observations
}

/// Sample one dense point (position, heading, speed) of the trajectory at time t.
pub fn sample_point(
    traj: &Trajectory,
    t: f64,
    duration_s: f64,
    speed: f64,
) -> ([f64; 2], f64, f64) {
    match traj {
        Trajectory::Straight { psi0 } => {
            let s = speed * t;
            let pos = [s * psi0.cos(), s * psi0.sin()];
            (pos, *psi0, speed)
        }
        Trajectory::Spiral {
            r0,
            r_growth,
            turns,
        } => {
            // Analytic derivatives (theta grows linearly in t).
            let omega_t = 2.0 * std::f64::consts::PI * *turns / duration_s; // d(theta)/dt
            let theta = omega_t * t;
            let r = r0 + r_growth * theta;
            let drdt = r_growth * omega_t;
            let dxdt = drdt * theta.cos() - r * omega_t * theta.sin();
            let dydt = drdt * theta.sin() + r * omega_t * theta.cos();
            let heading = dydt.atan2(dxdt);
            let sp = (dxdt * dxdt + dydt * dydt).sqrt();
            let pos = [r * theta.cos(), r * theta.sin()];
            (pos, heading, sp)
        }
        Trajectory::Sine {
            wavelength_m,
            amplitude_m,
            direction,
        } => {
            // Analytic derivatives. Distance along the baseline is `a = s`
            // (not `s * cos(direction)`), and `dadt = speed`.
            let s = speed * t;
            let a = s;
            let dadt = speed;
            let k = 2.0 * std::f64::consts::PI / wavelength_m;
            let ya = amplitude_m * (k * a).sin();
            let dyada = amplitude_m * k * (k * a).cos();
            let dyadt = dyada * dadt;
            
            let pos = [
                a * direction.cos() - ya * direction.sin(),
                a * direction.sin() + ya * direction.cos(),
            ];
            let dxdt = dadt * direction.cos() - dyadt * direction.sin();
            let dydt = dadt * direction.sin() + dyadt * direction.cos();
            let heading = dydt.atan2(dxdt);
            let sp = (dxdt * dxdt + dydt * dydt).sqrt();
            (pos, heading, sp)
        }
        Trajectory::Zigzag {
            segment_m,
            angle_rad,
            baseline_m,
        } => {
            let dx_leg = segment_m * angle_rad.cos();
            let h = segment_m * angle_rad.sin() / 2.0;
            // Time per segment.
            let per_seg = *segment_m / speed;
            // Vertex index i: the leg [i, i+1] containing t.
            let max_i = (duration_s / per_seg).floor() as usize;
            let i = (t / per_seg).floor() as usize;
            let i = i.min(max_i);
            let t0 = (i as f64) * per_seg;
            let t1 = (i + 1) as f64 * per_seg;
            let dt_seg = (t1 - t0).max(1e-9);
            let frac = ((t - t0) / dt_seg).min(1.0);
            // Vertex i and i+1 y-offsets alternate +h / -h.
            let y0 = baseline_m + if i % 2 == 0 { h } else { -h };
            let y1 = baseline_m + if (i + 1) % 2 == 0 { h } else { -h };
            let x0 = i as f64 * dx_leg;
            let x1 = (i + 1) as f64 * dx_leg;
            let pos = [x0 + (x1 - x0) * frac, y0 + (y1 - y0) * frac];
            // Heading along the leg: alternating -angle / +angle from +x.
            // Vertex i (even) is at y = baseline_m + h (above), vertex i+1
            // (odd) at y = baseline_m - h (below), so leg 0 descends => heading
            // is -angle_rad.
            let leg_angle = if i % 2 == 0 { -*angle_rad } else { *angle_rad };
            (pos, leg_angle, speed)
        }
    }
}

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
