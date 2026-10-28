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
    /// Archimedean spiral `r = r0 + r_growth * theta`, flown at constant
    /// ground speed `speed`. After `turns` full turns the UAV keeps orbiting
    /// at the final radius.
    Spiral {
        r0: f64,
        r_growth: f64,
        turns: f64,
    },
    /// Sinusoidal path: amplitude perpendicular to `direction` (radians),
    /// `wavelength_m` along the baseline, flown at constant ground speed
    /// `speed` (so it slows along the baseline where the path is steep).
    Sine {
        wavelength_m: f64,
        amplitude_m: f64,
        direction: f64,
    },
    /// Orbit: circle of `radius_m` flown at turn rate `omega` (rad/s, + = CCW)
    /// whose centre drifts at `drift_mps` along `drift_dir` (radians). Starts
    /// at the origin heading +x (circle centre at `[0, radius_m]` for CCW).
    /// `drift_mps = 0` is a loiter in place.
    Orbit {
        radius_m: f64,
        omega: f64,
        drift_mps: f64,
        drift_dir: f64,
    },
    /// Zigzag: legs of length `segment_m`, each tilted `angle_rad` from +x
    /// (alternating +angle / -angle), offset by `baseline_m` from +x.
    Zigzag {
        segment_m: f64,
        angle_rad: f64,
        baseline_m: f64,
    },
}

/// Check an analytic trajectory against the turn-table limits over
/// `[0, duration_s]`. Returns one human-readable warning per violated limit
/// (empty when the path is flyable).
///
/// A path the model is told is impossible (faster than `v_max`, tighter than
/// `a_lat` or `omega_cap`) makes the filter cap its own particles and fall
/// behind the track, so callers should surface these warnings.
pub fn check_limits(traj: &Trajectory, duration_s: f64, speed: f64, ttc: &Ttc) -> Vec<String> {
    let n = 800usize;
    let dt = duration_s / n as f64;
    let (mut v_max, mut a_lat_max, mut w_max) = (0.0f64, 0.0f64, 0.0f64);
    for i in 0..n {
        let t = i as f64 * dt;
        let (_, psi, v) = sample_point(traj, t, duration_s, speed);
        let (_, psi_n, _) = sample_point(traj, t + dt, duration_s, speed);
        // Turn rate from the wrapped heading change; a_lat = v * omega.
        let dpsi = (psi_n - psi + std::f64::consts::PI).rem_euclid(2.0 * std::f64::consts::PI)
            - std::f64::consts::PI;
        let w = dpsi.abs() / dt.max(1e-9);
        v_max = v_max.max(v);
        w_max = w_max.max(w);
        a_lat_max = a_lat_max.max(v * w);
    }
    let mut out = Vec::new();
    if v_max > ttc.v_max + 1e-6 {
        out.push(format!("speed reaches {:.1} m/s > v_max {:.1}", v_max, ttc.v_max));
    }
    if a_lat_max > ttc.a_lat + 1e-6 {
        out.push(format!("lateral acceleration reaches {:.1} m/s² > a_lat {:.1}", a_lat_max, ttc.a_lat));
    }
    if w_max > ttc.omega_cap + 1e-6 {
        out.push(format!("turn rate reaches {:.2} rad/s > omega_cap {:.2}", w_max, ttc.omega_cap));
    }
    out
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

    // Limits are checked by `check_limits` (callers surface the warnings).
    let _ = ttc;

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
            // Constant ground speed: integrate d(theta)/ds = 1 / sqrt(r^2 +
            // r_growth^2) along the arc to the travelled distance s = speed*t.
            let theta_end = 2.0 * std::f64::consts::PI * turns.max(0.0);
            let s_target = speed * t;
            let (mut theta, mut s) = (0.0f64, 0.0f64);
            let h = 1e-3;
            while theta < theta_end {
                let r = r0 + r_growth * theta;
                let ds = (r * r + r_growth * r_growth).sqrt() * h;
                if s + ds > s_target {
                    theta += h * (s_target - s) / ds;
                    s = s_target;
                    break;
                }
                s += ds;
                theta += h;
            }
            let theta = theta.min(theta_end);
            let r = r0 + r_growth * theta;
            // Past the last turn: keep orbiting at the final radius.
            let theta = if s < s_target { theta + (s_target - s) / r.max(1e-9) } else { theta };
            let growing = s >= s_target && theta < theta_end;
            let dr = if growing { *r_growth } else { 0.0 };
            // Tangent direction of d/dtheta [r cos, r sin].
            let tx = dr * theta.cos() - r * theta.sin();
            let ty = dr * theta.sin() + r * theta.cos();
            let pos = [r * theta.cos(), r * theta.sin()];
            (pos, ty.atan2(tx), speed)
        }
        Trajectory::Sine {
            wavelength_m,
            amplitude_m,
            direction,
        } => {
            // Constant ground speed: find the baseline distance `a` whose arc
            // length is s = speed * t (da/ds = 1 / sqrt(1 + y'(a)^2)).
            let k = 2.0 * std::f64::consts::PI / wavelength_m;
            let slope = |a: f64| amplitude_m * k * (k * a).cos();
            let s_target = speed * t;
            let (mut a, mut s) = (0.0f64, 0.0f64);
            let h = 0.05;
            while s < s_target {
                let ds = (1.0 + slope(a + 0.5 * h).powi(2)).sqrt() * h;
                if s + ds > s_target {
                    a += h * (s_target - s) / ds;
                    break;
                }
                s += ds;
                a += h;
            }
            let ya = amplitude_m * (k * a).sin();
            let pos = [
                a * direction.cos() - ya * direction.sin(),
                a * direction.sin() + ya * direction.cos(),
            ];
            let heading = direction + slope(a).atan();
            (pos, heading, speed)
        }
        Trajectory::Orbit {
            radius_m,
            omega,
            drift_mps,
            drift_dir,
        } => {
            let sign = if *omega < 0.0 { -1.0 } else { 1.0 };
            // Angle of the UAV around the centre; -pi/2 (CCW) puts it below the
            // centre, i.e. at the origin heading +x.
            let th = omega * t - sign * std::f64::consts::FRAC_PI_2;
            let (dcx, dcy) = (drift_mps * drift_dir.cos(), drift_mps * drift_dir.sin());
            let pos = [
                dcx * t + radius_m * th.cos(),
                sign * radius_m + dcy * t + radius_m * th.sin(),
            ];
            let dxdt = dcx - radius_m * omega * th.sin();
            let dydt = dcy + radius_m * omega * th.cos();
            (pos, dydt.atan2(dxdt), (dxdt * dxdt + dydt * dydt).sqrt())
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
