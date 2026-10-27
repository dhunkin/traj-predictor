# Context for Reviewer
This is a Particle Filter for UAV trajectory prediction. We recently added Stage 3: Bayesian Goal Inference. 

**The Problem:**
When we simulate a drone flying a complex trajectory (e.g., a sine wave or a spiral) towards a target (OOI), the prediction fan completely ignores the true physical trajectory (the dashed line in the UI) and just snaps straight to the OOI. The physics says "continue the spiral", but the intent says "aim at the OOI". 
We are passing `oois` and an `intent_hypothesis_prob` of 0.8. Inside the prediction loop, we force the turn rate (`forced_omega`) to steer towards the goal. 

**Please review:**
1. **Goal Initialization:** Are we biasing the initial heading too much towards the OOI if the drone is actually flying a curved path?
2. **Goal Steering (Prediction Loop):** We use a proportional navigation `cmd_omega = (k_turn * dpsi).clamp(...)`. Is this too aggressive? How do we make it so the particle tries to reach the goal *while maintaining its current physical maneuver* (e.g., if it's spiraling, it spirals towards the goal, rather than flying straight to it)?
3. **Filter Penalties:** We apply a penalty `p.weight -= 2.0` if a particle moves away from its goal. Is this killing the physical turning particles and forcing the filter to believe the drone is flying straight at the OOI?
4. **Physics vs Intent:** How should we properly blend the Ornstein-Uhlenbeck motion model with the goal-directed intent so that a spiraling drone doesn't have its fan teleport to the OOI?

## File: src/particle.rs
```rust
//! particle.rs — unified Particle Filter for track estimation.
//!
//! This module is the main entry point for heading prediction: the public
//! `predict()` function here delegates the whole pipeline (filter the track,
//! then fan forward to the azimuth histogram). It was previously split across
//! the now-removed `mc` and `filter` modules, which have been consolidated
//! into this single particle-engine entry point.
//!
//! Carries a set of weighted hypotheses (particles) over the UAV's full
//! kinematic state `(x, y, v, psi, omega)` and learns the state from the
//! entire observation track, rather than from just the last observation as
//! the Stage 1 Monte-Carlo engine did.
//!
//! Pipeline:
//!   1. INITIALISE N particles from the first observation's position
//!      covariance, a uniform velocity in `[v_min, v_max]`, and a uniform
//!      heading in `[0, 2*pi)` (stored as log-weights).
//!   2. FILTER over `track[1..]`: propagate each particle with the
//!      Ornstein-Uhlenbeck turn-rate model (`motion::propagate_step`),
//!      reweight by the 2D Mahalanobis likelihood of the new observation,
//!      normalise in the log domain, and systematic-resample when the
//!      effective sample size drops.
//!   3. PREDICT by resampling once, then fanning every particle forward to
//!      `horizon_s` and histogramming the final heading into the azimuth bins.
//!
//! Dependency-free: all math is manual, using `crate::motion`,
//! `crate::histogram`, `crate::vec2`, `crate::prng`, `crate::types`.
//!
//! TODO(Stage2): (reserved)

use crate::histogram::{bin_index, sector_width};
use crate::motion::propagate_step;
use crate::prng::Xoshiro256ss;
use crate::types::{Observation, Ooi, PredictConfig, Prediction, Ttc};
use crate::vec2::Mat2;
use std::f64::consts::PI;

/// A single particle hypothesis over the full kinematic state.
#[derive(Clone, Copy, Debug)]
pub struct Particle {
    pub x: f64,
    pub y: f64,
    pub v: f64,
    pub psi: f64,
    pub omega: f64,
    pub is_turning: bool,
    /// Stage 3 goal inference: index into `oois` if this particle drew a goal
    /// hypothesis, else `None`. Goal-assigned particles steer their prediction
    /// fan toward `oois[goal]` (see the prediction loop).
    pub goal: Option<usize>,
    pub weight: f64,
}

/// Systematic resampling of the particle set.
///
///   1. Build the cumulative-weight array.
///   2. Draw a single random start `u0 ~ U[0,1)` and set the step `u0 / N`.
///   3. Advance one particle index across the cumulative weights while the
///      k-th stratified position `step + k/N` lies inside it.
///   4. Replace the set with clones of the chosen particles and reset every
///      weight to `1 / N`.
pub fn resample(particles: &mut Vec<Particle>, rng: &mut Xoshiro256ss) {
    let n = particles.len();
    if n == 0 {
        return;
    }

    // 1. Cumulative weights.
    let mut cum = Vec::with_capacity(n);
    let mut total = 0.0;
    for p in particles.iter() {
        total += p.weight;
        cum.push(total);
    }

    // 2. Random start and stratified step.
    let step = rng.next_f64() / n as f64;

    // 3. Pick particles by walking the cumulative-weight array once.
    let mut chosen = Vec::with_capacity(n);
    let mut idx = 0usize;
    for k in 0..n {
        let u = step + (k as f64) / n as f64;
        while idx < n - 1 && u > cum[idx] {
            idx += 1;
        }
        chosen.push(particles[idx]);
    }

    // 4. Replace and reset weights.
    for p in chosen.iter_mut() {
        p.weight = 1.0 / n as f64;
    }
    *particles = chosen;
}

/// Weighted circular mean of the particle headings `psi`, used to anchor the
/// prediction fan. The position-only filter leaves the heading under-determined,
/// so the post-filter particle spread can cover the full 360 degrees; the mean
/// gives a single robust heading that the fan is seeded from.
fn weighted_circular_mean(particles: &[Particle]) -> f64 {
    let mut sum_sin = 0.0;
    let mut sum_cos = 0.0;
    let mut total = 0.0;
    for p in particles.iter() {
        sum_sin += p.weight * p.psi.sin();
        sum_cos += p.weight * p.psi.cos();
        total += p.weight;
    }
    if total <= 0.0 {
        return particles[0].psi;
    }
    sum_sin.atan2(sum_cos)
}

/// Unified particle filter over the full track.
///
/// See the module docstring for the pipeline. `track` is the ordered list of
/// observations; the first observation seeds the position prior and each
/// subsequent observation drives a filter (propagate -> reweight -> resample)
/// step before the final prediction fan.
///
/// This is the public entry point for heading prediction.
pub fn predict(
    track: &[Observation],
    ttc: &Ttc,
    oois: &[Ooi],
    cfg: &PredictConfig,
) -> Prediction {
    let n_samples = cfg.n_samples as usize;
    let n_bins = cfg.azimuth_bins as usize;

    assert!(track.len() >= 2, "Needs at least 2 observations");
    assert!(n_samples >= 2, "Needs at least 2 particles");

    let mut rng = Xoshiro256ss::from_seed(cfg.seed);

    // ---- 1. INITIALIZATION -------------------------------------------------
    // Sample initial position from track[0]'s covariance; velocity uniform in
    // [v_min, v_max]; heading anchored near the track direction; turn rate
    // zero. Weights are stored in the log domain.
    let mut particles: Vec<Particle> = Vec::with_capacity(n_samples);
    let cov_factor = Mat2::from_cov(track[0].cov);
    // Base heading from the first two observations. Every particle starts near
    // this direction (Stage 3 fix: the old code started psi uniform over
    // [0, 2*pi), which let the fan point in random directions — e.g. a sine
    // track's true -45 deg heading — because nothing anchored the initial
    // heading).
    let dx = track[1].pos[0] - track[0].pos[0];
    let dy = track[1].pos[1] - track[0].pos[1];
    let base_psi = dy.atan2(dx);
    for _ in 0..n_samples {
        let noise = cov_factor.sample(&mut rng);
        let pos = [
            track[0].pos[0] + noise.x,
            track[0].pos[1] + noise.y,
        ];
        let v = ttc.v_min + (ttc.v_max - ttc.v_min) * rng.next_f64();
        // 1. ALWAYS start near the base heading (std ~0.3 rad ~ 17 deg), never
        //    uniform over the full circle.
        let mut goal = None;
        let mut psi = base_psi + rng.gauss(0.0, 0.3);
        // 2. Stage 3 goal inference: with probability
        //    `intent_hypothesis_prob` a particle draws a goal hypothesis from
        //    `oois` and its initial heading is biased toward that goal instead.
        //    If `oois` is empty (or the draw misses) the particle stays on the
        //    base heading — psi is never left uniform over the full circle.
        if !oois.is_empty() && rng.next_f64() < cfg.intent_hypothesis_prob {
            let g_idx = (rng.next_f64() * oois.len() as f64) as usize % oois.len();
            goal = Some(g_idx);
            // Bias the initial heading toward the goal. Only do so when the goal
            // is somewhat in front of the base heading (within +/- 90 deg); if
            // the goal is behind, keep psi near the base heading so the filter
            // doesn't start a particle pointing 180 deg from its true track.
            let gx = oois[g_idx].pos[0] - track[0].pos[0];
            let gy = oois[g_idx].pos[1] - track[0].pos[1];
            let goal_psi = gy.atan2(gx);
            let dpsi = ((goal_psi - base_psi + PI).rem_euclid(2.0 * PI)) - PI;
            if dpsi.abs() < PI / 2.0 {
                psi = goal_psi + rng.gauss(0.0, 0.3);
            } else {
                psi = base_psi + rng.gauss(0.0, 0.3);
            }
        }
        // 20% of particles start in coordinated-turn mode so the filter can
        // track sustained maneuvers (spirals / orbits) as well as decaying turns.
        let is_turning = rng.next_f64() < 0.2;
        // Turning particles are seeded with a range of turn rates (gauss, std
        // 0.3 rad/s) so the filter can reweight them toward the true curvature.
        // The physics holds each turning particle's omega with only tiny drift,
        // so the init spread is what lets the filter discover the right turn rate
        // — a single omega = 0 would make every turning particle fly straight and
        // die on a curved track. Non-turning particles start at omega = 0 (straight).
        let omega = if is_turning { rng.gauss(0.0, 0.3) } else { 0.0 };
        particles.push(Particle {
            x: pos[0],
            y: pos[1],
            v,
            psi,
            omega,
            is_turning,
            goal,
            weight: (1.0 / n_samples as f64).ln(),
        });
    }

    // ---- 2. FILTERING LOOP (track[1..]) ------------------------------------
    for i in 1..track.len() {
        let obs = &track[i];
        let dt = obs.t - track[i - 1].t;
        if dt <= 0.0 { continue; }

        // Propagate + reweight each particle against the new observation.
        for p in particles.iter_mut() {
            let (pos_new, psi_new, v_new, omega_new) = propagate_step(
                [p.x, p.y],
                p.v,
                p.psi,
                p.omega,
                dt,
                cfg.speed_noise_mps,
                0.05, // tighter motion noise: trust the radar more
                p.is_turning,
                false, // filter phase: keep the filtered rate with tiny drift
                None,  // filter phase: no forced rate — OU dynamics run as before
                &mut rng,
                ttc,
            );
            p.x = pos_new[0];
            p.y = pos_new[1];
            p.psi = psi_new;
            p.v = v_new;
            p.omega = omega_new;

            // 2D Mahalanobis likelihood (with noise floor), applied in the log
            // domain so weights accumulate without underflowing to 0. Turning and
            // non-turning particles share the SAME position covariance so neither
            // is unfairly favoured: a non-turning particle flying straight on a
            // straight track is not penalised, and a turning particle curving away
            // is not over-rewarded (STEP 1). This keeps the surviving turning
            // fraction near its ~20% init rate instead of saturating at 100%.
            let floor = 0.1; // Measurement noise floor to prevent underflow
            let a = obs.cov[0] + floor;
            let b = obs.cov[1];
            let c = obs.cov[2] + floor;
            let det = a * c - b * b;
            let dx = p.x - obs.pos[0];
            let dy = p.y - obs.pos[1];
            let mut d2 = (c * dx * dx - 2.0 * b * dx * dy + a * dy * dy) / det.max(1e-9);
            // Heading term: the filter must follow the observed heading
            // (obs.psi) so the fan points along the true track, fixing the sine
            // lag. A moderate head_sigma (0.3) for EVERY particle anchors psi
            // tightly enough to stop the fan scattering chaotically on curves,
            // but not so tight (0.1) that turning particles are killed in a
            // single step.
            let head_sigma = 0.3;
            // Signed heading difference in [-pi, pi]: ((a - b + pi) mod 2pi) - pi.
            let dpsi = ((p.psi - obs.psi + PI).rem_euclid(2.0 * PI)) - PI;
            d2 += dpsi * dpsi / (head_sigma * head_sigma);

            // Survival bonus for turning particles: on a curve the position term
            // of a turning particle is legitimately off the dashed line, so a
            // single tight step would kill every turning particle at once
            // (the "all-or-nothing" problem). Reducing the penalty slightly lets
            // the turning hypothesis survive across steps; the spread that
            // matters comes from the different filtered omega per particle, not
            // from reweighting the whole class in or out at once.
            if p.is_turning {
                d2 *= 0.9;
            }
            let log_likelihood = -0.5 * d2;
            p.weight += log_likelihood;
        }

        // Log-domain normalisation (log-sum-exp); rescue to uniform if the
        // track is lost (weights collapse to -inf).
        let max_lw = particles.iter().map(|p| p.weight).fold(f64::NEG_INFINITY, f64::max);
        if !max_lw.is_finite() {
            // Track lost, rescue
            for p in particles.iter_mut() { p.weight = 1.0 / n_samples as f64; }
        } else {
            let mut sum = 0.0;
            for p in particles.iter_mut() {
                p.weight = (p.weight - max_lw).exp();
                sum += p.weight;
            }
            for p in particles.iter_mut() { p.weight /= sum; }
        }

        // Effective sample size; resample when it drops below N/2.
        let sum_sq: f64 = particles.iter().map(|p| p.weight * p.weight).sum();
        let n_eff = 1.0 / sum_sq;
        if n_eff < n_samples as f64 / 2.0 {
            resample(&mut particles, &mut rng);
        }
    }

    // ---- 3. PREDICTION LOOP (fan generation) -------------------------------
    // Compute the estimated speed and turn rate from the post-filter particle
    // set. `v_est` is the weighted mean speed over all particles; `omega_est`
    // is the weighted mean turn rate over the coordinated-turn particles only
    // (those with is_turning == true), since the non-turning particles drift
    // back toward zero and would bias the turn-rate estimate.
    let mut v_sum = 0.0;
    let mut omega_turn_sum = 0.0;
    let mut omega_turn_count = 0.0;
    for p in particles.iter() {
        v_sum += p.weight * p.v;
        if p.is_turning {
            omega_turn_sum += p.weight * p.omega;
            omega_turn_count += p.weight;
        }
    }
    let v_est = if n_samples > 0 { v_sum / n_samples as f64 } else { 0.0 };
    let omega_est = if omega_turn_count > 0.0 {
        omega_turn_sum / omega_turn_count
    } else {
        0.0
    };

    // Resample once so every particle has equal weight 1/N before prediction,
    // then fan each particle forward to the horizon with the OU turn-rate
    // model, histogramming the final heading. State is copied to locals so the
    // stored particle is not mutated.
    //
    // Because the filter uses a position-only likelihood, the heading `psi` of
    // the post-filter particle set is only weakly constrained (a nearly-straight
    // track leaves psi under-determined). To keep the fan from spreading over
    // the full 360 degrees we anchor every particle's initial heading to the
    // weighted circular mean `psi_est`, while still letting each particle keep
    // its own omega so the fan curls inward for a turning / spiral track.
    let psi_est = weighted_circular_mean(&particles);
    resample(&mut particles, &mut rng);
    let mut counts = vec![0u64; n_bins];
    let bin_width = 2.0 * PI / n_bins as f64;
    let mut point_cloud = Vec::with_capacity(n_samples);
    let mut path_cloud = Vec::with_capacity(n_samples);
    let mut sum_sin = 0.0;
    let mut sum_cos = 0.0;

    for p in particles.iter() {
        let mut pos = [p.x, p.y];
        let mut v = p.v;
        let mut psi = psi_est;

        // Stage 3 goal inference: goal-assigned particles steer their fan toward
        // their OOI by forcing the turn rate each step so the arc curves toward
        // the goal. Non-goal particles keep the OU baseline (forced_omega = None).
        let goal_info: Option<(usize, [f64; 2])> = p.goal.map(|g| (g, oois[g].pos));

        let mut time = 0.0;
        let n_steps = (cfg.horizon_s / 0.5).ceil() as usize;
        // Save the starting position before the loop so we can bin the actual
        // direction of travel (displacement bearing) from here to the horizon end,
        // rather than the final nose heading `psi` (which, because particles steer
        // toward a goal, curves away from the true path). This makes the histogram
        // fan match the actual point_cloud.
        let start_pos = [pos[0], pos[1]];
        // Sample the full per-particle trajectory so the demo can draw the
        // ribbon as distinct curved lines instead of a single straight sector.
        let mut path: Vec<[f64; 2]> = Vec::with_capacity(n_steps + 1);
        path.push(pos);
        // Prediction-fan motion noise. Kept tight (0.05, same as the filter) so
        // the fan spreads moderately over the horizon instead of re-opening the
        // heading sector the way 0.2 did (straight s90 205 deg -> 55 deg).
        let fan_sigma = 0.05;
        // Turning particles hold their own post-filter turn rate so the fan curls
        // inward like the true spiral; non-turning particles use the mean rate
        // for a tight, straight-ahead fan.
        let mut omega = if p.is_turning {
            p.omega // own: each particle keeps its post-filter rate
        } else {
            omega_est
        };
        for _ in 0..n_steps {
            let dt_step = (cfg.horizon_s - time).min(0.5);
            if dt_step <= 0.0 { break; }
            // STEP 2.3: on a curve the turn costs speed (centripetal loss), so
            // turning particles ease off slightly each step. Non-turning
            // particles keep their full speed. Applied before propagate so the
            // reduced speed feeds the arc geometry.
            if p.is_turning {
                v *= 0.995;
            }
            // Goal-steering (Stage 3): for a goal-assigned particle, force the
            // turn rate each step so the arc curves toward its OOI. Compute the
            // heading from the particle's current position to the goal and turn
            // toward it (proportional navigation: remaining angle per step,
            // clamped to the turn-table limit inside propagate_step). Non-goal
            // particles pass None so the OU baseline runs exactly as before.
            let mut forced = None;
            if let Some((_g, gp)) = goal_info {
                let gx = gp[0] - pos[0];
                let gy = gp[1] - pos[1];
                let psi_target = gy.atan2(gx);
                let dpsi = ((psi_target - psi + PI).rem_euclid(2.0 * PI)) - PI;

                // Proportional navigation: turn towards goal, but respect max turn rate
                let k_turn = 1.5;
                let cmd_omega = (k_turn * dpsi).clamp(-ttc.omega_cap, ttc.omega_cap);
                forced = Some(cmd_omega);
            }
            let (pos_new, psi_new, v_new, omega_new) = propagate_step(
                pos,
                v,
                psi,
                omega,
                dt_step,
                cfg.speed_noise_mps,
                fan_sigma,
                p.is_turning,
                true, // prediction phase: hold omega exactly, no random walk
                forced,
                &mut rng,
                ttc,
            );
            pos = pos_new;
            psi = psi_new;
            v = v_new;
            omega = omega_new;
            time += dt_step;
            path.push(pos);
        }

        point_cloud.push(pos);
        path_cloud.push(path);
        // Bin the actual direction of travel: the bearing from the start point
        // to the horizon-end point, not the final nose heading `psi`. Because
        // particles steer toward a goal their paths curve, so the final psi can
        // differ from the displacement direction; binning the bearing keeps the
        // histogram fan aligned with the drawn point_cloud.
        let dx = pos[0] - start_pos[0];
        let dy = pos[1] - start_pos[1];
        let bearing = dy.atan2(dx);
        let idx = bin_index(bearing, n_bins);
        counts[idx] += 1;
        sum_sin += bearing.sin();
        sum_cos += bearing.cos();
    }

    // ---- 4. RETURN ---------------------------------------------------------
    let bin_probs: Vec<f64> =
        counts.iter().map(|&c| c as f64 / n_samples as f64).collect();
    let bin_centers: Vec<f64> = (0..n_bins)
        .map(|i| (i as f64 + 0.5) * bin_width)
        .collect();

    let sector50 = sector_width(&counts, n_samples as u64, 0.5, n_bins);
    let sector90 = sector_width(&counts, n_samples as u64, 0.9, n_bins);

    // Circular mean of the final headings (robust to 2*pi wrap).
    let final_heading_mean = sum_sin.atan2(sum_cos);

    Prediction {
        bin_centers,
        bin_probs,
        sector50,
        sector90,
        point_cloud,
        path_cloud,
        final_heading_mean,
        v_est,
        omega_est,
    }
}
```

## File: src/motion.rs
```rust
//! motion.rs — propagate: CV + rate-limited turn, analytic arc.
//!
//! Analytic constant-turn-rate propagation over a single time step `t`.
//! The input turn rate `omega` drives the arc and heading update as-is;
//! the returned `omega_new` is that rate clamped to `[-omega_max,
//! omega_max]` (computed per §4.3). `omega_new` is returned so callers and
//! tests can inspect the rate-limit clamp; the arc geometry is driven by the
//! un-clamped input `omega`.
//!
//! TODO(Stage2): (reserved)

use crate::prng::Xoshiro256ss;
use crate::types::Ttc;

/// Propagate a kinematic state over one analytic arc.
///
/// Returns `(pos_new, psi_new, v_new, omega_new)`:
///   - `v_new = clamp(v + gauss(0, speed_noise), v_min, v_max)`
///   - arc via the chord identity: `half = omega * t / 2`,
///     `sinc(x) = sin(x)/x` (Taylor `1 - x²/6` near 0),
///     `chord = v_new * t * sinc(half)`, `mid = psi + half`,
///     `pos_new = pos + chord * [cos(mid), sin(mid)]` — exact for all `omega`,
///     including the straight-line limit `omega -> 0`.
///   - `psi_new = (psi + omega * t).rem_euclid(2*pi)`
///   - `omega_new = clamp(omega, -omega_max, omega_max)`
pub fn propagate(
    pos: [f64; 2],
    v: f64,
    psi: f64,
    omega: f64,
    t: f64,
    speed_noise: f64,
    rng: &mut Xoshiro256ss,
    ttc: &Ttc,
) -> ([f64; 2], f64, f64, f64) {
    let omega_clamped = omega.clamp(-ttc.omega_cap, ttc.omega_cap);
    let v_new = (v + rng.gauss(0.0, speed_noise)).clamp(ttc.v_min, ttc.v_max);
    
    let half = 0.5 * omega_clamped * t;
    let sinc = if half.abs() < 1e-4 { 1.0 - half * half / 6.0 } else { half.sin() / half };
    let chord = v_new * t * sinc;
    let mid = psi + half;
    
    let pos_new = [
        pos[0] + chord * mid.cos(),
        pos[1] + chord * mid.sin(),
    ];
    
    let psi_new = (psi + omega_clamped * t).rem_euclid(2.0 * std::f64::consts::PI);
    
    (pos_new, psi_new, v_new, omega_clamped)
}

/// Propagate the state one step forward using a time-correlated turn-rate
/// (Ornstein-Uhlenbeck) model with speed-dependent lateral-acceleration limits.
///
/// `is_turning` selects between two turn-rate dynamics:
///   - `false` (normal): the OU process mean-reverts to zero (a coordinated
///     turn that decays), with stationary std-dev `turn_noise_sigma` and
///     mean-reversion time `tau`.
///   - `true` (coordinated turn): the turn-rate holds its current value with
///     only a tiny random-walk drift (in the filter phase), so a sustained
///     turn (spiral / orbit) can be tracked.
///
/// `predicting` selects how the turning particle's omega is handled:
///   - `false` (filter phase): the turning particle keeps its current omega
///     with only a tiny `0.01` random-walk drift, so the filtered rate is
///     preserved across steps.
///   - `true` (prediction phase): the turning particle holds omega EXACTLY
///     (`omega_new = omega`, no random walk). The fan spread comes from
///     the different filtered omega each particle carries, not from adding
///     fresh noise that would scramble the arc into a chaotic scribble.
///
/// `forced_omega` overrides the turn-rate update entirely:
///   - `None` -> the turn-rate update runs exactly as above (OU mean-revert to
///     zero for non-turning particles; held/weak-drift for turning particles).
///   - `Some(w)` -> `omega_new = w` is used verbatim (clamped to the
///     speed-dependent limit afterwards), so the particle is steered along a
///     prescribed arc. Used by Stage 3 goal inference to aim a goal-assigned
///     particle's fan at its OOI.
///
/// Returns `(pos_new, psi_new, v_new, omega_new)`:
///   1. `v` is perturbed by longitudinal Gaussian noise (scaled by `sqrt(dt)`)
///      and clamped to `[v_min, v_max]`.
///   2. `omega` is updated (OU process, or held when turning + predicting),
///      then clamped to a speed-dependent limit `min(a_lat / v_new, omega_cap)`.
///   3. `psi` and `pos` are updated via the chord/sinc identity.
///
/// `turn_noise_sigma` is the OU stationary std-dev of `omega`; `speed_noise`
/// is the per-second longitudinal noise std-dev (applied per unit time).
pub fn propagate_step(
    pos: [f64; 2],
    v: f64,
    psi: f64,
    omega: f64,
    dt: f64,
    speed_noise: f64,
    turn_noise_sigma: f64,
    is_turning: bool,
    predicting: bool,
    forced_omega: Option<f64>,
    rng: &mut Xoshiro256ss,
    ttc: &Ttc,
) -> ([f64; 2], f64, f64, f64) {
    // 1. Update speed (longitudinal)
    let v_noise = rng.gauss(0.0, speed_noise) * dt.sqrt();
    let v_new = (v + v_noise).clamp(ttc.v_min, ttc.v_max);

    // 2. Update omega. If the caller supplied a forced rate, use it verbatim
    //    (clamped below) and skip the OU / held dynamics entirely — this is the
    //    Stage 3 goal-steering path. Otherwise the OU process runs exactly as
    //    before, so the baseline is preserved when `forced_omega` is `None`.
    let mut omega_new;
    if let Some(w) = forced_omega {
        omega_new = w;
    } else if is_turning {
        if predicting {
            // Prediction phase: hold omega exactly, no random walk.
            omega_new = omega;
        } else {
            // Filter phase: keep the filtered rate with a tiny 0.01
            // random-walk drift so the omega spread survives across steps.
            let drift = rng.gauss(0.0, 0.01);
            omega_new = omega + drift;
        }
    } else {
        let tau = 2.0;
        let alpha = (-dt / tau).exp();
        let noise = (1.0 - alpha * alpha).sqrt() * turn_noise_sigma * rng.gauss(0.0, 1.0);
        omega_new = alpha * omega + noise;
    }

    // 3. Clamp omega using speed-dependent limit (uses updated speed)
    let max_turn = ttc.a_lat / v_new.max(0.1);
    let limit = max_turn.min(ttc.omega_cap);
    omega_new = omega_new.clamp(-limit, limit);

    // 4. Update heading and position (chord identity)
    let half = 0.5 * omega_new * dt;
    let sinc = if half.abs() < 1e-4 { 1.0 - half * half / 6.0 } else { half.sin() / half };
    let chord = v_new * dt * sinc;
    let mid = psi + half;

    let pos_new = [
        pos[0] + chord * mid.cos(),
        pos[1] + chord * mid.sin(),
    ];
    let psi_new = (psi + omega_new * dt).rem_euclid(2.0 * std::f64::consts::PI);

    (pos_new, psi_new, v_new, omega_new)
}
```

## File: src/types.rs
```rust
//! types.rs — KinState, Observation, Ttc, Ooi, config, result.
//! TODO(Stage2): (reserved)

// ---- Placeholder items for Step 1 (filled in later steps) ----

#[derive(Debug, Clone, Copy)]
pub struct Ttc {
    pub v_min: f64,
    pub v_max: f64,
    pub a_lat: f64,      // max lateral acceleration (m/s^2), e.g., 9.0
    pub a_long: f64,     // max longitudinal accel/decel (m/s^2), e.g., 3.0
    pub omega_cap: f64,  // absolute max turn rate (rad/s), e.g., 1.0
}

#[derive(Debug, Clone, Copy)]
pub struct Observation {
    pub t: f64,           // seconds from scenario start
    pub pos: [f64; 2],
    pub v: f64,           // m/s, last speed (Stage 1 MC uses as-is)
    pub psi: f64,         // rad, last heading (Stage 1 MC uses as-is)
    pub cov: [f64; 3],    // [xx, xy, yy]
}

pub struct Ooi {
    pub pos: [f64; 2],
}

pub struct KinState {
    pub pos: [f64; 2],
    pub v: f64,           // m/s
    pub psi: f64,         // rad, 0 = +X
}

pub struct PredictConfig {
    pub horizon_s: f64,
    pub n_samples: u32,
    pub dt_s: f64,
    pub p_turn: f64,
    pub azimuth_bins: u32,
    pub speed_noise_mps: f64,
    /// Stage 3: probability a particle draws a goal hypothesis from `oois` at
    /// initialization. 0 disables goal inference (every particle stays on the
    /// OU baseline); ~0.8 keeps most particles anchored to a goal so the fan
    /// narrows toward it.
    pub intent_hypothesis_prob: f64,
    pub seed: u64,
}

pub struct Prediction {
    pub bin_centers: Vec<f64>,
    pub bin_probs: Vec<f64>,
    pub sector50: f64,
    pub sector90: f64,
    pub point_cloud: Vec<[f64; 2]>,
    /// Per-particle trajectory polylines (one `Vec<[f64;2]>` per particle), each
    /// sampled every prediction step from the post-filter position to the horizon
    /// end. `point_cloud` only keeps final positions; `path_cloud` retains the
    /// full arc so the demo can draw the ribbon as distinct curved lines.
    pub path_cloud: Vec<Vec<[f64; 2]>>,
    pub final_heading_mean: f64,
    pub v_est: f64,
    pub omega_est: f64,
}
```

## File: web/demo.js
```javascript
// web/demo.js — Stage 1 client/server heading-prediction demo.
//
// Maps the predictor core (Rust backend) onto a MapLibre GL JS map with a
// canvas overlay ("fan") that draws the track, point cloud, histogram fan,
// and 50%/90% sector contours. The prediction and simulation logic now runs
// in the Rust server over HTTP/JSON; the browser talks to it via fetch.
//
// Anchoring: scenario-local metres are converted to lng/lat relative to an
// arbitrary anchor point, then map.project() turns lng/lat into screen px.

const API_BASE = `${window.location.protocol}//${window.location.hostname}:8081`;

// ---- Anchoring: arbitrary origin of the scenario-local metre grid ----------
const ANCHOR_LNG = -79.4;
const ANCHOR_LAT = 43.7;
const METER_PER_DEG_LAT = 111320.0;

function metersToLngLat(x, y) {
  const lat = ANCHOR_LAT + y / METER_PER_DEG_LAT;
  const lng = ANCHOR_LNG + x / (METER_PER_DEG_LAT * Math.cos(ANCHOR_LAT * Math.PI / 180));
  return [lng, lat];
}
// Scenario-local metres (x, y) -> screen pixel [px, py].
function metersToScreen(x, y) {
  const p = map.project(metersToLngLat(x, y));
  return [p.x, p.y];
}

// ---- Map setup -------------------------------------------------------------
const map = new maplibregl.Map({
  container: 'map',
  style: {
    version: 8,
    sources: {
      osm: {
        type: 'raster',
        tiles: ['https://tile.openstreetmap.org/{z}/{x}/{y}.png'],
        tileSize: 256,
        attribution: '© OpenStreetMap contributors',
      },
    },
    layers: [{ id: 'osm', type: 'raster', source: 'osm' }],
  },
  center: [ANCHOR_LNG, ANCHOR_LAT],
  zoom: 18,
});

// ---- Zoom controls (FIX 1) -----------------------------------------------
map.addControl(new maplibregl.NavigationControl(), 'top-left');

// ---- Canvas overlay --------------------------------------------------------
const fan = document.getElementById('fan');
const ctx = fan.getContext('2d');

function resizeCanvas() {
  const c = map.getCanvas();
  const rect = c.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  fan.style.left = rect.left + 'px';
  fan.style.top = rect.top + 'px';
  fan.width = Math.round(rect.width * dpr);
  fan.height = Math.round(rect.height * dpr);
  fan.style.width = rect.width + 'px';
  fan.style.height = rect.height + 'px';
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
}

// ---- UI refs ---------------------------------------------------------------
const el = (id) => document.getElementById(id);
const inputs = {
  seed: 'seed', v_min: 'v_min', v_max: 'v_max', a_lat: 'a_lat', a_long: 'a_long', omega_cap: 'omega_cap',
  horizon_s: 'horizon_s', duration_s: 'duration_s', n_samples: 'n_samples',
  p_turn: 'p_turn', speed_noise_mps: 'speed_noise_mps', azimuth_bins: 'azimuth_bins', sim_speed: 'sim_speed',
  intent_hypothesis_prob: 'intent_hypothesis_prob', // <-- ADD THIS
};
function val(id) { return parseFloat(el(inputs[id]).value); }
function status(msg) { el('status').textContent = msg; }

// ---- State -----------------------------------------------------------------
let observations = []; // {t, x, y, cov_xx, cov_xy, cov_yy, v, psi}
let true_future = []; // the analytic truth past duration_s (for the dashed line)
let pred = null;
// OOIs (observable objective intent states) rendered + sent to the backend.
// Updated by predict(); consumed by draw() to render the red circles.
let oois = [];
// Last observation, hoisted to module scope so predict() can use it
// regardless of handler scope (fixes "last is not defined" on Predict).
let lastObservation = null;

// ---- HTTP helpers ----------------------------------------------------------
async function apiPost(path, payload) {
  const res = await fetch(API_BASE + path, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  if (!res.ok) {
    let detail = '';
    try { detail = await res.text(); } catch (_) { /* ignore */ }
    throw new Error('HTTP ' + res.status + (detail ? ': ' + detail : ''));
  }
  return res.json();
}

// ---- Rendering -------------------------------------------------------------
function draw() {
  ctx.clearRect(0, 0, fan.width, fan.height);
  const showTrack = el('show_track').checked;
  const showCloud = el('show_cloud').checked;
  const showRibbon = el('show_paths').checked;

  // Track polyline + observation dots.
  if (showTrack && observations.length) {
    const pts = observations.map((o) => metersToScreen(o.x, o.y));
    ctx.strokeStyle = '#1a5276';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(pts[0][0], pts[0][1]);
    for (let i = 1; i < pts.length; i++) ctx.lineTo(pts[i][0], pts[i][1]);
    ctx.stroke();
    ctx.fillStyle = '#154360';
    for (const p of pts) {
      ctx.beginPath();
      ctx.arc(p[0], p[1], 3, 0, Math.PI * 2);
      ctx.fill();
    }
  }

  if (!lastObservation) return;
  const last = lastObservation;

  // OOIs: render the goal points the backend used as large red circles so we
  // can see where the prediction fan is being steered toward.
  if (oois && oois.length > 0) {
    for (const o of oois) {
      const [px, py] = metersToScreen(o.x, o.y);
      ctx.beginPath();
      ctx.fillStyle = 'red';
      ctx.arc(px, py, 8, 0, Math.PI * 2);
      ctx.fill();
      ctx.fillStyle = 'white';
      ctx.font = '12px sans-serif';
      ctx.fillText('OOI', px + 10, py + 5);
    }
  }

  // True Future: the analytic truth past duration_s, drawn as a dashed black
  // line from the last observation to the end of truth. Drawn first so the fan
  // (drawn below) overlays it.
  if (true_future.length) {
    const pts = true_future.map((o) => metersToScreen(o.x, o.y));
    // Start exactly at the last observation's screen position.
    const start = metersToScreen(last.x, last.y);
    ctx.save();
    ctx.setLineDash([5, 5]);
    ctx.strokeStyle = '#000';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(start[0], start[1]);
    for (let i = 0; i < pts.length; i++) ctx.lineTo(pts[i][0], pts[i][1]);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.restore();
  }

  // Point cloud.
  if (showCloud && pred) {
    const cloud = pred.point_cloud;
    ctx.fillStyle = 'rgba(90,90,90,0.5)';
    for (let i = 0; i + 1 < cloud.length; i += 2) {
      const [px, py] = metersToScreen(cloud[i], cloud[i + 1]);
      ctx.beginPath();
      ctx.arc(px, py, 1.5, 0, Math.PI * 2);
      ctx.fill();
    }
  }

  // Trajectory ribbon: draw a subset of per-particle polylines (every Nth
  // particle) with higher alpha so the distinct curved lines are visible
  // instead of a solid cone. Drawn below the fan so the fan colors show.
  if (showRibbon && pred && pred.path_cloud) {
    drawPathRibbon(pred.path_cloud, showCloud);
  }

  if (!pred) return;

  // Apex = last observation's screen position.
  const [ax, ay] = metersToScreen(last.x, last.y);

  // Fan: filled thin sectors, one per non-zero bin.
  const binCenters = pred.bin_centers;
  const binProbs = pred.bin_probs;
  const azimuthBins = val('azimuth_bins');
  const binWidth = (2 * Math.PI) / azimuthBins;
  const vMean = (val('v_min') + val('v_max')) / 2;
  const horizon = val('horizon_s');
  const length = horizon * vMean;

  let maxP = 0;
  for (const p of binProbs) if (p > maxP) maxP = p;

  // FIX 4a: skip low-prob bins so the fan tail is cut, not a full disc.
  const REL_THRESHOLD = 0.05;

  ctx.save();
  ctx.lineWidth = 1;
  for (let i = 0; i < binCenters.length; i++) {
    const p = binProbs[i];
    if (p <= 0) continue;
    if (p < REL_THRESHOLD * maxP) continue;
    const a = binCenters[i];
    const a0 = a - binWidth / 2;
    const a1 = a + binWidth / 2;
    // Screen y is flipped vs. scenario y, so negate sin.
    const [x0, y0] = metersToScreen(
      last.x + length * Math.cos(a0), last.y + length * Math.sin(a0));
    const [x1, y1] = metersToScreen(
      last.x + length * Math.cos(a1), last.y + length * Math.sin(a1));

    const norm = maxP > 0 ? p / maxP : 0;
    const hue = 240 * (1 - norm); // blue (low) -> red (high)
    // FIX 4b: steepen alpha so low-prob bins fade toward 0.15.
    const alpha = 0.15 + 0.75 * Math.pow(norm, 1.5);
    ctx.beginPath();
    ctx.moveTo(ax, ay);
    ctx.lineTo(x0, y0);
    ctx.lineTo(x1, y1);
    ctx.closePath();
    ctx.fillStyle = `hsla(${hue}, 90%, 55%, ${alpha})`;
    ctx.fill();
    ctx.strokeStyle = `hsla(${hue}, 90%, 40%, ${alpha})`;
    ctx.stroke();
  }
  ctx.restore();

  // Sector contours: 50% and 90% as solid bright radial lines.
  const mean = pred.final_heading_mean;
  drawSectorContour(ax, ay, mean, pred.sector50, '#00ff00', last);
  drawSectorContour(ax, ay, mean, pred.sector90, '#ff00ff', last);
}

function drawSectorContour(ax, ay, mean, width, color, last) {
  const vMean = (val('v_min') + val('v_max')) / 2;
  const horizon = val('horizon_s');
  const length = horizon * vMean;
  const half = width / 2;
  const [x0, y0] = metersToScreen(
    last.x + length * Math.cos(mean - half),
    last.y + length * Math.sin(mean - half));
  const [x1, y1] = metersToScreen(
    last.x + length * Math.cos(mean + half),
    last.y + length * Math.sin(mean + half));
  ctx.strokeStyle = color;
  ctx.lineWidth = 3;
  ctx.beginPath();
  ctx.moveTo(ax, ay);
  ctx.lineTo(x0, y0);
  ctx.moveTo(ax, ay);
  ctx.lineTo(x1, y1);
  ctx.stroke();
}

// Draw the trajectory ribbon: a subset of per-particle polylines so the
// distinct curved lines show instead of a solid cone. Every Nth particle is
// drawn with a higher alpha so the lines stand out against the point cloud.
function drawPathRibbon(path_cloud, showCloud) {
  if (!path_cloud || path_cloud.length === 0) return;
  // Draw every Nth particle; fewer lines for large particle counts.
  const n = path_cloud.length;
  const step = n > 2000 ? 100 : n > 1000 ? 50 : 25;
  ctx.save();
  ctx.lineWidth = 1.5;
  ctx.strokeStyle = 'rgba(0, 0, 0, 0.3)';
  for (let i = 0; i < n; i += step) {
    const path = path_cloud[i];
    if (!path || path.length < 2) continue;
    ctx.beginPath();
    for (let j = 0; j < path.length; j++) {
      const [px, py] = metersToScreen(path[j][0], path[j][1]);
      if (j === 0) ctx.moveTo(px, py);
      else ctx.lineTo(px, py);
    }
    ctx.stroke();
  }
  ctx.restore();
}

// ---- Trajectory parameter UI -----------------------------------------------
// Build number inputs for the selected trajectory's parameters.
const TRAJ_DEFS = {
  straight: { fields: [
    { id: 'traj-psi0', label: 'psi0', value: 0 },
  ]},
  spiral: { fields: [
    { id: 'traj-r0', label: 'r0', value: 10 },
    { id: 'traj-r-growth', label: 'r_growth', value: 1 },
    { id: 'traj-turns', label: 'turns', value: 3 },
  ]},
  sine: { fields: [
    { id: 'traj-wavelength', label: 'wavelength', value: 100 },
    { id: 'traj-amplitude', label: 'amplitude', value: 20 },
    { id: 'traj-direction', label: 'direction', value: 0 },
  ]},
  zigzag: { fields: [
    { id: 'traj-segment', label: 'segment', value: 30 },
    { id: 'traj-angle', label: 'angle', value: 0.5 },
    { id: 'traj-baseline', label: 'baseline', value: 0 },
  ]},
};

function populateTrajParams() {
  const type = el('traj-type').value;
  const def = TRAJ_DEFS[type];
  const container = el('traj-params');
  container.innerHTML = '';
  if (!def) return;
  for (const f of def.fields) {
    const row = document.createElement('div');
    row.className = 'row';
    const label = document.createElement('label');
    label.setAttribute('for', f.id);
    label.textContent = f.label;
    const input = document.createElement('input');
    input.id = f.id;
    input.type = 'number';
    input.value = f.value;
    input.step = 'any';
    row.appendChild(label);
    row.appendChild(input);
    container.appendChild(row);
  }
}

// ---- Actions ---------------------------------------------------------------
async function simulate() {
  await W(async () => {
    const seed = BigInt(val('seed') | 0);
    const speed = val('sim_speed');
    const duration = val('duration_s');
    const horizon = val('horizon_s');
    const dt = 1.0;
    const cov_xx = 1.0;
    const cov_xy = 0.0;
    const cov_yy = 1.0;
    const trajType = el('traj-type').value;

    // Read trajectory parameters by id from the UI (only the ones for the
    // selected type exist; read all but tolerate missing inputs).
    const gp = (id) => { const n = el(id); return n ? parseFloat(n.value) : 0; };
    // Send only the params relevant to the selected trajectory type.
    // (The backend reads params[0..3] per type; a flat 10-element array
    // would misalign sine/zigzag params.)
    let params;
    switch (trajType) {
      case 'straight':
        params = [gp('traj-psi0')];
        break;
      case 'spiral':
        params = [gp('traj-r0'), gp('traj-r-growth'), gp('traj-turns')];
        break;
      case 'sine':
        params = [gp('traj-wavelength'), gp('traj-amplitude'), gp('traj-direction')];
        break;
      case 'zigzag':
        params = [gp('traj-segment'), gp('traj-angle'), gp('traj-baseline')];
        break;
      default:
        params = [0];
    }

    // POST to the server and parse the returned observations.
    const res = await apiPost('/api/simulate', {
      trajectory_type: trajType,
      params,
      speed,
      duration_s: duration,
      dt_obs_s: dt,
      cov_xx, cov_xy, cov_yy,
      seed: Number(seed),
      horizon_s: horizon,
      ttc: {
        v_min: val('v_min'),
        v_max: val('v_max'),
        a_lat: val('a_lat'),
        a_long: val('a_long'),
        omega_cap: val('omega_cap'),
      },
    });

    // Split the returned observations: the first part (t <= duration_s) is the
    // observed track fed to the filter; the remainder is the analytic truth
    // past duration_s, drawn as the dashed "True Future" line.
    const dur = duration;
    observations = res.observations
      .filter((o) => o.t <= dur)
      .map((o) => ({ t: o.t, x: o.x, y: o.y, v: o.v, psi: o.psi, cov_xx: o.cov_xx, cov_xy: o.cov_xy, cov_yy: o.cov_yy }));
    true_future = res.observations
      .filter((o) => o.t > dur)
      .map((o) => ({ t: o.t, x: o.x, y: o.y, v: o.v, psi: o.psi, cov_xx: o.cov_xx, cov_xy: o.cov_xy, cov_yy: o.cov_yy }));
    lastObservation = observations[observations.length - 1];
    pred = null;
    draw();
    status('track simulated: ' + observations.length + ' observations');

    // FIX 2: fit map to track + fan extents after Simulate.
    if (map.isStyleLoaded() && observations.length) {
      const vMean = (val('v_min') + val('v_max')) / 2;
      const fanLen = val('horizon_s') * vMean;
      const last = lastObservation;
      const pts = observations.map(o => metersToLngLat(o.x, o.y));
      pts.push(
        metersToLngLat(last.x + fanLen, last.y),
        metersToLngLat(last.x - fanLen, last.y),
        metersToLngLat(last.x, last.y + fanLen),
        metersToLngLat(last.x, last.y - fanLen),
      );
      const bounds = new maplibregl.LngLatBounds();
      pts.forEach(p => bounds.extend(p));
      map.fitBounds(bounds, { padding: 80, duration: 800 });
    }
  });
}

async function predict() {
  await W(async () => {
    if (!lastObservation) {
      console.warn('Predict pressed before Simulate; ignoring');
      status('simulate first');
      return;
    }
    const last = lastObservation;
    const azimuthBins = Math.round(val('azimuth_bins'));
    const t0 = performance.now();

    // Read OOIs safely so a missing element can't crash Predict.
    const ooiX = el('ooi_x') ? Number(el('ooi_x').value) : 0;
    const ooiY = el('ooi_y') ? Number(el('ooi_y').value) : 0;
    const currentOois = [{ x: ooiX, y: ooiY }];

    // Stage 2: send the full observations array as `track`. The server runs
    // the Kalman filter over it and feeds the smoothed state to predict().
    // `last_observation` is kept as the last point for backward compat.
    const res = await apiPost('/api/predict', {
      ttc: {
        v_min: val('v_min'),
        v_max: val('v_max'),
        a_lat: val('a_lat'),
        a_long: val('a_long'),
        omega_cap: val('omega_cap'),
      },
      track: observations.map((o) => ({
        t: o.t, x: o.x, y: o.y, v: o.v, psi: o.psi,
        cov_xx: o.cov_xx, cov_xy: o.cov_xy, cov_yy: o.cov_yy,
      })),
      last_observation: {
        t: last.t,
        x: last.x,
        y: last.y,
        v: last.v,
        psi: last.psi,
        cov_xx: last.cov_xx,
        cov_xy: last.cov_xy,
        cov_yy: last.cov_yy,
      },
      oois: currentOois,
      config: {
        horizon_s: val('horizon_s'),
        n_samples: val('n_samples'),
        dt_s: 1.0,
        p_turn: val('p_turn'),
        azimuth_bins: azimuthBins,
        speed_noise_mps: val('speed_noise_mps'),
        intent_hypothesis_prob: val('intent_hypothesis_prob'),
        seed: Number(val('seed') | 0),
      },
    });

    pred = {
      bin_centers: res.bin_centers,
      bin_probs: res.bin_probs,
      point_cloud: res.point_cloud,
      path_cloud: res.path_cloud, // [[ [x,y], ... ], ...] one polyline per particle
      final_heading_mean: res.heading_mean,
      sector50: res.sector50,
      sector90: res.sector90,
    };

    // Keep the OOIs the backend used so draw() can render them.
    oois = currentOois;

    // Stats (radians -> degrees).
    el('s50').textContent = (pred.sector50 * 180 / Math.PI).toFixed(1);
    el('s90').textContent = (pred.sector90 * 180 / Math.PI).toFixed(1);
    el('smean').textContent = (pred.final_heading_mean * 180 / Math.PI).toFixed(1);
    el('selapsed').textContent = Math.round(performance.now() - t0);
    // FIX 4c: n_bins_nonzero stat.
    const nz = pred.bin_probs.reduce((a, p) => a + (p > 0 ? 1 : 0), 0);
    el('snz').textContent = nz;

    draw();
    status('prediction complete');
  });
}

// ---- Wire up ---------------------------------------------------------------
let buttonsWired = false;
function wireButtons() {
  if (buttonsWired) return;
  buttonsWired = true;
  resizeCanvas();
  map.on('move', () => { resizeCanvas(); draw(); });
  map.on('zoom', () => { resizeCanvas(); draw(); });
  map.on('moveend', () => { resizeCanvas(); draw(); });
  window.addEventListener('resize', () => { resizeCanvas(); draw(); });
  el('btn_sim').addEventListener('click', simulate);
  el('btn_predict').addEventListener('click', predict);
  // Populate trajectory parameter inputs on load and when the selector changes.
  populateTrajParams();
  el('traj-type').addEventListener('change', populateTrajParams);
  status('Ready.');
}
map.on('load', wireButtons);
// Fallback: if the 'load' event doesn't fire (e.g., tiles not loading in a
// headless browser), wire up the buttons after a short delay.
setTimeout(wireButtons, 2000);

// Global error wrapper for the async action handlers.
async function W(fn) {
  try { return await fn(); }
  catch (e) { status('error: ' + (e && e.message ? e.message : e)); throw e; }
}
```
