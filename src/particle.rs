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
