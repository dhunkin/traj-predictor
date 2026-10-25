//! particle.rs — unified Particle Filter for track estimation.
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
use crate::types::{Observation, PredictConfig, Prediction, Ttc};
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

/// Unified particle filter over the full track.
///
/// See the module docstring for the pipeline. `track` is the ordered list of
/// observations; the first observation seeds the position prior and each
/// subsequent observation drives a filter (propagate -> reweight -> resample)
/// step before the final prediction fan.
pub fn run_particle_filter(
    track: &[Observation],
    ttc: &Ttc,
    cfg: &PredictConfig,
) -> Prediction {
    let n_samples = cfg.n_samples as usize;
    let n_bins = cfg.azimuth_bins as usize;

    assert!(track.len() >= 2, "Needs at least 2 observations");
    assert!(n_samples >= 2, "Needs at least 2 particles");

    let mut rng = Xoshiro256ss::from_seed(cfg.seed);

    // ---- 1. INITIALIZATION -------------------------------------------------
    // Sample initial position from track[0]'s covariance; velocity uniform in
    // [v_min, v_max]; heading uniform in [0, 2*pi); turn rate zero. Weights are
    // stored in the log domain.
    let mut particles: Vec<Particle> = Vec::with_capacity(n_samples);
    let cov_factor = Mat2::from_cov(track[0].cov);
    for _ in 0..n_samples {
        let noise = cov_factor.sample(&mut rng);
        let pos = [
            track[0].pos[0] + noise.x,
            track[0].pos[1] + noise.y,
        ];
        let v = ttc.v_min + (ttc.v_max - ttc.v_min) * rng.next_f64();
        let psi = 2.0 * PI * rng.next_f64();
        particles.push(Particle {
            x: pos[0],
            y: pos[1],
            v,
            psi,
            omega: 0.0,
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
                0.2,
                &mut rng,
                ttc,
            );
            p.x = pos_new[0];
            p.y = pos_new[1];
            p.psi = psi_new;
            p.v = v_new;
            p.omega = omega_new;

            // 2D Mahalanobis likelihood (with noise floor), applied in the log
            // domain so weights accumulate without underflowing to 0.
            let floor = 0.5; // Measurement noise floor to prevent underflow
            let a = obs.cov[0] + floor;
            let b = obs.cov[1];
            let c = obs.cov[2] + floor;
            let det = a * c - b * b;
            let dx = p.x - obs.pos[0];
            let dy = p.y - obs.pos[1];
            let d2 = (c * dx * dx - 2.0 * b * dx * dy + a * dy * dy) / det.max(1e-9);
            let log_likelihood = -0.5 * d2;
            p.weight = p.weight.ln() + log_likelihood;
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
    // Resample once so every particle has equal weight 1/N before prediction,
    // then fan each particle forward to the horizon with the OU turn-rate
    // model, histogramming the final heading. State is copied to locals so the
    // stored particle is not mutated.
    resample(&mut particles, &mut rng);
    let mut counts = vec![0u64; n_bins];
    let mut point_cloud = Vec::with_capacity(n_samples);
    let bin_width = 2.0 * PI / n_bins as f64;
    let mut sum_sin = 0.0;
    let mut sum_cos = 0.0;

    for p in particles.iter() {
        let mut pos = [p.x, p.y];
        let mut v = p.v;
        let mut psi = p.psi;
        let mut omega = p.omega;

        let mut time = 0.0;
        let n_steps = (cfg.horizon_s / 0.5).ceil() as usize;
        for _ in 0..n_steps {
            let dt_step = (cfg.horizon_s - time).min(0.5);
            if dt_step <= 0.0 { break; }
            let (pos_new, psi_new, v_new, omega_new) = propagate_step(
                pos,
                v,
                psi,
                omega,
                dt_step,
                cfg.speed_noise_mps,
                0.2,
                &mut rng,
                ttc,
            );
            pos = pos_new;
            psi = psi_new;
            v = v_new;
            omega = omega_new;
            time += dt_step;
        }

        point_cloud.push(pos);
        let idx = bin_index(psi, n_bins);
        counts[idx] += 1;
        sum_sin += psi.sin();
        sum_cos += psi.cos();
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
        final_heading_mean,
    }
}
