# Context for Reviewer
This is the core math for a UAV trajectory prediction engine, written in dependency-free Rust. It predicts future positions and headings of a drone based on noisy radar observations.

The core component is a Particle Filter (`particle.rs`) that:
1. Initializes N particles around the first observation.
2. Runs a filtering loop (predict -> weight by likelihood -> systematic resampling if N_eff < N/2).
3. Runs a prediction loop (propagate each particle forward to the horizon without resampling).
4. Bins the final headings into a histogram to find 50% and 90% confidence sectors.

The motion model (`motion.rs`) uses an Ornstein-Uhlenbeck process for turn rate and speed-dependent lateral acceleration limits.

Please review for:
1. **Particle Filter correctness**: Are the weights calculated correctly? Is the systematic resampling algorithm correct (no off-by-one errors)? Is the N_eff calculation correct?
2. **State mutation**: In the prediction phase, do we correctly copy the particle's state so we don't corrupt the filtered state?
3. **Numerical stability**: Are there divisions by zero (e.g., in likelihood if variance is 0), or underflow issues (exp of large negative numbers)? Is the circular mean calculated correctly?
4. **Physics math**: Is the propagate_step (Ornstein-Uhlenbeck + chord approximation) implemented correctly?

---

## File: src/particle.rs

```rust
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
//!      heading in `[0, 2*pi)`.
//!   2. FILTER over `track[1..]`: propagate each particle with the
//!      Ornstein-Uhlenbeck turn-rate model (`motion::propagate_step`),
//!      reweight by distance to the new observation, normalise, and
//!      systematic-resample when the effective sample size drops.
//!   3. PREDICT by fanning every particle forward to `horizon_s` and
//!      histogramming the final heading into the azimuth bins.
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

    let mut rng = Xoshiro256ss::from_seed(cfg.seed);
    eprintln!("PF_DEBUG seed={} track_len={}", cfg.seed, track.len());

    // ---- 1. INITIALIZATION -------------------------------------------------
    // Sample initial position from track[0]'s covariance; velocity uniform in
    // [v_min, v_max]; heading uniform in [0, 2*pi); turn rate zero.
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
            weight: 1.0 / n_samples as f64,
        });
    }

    eprintln!("PF_DEBUG init p0={:?} p1={:?}", particles[0], particles[1]);
    // ---- 2. FILTERING LOOP (track[1..]) ------------------------------------
    eprintln!("PF_DEBUG filter_steps={}", track.len() - 1);
    for i in 1..track.len() {
        let obs = &track[i];
        let dt = obs.t - track[i - 1].t;
        eprintln!("PF_DEBUG step i={} dt={}", i, dt);

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

            // Reweight by the (1-D) Gaussian likelihood of the position, using
            // obs.cov[0] as the variance (clamped to 1e-10 if zero).
            let d2 = (p.x - obs.pos[0]).powi(2) + (p.y - obs.pos[1]).powi(2);
            let var = obs.cov[0].max(1e-10);
            p.weight *= (-0.5 * d2 / var).exp();
        }

        // Normalise weights; rescue to uniform if the total collapsed to 0.
        let sum: f64 = particles.iter().map(|p| p.weight).sum();
        if sum == 0.0 {
            for p in particles.iter_mut() {
                p.weight = 1.0 / n_samples as f64;
            }
        } else {
            for p in particles.iter_mut() {
                p.weight /= sum;
            }
        }

        // Effective sample size; resample when it drops below N/2.
        let sum_sq: f64 = particles.iter().map(|p| p.weight * p.weight).sum();
        let n_eff = 1.0 / sum_sq;
        if n_eff < n_samples as f64 / 2.0 {
            resample(&mut particles, &mut rng);
        }
    }

    eprintln!("PF_DEBUG after_filter p0={:?} p1={:?}", particles[0], particles[1]);
    // ---- 3. PREDICTION LOOP (fan generation) -------------------------------
    // Fan each particle forward to the horizon with the OU turn-rate model,
    // histogramming the final heading. State is copied to locals so the
    // stored particle is not mutated.
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
        let dt_step = 0.5;
        while time < cfg.horizon_s {
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

    eprintln!("PF_DEBUG n_samples={} n_bins={} counts_sum={} nz={}", n_samples, n_bins, counts.iter().map(|&c| c as u64).sum::<u64>(), {let mut z=0; for c in counts.iter(){if *c>0 {z+=1;}}; z});
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
/// Returns `(pos_new, psi_new, v_new, omega_new)`:
///   1. `omega` is updated via an OU process (mean-reversion time `tau = 2.0s`),
///      then clamped to a speed-dependent limit `min(a_lat / v, omega_cap)`.
///   2. `v` is perturbed by longitudinal Gaussian noise and clamped to
///      `[v_min, v_max]`.
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
    rng: &mut Xoshiro256ss,
    ttc: &Ttc,
) -> ([f64; 2], f64, f64, f64) {
    // 1. Update omega using Ornstein-Uhlenbeck process (tau = 2.0s)
    let tau = 2.0;
    let alpha = (-dt / tau).exp();
    let noise = (1.0 - alpha * alpha).sqrt() * turn_noise_sigma * rng.gauss(0.0, 1.0);
    let mut omega_new = alpha * omega + noise;

    // 2. Clamp omega using speed-dependent limit
    let max_turn = ttc.a_lat / v.max(0.1);
    let limit = max_turn.min(ttc.omega_cap);
    omega_new = omega_new.clamp(-limit, limit);

    // 3. Update speed (longitudinal)
    let v_noise = rng.gauss(0.0, speed_noise) * dt;
    let v_new = (v + v_noise).clamp(ttc.v_min, ttc.v_max);

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

## File: src/vec2.rs

```rust
//! vec2.rs — Vec2, Mat2, cholesky_sample.
//!
//! A tiny 2D vector/matrix algebra core, dependency-free, no `unsafe`.
//! Mat2 wraps a lower-triangular Cholesky factor of a 2x2 covariance
//! matrix `[xx, xy, yy]`; `sample` applies it to two standard normals.
//!
//! TODO(Stage2): (reserved)

use crate::prng::Xoshiro256ss;

/// 2D vector.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vec2 {
    pub x: f64,
    pub y: f64,
}

impl Vec2 {
    pub fn new(x: f64, y: f64) -> Self {
        Vec2 { x, y }
    }

    pub fn add(self, o: Vec2) -> Vec2 {
        Vec2 {
            x: self.x + o.x,
            y: self.y + o.y,
        }
    }

    pub fn sub(self, o: Vec2) -> Vec2 {
        Vec2 {
            x: self.x - o.x,
            y: self.y - o.y,
        }
    }

    pub fn scale(self, s: f64) -> Vec2 {
        Vec2 {
            x: self.x * s,
            y: self.y * s,
        }
    }

    pub fn dot(self, o: Vec2) -> f64 {
        self.x * o.x + self.y * o.y
    }

    pub fn len(self) -> f64 {
        (self.dot(self)).sqrt()
    }

    /// Rotate the vector by `angle` radians (counterclockwise).
    pub fn rotate(self, angle: f64) -> Vec2 {
        let (s, c) = angle.sin_cos();
        Vec2 {
            x: self.x * c - self.y * s,
            y: self.x * s + self.y * c,
        }
    }
}

/// Lower-triangular Cholesky factor of a 2x2 covariance matrix.
///
/// Built from `[xx, xy, yy]` (variances + covariance) via the
/// correlation-parametrized decomposition:
///   sx = sqrt(xx), sy = sqrt(yy)
///   rho = clamp(xy / (sx * sy), -1, 1)
///   L11 = sx, L21 = rho * sy, L22 = sy * sqrt(1 - rho^2)
///
/// Degenerate handling: if either standard deviation is zero the factor is
/// diagonal (`L11 = sx`, `L22 = sy`, `L21 = 0`).
#[derive(Debug, Clone, Copy)]
pub struct Mat2 {
    pub l11: f64,
    pub l21: f64,
    pub l22: f64,
}

impl Mat2 {
    /// Build a Cholesky factor from `[xx, xy, yy]`.
    pub fn from_cov(cov: [f64; 3]) -> Self {
        let sx = cov[0].max(0.0).sqrt();
        let sy = cov[2].max(0.0).sqrt();
        if sx == 0.0 || sy == 0.0 {
            return Mat2 { l11: sx, l21: 0.0, l22: sy };
        }
        let rho = (cov[1] / (sx * sy)).clamp(-1.0, 1.0);
        Mat2 { l11: sx, l21: rho * sy, l22: sy * (1.0 - rho * rho).sqrt() }
    }

    /// Apply the factor to a standard-normal pair: `L * [a, b]`.
    pub fn apply(self, a: f64, b: f64) -> Vec2 {
        Vec2 {
            x: self.l11 * a,
            y: self.l21 * a + self.l22 * b,
        }
    }

    /// Sample `L * [gauss(0,1), gauss(0,1)]` using the given RNG.
    pub fn sample(&self, rng: &mut Xoshiro256ss) -> Vec2 {
        let a = rng.gauss(0.0, 1.0);
        let b = rng.gauss(0.0, 1.0);
        self.apply(a, b)
    }
}
```

## File: src/histogram.rs

```rust
//! histogram.rs — azimuth bins, 50/90% sector extraction.
//!
//! Azimuth histogram built from a set of final headings (radians), and the
//! extraction of the smallest circular sector whose cumulative density reaches
//! a given threshold. See §4.5.
//!
//! TODO(Stage2): (reserved)

use std::f64::consts::PI;

/// Reduce an angle to `[0, 2*pi)` (Euclidean/positive remainder).
pub fn psi_rem_euclid(angle: f64) -> f64 {
    let twopi = 2.0 * PI;
    let r = angle.rem_euclid(twopi);
    if r >= twopi { 0.0 } else { r }
}

/// Azimuth bin index for a heading, given the number of azimuth bins.
///
/// `floor(psi_rem_euclid(2*pi) / (2*pi / N)) % N`.
pub fn bin_index(psi: f64, n_bins: usize) -> usize {
    if n_bins == 0 {
        return 0;
    }
    let bin_width = 2.0 * PI / n_bins as f64;
    let rem = psi_rem_euclid(psi);
    let idx = (rem / bin_width) as usize;
    idx % n_bins
}

/// Smallest circular sector (in radians) whose cumulative density reaches
/// `threshold`, over `n_bins` azimuth bins.
///
/// `counts` holds the per-bin integer counts; `n_samples` is the total number
/// of samples. The density array is treated cyclically (doubled), so a sector
/// may wrap around the 0/2*pi seam.
///
/// Algorithm (§4.5): build the doubled counts array, then slide a window of
/// width k (in bins) starting at 0 with an incrementally maintained sum. Find
/// the smallest k for which some window of width k has sum >= need, where
/// `need = ceil(threshold * n_samples - 1e-9)`. Return width = k * (2*pi /
/// n_bins).
///
/// Discrete approximation: the returned width is an integer multiple of one
/// bin width (2*pi / n_bins). Integer counts avoid float drift in the density
/// threshold.
pub fn sector_width(counts: &[u64], n_samples: u64, threshold: f64, n_bins: usize) -> f64 {
    if n_bins == 0 || n_samples == 0 { return 0.0; }
    assert!(counts.len() == n_bins, "counts length must match n_bins");
    
    let bin_width = 2.0 * PI / n_bins as f64;
    let need = ((threshold * n_samples as f64) - 1e-9).ceil() as u64;

    let mut doubled = Vec::with_capacity(2 * n_bins);
    doubled.extend_from_slice(counts);
    doubled.extend_from_slice(counts);

    for k in 1..=n_bins {
        let mut sum: u64 = doubled[0..k].iter().sum();
        if sum >= need { return k as f64 * bin_width; }
        for start in 1..n_bins {
            sum = sum + doubled[start + k - 1] - doubled[start - 1];
            if sum >= need { return k as f64 * bin_width; }
        }
    }
    n_bins as f64 * bin_width
}
```

## File: src/prng.rs

```rust
//! prng.rs — xoshiro256**, splitmix64, gaussian.
//!
//! Deterministic pseudo-random generation with no OS randomness, no
//! `unsafe`, and no external crates. All RNG state lives in the struct.
//!
//! TODO(Stage2): (reserved)

/// splitmix64: expand a single u64 seed into a stream of u64 words.
fn splitmix64(seed: &mut u64) -> u64 {
    *seed = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *seed;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// xoshiro256** — 4-word state, standard rotation-based transition.
#[derive(Debug, Clone)]
pub struct Xoshiro256ss {
    state: [u64; 4],
    /// Cached second Marsaglia polar sample (raw standard normal `v*factor`).
    gauss_cache: Option<f64>,
}

impl Xoshiro256ss {
    /// Build an RNG from a single u64 seed, expanded via splitmix64.
    pub fn from_seed(seed: u64) -> Self {
        let mut s = seed;
        let mut state = [0u64; 4];
        for word in &mut state {
            *word = splitmix64(&mut s);
        }
        Xoshiro256ss {
            state,
            gauss_cache: None,
        }
    }

    /// Next raw u64 from the standard xoshiro256** transition.
    pub fn next_u64(&mut self) -> u64 {
        let rotl = |x: u64, k: u32| (x << k) | (x >> (64 - k));
        let result = rotl(self.state[1].wrapping_mul(5), 7).wrapping_mul(9);
        let t = self.state[1] << 17;
        self.state[2] ^= self.state[0];
        self.state[3] ^= self.state[1];
        self.state[1] ^= self.state[2];
        self.state[0] ^= self.state[3];
        self.state[2] ^= t;
        self.state[3] = rotl(self.state[3], 45);
        result
    }

    /// Uniform f64 in [0, 1): top 53 bits of next_u64() / 2^53.
    pub fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / 9007_1992_5474_0992.0
    }

    /// Gaussian sample via the Marsaglia polar method.
    ///
    /// The second of the two polar samples is cached in `Option<f64>` as a
    /// raw standard normal (`v * factor`); the caller's `mean`/`std` are
    /// applied (`mean + std * z`) on every return path, never baked into the
    /// cached value.
    pub fn gauss(&mut self, mean: f64, std: f64) -> f64 {
        let z = if let Some(z) = self.gauss_cache.take() {
            z
        } else {
            let mut u;
            let mut v;
            let mut s;
            loop {
                u = 2.0 * self.next_f64() - 1.0;
                v = 2.0 * self.next_f64() - 1.0;
                s = u * u + v * v;
                if s > 0.0 && s < 1.0 {
                    break;
                }
            }
            let factor = (-2.0 * s.ln() / s).sqrt();
            self.gauss_cache = Some(v * factor);
            u * factor
        };
        mean + std * z
    }
}
```
