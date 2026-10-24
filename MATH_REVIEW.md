# Context for Reviewer
This is a Rust library for predicting UAV trajectories using
Monte Carlo methods. It uses no external math libraries
(dependencies-free). It implements:
- xoshiro256** PRNG and Marsaglia polar Gaussian sampling.
- 2x2 Cholesky decomposition for covariance sampling.
- Analytic constant-turn-rate arc propagation.
- Azimuth histogramming and minimal cyclic sector extraction.
- (If present) a 4x4 Constant Velocity Kalman Filter.
Please review the math for correctness, numerical stability,
and edge cases (e.g., division by zero, negative variances).

---

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
/// Built from `[xx, xy, yy]` (variances + covariance) as:
///   L11 = sqrt(xx)
///   L21 = xy / L11
///   L22 = sqrt(yy - L21^2)
///
/// Degenerate handling:
///   - xx == 0 && yy == 0 -> zero noise (identity-free zero factor).
///   - xx == 0 && yy > 0   -> swap axes so the non-zero variance lands on
///     the diagonal (documented choice; see below).
#[derive(Debug, Clone, Copy)]
pub struct Mat2 {
    pub l11: f64,
    pub l21: f64,
    pub l22: f64,
}

impl Mat2 {
    /// Build a Cholesky factor from `[xx, xy, yy]`.
    pub fn from_cov(cov: [f64; 3]) -> Self {
        let (xx, xy, yy) = (cov[0], cov[1], cov[2]);
        debug_assert!(xx >= 0.0, "xx must be a variance, got {xx}");
        debug_assert!(yy >= 0.0, "yy must be a variance, got {yy}");
        debug_assert!(
            xy * xy <= xx * yy + 1e-12,
            "covariance is not PSD: xy^2 = {}, xx*yy = {}",
            xy * xy,
            xx * yy
        );
        if xx == 0.0 && yy == 0.0 {
            // Degenerate: zero noise.
            return Mat2 {
                l11: 0.0,
                l21: 0.0,
                l22: 0.0,
            };
        }
        if xx == 0.0 {
            // xx == 0, yy > 0: swap axes. The variance now lives on the
            // y-axis; L22 = sqrt(yy), off-diagonal zero.
            return Mat2 {
                l11: 0.0,
                l21: 0.0,
                l22: yy.sqrt(),
            };
        }
        let l11 = xx.sqrt();
        let l21 = xy / l11;
        let l22 = (yy - l21 * l21).max(0.0).sqrt();
        Mat2 {
            l11,
            l21,
            l22,
        }
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
    let mut rem = angle % twopi;
    if rem < 0.0 {
        rem += twopi;
    }
    rem
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
/// `counts` holds the per-bin counts; `n_samples` is the total number of
/// samples; density[i] = counts[i] / n_samples. The density array is treated
/// cyclically (doubled), so a sector may wrap around the 0/2*pi seam.
///
/// Algorithm (§4.5): build the doubled density array, then slide a window of
/// width k (in bins) starting at 0 with an incrementally maintained sum. Find
/// the smallest k for which some window of width k has sum >= threshold. Return
/// width = k * (2*pi / n_bins).
///
/// Discrete approximation: the returned width is an integer multiple of one
/// bin width (2*pi / n_bins).
pub fn sector_width(
    counts: &[f64],
    n_samples: usize,
    threshold: f64,
    n_bins: usize,
) -> f64 {
    if n_bins == 0 || n_samples == 0 {
        return 0.0;
    }
    let bin_width = 2.0 * PI / n_bins as f64;

    // Density array p[0..N].
    let p: Vec<f64> = counts.iter().map(|&c| c / n_samples as f64).collect();

    // Doubled array p[0..2N] for cyclic windows.
    let mut doubled: Vec<f64> = Vec::with_capacity(2 * n_bins);
    doubled.extend_from_slice(&p);
    doubled.extend_from_slice(&p);

    // Smallest window width k (in bins) with some cyclic window summing to
    // >= threshold.
    for k in 1..=n_bins {
        // Sliding window of width k, starting at 0, sum maintained incrementally.
        let mut sum: f64 = doubled[0..k].iter().sum::<f64>();
        if sum >= threshold {
            return k as f64 * bin_width;
        }
        for start in 1..n_bins {
            sum += doubled[start + k - 1] - doubled[start - 1];
            if sum >= threshold {
                return k as f64 * bin_width;
            }
        }
    }

    // Fallback: full circle.
    n_bins as f64 * bin_width
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
///   - `psi_new = psi + omega * t`
///   - straight line if `|omega| < 1e-9`, else the analytic constant-turn arc
///   - `v_new = clamp(v + gauss(0, speed_noise), v_min, v_max)`
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
    let omega = omega.clamp(-ttc.omega_max, ttc.omega_max);
    let v_new = (v + rng.gauss(0.0, speed_noise))
        .clamp(ttc.v_min, ttc.v_max);
    let psi_new = psi + omega * t;
    let pos_new = if omega.abs() < 1e-9 {
        [
            pos[0] + v_new * t * psi.cos(),
            pos[1] + v_new * t * psi.sin(),
        ]
    } else {
        let ratio = v_new / omega;
        [
            pos[0] + ratio * (psi_new.sin() - psi.sin()),
            pos[1] + ratio * (-psi_new.cos() + psi.cos()),
        ]
    };
    (pos_new, psi_new, v_new, omega)
}
```

## File: src/simulator.rs

```rust
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
    /// (alternating +angle / -angle), offset by `baseline` from +x.
    Zigzag {
        segment_m: f64,
        angle_rad: f64,
        baseline: f64,
    },
}

/// Generate a synthetic track for a curved (or straight) trajectory.
///
/// The path is sampled analytically at a dense time grid
/// (`dt_obs_s / 20`) so per-observation position, speed, and heading are
/// computed directly from the trajectory formula — `propagate()` is never
/// called. Per-observation Gaussian noise of covariance `obs_cov` is drawn
/// from `Mat2::from_cov(obs_cov).sample(&mut rng)`.
///
/// `speed` is the forward speed used for `Straight` (arc length `s = speed*t`)
/// and for the Zigzag timing (`t_i = i * segment_m / speed`). For `Spiral`
/// the tangential speed is not constant; the time-uniform parametrization
/// (theta grows linearly in t) is used and `v` is derived from finite
/// differences of consecutive dense points.
pub fn generate_curved_track(
    traj: &Trajectory,
    speed: f64,
    duration_s: f64,
    dt_obs_s: f64,
    obs_cov: [f64; 3],
    seed: u64,
) -> Vec<Observation> {
    assert!(dt_obs_s > 0.0, "dt_obs_s must be positive");

    // Dense sampling: 20 samples per observation interval.
    let dt_dense = dt_obs_s / 20.0;
    let n_dense = (duration_s / dt_dense).floor() as usize;

    // Analytic dense positions + per-point heading and speed.
    let mut dense_pos: Vec<[f64; 2]> = Vec::with_capacity(n_dense + 1);
    let mut dense_heading: Vec<f64> = Vec::with_capacity(n_dense + 1);
    let mut dense_speed: Vec<f64> = Vec::with_capacity(n_dense + 1);

    // Compute dense points.
    for i in 0..=n_dense {
        let t = i as f64 * dt_dense;
        let (pos, heading, sp) = sample_point(traj, t, duration_s, speed);
        dense_pos.push(pos);
        dense_heading.push(heading);
        dense_speed.push(sp);
    }

    let mut rng = Xoshiro256ss::from_seed(seed);
    let mat = Mat2::from_cov(obs_cov);

    // Observation times: t_k = k * dt_obs_s, k = 0..=floor(duration / dt_obs).
    let n_obs = (duration_s / dt_obs_s).floor() as usize;

    let mut observations = Vec::with_capacity(n_obs + 1);
    for k in 0..=n_obs {
        let t = k as f64 * dt_obs_s;
        // Nearest dense sample to t_k.
        let idx = (t / dt_dense).round() as usize;
        let idx = idx.min(n_dense);
        let pos = dense_pos[idx];

        // Heading psi = atan2(pos[k] - pos[k-1], ...) for k >= 1; for k = 0
        // use the dense heading at t = 0.
        let psi = if k == 0 {
            dense_heading[0]
        } else {
            let prev = dense_pos[idx - 1];
            (pos[1] - prev[1]).atan2(pos[0] - prev[0])
        };

        let v = dense_speed[idx];

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
fn sample_point(
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
            // Analytic derivatives.
            let s = speed * t;
            let a = s * direction.cos();
            let dadt = speed * direction.cos();
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
            baseline,
        } => {
            let dx_leg = segment_m * angle_rad.cos();
            let h = segment_m * angle_rad.sin() / 2.0;
            // Time at vertex i: t_i = i * segment_m / speed.
            let vertex_time = |i: usize| (i as f64) * *segment_m / speed;
            let per_seg = *segment_m / speed;
            // Walk to the vertex interval [i, i+1] that contains t.
            let max_i = (duration_s / per_seg).floor() as usize;
            let mut i = 0usize;
            while i < max_i && vertex_time(i + 1) <= t {
                i += 1;
            }
            let i = i.min(max_i);
            let t0 = vertex_time(i);
            let t1 = vertex_time(i + 1);
            let dt_seg = (t1 - t0).max(1e-9);
            let frac = ((t - t0) / dt_seg).min(1.0);
            // Vertex i and i+1 y-offsets alternate +h / -h.
            let y0 = baseline + if i % 2 == 0 { h } else { -h };
            let y1 = baseline + if (i + 1) % 2 == 0 { h } else { -h };
            let x0 = i as f64 * dx_leg;
            let x1 = (i + 1) as f64 * dx_leg;
            let pos = [x0 + (x1 - x0) * frac, y0 + (y1 - y0) * frac];
            // Heading along the leg: alternating -angle / +angle from +x.
            // Vertex i (even) is at y = baseline + h (above), vertex i+1
            // (odd) at y = baseline - h (below), so leg 0 descends => heading
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
```

## File: src/mc.rs

```rust
//! mc.rs — Monte-Carlo engine.
//!
//! Monte-Carlo propagation of the UAV heading over `horizon_s`. For each of
//! `n_samples`:
//!   1. Sample initial position from `last.pos + Cholesky(last.cov)`.
//!      Last heading and speed are used as-is (Stage 1).
//!      // TODO(Stage2): replace with state estimate from filter over track.
//!   2. With probability `p_turn`, draw omega ~ gauss(0, omega_max/3),
//!      clamped to [-omega_max, omega_max]; otherwise omega = 0.
//!   3. Propagate analytically for `horizon_s` via `motion::propagate`.
//!   4. Store the final (pos, psi).
//!
//! Returns the final point cloud, the azimuth histogram built from the final
//! headings, and the 50%/90% sectors. See §4.4.
//!
//! TODO(Stage2): replace initial-position sampling with filter state estimate.

use crate::histogram::{bin_index, sector_width};
use crate::motion::propagate;
use crate::prng::Xoshiro256ss;
use crate::types::{Observation, Ooi, PredictConfig, Prediction, Ttc};
use crate::vec2::Mat2;

pub fn predict(
    last: &Observation,
    ttc: &Ttc,
    oois: &[Ooi],
    cfg: &PredictConfig,
) -> Prediction {
    let _ = oois; // Stage 1: OOIs are an extension point, not used for inference.

    let n_samples = cfg.n_samples as usize;
    let n_bins = cfg.azimuth_bins as usize;

    let mut rng = Xoshiro256ss::from_seed(cfg.seed);
    let cov_factor = Mat2::from_cov(last.cov);

    // Accumulate final headings for the histogram and circular mean.
    let mut bin_counts = vec![0.0; n_bins];
    let mut sum_cos = 0.0;
    let mut sum_sin = 0.0;
    let mut point_cloud = Vec::with_capacity(n_samples);

    let omega_std = ttc.omega_max / 3.0;

    for _ in 0..n_samples {
        // 1. Sample initial position from last.pos + Cholesky(last.cov).
        let noise = cov_factor.sample(&mut rng);
        let pos0 = [
            last.pos[0] + noise.x,
            last.pos[1] + noise.y,
        ];

        // 2. Sample turn rate.
        let omega = if rng.next_f64() < cfg.p_turn {
            let w = rng.gauss(0.0, omega_std);
            w.clamp(-ttc.omega_max, ttc.omega_max)
        } else {
            0.0
        };

        // 3. Propagate analytically for horizon_s.
        let (pos_new, psi_new, _v_new, _omega_new) = propagate(
            pos0,
            last.v,
            last.psi,
            omega,
            cfg.horizon_s,
            cfg.speed_noise_mps,
            &mut rng,
            ttc,
        );

        // 4. Store final (pos, psi).
        point_cloud.push(pos_new);
        let idx = bin_index(psi_new, n_bins);
        bin_counts[idx] += 1.0;

        sum_cos += psi_new.cos();
        sum_sin += psi_new.sin();
    }

    // Histogram density (sums to 1).
    let bin_probs: Vec<f64> = bin_counts
        .iter()
        .map(|&c| c / n_samples as f64)
        .collect();

    // Bin centers: center of bin i = (i + 0.5) * bin_width.
    let bin_width = 2.0 * std::f64::consts::PI / n_bins as f64;
    let bin_centers: Vec<f64> = (0..n_bins)
        .map(|i| (i as f64 + 0.5) * bin_width)
        .collect();

    let sector50 = sector_width(&bin_counts, n_samples, 0.5, n_bins);
    let sector90 = sector_width(&bin_counts, n_samples, 0.9, n_bins);

    // Circular mean of final headings (robust to 2*pi wrap).
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

## File: src/filter.rs

```rust
//! filter.rs — 2D Constant-Velocity Kalman Filter (dependency-free).
//!
//! Estimates a 4-state kinematic model `X = [x, y, vx, vy]^T` from a noisy
//! sequence of 2D position observations. Position is the measured output
//! (`H` selects x, y); velocity is unobserved and inferred from the track.
//! This smooths radar noise and yields a much better (position, velocity,
//! heading) estimate for the Monte-Carlo fan in `mc::predict`.
//!
//! All matrix algebra is written out by hand for fixed 4x4 / 2x2 sizes —
//! no nalgebra, no external crates — so the core library stays
//! dependency-free (see ORNITH.MD §0). No `unsafe`, no OS randomness.
//!
//! Array convention: `[[f64; COLS]; ROWS]` = ROWS rows, COLS columns.
//!
//! TODO(Stage2): wired into `bin/server.rs` via `filter::filter_track`.

use crate::types::Observation;

/// Acceleration-noise standard deviation (m/s^2) for the CV process-noise
/// model. Documented default from the spec.
const Q_SIGMA: f64 = 0.1;

/// Large initial velocity variance so the filter can learn velocity from a
/// short track before a couple of measurements arrive.
const INIT_V_VAR: f64 = 100.0;

/// 4x4 constant-velocity transition matrix for one time step `dt`:
///   [1, 0, dt, 0]
///   [0, 1, 0, dt]
///   [0, 0, 1,  0]
///   [0, 0, 0,  1]
fn transition(dt: f64) -> [[f64; 4]; 4] {
    [
        [1.0, 0.0, dt, 0.0],
        [0.0, 1.0, 0.0, dt],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
}

/// Process noise Q (4x4) for the discrete CV white-noise-acceleration model,
/// scaled by `q_sigma^2`:
///   q^2 * [dt^4/4,      0,     dt^3/2,   0    ]
///         [0,     dt^4/4,       0,     dt^3/2]
///         [dt^3/2,   0,      dt^2,      0    ]
///         [0,      dt^3/2,       0,     dt^2  ]
fn process_noise(dt: f64, q_sigma: f64) -> [[f64; 4]; 4] {
    let q2 = q_sigma * q_sigma;
    let dt2 = dt * dt;
    let dt3 = dt2 * dt;
    let dt4 = dt3 * dt;
    let q_dt4_4 = q2 * dt4 / 4.0;
    let q_dt3_2 = q2 * dt3 / 2.0;
    let q_dt2 = q2 * dt2;
    [
        [q_dt4_4, 0.0, q_dt3_2, 0.0],
        [0.0, q_dt4_4, 0.0, q_dt3_2],
        [q_dt3_2, 0.0, q_dt2, 0.0],
        [0.0, q_dt3_2, 0.0, q_dt2],
    ]
}

/// 2x2 matrix inverse with a small diagonal regularizer so a (near-)singular
/// `S` still yields a finite gain.
fn inverse22(a: [[f64; 2]; 2]) -> [[f64; 2]; 2] {
    let eps = 1e-9;
    let a00 = a[0][0] + eps;
    let a11 = a[1][1] + eps;
    let det = a00 * a11 - a[0][1] * a[1][0];
    [
        [a11 / det, -a[0][1] / det],
        [-a[1][0] / det, a00 / det],
    ]
}

/// 2x4 * 4x4 -> 2x4.
fn mul_2x4_by_4x4(a: [[f64; 4]; 2], b: [[f64; 4]; 4]) -> [[f64; 4]; 2] {
    let mut c = [[0.0; 4]; 2];
    for i in 0..2 {
        for j in 0..4 {
            let mut s = 0.0;
            for k in 0..4 {
                s += a[i][k] * b[k][j];
            }
            c[i][j] = s;
        }
    }
    c
}

/// 2x4 * 4x2 -> 2x2.
fn mul_2x4_by_4x2(a: [[f64; 4]; 2], b: [[f64; 2]; 4]) -> [[f64; 2]; 2] {
    let mut c = [[0.0; 2]; 2];
    for i in 0..2 {
        for j in 0..2 {
            let mut s = 0.0;
            for k in 0..4 {
                s += a[i][k] * b[k][j];
            }
            c[i][j] = s;
        }
    }
    c
}

/// 4x4 * 4x4 -> 4x4.
fn mul_4x4(a: [[f64; 4]; 4], b: [[f64; 4]; 4]) -> [[f64; 4]; 4] {
    let mut c = [[0.0; 4]; 4];
    for i in 0..4 {
        for j in 0..4 {
            let mut s = 0.0;
            for k in 0..4 {
                s += a[i][k] * b[k][j];
            }
            c[i][j] = s;
        }
    }
    c
}

/// 4x4 * 4x2 -> 4x2.
fn mul_4x4_by_4x2(a: [[f64; 4]; 4], b: [[f64; 2]; 4]) -> [[f64; 2]; 4] {
    let mut c = [[0.0; 2]; 4];
    for i in 0..4 {
        for j in 0..2 {
            let mut s = 0.0;
            for k in 0..4 {
                s += a[i][k] * b[k][j];
            }
            c[i][j] = s;
        }
    }
    c
}

/// 4x2 * 2x2 -> 4x2.
fn mul_4x2_by_2x2(a: [[f64; 2]; 4], b: [[f64; 2]; 2]) -> [[f64; 2]; 4] {
    let mut c = [[0.0; 2]; 4];
    for i in 0..4 {
        for j in 0..2 {
            let mut s = 0.0;
            for k in 0..2 {
                s += a[i][k] * b[k][j];
            }
            c[i][j] = s;
        }
    }
    c
}

/// 4x2 * 2x4 -> 4x4.
fn mul_4x2_by_2x4(a: [[f64; 2]; 4], b: [[f64; 4]; 2]) -> [[f64; 4]; 4] {
    let mut c = [[0.0; 4]; 4];
    for i in 0..4 {
        for j in 0..4 {
            let mut s = 0.0;
            for k in 0..2 {
                s += a[i][k] * b[k][j];
            }
            c[i][j] = s;
        }
    }
    c
}

/// Transpose of a 4x4 matrix.
fn transpose44(a: [[f64; 4]; 4]) -> [[f64; 4]; 4] {
    let mut t = [[0.0; 4]; 4];
    for i in 0..4 {
        for j in 0..4 {
            t[j][i] = a[i][j];
        }
    }
    t
}

/// 4x4 * 4x1 -> 4x1.
fn mul44_vec(a: [[f64; 4]; 4], b: [f64; 4]) -> [f64; 4] {
    let mut c = [0.0; 4];
    for i in 0..4 {
        let mut s = 0.0;
        for k in 0..4 {
            s += a[i][k] * b[k];
        }
        c[i] = s;
    }
    c
}

/// Element-wise 4x4 addition.
fn add44(a: [[f64; 4]; 4], b: [[f64; 4]; 4]) -> [[f64; 4]; 4] {
    let mut c = [[0.0; 4]; 4];
    for i in 0..4 {
        for j in 0..4 {
            c[i][j] = a[i][j] + b[i][j];
        }
    }
    c
}

/// `I - a` for a 4x4 matrix (used for `(I - K H) P`).
fn sub_from_identity(a: [[f64; 4]; 4]) -> [[f64; 4]; 4] {
    let mut c = [[0.0; 4]; 4];
    for i in 0..4 {
        for j in 0..4 {
            c[i][j] = if i == j { 1.0 - a[i][j] } else { -a[i][j] };
        }
    }
    c
}

/// 2x2 addition.
fn add22(a: [[f64; 2]; 2], b: [[f64; 2]; 2]) -> [[f64; 2]; 2] {
    [
        [a[0][0] + b[0][0], a[0][1] + b[0][1]],
        [a[1][0] + b[1][0], a[1][1] + b[1][1]],
    ]
}

/// 2D Constant-Velocity Kalman Filter over a track of position observations.
pub struct KalmanFilter {
    /// State: [x, y, vx, vy].
    x: [f64; 4],
    /// Covariance: 4x4.
    p: [[f64; 4]; 4],
    /// Time of the last processed observation.
    last_t: f64,
}

impl KalmanFilter {
    /// Initialize the filter with the first observation: seed the state at the
    /// measured position (velocity unknown), build `P` from the observation
    /// covariance for the position block and a large variance for velocity,
    /// then incorporate the first measurement.
    pub fn new(initial_obs: &Observation) -> Self {
        let cov = initial_obs.cov;
        let mut kf = KalmanFilter {
            x: [initial_obs.pos[0], initial_obs.pos[1], 0.0, 0.0],
            p: [[0.0; 4]; 4],
            last_t: initial_obs.t,
        };
        // Position covariance block.
        kf.p[0][0] = cov[0];
        kf.p[0][1] = cov[1];
        kf.p[1][0] = cov[1];
        kf.p[1][1] = cov[2];
        // Unknown velocity -> large variance.
        kf.p[2][2] = INIT_V_VAR;
        kf.p[3][3] = INIT_V_VAR;
        // Incorporate the first measurement.
        kf.update(initial_obs);
        kf
    }

    /// Predict the state forward by `dt` seconds using the CV model with
    /// acceleration noise `q_sigma`.
    pub fn predict(&mut self, dt: f64, q_sigma: f64) {
        if dt <= 0.0 {
            return;
        }
        let f = transition(dt);
        let ft = transpose44(f);
        // x' = F x.
        self.x = mul44_vec(f, self.x);
        // P' = F P F^T + Q.
        let fp = mul_4x4(f, self.p);
        let fpf = mul_4x4(fp, ft);
        let q = process_noise(dt, q_sigma);
        self.p = add44(fpf, q);
        self.last_t += dt;
    }

    /// Incorporate a position measurement (correction step).
    pub fn update(&mut self, obs: &Observation) {
        // H (2x4): selects position.
        let h: [[f64; 4]; 2] = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]];
        let ht: [[f64; 2]; 4] = [[1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.0, 0.0]];
        // R (2x2) from observation covariance.
        let r = [[obs.cov[0], obs.cov[1]], [obs.cov[1], obs.cov[2]]];
        // Innovation y = z - H x.
        let y = [obs.pos[0] - self.x[0], obs.pos[1] - self.x[1]];
        // S = H P H^T + R (2x2).
        let hp = mul_2x4_by_4x4(h, self.p);
        let hph = mul_2x4_by_4x2(hp, ht);
        let s = add22(hph, r);
        let s_inv = inverse22(s);
        // K = P H^T S^-1 (4x2).
        let pht = mul_4x4_by_4x2(self.p, ht);
        let k = mul_4x2_by_2x2(pht, s_inv);
        // x' = x + K y.
        for i in 0..4 {
            self.x[i] += k[i][0] * y[0] + k[i][1] * y[1];
        }
        // P' = (I - K H) P.
        let kh = mul_4x2_by_2x4(k, h);
        let ikh = sub_from_identity(kh);
        self.p = mul_4x4(ikh, self.p);
        self.last_t = obs.t;
    }

    /// Return the smoothed state as an `Observation`:
    ///   pos  = [x, y] (filtered)
    ///   v    = sqrt(vx^2 + vy^2) (filtered)
    ///   psi  = atan2(vy, vx) (filtered)
    ///   cov  = P[0..2, 0..2] (2x2 position covariance)
    ///   t    = last processed observation's t
    pub fn get_state(&self) -> Observation {
        let vx = self.x[2];
        let vy = self.x[3];
        Observation {
            t: self.last_t,
            pos: [self.x[0], self.x[1]],
            v: (vx * vx + vy * vy).sqrt(),
            psi: vy.atan2(vx),
            cov: [self.p[0][0], self.p[0][1], self.p[1][1]],
        }
    }
}

/// Run the CV Kalman filter over a whole track and return the smoothed final
/// state as an `Observation`.
///
/// The filter is initialized with `track[0]`, then for each subsequent
/// observation it predicts forward by the inter-sample interval `dt` and
/// corrects with the measurement.
pub fn filter_track(track: &[Observation]) -> Observation {
    assert!(
        !track.is_empty(),
        "filter_track needs at least one observation"
    );
    let mut kf = KalmanFilter::new(&track[0]);
    for obs in &track[1..] {
        let dt = obs.t - kf.last_t;
        kf.predict(dt, Q_SIGMA);
        kf.update(obs);
    }
    kf.get_state()
}
```

## File: tests/prng.rs

```rust
//! Tests for prng.rs — determinism by seed, and gaussian statistics.

use predictor::prng::Xoshiro256ss;

#[test]
fn determinism_by_seed() {
    // Same seed + same sequence of draws => bit-identical output.
    let mut a = Xoshiro256ss::from_seed(12345);
    let mut b = Xoshiro256ss::from_seed(12345);

    for _ in 0..1000 {
        let fa = a.next_f64();
        let fb = b.next_f64();
        assert_eq!(fa.to_bits(), fb.to_bits(), "f64 streams diverged");
    }
    for _ in 0..1000 {
        let ua = a.next_u64();
        let ub = b.next_u64();
        assert_eq!(ua, ub, "u64 streams diverged");
    }
    for _ in 0..500 {
        let ga = a.gauss(3.0, 2.0);
        let gb = b.gauss(3.0, 2.0);
        assert_eq!(ga.to_bits(), gb.to_bits(), "gaussian streams diverged");
    }
}

#[test]
fn different_seed_differs() {
    let mut a = Xoshiro256ss::from_seed(1);
    let mut b = Xoshiro256ss::from_seed(2);
    let mut diffs = 0;
    for _ in 0..1000 {
        if a.next_u64() != b.next_u64() {
            diffs += 1;
        }
    }
    assert!(diffs > 500, "different seeds should mostly differ");
}

#[test]
fn next_f64_in_unit_interval() {
    let mut rng = Xoshiro256ss::from_seed(42);
    for _ in 0..100_000 {
        let x = rng.next_f64();
        assert!((0.0..1.0).contains(&x), "f64 out of [0,1): {x}");
    }
}

#[test]
fn gaussian_mean_std() {
    // ~1e6 draws: |mean| < 0.01, |std - 1| < 0.01 for standard normal.
    let mut rng = Xoshiro256ss::from_seed(7);
    let n = 1_000_000u64;
    let mut sum = 0.0f64;
    let mut sum2 = 0.0f64;
    for _ in 0..n {
        let x = rng.gauss(0.0, 1.0);
        sum += x;
        sum2 += x * x;
    }
    let mean = sum / n as f64;
    let var = sum2 / n as f64 - mean * mean;
    let std = var.sqrt();
    assert!(mean.abs() < 0.01, "mean too far from 0: {mean}");
    assert!((std - 1.0).abs() < 0.01, "std too far from 1: {std}");
}

#[test]
fn gaussian_scale_and_shift() {
    // gauss(mean, std) should have the right mean and std.
    let mut rng = Xoshiro256ss::from_seed(99);
    let n = 200_000u64;
    let mut sum = 0.0f64;
    let mut sum2 = 0.0f64;
    for _ in 0..n {
        let x = rng.gauss(10.0, 3.0);
        sum += x;
        sum2 += x * x;
    }
    let mean = sum / n as f64;
    let var = sum2 / n as f64 - mean * mean;
    let std = var.sqrt();
    assert!((mean - 10.0).abs() < 0.05, "bad mean: {mean}");
    assert!((std - 3.0).abs() < 0.05, "bad std: {std}");
}
```

## File: tests/vec2.rs

```rust
//! Tests for vec2.rs — Vec2 ops, Mat2 Cholesky, and sampling.

use predictor::prng::Xoshiro256ss;
use predictor::vec2::{Mat2, Vec2};

#[test]
fn vec2_basic_ops() {
    let a = Vec2::new(3.0, 4.0);
    let b = Vec2::new(1.0, 2.0);
    assert_eq!(a.add(b), Vec2::new(4.0, 6.0));
    assert_eq!(a.sub(b), Vec2::new(2.0, 2.0));
    assert_eq!(a.scale(2.0), Vec2::new(6.0, 8.0));
    assert_eq!(a.dot(b), 11.0);
    assert!((a.len() - 5.0).abs() < 1e-12);
}

#[test]
fn vec2_rotate_by_90() {
    let a = Vec2::new(1.0, 0.0);
    let r = a.rotate(std::f64::consts::PI / 2.0);
    assert!((r.x - 0.0).abs() < 1e-12, "x: {}", r.x);
    assert!((r.y - 1.0).abs() < 1e-12, "y: {}", r.y);
}

#[test]
fn vec2_rotate_identity() {
    let a = Vec2::new(3.0, 4.0);
    let r = a.rotate(0.0);
    assert_eq!(r, a);
}

#[test]
fn mat2_cholesky_reconstructs_cov() {
    // L * L^T should reproduce [xx, xy, yy] on the diagonal-ish form.
    let cov = [4.0, 2.0, 5.0]; // xx=4, xy=2, yy=5
    let l = Mat2::from_cov(cov);
    assert!((l.l11 - 2.0).abs() < 1e-12); // sqrt(4)
    assert!((l.l21 - 1.0).abs() < 1e-12); // 2/2
    assert!((l.l22 - 2.0).abs() < 1e-12); // sqrt(5 - 1)

    // L * [1,0] = first column gives the covariance's x-part.
    let v = l.apply(1.0, 0.0);
    assert!((v.x - l.l11).abs() < 1e-12);
    assert!((v.y - l.l21).abs() < 1e-12);
}

#[test]
fn mat2_zero_cov_is_zero_noise() {
    let l = Mat2::from_cov([0.0, 0.0, 0.0]);
    assert_eq!((l.l11, l.l21, l.l22), (0.0, 0.0, 0.0));
    let mut rng = Xoshiro256ss::from_seed(1);
    let s = l.sample(&mut rng);
    assert_eq!((s.x, s.y), (0.0, 0.0));
}

#[test]
fn mat2_xx_zero_swaps_axes() {
    // xx == 0, yy > 0 -> axes swapped: l22 = sqrt(yy).
    let l = Mat2::from_cov([0.0, 0.0, 9.0]);
    assert!((l.l11 - 0.0).abs() < 1e-12);
    assert!((l.l21 - 0.0).abs() < 1e-12);
    assert!((l.l22 - 3.0).abs() < 1e-12);
}

#[test]
fn mat2_sample_is_centered() {
    // Over many draws, the mean of L * N(0, I) should be ~0.
    let l = Mat2::from_cov([4.0, 0.0, 9.0]);
    let mut rng = Xoshiro256ss::from_seed(55);
    let n = 100_000u64;
    let mut sx = 0.0f64;
    let mut sy = 0.0f64;
    for _ in 0..n {
        let s = l.sample(&mut rng);
        sx += s.x;
        sy += s.y;
    }
    assert!((sx / n as f64).abs() < 0.05, "mean x: {}", sx / n as f64);
    assert!((sy / n as f64).abs() < 0.05, "mean y: {}", sy / n as f64);
}
```

## File: tests/histogram.rs

```rust
//! Tests for histogram.rs — azimuth bins and sector extraction (§4.5).

use predictor::histogram::{bin_index, psi_rem_euclid, sector_width};
use std::f64::consts::PI;

#[test]
fn psi_rem_euclid_wraps_into_zero_to_twopi() {
    assert!((psi_rem_euclid(0.5) - 0.5).abs() < 1e-9);
    assert!((psi_rem_euclid(2.0 * PI)).abs() < 1e-9);
    assert!((psi_rem_euclid(-0.5) - (2.0 * PI - 0.5)).abs() < 1e-9);
    assert!((psi_rem_euclid(3.0 * PI) - PI).abs() < 1e-9);
}

#[test]
fn bin_index_maps_heading_into_range() {
    let n_bins = 72;
    assert_eq!(bin_index(0.0, n_bins), 0);
    assert_eq!(bin_index(PI, n_bins), 36);
    assert_eq!(bin_index(2.0 * PI, n_bins), 0);
    // Wraps correctly.
    assert_eq!(bin_index(2.0 * PI + 0.001, n_bins), bin_index(0.001, n_bins));
    // Out of range is reduced.
    assert!(bin_index(-0.001, n_bins) == n_bins - 1);
}

#[test]
fn delta_function_sector_is_one_bin() {
    // All samples in bin 0.
    let counts = [100.0, 0.0, 0.0, 0.0];
    let n_samples = 100usize;
    let n_bins = 4usize;
    let bin_width = 2.0 * PI / n_bins as f64;

    let s50 = sector_width(&counts, n_samples, 0.5, n_bins);
    let s90 = sector_width(&counts, n_samples, 0.9, n_bins);

    assert!((s50 - bin_width).abs() < 1e-9, "s50 = {}", s50);
    assert!((s90 - bin_width).abs() < 1e-9, "s90 = {}", s90);
}

#[test]
fn uniform_sector_is_half_and_ninety() {
    // 72 bins, uniform density.
    let n_bins = 72usize;
    let n_samples = 7200usize;
    let counts = vec![100.0; n_bins];

    let s50 = sector_width(&counts, n_samples, 0.5, n_bins);
    let s90 = sector_width(&counts, n_samples, 0.9, n_bins);

    // sector50 ~ 180 deg, sector90 ~ 324 deg (within 1 bin).
    let bin_width_deg = 360.0 / n_bins as f64;
    let one_bin_rad = bin_width_deg * PI / 180.0;
    assert!(
        (s50 - PI).abs() < one_bin_rad,
        "s50 = {} rad (want ~{})",
        s50,
        PI
    );
    assert!(
        (s90 - 324.0 * PI / 180.0).abs() < one_bin_rad,
        "s90 = {} rad (want ~324 deg)",
        s90
    );
}

#[test]
fn symmetric_bimodal_sector() {
    // Two modes at bin 0 and bin 36, each with 50 samples (72 bins).
    let n_bins = 72usize;
    let n_samples = 100usize;
    let mut counts = vec![0.0; n_bins];
    counts[0] = 50.0;
    counts[36] = 50.0;

    let s50 = sector_width(&counts, n_samples, 0.5, n_bins);
    let s90 = sector_width(&counts, n_samples, 0.9, n_bins);

    let bin_width_deg = 360.0 / n_bins as f64;
    let one_bin_rad = bin_width_deg * PI / 180.0;

    // sector50 = 1 bin width (one mode).
    assert!(
        (s50 - one_bin_rad).abs() < 1e-9,
        "s50 = {} rad (want 1 bin)",
        s50
    );
    // sector90 must span both modes: 37 bins (185 deg).
    assert!(
        (s90 - 37.0 * one_bin_rad).abs() < 1e-9,
        "s90 = {} rad (want 37 bins)",
        s90
    );
}

#[test]
fn sector50_leq_sector90() {
    let counts = vec![10.0, 20.0, 30.0, 40.0];
    let n_samples = 100usize;
    let n_bins = 4usize;
    let s50 = sector_width(&counts, n_samples, 0.5, n_bins);
    let s90 = sector_width(&counts, n_samples, 0.9, n_bins);
    assert!(s50 <= s90, "s50 = {} should be <= s90 = {}", s50, s90);
}
```

## File: tests/motion.rs

```rust
//! Tests for motion.rs — analytic arc propagation, straight line, circle
//! radius, and speed/omega clamping.

use predictor::motion::propagate;
use predictor::prng::Xoshiro256ss;
use predictor::types::Ttc;

/// Straight line: omega == 0 -> pos advances by v*t along the heading.
#[test]
fn omega_zero_is_straight_line() {
    let ttc = Ttc {
        v_min: 0.0,
        v_max: 100.0,
        omega_max: 1.0,
    };
    let mut rng = Xoshiro256ss::from_seed(1);
    // psi = 0 -> move along +X. No speed noise so it's exact.
    let (pos, psi, v, omega) = propagate(
        [0.0, 0.0], 10.0, 0.0, 0.0, 5.0, 0.0, &mut rng, &ttc,
    );
    assert!((pos[0] - 50.0).abs() < 1e-9, "x: {}", pos[0]);
    assert!((pos[1] - 0.0).abs() < 1e-9, "y: {}", pos[1]);
    assert!((psi - 0.0).abs() < 1e-9, "psi: {}", psi);
    assert!((v - 10.0).abs() < 1e-9, "v: {}", v);
    assert!((omega - 0.0).abs() < 1e-9, "omega: {}", omega);
}

/// omega == omega_max -> after one full turn (t = 2*pi/omega) the UAV should
/// return to the start, tracing a circle of radius r = v/omega_max.
#[test]
fn omega_max_traces_circle_of_radius_v_over_omega() {
    let omega_max = 0.5;
    let v = 10.0;
    let ttc = Ttc {
        v_min: 0.0,
        v_max: 100.0,
        omega_max,
    };
    let mut rng = Xoshiro256ss::from_seed(2);
    let t = 2.0 * std::f64::consts::PI / omega_max; // one full revolution
    let (pos, psi, _v, omega) = propagate(
        [0.0, 0.0],
        v,
        std::f64::consts::PI / 2.0,
        omega_max,
        t,
        0.0,
        &mut rng,
        &ttc,
    );
    // Back to the start after a full circle.
    assert!(pos[0].abs() < 1e-6, "x: {}", pos[0]);
    assert!(pos[1].abs() < 1e-6, "y: {}", pos[1]);
    // Heading advanced by exactly 2*pi.
    assert!((psi - (std::f64::consts::PI / 2.0 + 2.0 * std::f64::consts::PI))
        .abs()
        < 1e-6,
        "psi: {}",
        psi
    );
    // Radius check via the half-turn diameter: after omega*t = pi the UAV
    // is diametrically opposite the start, so the distance from start is
    // 2r where r = v / omega_max.
    let r = v / omega_max;
    let (half_pos, _half_psi, _half_v, _half_omega) = propagate(
        [0.0, 0.0],
        v,
        std::f64::consts::PI / 2.0,
        omega_max,
        t / 2.0,
        0.0,
        &mut rng,
        &ttc,
    );
    let chord = (half_pos[0] * half_pos[0] + half_pos[1] * half_pos[1]).sqrt();
    assert!((chord - 2.0 * r).abs() < 1e-6,
        "diameter: {}, expected 2r = {}",
        chord,
        2.0 * r
    );
    assert!((omega - omega_max).abs() < 1e-9, "omega: {}", omega);
}

/// Speed clamping: v + noise is clamped to [v_min, v_max].
#[test]
fn speed_is_clamped() {
    let ttc = Ttc {
        v_min: 2.0,
        v_max: 8.0,
        omega_max: 1.0,
    };
    let mut rng = Xoshiro256ss::from_seed(3);
    // v=100, huge noise, omega=0 -> straight line. Per BUG 1 fix, position
    // uses the clamped v_new (sampled velocity), not the input v.
    let (pos, _psi, v, _omega) = propagate(
        [0.0, 0.0], 100.0, 0.0, 0.0, 1.0, 1000.0, &mut rng, &ttc,
    );
    assert!((v - 8.0).abs() < 1e-9, "v clamped high: {}", v);
    // omega=0, t=1, psi=0 => pos[0] = v_new * t * cos(psi) = v_new.
    assert!((pos[0] - v).abs() < 1e-6, "x: {}", pos[0]);

    // v = -100 -> v_new clamped to v_min.
    let (_pos2, _psi2, v2, _omega2) = propagate(
        [0.0, 0.0], -100.0, 0.0, 0.0, 1.0, 0.0, &mut rng, &ttc,
    );
    assert!((v2 - 2.0).abs() < 1e-9, "v clamped low: {}", v2);
}

/// Omega clamping: omega beyond omega_max is returned clamped in omega_new,
/// while the arc geometry still uses the (un-clamped) input omega.
#[test]
fn omega_is_clamped_in_return_value() {
    let ttc = Ttc {
        v_min: 0.0,
        v_max: 100.0,
        omega_max: 0.3,
    };
    let mut rng = Xoshiro256ss::from_seed(4);
    // omega = 5.0 -> omega_new clamped to 0.3.
    let (_pos, _psi, _v, omega) = propagate(
        [0.0, 0.0], 10.0, 0.0, 5.0, 1.0, 0.0, &mut rng, &ttc,
    );
    assert!((omega - 0.3).abs() < 1e-9, "omega clamped high: {}", omega);

    // omega = -5.0 -> omega_new clamped to -0.3.
    let (_pos2, _psi2, _v2, omega2) = propagate(
        [0.0, 0.0], 10.0, 0.0, -5.0, 1.0, 0.0, &mut rng, &ttc,
    );
    assert!((omega2 - (-0.3)).abs() < 1e-9, "omega clamped low: {}", omega2);
}

/// Determinism: same seed + same input => bit-identical output.
#[test]
fn propagation_is_deterministic() {
    let ttc = Ttc {
        v_min: 0.0,
        v_max: 100.0,
        omega_max: 0.4,
    };
    let mut a = Xoshiro256ss::from_seed(7);
    let mut b = Xoshiro256ss::from_seed(7);
    let (pa, psi_a, va, oa) = propagate(
        [1.0, 2.0], 12.0, 0.7, 0.25, 3.0, 1.5, &mut a, &ttc,
    );
    let (pb, psi_b, vb, ob) = propagate(
        [1.0, 2.0], 12.0, 0.7, 0.25, 3.0, 1.5, &mut b, &ttc,
    );
    assert_eq!(pa, pb);
    assert_eq!(psi_a.to_bits(), psi_b.to_bits());
    assert_eq!(va.to_bits(), vb.to_bits());
    assert_eq!(oa.to_bits(), ob.to_bits());
}
```

## File: tests/mc.rs

```rust
//! Tests for mc.rs — Monte-Carlo prediction engine (§4.4).

use predictor::mc::predict;
use predictor::types::{Observation, Ooi, PredictConfig, Ttc};
use std::f64::consts::PI;

fn ttc() -> Ttc {
    Ttc {
        v_min: 2.0,
        v_max: 8.0,
        omega_max: 0.5,
    }
}

fn cfg(n_samples: u32, p_turn: f64, seed: u64, azimuth_bins: u32) -> PredictConfig {
    PredictConfig {
        horizon_s: 10.0,
        n_samples,
        dt_s: 1.0, // reserved, unused in Stage 1
        p_turn,
        azimuth_bins,
        speed_noise_mps: 0.0,
        seed,
    }
}

fn obs() -> Observation {
    Observation {
        t: 0.0,
        pos: [0.0, 0.0],
        v: 5.0,
        psi: 0.0,
        cov: [1.0, 0.0, 1.0],
    }
}

fn oois() -> Vec<Ooi> {
    vec![Ooi { pos: [10.0, 0.0] }]
}

#[test]
fn determinism_same_seed_bit_identical() {
    let o = obs();
    let t = ttc();
    let c = cfg(1000, 0.3, 42, 72);
    let a = predict(&o, &t, &oois(), &c);
    let b = predict(&o, &t, &oois(), &c);

    assert_eq!(a.point_cloud.len(), b.point_cloud.len());
    for (p, q) in a.point_cloud.iter().zip(b.point_cloud.iter()) {
        assert_eq!(*p, *q);
    }
    assert_eq!(a.bin_centers, b.bin_centers);
    assert_eq!(a.bin_probs, b.bin_probs);
    assert_eq!(a.sector50, b.sector50);
    assert_eq!(a.sector90, b.sector90);
    assert_eq!(a.final_heading_mean, b.final_heading_mean);
}

#[test]
fn bin_centers_count_and_probs_sum_to_one() {
    let o = obs();
    let t = ttc();
    let c = cfg(1000, 0.3, 1, 72);
    let p = predict(&o, &t, &oois(), &c);

    assert_eq!(p.bin_centers.len(), 72);
    assert_eq!(p.bin_probs.len(), 72);

    let sum: f64 = p.bin_probs.iter().sum::<f64>();
    assert!((sum - 1.0).abs() < 1e-9, "bin_probs sum = {}", sum);
}

#[test]
fn point_cloud_has_n_samples() {
    let o = obs();
    let t = ttc();
    let c = cfg(500, 0.3, 7, 72);
    let p = predict(&o, &t, &oois(), &c);
    assert_eq!(p.point_cloud.len(), 500);
}

#[test]
fn zero_turn_probability_concentrates_all_samples() {
    // p_turn = 0 => omega = 0 for all samples => psi_new = last.psi = 0
    // for every sample. All samples fall in bin 0, so both sectors are
    // exactly one bin width (5 deg).
    let o = obs();
    let t = ttc();
    let c = cfg(2000, 0.0, 1, 72);
    let p = predict(&o, &t, &oois(), &c);

    let bin_width = 2.0 * PI / 72.0;
    let s50 = p.sector50;
    let s90 = p.sector90;

    assert!((s50 - bin_width).abs() < 1e-9, "s50 = {}", s50);
    assert!((s90 - bin_width).abs() < 1e-9, "s90 = {}", s90);

    // Circular mean should be near 0 (last.psi).
    assert!(p.final_heading_mean.abs() < 1e-6);

    // Exactly one non-empty bin.
    let nonzero = p.bin_probs.iter().filter(|&&x| x > 0.0).count();
    assert_eq!(nonzero, 1);
}

#[test]
fn nonzero_turn_probability_spreads_sectors() {
    // p_turn = 1 => omega drawn every time => heading distribution is
    // spread over more bins => sector90 > 1 bin width.
    let o = obs();
    let t = ttc();
    let c = cfg(5000, 1.0, 123, 72);
    let p = predict(&o, &t, &oois(), &c);

    let bin_width = 2.0 * PI / 72.0;
    assert!(
        p.sector90 > bin_width * 2.0,
        "s90 = {} should be > 2 bins",
        p.sector90
    );
}

#[test]
fn initial_position_noise_samples_from_cov() {
    // With cov = [1, 0, 1], the initial position noise has std 1 in each axis.
    // p_turn = 0 => omega = 0 => psi_new = last.psi = 0 for all samples, so
    // each final position = pos0 + v*T*[1, 0] = pos0 + [50, 0] + noise.
    // The cloud is therefore centered near [50, 0] with std ~1 in x and y.
    let o = obs();
    let t = ttc();
    let c = cfg(2000, 0.0, 5, 72);
    let p = predict(&o, &t, &oois(), &c);

    let mean_x: f64 = p.point_cloud.iter().map(|p| p[0]).sum::<f64>() / 2000.0;
    let mean_y: f64 = p.point_cloud.iter().map(|p| p[1]).sum::<f64>() / 2000.0;

    // Noise is zero-mean, so means should be near the deterministic
    // propagation endpoint [50, 0].
    assert!((mean_x - 50.0).abs() < 0.2, "mean_x = {}", mean_x);
    assert!(mean_y.abs() < 0.2, "mean_y = {}", mean_y);
}

#[test]
fn different_seed_differs() {
    let o = obs();
    let t = ttc();
    let c1 = cfg(1000, 0.3, 1, 72);
    let c2 = cfg(1000, 0.3, 2, 72);
    let p1 = predict(&o, &t, &oois(), &c1);
    let p2 = predict(&o, &t, &oois(), &c2);

    // At least one point in the cloud differs.
    let mut any_diff = false;
    for (p, q) in p1.point_cloud.iter().zip(p2.point_cloud.iter()) {
        if *p != *q {
            any_diff = true;
        }
    }
    assert!(any_diff, "different seeds should yield different clouds");
}
```

## File: tests/simulator.rs

```rust
//! Determinism + basic correctness tests for `simulator::generate_track`.
//!
//! Stage 1 truth is straight-line (omega = 0, speed_noise = 0) starting at
//! truth_waypoints[0]; each observation carries Gaussian noise of covariance
//! obs_cov (via Mat2::from_cov(obs_cov).sample(&mut rng)). With obs_cov =
//! [0, 0, 0] the noise is zero and observations lie exactly on the truth line.

use predictor::simulator::{generate_curved_track, generate_track, Trajectory};
use predictor::types::{KinState, Ttc};

fn ttc() -> Ttc {
    Ttc {
        v_min: 2.0,
        v_max: 8.0,
        omega_max: 0.5,
    }
}

/// Two runs with the same seed produce bit-identical observations.
#[test]
fn determinism_same_seed() {
    let waypoints = vec![
        KinState {
            pos: [0.0, 0.0],
            v: 5.0,
            psi: 0.0,
        },
        KinState {
            pos: [5.0, 0.0],
            v: 5.0,
            psi: 0.0,
        },
        KinState {
            pos: [10.0, 0.0],
            v: 5.0,
            psi: 0.0,
        },
    ];
    let ttc = ttc();
    let obs_cov = [1.0, 0.0, 1.0];
    let dt = 1.0;
    let seed = 42;

    let a = generate_track(&waypoints, &ttc, dt, obs_cov, seed);
    let b = generate_track(&waypoints, &ttc, dt, obs_cov, seed);

    assert_eq!(a.len(), b.len(), "observation counts must match");
    for (x, y) in a.iter().zip(b.iter()) {
        assert_eq!(x.t, y.t, "time mismatch at t={}", x.t);
        assert_eq!(x.pos, y.pos, "position mismatch at t={}", x.t);
        assert_eq!(x.cov, y.cov, "cov mismatch at t={}", x.t);
    }
}

/// Straight-line truth with omega = 0: position advances along psi by v*dt per
/// leg. With obs_cov = [0, 0, 0] the noise is zero and observations lie
/// exactly on the truth line.
#[test]
fn straight_line_truth_geometry() {
    // Three legs, each 1s, v = 4, psi = 0 -> advance 4 m along +x per leg.
    let waypoints = vec![
        KinState {
            pos: [0.0, 0.0],
            v: 4.0,
            psi: 0.0,
        },
        KinState {
            pos: [4.0, 0.0],
            v: 4.0,
            psi: 0.0,
        },
        KinState {
            pos: [8.0, 0.0],
            v: 4.0,
            psi: 0.0,
        },
    ];
    let ttc = ttc();
    let obs_cov = [0.0, 0.0, 0.0]; // zero noise -> exact truth line
    let dt = 1.0;

    let obs = generate_track(&waypoints, &ttc, dt, obs_cov, 1);

    assert_eq!(obs.len(), 3, "3 waypoints -> 3 observations");
    assert_eq!(obs[0].pos, [0.0, 0.0]);
    assert_eq!(obs[1].pos, [4.0, 0.0]);
    assert_eq!(obs[2].pos, [8.0, 0.0]);
    // cov field carries obs_cov.
    for o in &obs {
        assert_eq!(o.cov, [0.0, 0.0, 0.0]);
    }
}

/// Observation count equals waypoints.len().
#[test]
fn observation_count_matches_legs() {
    let waypoints = vec![
        KinState {
            pos: [0.0, 0.0],
            v: 5.0,
            psi: 0.0,
        },
        KinState {
            pos: [5.0, 0.0],
            v: 5.0,
            psi: 0.0,
        },
        KinState {
            pos: [10.0, 0.0],
            v: 5.0,
            psi: 0.0,
        },
        KinState {
            pos: [15.0, 0.0],
            v: 5.0,
            psi: 0.0,
        },
    ];
    let ttc = ttc();
    let obs_cov = [1.0, 0.0, 1.0];
    let dt = 1.0;

    let obs = generate_track(&waypoints, &ttc, dt, obs_cov, 3);
    assert_eq!(obs.len(), 4, "4 waypoints -> 4 observations");
}

/// Single waypoint -> single observation at t = 0, exactly on the truth point
/// (obs_cov = [0, 0, 0]).
#[test]
fn single_waypoint() {
    let waypoints = vec![KinState {
        pos: [2.0, 3.0],
        v: 5.0,
        psi: 0.0,
    }];
    let ttc = ttc();
    let obs_cov = [0.0, 0.0, 0.0]; // zero noise -> exact truth point
    let dt = 1.0;

    let obs = generate_track(&waypoints, &ttc, dt, obs_cov, 99);
    assert_eq!(obs.len(), 1);
    assert_eq!(obs[0].t, 0.0);
    assert_eq!(obs[0].pos, [2.0, 3.0]);
}

/// With obs_cov = [1, 0, 1] along a straight +X line, the per-observation noise
/// variance in y should be close to 1.0 (not 0). This test fails on the
/// noiseless (pre-fix) simulator and passes after BUG 1 is fixed.
#[test]
fn noise_variance_matches_obs_cov() {
    // 500 waypoints -> 500 observations (one per waypoint), all on the +X line
    // since v is along psi = 0 for every leg.
    let waypoints: Vec<KinState> = (0..500)
        .map(|i| KinState {
            pos: [i as f64 * 10.0, 0.0],
            v: 10.0,
            psi: 0.0,
        })
        .collect();
    let ttc = ttc();
    let obs_cov = [1.0, 0.0, 1.0];
    let dt = 1.0;
    let n = 500;

    let obs = generate_track(&waypoints, &ttc, dt, obs_cov, 123);
    assert_eq!(obs.len(), n);

    // truth.y = 0 for all obs (psi = 0), so Var(obs.y) ~= obs_cov[2] = 1.0.
    let y: Vec<f64> = obs.iter().map(|o| o.pos[1]).collect();
    let mean = y.iter().sum::<f64>() / n as f64;
    let var: f64 = y.iter().map(|yi| (yi - mean) * (yi - mean)).sum::<f64>() / n as f64;
    assert!(
        (var - 1.0).abs() < 0.15,
        "Var(y) = {} should be close to 1.0",
        var
    );
}

/// RNG order: two runs with the same seed produce identical noise; a different
/// seed produces different noise. Verifies BUG 3 (no spurious gauss(0,0) draw)
/// keeps the RNG stream deterministic and seed-dependent.
#[test]
fn rng_order_is_seed_dependent() {
    let waypoints = vec![KinState {
        pos: [0.0, 0.0],
        v: 10.0,
        psi: 0.0,
    }];
    let ttc = ttc();
    let obs_cov = [1.0, 0.0, 1.0];
    let dt = 1.0;

    let a = generate_track(&waypoints, &ttc, dt, obs_cov, 7);
    let b = generate_track(&waypoints, &ttc, dt, obs_cov, 7);
    let c = generate_track(&waypoints, &ttc, dt, obs_cov, 8);

    // Same seed -> identical noise on every observation.
    for (x, y) in a.iter().zip(b.iter()) {
        assert_eq!(x.pos, y.pos, "same seed must give identical noise");
    }

    // Different seed -> at least one observation's noise differs.
    let mut any_diff = false;
    for (x, y) in a.iter().zip(c.iter()) {
        if x.pos != y.pos {
            any_diff = true;
        }
    }
    assert!(any_diff, "different seed should produce different noise");
}

/// Curved straight-line with psi0 = 0 and zero covariance: every observation
/// lies on the +X axis (|y| < 1e-9).
#[test]
fn curved_straight_equals_linear() {
    let traj = Trajectory::Straight { psi0: 0.0 };
    let obs_cov = [0.0, 0.0, 0.0];
    let speed = 5.0;
    let duration_s = 60.0;
    let dt_obs_s = 1.0;
    let seed = 42;

    let obs = generate_curved_track(&traj, speed, duration_s, dt_obs_s, obs_cov, seed);
    assert!(obs.len() >= 2, "expected multiple observations");

    for o in &obs {
        assert!(
            o.pos[1].abs() < 1e-9,
            "obs at t={} off the +X axis: y={}",
            o.t,
            o.pos[1]
        );
    }
    // Sanity: x advances ~ speed*dt per leg.
    assert!(obs[1].pos[0] > 4.0 && obs[1].pos[0] < 6.0, "x at t=1 = {}", obs[1].pos[0]);
}

/// Curved Sine with amplitude 5: |y_k| <= 5 + 1e-9 for all k (zero cov).
#[test]
fn curved_sine_y_bounded() {
    let traj = Trajectory::Sine {
        wavelength_m: 100.0,
        amplitude_m: 5.0,
        direction: 0.0,
    };
    let obs_cov = [0.0, 0.0, 0.0];
    let speed = 5.0;
    let duration_s = 200.0; // several wavelengths
    let dt_obs_s = 1.0;
    let seed = 7;

    let obs = generate_curved_track(&traj, speed, duration_s, dt_obs_s, obs_cov, seed);
    let mut max_y: f64 = 0.0;
    for o in &obs {
        max_y = max_y.max(o.pos[1].abs());
    }
    assert!(
        max_y <= 5.0 + 1e-9,
        "max |y| = {} exceeds amplitude + eps",
        max_y
    );
}

/// Curved Zigzag: consecutive psi values alternate sign; at least one sign
/// flip within the first 5 observations (zero cov).
#[test]
fn curved_zigzag_alternates() {
    let traj = Trajectory::Zigzag {
        segment_m: 100.0,
        angle_rad: 0.3,
        baseline: 0.0,
    };
    let obs_cov = [0.0, 0.0, 0.0];
    let speed = 5.0;
    let duration_s = 60.0;
    let dt_obs_s = 1.0;
    let seed = 3;

    let obs = generate_curved_track(&traj, speed, duration_s, dt_obs_s, obs_cov, seed);
    assert!(obs.len() >= 5, "expected at least 5 observations");

    // Each leg spans segment_m/speed = 100/5 = 20 observation intervals, so the
    // first 5 observations all lie in leg 0 (same sign). Check a wider window
    // to catch the sign flip at the leg 0 -> leg 1 boundary (obs[21]).
    let mut flips = 0usize;
    for i in 1..25 {
        let (s0, s1) = (obs[i - 1].psi.signum(), obs[i].psi.signum());
        if s0 != s1 && s0 != 0.0 && s1 != 0.0 {
            flips += 1;
        }
    }
    assert!(flips >= 1, "expected at least one psi sign flip in first 25 obs");
}

/// Curved Spiral with r0 = 10, r_growth = 5, turns = 2: radius grows, so
/// |pos[last]| - |pos[0]| > 2.0 (zero cov).
#[test]
fn curved_spiral_radius_grows() {
    let traj = Trajectory::Spiral {
        r0: 10.0,
        r_growth: 5.0,
        turns: 2.0,
    };
    let obs_cov = [0.0, 0.0, 0.0];
    let speed = 5.0;
    let duration_s = 60.0;
    let dt_obs_s = 1.0;
    let seed = 9;

    let obs = generate_curved_track(&traj, speed, duration_s, dt_obs_s, obs_cov, seed);
    let pos0 = obs[0].pos;
    let last = &obs[obs.len() - 1];
    let r0 = (pos0[0] * pos0[0] + pos0[1] * pos0[1]).sqrt();
    let rlast = (last.pos[0] * last.pos[0] + last.pos[1] * last.pos[1]).sqrt();
    assert!(
        rlast - r0 > 2.0,
        "radius growth = {} should be > 2.0 (r0={}, rlast={})",
        rlast - r0,
        r0,
        rlast
    );
}

/// Curved generate_curved_track: same seed + input -> bit-identical positions.
#[test]
fn curved_deterministic_from_seed() {
    let traj = Trajectory::Spiral {
        r0: 10.0,
        r_growth: 5.0,
        turns: 2.0,
    };
    let obs_cov = [1.0, 0.0, 1.0];
    let speed = 5.0;
    let duration_s = 60.0;
    let dt_obs_s = 1.0;

    let a = generate_curved_track(&traj, speed, duration_s, dt_obs_s, obs_cov, 12345);
    let b = generate_curved_track(&traj, speed, duration_s, dt_obs_s, obs_cov, 12345);

    assert_eq!(a.len(), b.len(), "observation counts must match");
    for (x, y) in a.iter().zip(b.iter()) {
        assert_eq!(x.pos, y.pos, "position mismatch at t={}", x.t);
        assert_eq!(x.psi, y.psi, "psi mismatch at t={}", x.t);
    }
}

/// Sanity: legacy generate_track still works (unchanged behaviour).
#[test]
fn legacy_generate_track_still_runs() {
    let waypoints = vec![KinState {
        pos: [0.0, 0.0],
        v: 5.0,
        psi: 0.0,
    }];
    let ttc = ttc();
    let obs = generate_track(&waypoints, &ttc, 1.0, [0.0, 0.0, 0.0], 5);
    assert_eq!(obs.len(), 1);
    assert_eq!(obs[0].pos, [0.0, 0.0]);
}

/// Zigzag psi sign matches geometry: leg 0 descends (vertex 0 at +h, vertex 1
/// at -h), so obs[0] and obs[1] have negative psi; leg 1 ascends, so obs[21]
/// has positive psi. Regression for the inverted-sign fix.
#[test]
fn zigzag_psi_sign_matches_geometry() {
    let traj = Trajectory::Zigzag {
        segment_m: 100.0,
        angle_rad: 0.3,
        baseline: 0.0,
    };
    let obs_cov = [0.0, 0.0, 0.0];
    let speed = 5.0;
    let duration_s = 60.0;
    let dt_obs_s = 1.0;
    let seed = 1;

    let obs = generate_curved_track(&traj, speed, duration_s, dt_obs_s, obs_cov, seed);
    assert!(obs.len() >= 22, "expected at least 22 observations");

    // obs[0] and obs[1] are in leg 0 (descends) -> psi < 0.
    assert!(
        obs[0].psi < 0.0,
        "obs[0].psi = {} should be negative (leg 0 descends)",
        obs[0].psi
    );
    assert!(
        obs[1].psi < 0.0,
        "obs[1].psi = {} should be negative (still in leg 0)",
        obs[1].psi
    );
    // obs[21] is in leg 1 (ascends) -> psi > 0.
    assert!(
        obs[21].psi > 0.0,
        "obs[21].psi = {} should be positive (leg 1 ascends)",
        obs[21].psi
    );
}

/// Spiral psi at t = 0 equals atan2(r0, r_growth) (the tangent direction),
/// within 1e-6. Regression for the analytic-spiral-heading fix.
#[test]
fn spiral_heading_at_t0_is_tangent() {
    let traj = Trajectory::Spiral {
        r0: 10.0,
        r_growth: 5.0,
        turns: 2.0,
    };
    let obs_cov = [0.0, 0.0, 0.0];
    let speed = 5.0;
    let duration_s = 60.0;
    let dt_obs_s = 1.0;
    let seed = 1;

    let obs = generate_curved_track(&traj, speed, duration_s, dt_obs_s, obs_cov, seed);
    let expected = (10.0_f64).atan2(5.0);
    assert!(
        (obs[0].psi - expected).abs() < 1e-6,
        "obs[0].psi = {} should equal atan2(10, 5) = {} within 1e-6",
        obs[0].psi,
        expected
    );
}
```

## File: tests/scenarios.rs

```rust
//! scenarios.rs — end-to-end Monte-Carlo prediction across the reference
//! scenarios from spec §7.
//!
//! Each test builds a scenario (same parser logic the CLI uses), runs
//! `mc::predict` from the last observation, and checks the published Stage 1
//! behaviour. Fixed seeds are used throughout; assertions use tolerances,
//! never exact floats (per spec §7).
//!
//! Stage 1 note: `oois` is an extension point and is NOT used for inference
//! (see `mc::predict`), so the "two OOIs" scenario is exercised as a
//! symmetric two-sided heading spread, and heading-sector width is driven by
//! turn-rate uncertainty (`p_turn`), not by position (`cov`) noise.

use predictor::histogram::psi_rem_euclid;
use predictor::mc::predict;
use predictor::prng::Xoshiro256ss;
use predictor::simulator::generate_track;
use predictor::types::{KinState, Observation, PredictConfig, Ttc};
use std::f64::consts::PI;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn ttc(v_min: f64, v_max: f64, omega_max: f64) -> Ttc {
    Ttc {
        v_min,
        v_max,
        omega_max,
    }
}

fn cfg(
    horizon: f64,
    n_samples: u32,
    p_turn: f64,
    azimuth_bins: u32,
    seed: u64,
    speed_noise: f64,
) -> PredictConfig {
    PredictConfig {
        horizon_s: horizon,
        n_samples,
        dt_s: 1.0, // reserved for Stage 2, unused in Stage 1
        p_turn,
        azimuth_bins,
        speed_noise_mps: speed_noise,
        seed,
    }
}

/// Last observation of a straight +X track ending at (x, y): heading 0,
/// speed 5 m/s. `predict` uses last.v / last.psi as-is (Stage 1).
fn straight_last_obs(x: f64, y: f64, cov: [f64; 3]) -> Observation {
    Observation {
        t: 0.0,
        pos: [x, y],
        v: 5.0,
        psi: 0.0,
        cov,
    }
}

/// Smallest circular sector window (start bin, width in bins) whose cumulative
/// density reaches `threshold`. Mirrors histogram::sector_width's algorithm so
/// callers can test membership of a heading in the sector.
fn smallest_window(
    bin_probs: &[f64],
    threshold: f64,
    n_bins: usize,
) -> (usize, usize) {
    let doubled: Vec<f64> =
        bin_probs.iter().cloned().chain(bin_probs.iter().cloned()).collect();
    for k in 1..=n_bins {
        let mut sum: f64 = doubled[0..k].iter().sum::<f64>();
        if sum >= threshold {
            return (0, k);
        }
        for start in 1..n_bins {
            sum += doubled[start + k - 1] - doubled[start - 1];
            if sum >= threshold {
                return (start, k);
            }
        }
    }
    (0, n_bins)
}

/// True (heading 0 -> bin 0) inside a sector window, with 0/2*pi seam wrap.
fn truth_in_sector(window: (usize, usize), n_bins: usize) -> bool {
    let (start, width) = window;
    for k in 0..width {
        if (start + k) % n_bins == 0 {
            return true;
        }
    }
    false
}

/// Sample a bin index from a probability mass function (inverse-CDF).
fn sample_bin(bin_probs: &[f64], rng: &mut Xoshiro256ss) -> usize {
    let u = rng.next_f64();
    let mut cum = 0.0;
    for (i, &p) in bin_probs.iter().enumerate() {
        cum += p;
        if u <= cum {
            return i;
        }
    }
    bin_probs.len() - 1
}

// ---------------------------------------------------------------------------
// 1. straight_1_ooi_no_noise
// ---------------------------------------------------------------------------

#[test]
fn straight_1_ooi_no_noise() {
    // Zero sensor noise (cov = [0,0,0]) and no turn probability -> the point
    // cloud collapses to a single arc, so the 90% sector is a single bin.
    let obs = straight_last_obs(20.0, 0.0, [0.0, 0.0, 0.0]);
    let t = ttc(2.0, 8.0, 0.5);
    let c = cfg(10.0, 10000, 0.0, 72, 42, 0.0);
    let prediction = predict(&obs, &t, &[], &c);

    let sector90_deg = prediction.sector90 * 180.0 / PI;
    assert!(
        sector90_deg < 10.0,
        "sector90_deg = {}, expected < 10 (no noise)",
        sector90_deg
    );

    // Ground truth (heading 0) must be inside the 90% sector.
    let window = smallest_window(&prediction.bin_probs, 0.9, 72);
    assert!(
        truth_in_sector(window, 72),
        "ground truth not inside the 90% sector"
    );
}

// ---------------------------------------------------------------------------
// 2. two_oois_between
// ---------------------------------------------------------------------------

#[test]
fn two_oois_between() {
    // Two symmetric OOIs with the UAV between them -> in Stage 1 (oois unused
    // for inference) this shows up as a symmetric two-sided heading spread:
    // both the + and - heading regions carry significant probability mass.
    let obs = straight_last_obs(20.0, 0.0, [1.0, 0.0, 1.0]);
    let t = ttc(2.0, 8.0, 0.5);
    let c = cfg(10.0, 20000, 1.0, 72, 7, 0.0);
    let prediction = predict(&obs, &t, &[], &c);

    let n = prediction.bin_probs.len();
    let bin_width = 2.0 * PI / n as f64;
    let mut right: f64 = 0.0; // heading in (0, pi)
    let mut left: f64 = 0.0; // heading in (-pi, 0)
    for i in 0..n {
        let heading = (i as f64 + 0.5) * bin_width; // bin center in [0, 2*pi)
        let h = if heading > PI { heading - 2.0 * PI } else { heading };
        let p = prediction.bin_probs[i];
        if h > 0.0 {
            right = right.max(p);
        } else {
            left = left.max(p);
        }
    }

    assert!(
        right > 0.01 && left > 0.01,
        "expected mass on both sides: right max = {right}, left max = {left}"
    );
    // Two-sided spread: the 90% sector is wide (well over a quarter circle).
    let sector90_deg = prediction.sector90 * 180.0 / PI;
    assert!(
        sector90_deg > 90.0,
        "sector90_deg = {}, expected wide two-sided spread",
        sector90_deg
    );
}

// ---------------------------------------------------------------------------
// 3. noisy_track_wide
// ---------------------------------------------------------------------------

#[test]
fn noisy_track_wide() {
    // Large sensor noise (cov = [9,0,9], sigma = 3 m/axis) plus turn
    // uncertainty -> wide point cloud and wide heading sector, but the ground
    // truth (heading 0) stays inside the 90% sector.
    let obs = straight_last_obs(20.0, 0.0, [9.0, 0.0, 9.0]);
    let t = ttc(2.0, 8.0, 0.5);
    let c = cfg(10.0, 20000, 1.0, 72, 13, 0.0);
    let prediction = predict(&obs, &t, &[], &c);

    // Wide heading sector.
    let sector90_deg = prediction.sector90 * 180.0 / PI;
    assert!(
        sector90_deg > 60.0,
        "sector90_deg = {}, expected wide (> 60)",
        sector90_deg
    );

    // Wide point cloud: max displacement of a final point from the start.
    let start = [20.0, 0.0];
    let max_dev: f64 = prediction
        .point_cloud
        .iter()
        .map(|&p| (p[0] - start[0]).powi(2) + (p[1] - start[1]).powi(2))
        .fold(0.0, f64::max)
        .sqrt();
    assert!(
        max_dev > 20.0,
        "point cloud max dev = {}, expected wide (> 20)",
        max_dev
    );

    // Ground truth still inside the 90% sector.
    let window = smallest_window(&prediction.bin_probs, 0.9, 72);
    assert!(
        truth_in_sector(window, 72),
        "ground truth not inside the 90% sector despite wide noise"
    );
}

// ---------------------------------------------------------------------------
// 4. low_vs_high_turn_rate
// ---------------------------------------------------------------------------

#[test]
fn low_vs_high_turn_rate() {
    // omega_max = 0.05 vs 1.0, all else equal. The high-turn-rate run must
    // produce a strictly wider 90% sector.
    let obs = straight_last_obs(20.0, 0.0, [1.0, 0.0, 1.0]);

    let t_low = ttc(2.0, 8.0, 0.05);
    let c_low = cfg(10.0, 20000, 1.0, 72, 42, 0.0);
    let p_low = predict(&obs, &t_low, &[], &c_low);
    let sector90_low = p_low.sector90 * 180.0 / PI;

    let t_high = ttc(2.0, 8.0, 1.0);
    let c_high = cfg(10.0, 20000, 1.0, 72, 42, 0.0);
    let p_high = predict(&obs, &t_high, &[], &c_high);
    let sector90_high = p_high.sector90 * 180.0 / PI;

    assert!(
        sector90_high > sector90_low,
        "high-turn sector90 ({}) must be strictly wider than low-turn ({})",
        sector90_high,
        sector90_low
    );
}

// Sector-mass consistency check: sampling a bin from the
// predicted distribution and testing membership in the
// extracted sector must succeed with probability equal to
// that sector's accumulated mass. Because the sector is
// chosen to cover >= threshold of the distribution, this
// succeeds in [0.85, 0.95] for 90% and [0.45, 0.55] for
// 50% by construction. This does NOT test calibration of
// predict(); true coverage against independent ground
// truth requires heading uncertainty that Stage 1 does
// not produce.
//
// TODO(Stage2): replace with a calibration test that
// simulates a noisy truth track, predicts from its last
// observation, and checks whether the truth heading falls
// inside the extracted sector.

#[test]
#[ignore]
fn sector_mass_consistency_90() {
    // 1000 different seeds. For each seed: extract the 90% sector,
    // sample a heading from the same distribution, count membership.
    // Expected fraction is ~0.90 by construction (sector covers 90%
    // of the distribution). This is a consistency check, not a
    // calibration test; see the docstring above for why.
    let obs = straight_last_obs(20.0, 0.0, [1.0, 0.0, 1.0]);
    let t = ttc(2.0, 8.0, 0.5);
    let n_tracks = 1000u64;
    let mut inside = 0u64;

    for seed in 1..=n_tracks {
        let c = cfg(10.0, 20000, 1.0, 72, seed, 0.0);
        let prediction = predict(&obs, &t, &[], &c);
        let window = smallest_window(&prediction.bin_probs, 0.9, 72);
        let mut rng = Xoshiro256ss::from_seed(seed.wrapping_mul(2654435761));
        let j = sample_bin(&prediction.bin_probs, &mut rng);
        let (start, width) = window;
        let mut in_win = false;
        for k in 0..width {
            if (start + k) % 72 == j {
                in_win = true;
                break;
            }
        }
        if in_win {
            inside += 1;
        }
    }

    let frac = inside as f64 / n_tracks as f64;
    assert!(
        frac >= 0.85 && frac <= 0.95,
        "sector_mass_consistency_90 = {}, expected in [0.85, 0.95]",
        frac
    );
}

// Sector-mass consistency check: sampling a bin from the
// predicted distribution and testing membership in the
// extracted sector must succeed with probability equal to
// that sector's accumulated mass. Because the sector is
// chosen to cover >= threshold of the distribution, this
// succeeds in [0.85, 0.95] for 90% and [0.45, 0.55] for
// 50% by construction. This does NOT test calibration of
// predict(); true coverage against independent ground
// truth requires heading uncertainty that Stage 1 does
// not produce.
//
// TODO(Stage2): replace with a calibration test that
// simulates a noisy truth track, predicts from its last
// observation, and checks whether the truth heading falls
// inside the extracted sector.

#[test]
#[ignore]
fn sector_mass_consistency_50() {
    // Same as sector_mass_consistency_90 for the 50% sector.
    // Expected fraction ~0.50 by construction.
    let obs = straight_last_obs(20.0, 0.0, [1.0, 0.0, 1.0]);
    let t = ttc(2.0, 8.0, 0.5);
    let n_tracks = 1000u64;
    let mut inside = 0u64;

    for seed in 1..=n_tracks {
        let c = cfg(10.0, 20000, 1.0, 72, seed, 0.0);
        let prediction = predict(&obs, &t, &[], &c);
        let window = smallest_window(&prediction.bin_probs, 0.5, 72);
        let mut rng = Xoshiro256ss::from_seed(seed.wrapping_mul(2654435761));
        let j = sample_bin(&prediction.bin_probs, &mut rng);
        let (start, width) = window;
        let mut in_win = false;
        for k in 0..width {
            if (start + k) % 72 == j {
                in_win = true;
                break;
            }
        }
        if in_win {
            inside += 1;
        }
    }

    let frac = inside as f64 / n_tracks as f64;
    assert!(
        frac >= 0.45 && frac <= 0.55,
        "sector_mass_consistency_50 = {}, expected in [0.45, 0.55]",
        frac
    );
}

// ---------------------------------------------------------------------------
// Cross-check against the committed CLI reference output (scenarios/simple.txt)
// ---------------------------------------------------------------------------

#[test]
fn reference_scenario_numbers() {
    // Last obs of simple.txt: straight +X, v = 5, psi = 0, cov = [1,0,1].
    let obs = straight_last_obs(20.0, 0.0, [1.0, 0.0, 1.0]);
    let t = ttc(2.0, 8.0, 0.5);
    let c = cfg(10.0, 10000, 0.3, 72, 42, 0.5);
    let prediction = predict(&obs, &t, &[], &c);

    let sector50_deg = prediction.sector50 * 180.0 / PI;
    let sector90_deg = prediction.sector90 * 180.0 / PI;
    let heading_mean_deg = psi_rem_euclid(prediction.final_heading_mean) * 180.0 / PI;
    let n_bins_nonzero = prediction
        .bin_probs
        .iter()
        .filter(|&&p| p > 0.0)
        .count();

    assert!(
        (sector50_deg - 5.0).abs() < 1.0,
        "sector50_deg = {}, expected ~5",
        sector50_deg
    );
    assert!(
        (sector90_deg - 180.0).abs() < 3.0,
        "sector90_deg = {}, expected ~180",
        sector90_deg
    );
    let h = if heading_mean_deg > 180.0 {
        heading_mean_deg - 360.0
    } else {
        heading_mean_deg
    };
    assert!(h.abs() < 2.0, "heading_mean_deg = {}, expected near 0", heading_mean_deg);
    assert!(
        n_bins_nonzero >= 70,
        "n_bins_nonzero = {}, expected ~72",
        n_bins_nonzero
    );
}

// ---------------------------------------------------------------------------
// Simulator + predictor integration (exercises generate_track end to end)
// ---------------------------------------------------------------------------

#[test]
fn simulator_then_predict() {
    // Build a straight +X truth track with the simulator, then predict from
    // its last observation.
    let start = KinState {
        pos: [0.0, 0.0],
        v: 5.0,
        psi: 0.0,
    };
    let t = ttc(2.0, 8.0, 0.5);
    let track = generate_track(&[start], &t, 1.0, [1.0, 0.0, 1.0], 42);
    assert_eq!(track.len(), 1, "one waypoint -> one observation");

    let last = &track[0];
    // First observation carries v = start.v, psi = start.psi
    // (simulator copies the initial KinState into every
    // observation for Stage 1).
    assert!(
        (last.v - 5.0).abs() < 1e-9,
        "simulator sets v from start: got {}",
        last.v
    );
    assert!(
        (last.psi - 0.0).abs() < 1e-9,
        "simulator sets psi from start: got {}",
        last.psi
    );

    let c = cfg(10.0, 10000, 0.3, 72, 42, 0.5);
    let prediction = predict(last, &t, &[], &c);
    let sum: f64 = prediction.bin_probs.iter().sum();
    assert!((sum - 1.0).abs() < 1e-6, "bin_probs sum = {}, expected ~1", sum);
    assert_eq!(
        prediction.bin_probs.len(),
        c.azimuth_bins as usize,
        "bin count == azimuth_bins"
    );
}
```

## File: tests/filter.rs

```rust
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
```
