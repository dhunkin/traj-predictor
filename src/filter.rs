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

/// 2x2 matrix inverse with a scaled diagonal regularizer so a (near-)singular
/// `S` still yields a finite gain.
fn inverse22(a: [[f64; 2]; 2]) -> [[f64; 2]; 2] {
    let eps = 1e-12 * a[0][0].abs().max(a[1][1].abs()).max(1e-12);
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
    /// covariance for the position block and a large variance for velocity.
    /// The initialization already represents the posterior (no extra update).
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
        // P' = (I - K H) P (I - K H)^T + K R K^T
        let kh = mul_4x2_by_2x4(k, h);
        let ikh = sub_from_identity(kh);
        let ikh_t = transpose44(ikh);

        // (I - KH) P (I - KH)^T
        let mut p = mul_4x4(mul_4x4(ikh, self.p), ikh_t);

        // + K R K^T
        let kr = mul_4x2_by_2x2(k, r);
        for i in 0..4 {
            for j in 0..4 {
                p[i][j] += kr[i][0] * k[j][0] + kr[i][1] * k[j][1];
            }
        }

        // Enforce symmetry
        for i in 0..4 {
            for j in (i + 1)..4 {
                let m = 0.5 * (p[i][j] + p[j][i]);
                p[i][j] = m;
                p[j][i] = m;
            }
        }
        self.p = p;
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
    for i in 1..track.len() {
        assert!(track[i].t > track[i-1].t, "Track timestamps must be strictly increasing");
    }
    let mut kf = KalmanFilter::new(&track[0]);
    for obs in &track[1..] {
        let dt = obs.t - kf.last_t;
        kf.predict(dt, Q_SIGMA);
        kf.update(obs);
    }
    kf.get_state()
}
