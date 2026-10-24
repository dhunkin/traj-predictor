//! wasm.rs — #[cfg(feature = "wasm")] wasm-bindgen wrappers.
//!
//! Thin, dependency-free adapters that expose the native core
//! (`Simulator` + `Predictor`) to JavaScript through typed-array
//! boundaries. No JSON, no serde: everything crosses as `Box<[f64]>`
//! (Float64Array) or `Box<[f64]>` for the point cloud.
//!
//! Public API (see spec §3):
//!   - `init()`            — wasm-bindgen start hook (installs panic hook).
//!   - `Simulator`         — synthetic straight-line track generator.
//!   - `Predictor`         — Monte-Carlo heading predictor.
//!   - `PredictionResult`  — typed-array getters over a `Prediction`.
//!
//! TODO(Stage2): (reserved)
//!
//! NOTE: This module is only compiled when the `wasm` feature is active,
//! so `wasm_bindgen` and `js_sys` are the only deps it touches.

use wasm_bindgen::prelude::*;

use crate::mc::predict;
use crate::prng::Xoshiro256ss;
use crate::simulator::{generate_curved_track, Trajectory};
use crate::types::{Observation, Ooi, PredictConfig, Ttc};
use crate::vec2::Mat2;

/// wasm-bindgen start hook.
///
/// Called automatically when the module is imported into a JS environment.
/// Installs a panic hook so Rust panics surface as JS exceptions (via the
/// global `console.error`) instead of aborting the whole WASM module.
#[wasm_bindgen(start)]
pub fn init() {
    std::panic::set_hook(Box::new(|info| {
        let msg = info.to_string();
        // `{:?}` on a Rust string produces a valid JS string literal with
        // proper escaping; eval it against the JS `console`.
        let code = format!("console.error({:?});", msg);
        if let Err(e) = js_sys::eval(&code) {
            let _ = e;
        }
    }));
}

/// Synthetic track simulator (Stage 1: straight-line truth only).
///
/// Layout of `generate_track_from` output: flat `Box<[f64]>` of
/// `[t, x, y, cov_xx, cov_xy, cov_yy]` triples, one per observation.
#[wasm_bindgen]
pub struct Simulator {
    seed: u64,
    /// Turn-rate limits, accepted by the constructor for Stage 2 curved-path
    /// support; unused in Stage 1 (straight-line truth only).
    #[allow(dead_code)]
    ttc: Ttc,
    dt_obs_s: f64,
    obs_cov: [f64; 3],
}

#[wasm_bindgen]
impl Simulator {
    #[wasm_bindgen(constructor)]
    pub fn new(
        seed: u64,
        v_min: f64,
        v_max: f64,
        a_lat: f64,
        a_long: f64,
        omega_cap: f64,
        dt_obs: f64,
        cov_xx: f64,
        cov_xy: f64,
        cov_yy: f64,
    ) -> Simulator {
        Simulator {
            seed,
            ttc: Ttc {
                v_min,
                v_max,
                a_lat,
                a_long,
                omega_cap,
            },
            dt_obs_s: dt_obs,
            obs_cov: [cov_xx, cov_xy, cov_yy],
        }
    }

    /// Generate a straight-line truth track from `start`.
    ///
    /// Returns a flat `Box<[f64]>`: `[t, x, y, cov_xx, cov_xy, cov_yy, ...]`.
    pub fn generate_track_from(
        &self,
        start_x: f64,
        start_y: f64,
        heading_rad: f64,
        speed: f64,
        duration_s: f64,
    ) -> Box<[f64]> {
        let n_legs = (duration_s / self.dt_obs_s).floor() as usize;
        let mut rng = Xoshiro256ss::from_seed(self.seed);
        let mat = Mat2::from_cov(self.obs_cov);

        let mut out: Vec<f64> = Vec::with_capacity((n_legs + 1) * 6);
        for i in 0..=n_legs {
            let t = i as f64 * self.dt_obs_s;
            let truth_pos = [
                start_x + speed * t * heading_rad.cos(),
                start_y + speed * t * heading_rad.sin(),
            ];
            let noise = mat.sample(&mut rng);
            out.push(t);
            out.push(truth_pos[0] + noise.x);
            out.push(truth_pos[1] + noise.y);
            out.push(self.obs_cov[0]);
            out.push(self.obs_cov[1]);
            out.push(self.obs_cov[2]);
        }
        out.into_boxed_slice()
    }

    /// Generate a curved track using the `Spiral` trajectory.
    ///
    /// Returns a flat `Box<[f64]>` of `[t, x, y, v, psi, cov_xx, cov_xy, cov_yy]`
    /// repeated, one 8-field group per observation.
    pub fn generate_spiral(
        &self,
        r0: f64,
        r_growth: f64,
        turns: f64,
        speed: f64,
        duration: f64,
    ) -> Box<[f64]> {
        let traj = Trajectory::Spiral {
            r0,
            r_growth,
            turns,
        };
        generate_curved_track(&traj, speed, duration, self.dt_obs_s, self.obs_cov, self.seed)
            .into_flat()
    }

    /// Generate a curved track using the `Sine` trajectory.
    ///
    /// Returns a flat `Box<[f64]>` of `[t, x, y, v, psi, cov_xx, cov_xy, cov_yy]`
    /// repeated, one 8-field group per observation.
    pub fn generate_sine(
        &self,
        wavelength: f64,
        amplitude: f64,
        direction: f64,
        speed: f64,
        duration: f64,
    ) -> Box<[f64]> {
        let traj = Trajectory::Sine {
            wavelength_m: wavelength,
            amplitude_m: amplitude,
            direction,
        };
        generate_curved_track(&traj, speed, duration, self.dt_obs_s, self.obs_cov, self.seed)
            .into_flat()
    }

    /// Generate a curved track using the `Zigzag` trajectory.
    ///
    /// Returns a flat `Box<[f64]>` of `[t, x, y, v, psi, cov_xx, cov_xy, cov_yy]`
    /// repeated, one 8-field group per observation.
    pub fn generate_zigzag(
        &self,
        segment: f64,
        angle: f64,
        baseline: f64,
        speed: f64,
        duration: f64,
    ) -> Box<[f64]> {
        let traj = Trajectory::Zigzag {
            segment_m: segment,
            angle_rad: angle,
            baseline_m: baseline,
        };
        generate_curved_track(&traj, speed, duration, self.dt_obs_s, self.obs_cov, self.seed)
            .into_flat()
    }

    /// Generate a straight track using the `Straight` trajectory.
    ///
    /// Returns a flat `Box<[f64]>` of `[t, x, y, v, psi, cov_xx, cov_xy, cov_yy]`
    /// repeated, one 8-field group per observation.
    pub fn generate_straight(
        &self,
        psi0: f64,
        speed: f64,
        duration: f64,
    ) -> Box<[f64]> {
        let traj = Trajectory::Straight { psi0 };
        generate_curved_track(&traj, speed, duration, self.dt_obs_s, self.obs_cov, self.seed)
            .into_flat()
    }
}

/// Flatten an `Observation` into the 8-field group
/// `[t, x, y, v, psi, cov_xx, cov_xy, cov_yy]`.
trait IntoFlat {
    fn into_flat(self) -> Box<[f64]>;
}

impl IntoFlat for Vec<Observation> {
    fn into_flat(self) -> Box<[f64]> {
        let mut out: Vec<f64> = Vec::with_capacity(self.len() * 8);
        for o in self {
            out.push(o.t);
            out.push(o.pos[0]);
            out.push(o.pos[1]);
            out.push(o.v);
            out.push(o.psi);
            out.push(o.cov[0]);
            out.push(o.cov[1]);
            out.push(o.cov[2]);
        }
        out.into_boxed_slice()
    }
}

/// Monte-Carlo heading predictor.
#[wasm_bindgen]
pub struct Predictor {
    ttc: Ttc,
}

#[wasm_bindgen]
impl Predictor {
    #[wasm_bindgen(constructor)]
    pub fn new(
        v_min: f64,
        v_max: f64,
        a_lat: f64,
        a_long: f64,
        omega_cap: f64,
    ) -> Predictor {
        Predictor {
            ttc: Ttc {
                v_min,
                v_max,
                a_lat,
                a_long,
                omega_cap,
            },
        }
    }

    /// Run the Monte-Carlo prediction from a single observation.
    ///
    /// Arguments mirror the scenario's last observation + config. Returns a
    /// `PredictionResult` whose getters expose typed arrays.
    #[wasm_bindgen(js_name = predict)]
    pub fn predict(
        &self,
        last_t: f64,
        last_x: f64,
        last_y: f64,
        cov_xx: f64,
        cov_xy: f64,
        cov_yy: f64,
        heading_rad: f64,
        speed: f64,
        horizon_s: f64,
        n_samples: u32,
        p_turn: f64,
        azimuth_bins: u32,
        speed_noise_mps: f64,
        seed: u64,
    ) -> PredictionResult {
        let last = Observation {
            t: last_t,
            pos: [last_x, last_y],
            v: speed,
            psi: heading_rad,
            cov: [cov_xx, cov_xy, cov_yy],
        };
        let oois: Vec<Ooi> = Vec::new(); // Stage 1: OOIs are an extension point.
        let cfg = PredictConfig {
            horizon_s,
            n_samples,
            dt_s: 0.0, // reserved for Stage 2
            p_turn,
            azimuth_bins,
            speed_noise_mps,
            seed,
        };

        let prediction = predict(&last, &self.ttc, &oois, &cfg);
        PredictionResult::from_prediction(prediction)
    }
}

/// Result of a Monte-Carlo prediction, exposed to JS via typed arrays.
#[wasm_bindgen]
pub struct PredictionResult {
    bin_centers: Box<[f64]>,
    bin_probs: Box<[f64]>,
    point_cloud: Box<[f64]>,
    sector50: f64,
    sector90: f64,
    final_heading_mean: f64,
}

#[wasm_bindgen]
impl PredictionResult {
    fn from_prediction(p: crate::types::Prediction) -> PredictionResult {
        let mut point_cloud = Vec::with_capacity(p.point_cloud.len() * 2);
        for pt in &p.point_cloud {
            point_cloud.push(pt[0]);
            point_cloud.push(pt[1]);
        }

        PredictionResult {
            bin_centers: p.bin_centers.into_boxed_slice(),
            bin_probs: p.bin_probs.into_boxed_slice(),
            point_cloud: point_cloud.into_boxed_slice(),
            sector50: p.sector50,
            sector90: p.sector90,
            final_heading_mean: p.final_heading_mean,
        }
    }

    /// `Float64Array` of azimuth bin centers (radians), length == azimuth_bins.
    pub fn bin_centers(&self) -> Box<[f64]> {
        self.bin_centers.clone()
    }

    /// `Float64Array` of bin densities (sum to 1), length == azimuth_bins.
    pub fn bin_probs(&self) -> Box<[f64]> {
        self.bin_probs.clone()
    }

    /// `Float64Array` of the point cloud as `[x0, y0, x1, y1, ...]`.
    pub fn point_cloud(&self) -> Box<[f64]> {
        self.point_cloud.clone()
    }

    /// 50% sector width (radians).
    pub fn sector50(&self) -> f64 {
        self.sector50
    }

    /// 90% sector width (radians).
    pub fn sector90(&self) -> f64 {
        self.sector90
    }

    /// Circular-mean final heading (radians, in (-pi, pi]).
    pub fn final_heading_mean(&self) -> f64 {
        self.final_heading_mean
    }
}
