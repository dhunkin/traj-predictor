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
    pub seed: u64,
}

pub struct Prediction {
    pub bin_centers: Vec<f64>,
    pub bin_probs: Vec<f64>,
    pub sector50: f64,
    pub sector90: f64,
    pub point_cloud: Vec<[f64; 2]>,
    pub final_heading_mean: f64,
}
