//! types.rs — KinState, Observation, Ttc, Ooi, config, result.
//! TODO(Stage2): (reserved)

// ---- Placeholder items for Step 1 (filled in later steps) ----

#[derive(Clone)]
pub struct KinState {
    pub pos: [f64; 2],
    pub v: f64,
    pub psi: f64,
}

pub struct Observation {
    pub t: f64,
    pub pos: [f64; 2],
    pub cov: [f64; 3],
}

pub struct Ttc {
    pub v_min: f64,
    pub v_max: f64,
    pub omega_max: f64,
}

pub struct Ooi {
    pub pos: [f64; 2],
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
