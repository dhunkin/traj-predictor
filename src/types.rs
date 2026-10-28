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
    /// Stationary probability of the coordinated-turn mode (|omega| >= 0.1
    /// rad/s). Also the initial P(Turn). 1.0 forbids straight flight.
    pub p_turn: f64,
    pub azimuth_bins: u32,
    pub speed_noise_mps: f64,
    /// Stage 3: prior probability that the UAV is heading for one of the
    /// `oois` (split evenly between them); `1 - intent_hypothesis_prob` is the
    /// prior of "no goal". It is only a prior: the filter updates it from the
    /// track, and the posterior is reported as `Prediction::goal_probs`.
    pub intent_hypothesis_prob: f64,
    pub seed: u64,
}

pub struct Prediction {
    pub bin_centers: Vec<f64>,
    pub bin_probs: Vec<f64>,
    pub sector50: f64,
    pub sector90: f64,
    /// Start angle (radians, [0, 2*pi)) of the 50% / 90% sectors. A sector
    /// spans `[start, start + width]` counter-clockwise. Found from the
    /// histogram, so it is correct for a multi-modal fan where the circular
    /// mean may lie between the modes.
    pub sector50_start: f64,
    pub sector90_start: f64,
    pub point_cloud: Vec<[f64; 2]>,
    /// Per-particle trajectory polylines (one `Vec<[f64;2]>` per particle), each
    /// sampled every prediction step from the post-filter position to the horizon
    /// end. `point_cloud` only keeps final positions; `path_cloud` retains the
    /// full arc so the demo can draw the ribbon as distinct curved lines.
    pub path_cloud: Vec<Vec<[f64; 2]>>,
    /// Goal label of each path in `path_cloud` (index into `oois`, or `None`).
    pub path_goal: Vec<Option<usize>>,
    pub final_heading_mean: f64,
    pub v_est: f64,
    pub omega_est: f64,
    /// Posterior P(goal = oois[k] | track) at the last observation, one entry
    /// per OOI. `1 - sum` is the posterior of "no goal".
    pub goal_probs: Vec<f64>,
    /// Posterior probability of the coordinated-turn mode at the last
    /// observation.
    pub turn_prob: f64,
    /// How many times the filter lost the track and re-seeded from the
    /// latest observations (0 when the target stays within the model).
    pub recoveries: u32,
}
