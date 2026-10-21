//! mc.rs — Monte-Carlo engine.
//! TODO(Stage2): replace initial-position sampling with filter state estimate.

use crate::types::{Observation, Ooi, PredictConfig, Prediction, Ttc};

pub fn predict(
    _last: &Observation,
    _ttc: &Ttc,
    _oois: &[Ooi],
    _cfg: &PredictConfig,
) -> Prediction {
    // Placeholder body for Step 1; real impl in later steps.
    Prediction {
        bin_centers: Vec::new(),
        bin_probs: Vec::new(),
        sector50: 0.0,
        sector90: 0.0,
        point_cloud: Vec::new(),
        final_heading_mean: 0.0,
    }
}
