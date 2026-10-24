//! mc.rs — Monte-Carlo engine.
//!
//! Public entry point for heading prediction over `horizon_s`. Delegates to
//! the unified particle filter (`particle::run_particle_filter`), which learns
//! the UAV state from the full `track` and fans each particle forward to build
//! the final point cloud, the azimuth histogram, and the 50%/90% sectors.
//! See §4.4.
//!
//! TODO(Stage2): use OOIs for inference.

use crate::particle::run_particle_filter;
use crate::types::{Observation, Ooi, PredictConfig, Prediction, Ttc};

pub fn predict(
    track: &[Observation],
    ttc: &Ttc,
    oois: &[Ooi],
    cfg: &PredictConfig,
) -> Prediction {
    let _ = oois; // Stage 2: OOIs are an extension point, not yet used for inference.

    // Unified particle filter over the full track: learns the state from all
    // observations, then fans forward to the azimuth histogram.
    run_particle_filter(track, ttc, cfg)
}
