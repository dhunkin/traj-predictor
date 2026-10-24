//! Stage 1 baseline — UAV heading prediction (Rust + WASM).
//!
//! Public re-exports and feature gates for the predictor core.
//! The native (non-wasm) target compiles with `default = []`.
//! The `wasm` feature pulls in wasm-bindgen + js-sys only.

pub mod filter;
pub mod histogram;
pub mod mc;
pub mod motion;
pub mod prng;
pub mod simulator;
pub mod types;
pub mod vec2;

#[cfg(feature = "wasm")]
pub mod wasm;

pub use types::{
    KinState, Observation, Ooi, PredictConfig, Prediction, Ttc,
};

pub use mc::predict;
pub use simulator::generate_track;
pub use filter::filter_track;
