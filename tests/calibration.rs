//! calibration.rs — coverage of the heading sectors against the true future,
//! over the scenario suite in tests/common/calib.rs. A small-sample guard
//! against regressions; tune with `cargo run --release --example calibrate`.
//!
//! The suite pools families with very different predictability (a straight
//! line always lands in the core of the fan; a turn that starts after the
//! last observation cannot be foreseen), so only the pooled coverage is held
//! near nominal; per-family coverage gets loose floors.

mod common;

use common::calib::*;
use predictor::particle::ModelParams;

#[test]
fn sectors_are_calibrated_on_the_scenario_suite() {
    let mp = ModelParams::default();
    let all = core_families();
    let per = run(&mp, 40, 2000, &all);
    let real: Vec<Outcome> = all
        .iter()
        .filter(|&&f| f != model_family())
        .flat_map(|&f| per[f].clone())
        .collect();
    let pooled = summarise(&real);
    assert!(
        (0.85..=0.97).contains(&pooled.cov90),
        "pooled 90% coverage = {:.2}",
        pooled.cov90
    );
    assert!(
        (0.45..=0.75).contains(&pooled.cov50),
        "pooled 50% coverage = {:.2}",
        pooled.cov50
    );
    for &f in all.iter() {
        let s = summarise(&per[f]);
        assert!(s.cov90 >= 0.6, "{}: 90% coverage = {:.2}", FAMILIES[f], s.cov90);
    }
    // Self-consistency: truth drawn from the model itself.
    let m = summarise(&per[model_family()]);
    assert!(m.cov90 >= 0.75 && m.cov50 >= 0.3, "model family: {:.2} / {:.2}", m.cov50, m.cov90);
}
