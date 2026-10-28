//! intent.rs — regression tests for goal (intent) inference.
//!
//! The posterior P(goal | track) must respond to the data: it rises when the
//! track behaves like the goal model and falls when it does not. Each check
//! averages a few seeds; thresholds leave a margin around the values printed by
//! `cargo run --release --example intent_probe` (prior P(goal) = 0.5).

mod common;

use common::intent::*;
use predictor::particle::predict;
use predictor::types::{Ooi, Prediction};
use std::f64::consts::PI;

const SEEDS: [u64; 5] = [1, 2, 3, 4, 5];

/// Mean P(goal) and P(turn) over SEEDS, using the first `t_cut` seconds.
fn posterior(duration: f64, t_cut: f64, oois: &[Ooi], truth: impl Fn(f64) -> [f64; 2]) -> (f64, f64) {
    let n = (t_cut / DT) as usize + 1;
    let (mut pg, mut pt) = (0.0, 0.0);
    for &s in SEEDS.iter() {
        let tr = observe(duration, s, &truth);
        let p = predict(&tr[..n], &ttc(), oois, &cfg(s));
        pg += p.goal_probs.iter().sum::<f64>();
        pt += p.turn_prob;
    }
    let k = SEEDS.len() as f64;
    (pg / k, pt / k)
}

#[test]
fn straight_at_ooi_keeps_goal() {
    // A straight line at the OOI is explained equally well by "no goal, flying
    // straight", so the posterior must not fall (it stays near / above prior).
    let (pg, pt) = posterior(40.0, 40.0, &ooi(600.0, 0.0), straight(8.0, 0.0));
    assert!(pg > 0.4, "P(goal) = {pg}, expected >= prior-ish");
    assert!(pt < 0.3, "P(turn) = {pt}, expected low on a straight line");
}

#[test]
fn straight_away_from_ooi_drops_goal() {
    let (pg, _) = posterior(40.0, 40.0, &ooi(-300.0, 0.0), straight(8.0, 0.0));
    assert!(pg < 0.3, "P(goal) = {pg}, expected to fall below the prior");
}

#[test]
fn straight_past_ooi_falls_after_passing() {
    // OOI 30 m off the line, closest approach at t ~ 19 s. Early on the OOI is
    // nearly ahead, so the goal is plausible; as its bearing swings off the
    // nose without the UAV turning toward it, the goal loses support (already
    // before the closest approach), and it is ruled out once passed.
    let oois = ooi(150.0, 30.0);
    let (before, _) = posterior(40.0, 10.0, &oois, straight(8.0, 0.0));
    let (after, _) = posterior(40.0, 40.0, &oois, straight(8.0, 0.0));
    assert!(before > 0.3, "P(goal) at t = 10 s = {before}");
    assert!(after < 0.1, "P(goal) after passing = {after}");
    assert!(after < before - 0.2, "expected a clear drop: {before} -> {after}");
}

#[test]
fn homing_onto_ooi_raises_goal() {
    // Truth follows the goal model (heading bends toward the OOI).
    let g = [200.0, 250.0];
    let (pg, _) = posterior(40.0, 40.0, &ooi(g[0], g[1]), homing(8.0, 0.0, g, 0.1));
    assert!(pg > 0.55, "P(goal) = {pg}, expected above the prior");
}

#[test]
fn circle_in_place_drops_goal_keeps_turning() {
    let (pg, pt) = posterior(40.0, 40.0, &ooi(0.0, 300.0), drifting_circle(20.0, 0.4, [0.0, 20.0], 0.0, 0.0));
    assert!(pg < 0.3, "P(goal) = {pg}, expected to fall toward 0");
    // With cruise_tau_s = 8 s a cruise particle can hold a turn for a while,
    // so the modes overlap more; Turn must still be the majority.
    assert!(pt > 0.5, "P(turn) = {pt}, expected the turn mode to dominate");
}

#[test]
fn circle_drifting_toward_ooi_raises_goal() {
    let (pg, pt) = posterior(40.0, 40.0, &ooi(300.0, 20.0), drifting_circle(20.0, 0.4, [0.0, 20.0], 0.6, 0.0));
    assert!(pg > 0.7, "P(goal) = {pg}, expected high");
    assert!(pt > 0.6, "P(turn) = {pt}");
}

#[test]
fn circle_drifting_away_drops_goal() {
    let (pg, _) = posterior(40.0, 40.0, &ooi(-300.0, 20.0), drifting_circle(20.0, 0.4, [0.0, 20.0], 0.6, 0.0));
    assert!(pg < 0.2, "P(goal) = {pg}, expected low");
}

#[test]
fn no_oois_means_no_goal_mass() {
    let tr = observe(20.0, 1, straight(8.0, 0.0));
    let p = predict(&tr, &ttc(), &[], &cfg(1));
    assert!(p.goal_probs.is_empty());
    assert!(p.path_goal.iter().all(|g| g.is_none()));
}

/// True 10 s displacement bearing inside the reported 90% sector?
fn truth_in_90(p: &Prediction, a: [f64; 2], b: [f64; 2]) -> bool {
    let brg = (b[1] - a[1]).atan2(b[0] - a[0]);
    (brg - p.sector90_start).rem_euclid(2.0 * PI) <= p.sector90
}

#[test]
fn ninety_percent_sector_covers_true_future() {
    // Coverage against the unobserved truth, not just self-consistency. The
    // sector is located from the histogram (start + width), so it stays on the
    // right mode even when the fan is a mixture.
    let cases: Vec<(Vec<Ooi>, Box<dyn Fn(f64) -> [f64; 2]>)> = vec![
        (ooi(150.0, 30.0), Box::new(straight(8.0, 0.0))),
        (ooi(300.0, 20.0), Box::new(drifting_circle(20.0, 0.4, [0.0, 20.0], 0.6, 0.0))),
        (ooi(0.0, 300.0), Box::new(drifting_circle(20.0, 0.4, [0.0, 20.0], 0.0, 0.0))),
    ];
    for (k, (oois, truth)) in cases.iter().enumerate() {
        let mut inside = 0;
        let n = 20u64;
        for s in 1..=n {
            let tr = observe(40.0, s, |t| truth(t));
            let c = cfg(s);
            let p = predict(&tr, &ttc(), oois, &c);
            if truth_in_90(&p, truth(40.0), truth(40.0 + c.horizon_s)) {
                inside += 1;
            }
        }
        assert!(inside >= 17, "case {k}: truth in 90% sector {inside}/{n}");
    }
}
