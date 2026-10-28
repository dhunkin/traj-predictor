//! calibrate — coverage of the predicted heading sectors against the true
//! future, per scenario family (see tests/common/calib.rs).
//!
//! Run:
//!   cargo run --release --example calibrate -- [--seeds 60] [--n 2000]
//!       [--set key=val,key=val] [--sweep key=v1,v2,...]
//!
//! `--stress` also runs the platform stress families (multicopter /
//! fixed-wing), reported separately and never part of the tuning score.
//!
//! `key` is a `ModelParams` field name. `--set` overrides the defaults;
//! `--sweep` reruns the suite for each value and prints one line per value.
//! cov50/cov90 use the reported sectors (target 0.50 / 0.90); hpd50/hpd90 use
//! the HPD level of the truth (free of the 5 deg bin rounding of the sectors).
//! logS is the log score (-ln density at the true bearing; lower is better):
//! the main tuning target, since coverage alone also rewards a wide or biased
//! fan. Coverage is the calibration check.

#[path = "../tests/common/mod.rs"]
mod common;

use common::calib::*;
use predictor::particle::ModelParams;

fn set(mp: &mut ModelParams, key: &str, v: f64) {
    match key {
        "cruise_tau_s" => mp.cruise_tau_s = v,
        "cruise_omega_sigma" => {
            mp.cruise_omega_sigma_min = v;
            mp.cruise_omega_sigma_max = v;
        }
        "cruise_omega_sigma_min" => mp.cruise_omega_sigma_min = v,
        "cruise_omega_sigma_max" => mp.cruise_omega_sigma_max = v,
        "turn_omega_walk" => mp.turn_omega_walk = v,
        "omega_turn_min" => mp.omega_turn_min = v,
        "turn_entry_sigma" => mp.turn_entry_sigma = v,
        "mode_switch_rate" => mp.mode_switch_rate = v,
        "turn_persistence_s" => mp.turn_persistence_s = v,
        "goal_switch_rate" => mp.goal_switch_rate = v,
        "k_goal" => mp.k_goal = v,
        "k_course" => mp.k_course = v,
        "course_walk" => mp.course_walk = v,
        "p_weave" => mp.p_weave = v,
        "weave_tau_s" => mp.weave_tau_s = v,
        "weave_noise" => mp.weave_noise = v,
        "weave_freq_min" => mp.weave_freq_min = v,
        "weave_freq_max" => mp.weave_freq_max = v,
        "recovery_d2" => mp.recovery_d2 = v,
        "goal_drift_frac" => mp.goal_drift_frac = v,
        "omega_loop" => mp.omega_loop = v,
        "filter_speed_noise_floor" => mp.filter_speed_noise_floor = v,
        "min_mode_share" => mp.min_mode_share = v,
        "meas_floor_m2" => mp.meas_floor_m2 = v,
        _ => panic!("unknown ModelParams field: {key}"),
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let arg = |name: &str| args.iter().position(|a| a == name).map(|i| args[i + 1].clone());
    let seeds: u64 = arg("--seeds").map(|s| s.parse().unwrap()).unwrap_or(60);
    let n: u32 = arg("--n").map(|s| s.parse().unwrap()).unwrap_or(2000);
    let mut mp = ModelParams::default();
    if let Some(sets) = arg("--set") {
        for kv in sets.split(',') {
            let (k, v) = kv.split_once('=').expect("--set key=val");
            set(&mut mp, k, v.parse().unwrap());
        }
    }
    let stress = args.iter().any(|a| a == "--stress");
    let mut all = core_families();
    if stress {
        all.extend(stress_families());
    }
    let tuned: Vec<usize> = core_families().into_iter().filter(|&f| FAMILIES[f] != "model").collect();

    if let Some(sw) = arg("--sweep") {
        let (k, vals) = sw.split_once('=').expect("--sweep key=v1,v2");
        for v in vals.split(',') {
            let mut m = mp;
            set(&mut m, k, v.parse().unwrap());
            let per = run(&m, seeds, n, &all);
            let s = summarise(&tuned.iter().flat_map(|&f| per[f].clone()).collect::<Vec<_>>());
            let worst = tuned.iter().map(|&f| (summarise(&per[f]).cov90 - 0.9).abs()).fold(0.0, f64::max);
            println!(
                "{k}={v:<8} logS {:.3}  obj {:.3}  cov50 {:.2} cov90 {:.2}  hpd50 {:.2} hpd90 {:.2}  w50 {:5.1} w90 {:5.1}  worst|cov90-0.9| {:.2}  model cov50 {:.2} cov90 {:.2}",
                mean_log_score(&per, &tuned), objective(&per, &tuned), s.cov50, s.cov90, s.hpd50, s.hpd90, s.w50_deg, s.w90_deg, worst,
                summarise(&per[model_family()]).cov50, summarise(&per[model_family()]).cov90
            );
        }
        return;
    }

    let t0 = std::time::Instant::now();
    let per = run(&mp, seeds, n, &all);
    println!("{:<16} {:>4} {:>6} {:>6} {:>6} {:>6} {:>7} {:>7} {:>6}", "family", "n", "cov50", "cov90", "hpd50", "hpd90", "w50°", "w90°", "logS");
    for &f in all.iter() {
        if f == N_CORE {
            println!("--- platform stress families (not tuned on) ---");
        }
        let s = summarise(&per[f]);
        println!(
            "{:<16} {:>4} {:>6.2} {:>6.2} {:>6.2} {:>6.2} {:>7.1} {:>7.1} {:>6.2}",
            FAMILIES[f], s.n, s.cov50, s.cov90, s.hpd50, s.hpd90, s.w50_deg, s.w90_deg, s.log_score
        );
    }
    let s = summarise(&tuned.iter().flat_map(|&f| per[f].clone()).collect::<Vec<_>>());
    println!(
        "{:<16} {:>4} {:>6.2} {:>6.2} {:>6.2} {:>6.2} {:>7.1} {:>7.1} {:>6.2}",
        "TUNING (excl model)", s.n, s.cov50, s.cov90, s.hpd50, s.hpd90, s.w50_deg, s.w90_deg, s.log_score
    );
    println!(
        "mean log score over families = {:.3} (lower is better); coverage objective = {:.3}   [{:.1} s]",
        mean_log_score(&per, &tuned), objective(&per, &tuned), t0.elapsed().as_secs_f64()
    );
}
