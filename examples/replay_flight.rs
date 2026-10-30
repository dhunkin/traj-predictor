//! replay_flight — re-run every live prediction of a flight downloaded from
//! the web demo ("Download flight") through the engine, offline.
//!
//! For each scored prediction it rebuilds the exact observation window the
//! browser sent, re-predicts with the recorded settings, and reports:
//!   - P(goal) as recorded and as replayed (same seed: they must agree),
//!   - track-loss recoveries,
//!   - P(goal) spread over other seeds (Monte-Carlo stability),
//!   - whether the true direction fell in the replayed 50% / 90% sectors.
//!
//! Run: cargo run --release --features server --example replay_flight -- <flight.json> [seeds] [key=value ...]
//!
//! `key=value` overrides any `ModelParams` field for the replay (booleans
//! 0/1); plus n_samples=N (particles) and every=K (replay every K-th
//! prediction). Add `quiet` to print only the
//! summary.

use predictor::particle::{predict_with, ModelParams};
use predictor::types::{Observation, Ooi, PredictConfig, Ttc};
use serde_json::Value;
use std::f64::consts::PI;

fn num(v: &Value, k: &str) -> f64 {
    v[k].as_f64().unwrap_or_else(|| panic!("missing number {k}"))
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let path = args.get(1).expect("usage: replay_flight <flight.json> [seeds]");
    let n_seeds: u64 = args.get(2).map(|s| s.parse().unwrap()).unwrap_or(6);
    let d: Value = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    let s = &d["settings"];
    let ttc = Ttc {
        v_min: num(s, "v_min"),
        v_max: num(s, "v_max"),
        a_lat: num(s, "a_lat"),
        a_long: num(s, "a_long"),
        omega_cap: num(s, "omega_cap"),
    };
    // Optional overrides that are not model parameters.
    let opt = |k: &str| args.iter().filter_map(|a| a.split_once('=')).find(|kv| kv.0 == k).map(|kv| kv.1.parse::<f64>().unwrap());
    let n_override = opt("n_samples");
    let every = opt("every").unwrap_or(1.0).max(1.0) as usize;
    let cfg = |seed: u64| PredictConfig {
        horizon_s: num(s, "horizon_s"),
        n_samples: n_override.unwrap_or(num(s, "n_samples")) as u32,
        dt_s: 1.0,
        p_turn: num(s, "p_turn"),
        azimuth_bins: num(s, "azimuth_bins") as u32,
        speed_noise_mps: num(s, "speed_noise_mps"),
        intent_hypothesis_prob: num(s, "intent_hypothesis_prob"),
        seed,
    };
    let seed0 = num(s, "seed") as u64;
    let window = num(s, "duration_s");
    let oois: Vec<Ooi> = d["oois"].as_array().unwrap().iter().map(|o| Ooi { pos: [num(o, "x"), num(o, "y")] }).collect();
    let obs: Vec<Observation> = d["observations"]
        .as_array()
        .unwrap()
        .iter()
        .map(|o| Observation {
            t: num(o, "t"),
            pos: [num(o, "x"), num(o, "y")],
            v: 0.0,
            psi: 0.0,
            cov: [num(o, "cov_xx"), num(o, "cov_xy"), num(o, "cov_yy")],
        })
        .collect();
    let mut mp = ModelParams::default();
    let quiet = args.iter().any(|a| a == "quiet");
    for kv in args.iter().skip(3).filter_map(|a| a.split_once('=')).filter(|kv| kv.0 != "n_samples" && kv.0 != "every") {
        let v: f64 = kv.1.parse().unwrap();
        mp.set(kv.0, v).unwrap_or_else(|e| panic!("{e}"));
    }
    let mut spread_sum = 0.0;
    let mut leg = Vec::new(); // P(goal) on the straight leg t = 17-32 s
    let mut log_sum = 0.0;

    if !quiet {
        println!("{:>5} {:>7} {:>7} {:>4} {:>17} {:>5} {:>5}", "t", "Pg rec", "Pg rep", "rec", "Pg seeds min/avg/max", "in50", "in90");
    }
    let (mut h50, mut h90, mut n) = (0, 0, 0);
    for p in d["predictions"].as_array().unwrap().iter().step_by(every) {
        let t = num(p, "t");
        let track: Vec<Observation> = obs.iter().cloned().filter(|o| o.t >= t - window - 1e-9 && o.t <= t + 1e-9).collect();
        let r = predict_with(&track, &ttc, &oois, &cfg(seed0), &mp);
        let pg = |pr: &predictor::types::Prediction| pr.goal_probs.iter().sum::<f64>();
        let mut spread = Vec::new();
        for k in 1..=n_seeds {
            spread.push(pg(&predict_with(&track, &ttc, &oois, &cfg(seed0 + 1000 * k), &mp)));
        }
        let (lo, hi) = spread.iter().fold((1.0f64, 0.0f64), |(a, b), &x| (a.min(x), b.max(x)));
        let avg = spread.iter().sum::<f64>() / spread.len().max(1) as f64;
        let brg = num(p, "true_bearing");
        let inside = |start: f64, w: f64| (brg - start).rem_euclid(2.0 * PI) < w;
        let (i50, i90) = (inside(r.sector50_start, r.sector50), inside(r.sector90_start, r.sector90));
        spread_sum += hi - lo;
        if (17.0..=32.0).contains(&t) {
            leg.push(avg);
        }
        let nb = r.bin_probs.len();
        let j = ((brg.rem_euclid(2.0 * PI) / (2.0 * PI) * nb as f64) as usize).min(nb - 1);
        log_sum += -((0.99 * r.bin_probs[j] + 0.01 / nb as f64) / (2.0 * PI / nb as f64)).ln();
        h50 += i50 as u32;
        h90 += i90 as u32;
        n += 1;
        let rec_pg: f64 = p["p_goal"].as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).sum();
        if !quiet {
            println!(
                "{:>5.0} {:>7.2} {:>7.2} {:>4} {:>6.2}/{:.2}/{:.2} {:>5} {:>5}",
                t, rec_pg, pg(&r), r.recoveries, lo, avg, hi, i50 as u8, i90 as u8
            );
        }
    }
    println!(
        "replayed: 50% sector {}/{n}, 90% sector {}/{n}, log score {:.2}; P(goal) seed spread (max-min) {:.2} avg; straight leg t=17-32 mean P(goal) {:.2}",
        h50, h90, log_sum / n as f64, spread_sum / n as f64,
        leg.iter().sum::<f64>() / leg.len().max(1) as f64
    );
}
