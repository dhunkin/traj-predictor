//! straight_anatomy — what widens the fan on a perfectly straight track?
//! Straight east at 12 m/s, 30 s observed, OOI at (550, -100) as in the UI.
//! Each variant switches one model ingredient off and reports the mean sector
//! widths and the share of predicted endpoints more than 20 degrees off the
//! true (straight-ahead) direction, over 20 seeds.
//!
//! Run: cargo run --release --example straight_anatomy

use predictor::particle::{predict_with, ModelParams};
use predictor::simulator::{generate_curved_track, Trajectory};
use predictor::types::{Ooi, PredictConfig, Ttc};

fn main() {
    let ttc = Ttc { v_min: 3.0, v_max: 12.0, a_lat: 9.0, a_long: 3.0, omega_cap: 1.0 };
    let d = ModelParams::default();
    let variants: Vec<(&str, ModelParams)> = vec![
        ("default", d),
        ("no weaver births", ModelParams { weave_switch_rate: 0.0, ..d }),
        ("no weavers at all", ModelParams { weave_switch_rate: 0.0, p_weave: 0.0, ..d }),
        ("turn onsets /3", ModelParams { mode_switch_rate: d.mode_switch_rate / 3.0, ..d }),
        ("both of the above", ModelParams { weave_switch_rate: 0.0, mode_switch_rate: d.mode_switch_rate / 3.0, ..d }),
        ("gentle weave births", ModelParams { weave_birth_amp: 0.2, ..d }),
        ("learned restlessness", ModelParams { restless_min: 0.1, restless_max: 3.0, ..d }),
        ("gentle + restless", ModelParams { weave_birth_amp: 0.2, restless_min: 0.1, restless_max: 3.0, ..d }),
    ];
    println!("{:<20} {:>6} {:>6} {:>10}", "variant", "w50°", "w90°", ">20° off");
    for (name, mp) in variants {
        let (mut w50, mut w90, mut off) = (0.0, 0.0, 0.0);
        let n = 20;
        for seed in 1..=n {
            let obs = generate_curved_track(&Trajectory::Straight { psi0: 0.0 }, 12.0, 30.0, 1.0, [1.0, 0.0, 1.0], seed, &ttc);
            let cfg = PredictConfig { horizon_s: 10.0, n_samples: 5000, dt_s: 1.0, p_turn: 0.3, azimuth_bins: 72, speed_noise_mps: 0.5, intent_hypothesis_prob: 0.5, seed };
            let p = predict_with(&obs, &ttc, &[Ooi { pos: [550.0, -100.0] }], &cfg, &mp);
            w50 += p.sector50.to_degrees();
            w90 += p.sector90.to_degrees();
            off += p.path_cloud.iter().filter(|path| {
                let (s, e) = (path[0], path[path.len() - 1]);
                (e[1] - s[1]).atan2(e[0] - s[0]).abs().to_degrees() > 20.0
            }).count() as f64 / p.path_cloud.len() as f64;
        }
        let k = n as f64;
        println!("{:<20} {:>6.1} {:>6.1} {:>9.0}%", name, w50 / k, w90 / k, off / k * 100.0);
    }
}
