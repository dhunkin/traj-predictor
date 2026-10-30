//! ui_cases — the demo-UI scenarios from manual testing (seed 42, 30 s track,
//! 10 s horizon, 1 m noise, 5000 particles), scored against the noise-free
//! truth: is the true 10 s bearing inside the 50% / 90% sector, how far is the
//! fan's mean heading off, and did the filter keep the track (fan start vs
//! true position, v_est vs true speed).
//!
//! Run: cargo run --release --example ui_cases

use predictor::particle::predict;
use predictor::simulator::{check_limits, generate_curved_track, sample_point, Trajectory};
use predictor::types::{Ooi, PredictConfig, Ttc};
use std::f64::consts::PI;

fn main() {
    // `ui_cases <seed>` runs every case with that seed (default 42).
    let seed: u64 = std::env::args().nth(1).map(|a| a.parse().unwrap()).unwrap_or(42);
    let cases: Vec<(&str, Trajectory, f64, [f64; 2])> = vec![
        ("zigzag 30/0.5", Trajectory::Zigzag { segment_m: 30.0, angle_rad: 0.5, baseline_m: 0.0 }, 40.0, [550.0, 0.0]),
        ("sine 100/20, OOI ahead", Trajectory::Sine { wavelength_m: 100.0, amplitude_m: 20.0, direction: 0.0 }, 40.0, [550.0, 0.0]),
        ("sine 100/20, OOI behind", Trajectory::Sine { wavelength_m: 100.0, amplitude_m: 20.0, direction: 0.0 }, 40.0, [-100.0, 0.0]),
        ("sine, a_lat 15 (in limits)", Trajectory::Sine { wavelength_m: 100.0, amplitude_m: 20.0, direction: 0.0 }, -15.0, [550.0, 0.0]),
        ("zigzag 60/0.3 (gentler)", Trajectory::Zigzag { segment_m: 60.0, angle_rad: 0.3, baseline_m: 0.0 }, 12.0, [550.0, 0.0]),
        ("spiral 10/1/3", Trajectory::Spiral { r0: 10.0, r_growth: 1.0, turns: 3.0 }, 40.0, [-100.0, 0.0]),
        ("orbit, OOI behind drift", Trajectory::Orbit { radius_m: 20.0, omega: 0.4, drift_mps: 0.6, drift_dir: 0.0 }, 12.0, [-100.0, 0.0]),
        ("orbit, OOI ahead of drift", Trajectory::Orbit { radius_m: 20.0, omega: 0.4, drift_mps: 0.6, drift_dir: 0.0 }, 12.0, [150.0, 0.0]),
        ("straight, OOI ahead", Trajectory::Straight { psi0: 0.0 }, 12.0, [550.0, 0.0]),
        // Second round of manual tests (OOI off to the side).
        ("sine, OOI (250,100)", Trajectory::Sine { wavelength_m: 100.0, amplitude_m: 20.0, direction: 0.0 }, 12.0, [250.0, 100.0]),
        ("spiral, OOI (250,100)", Trajectory::Spiral { r0: 10.0, r_growth: 1.0, turns: 3.0 }, 12.0, [250.0, 100.0]),
        ("orbit, OOI (250,100)", Trajectory::Orbit { radius_m: 20.0, omega: 0.4, drift_mps: 0.6, drift_dir: 0.0 }, 12.0, [250.0, 100.0]),
        ("straight, OOI (550,100)", Trajectory::Straight { psi0: 0.0 }, 12.0, [550.0, 100.0]),
    ];
    let (dur, hor, speed) = (30.0, 10.0, 12.0);
    println!("{:<27} {:>4} {:>4} {:>7} {:>6} {:>6} {:>6} {:>6} {:>10} {:>4} {:>5}  limits", "case", "in50", "in90", "bias°", "w90°", "Pgoal", "Pturn", "v_est", "lag m/v", "rec", "loop%");
    for (name, traj, v_max, ooi) in cases {
        // A negative third field means: v_max 12 with that a_lat instead of 9.
        let (v_max, a_lat) = if v_max < 0.0 { (12.0, -v_max) } else { (v_max, 9.0) };
        let ttc = Ttc { v_min: 3.0, v_max, a_lat, a_long: 3.0, omega_cap: 1.0 };
        let obs = generate_curved_track(&traj, speed, dur + hor, 1.0, [1.0, 0.0, 1.0], seed, &ttc);
        let track: Vec<_> = obs.into_iter().filter(|o| o.t <= dur).collect();
        let cfg = PredictConfig { horizon_s: hor, n_samples: 5000, dt_s: 1.0, p_turn: 0.3, azimuth_bins: 72, speed_noise_mps: 0.5, intent_hypothesis_prob: 0.5, seed };
        let p = predict(&track, &ttc, &[Ooi { pos: ooi }], &cfg);
        let (a, va) = { let (pos, _, v) = sample_point(&traj, dur, dur + hor, speed); (pos, v) };
        let b = sample_point(&traj, dur + hor, dur + hor, speed).0;
        let brg = (b[1] - a[1]).atan2(b[0] - a[0]);
        let inside = |s: f64, w: f64| (brg - s).rem_euclid(2.0 * PI) < w;
        let bias = ((p.final_heading_mean - brg + PI).rem_euclid(2.0 * PI) - PI).to_degrees();
        // Share of predicted paths that turn through more than a full circle
        // within the horizon (sum of absolute heading changes > 2 pi).
        let loops = p.path_cloud.iter().filter(|path| {
            let mut tot = 0.0;
            for w in path.windows(3) {
                let a1 = (w[1][1] - w[0][1]).atan2(w[1][0] - w[0][0]);
                let a2 = (w[2][1] - w[1][1]).atan2(w[2][0] - w[1][0]);
                tot += ((a2 - a1 + PI).rem_euclid(2.0 * PI) - PI).abs();
            }
            tot > 2.0 * PI
        }).count() as f64 / p.path_cloud.len() as f64 * 100.0;
        let lag = p.path_cloud.iter().map(|q| ((q[0][0] - a[0]).powi(2) + (q[0][1] - a[1]).powi(2)).sqrt()).sum::<f64>() / p.path_cloud.len() as f64;
        println!(
            "{:<27} {:>4} {:>4} {:>+7.0} {:>6.0} {:>6.2} {:>6.2} {:>6.1} {:>5.1}/{:<4.1} {:>4} {:>5.0}  {}",
            name, inside(p.sector50_start, p.sector50) as u8, inside(p.sector90_start, p.sector90) as u8, bias,
            p.sector90.to_degrees(), p.goal_probs[0], p.turn_prob, p.v_est, lag, va, p.recoveries, loops,
            check_limits(&traj, dur + hor, speed, &ttc).join("; ")
        );
    }
}
