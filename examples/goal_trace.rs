//! goal_trace — P(goal) along a straight track as it passes an OOI, with the
//! OOI's bearing off the nose and distance at each time. Default: the UI case
//! straight east at 12 m/s, OOI at (400, -100), seed 42.
//!
//! Run: cargo run --release --example goal_trace -- [ooi_x ooi_y]

use predictor::particle::predict;
use predictor::simulator::{generate_curved_track, Trajectory};
use predictor::types::{Ooi, PredictConfig, Ttc};

fn main() {
    let a: Vec<f64> = std::env::args().skip(1).map(|x| x.parse().unwrap()).collect();
    let ooi = if a.len() == 2 { [a[0], a[1]] } else { [400.0, -100.0] };
    let ttc = Ttc { v_min: 3.0, v_max: 12.0, a_lat: 9.0, a_long: 3.0, omega_cap: 1.0 };
    let obs = generate_curved_track(&Trajectory::Straight { psi0: 0.0 }, 12.0, 30.0, 1.0, [1.0, 0.0, 1.0], 42, &ttc);
    let cfg = PredictConfig { horizon_s: 10.0, n_samples: 5000, dt_s: 1.0, p_turn: 0.3, azimuth_bins: 72, speed_noise_mps: 0.5, intent_hypothesis_prob: 0.5, seed: 42 };
    println!("OOI at ({}, {}); UAV flies east at 12 m/s", ooi[0], ooi[1]);
    println!("{:>5} {:>8} {:>12} {:>9} {:>8}", "t s", "x m", "OOI off nose", "dist m", "P(goal)");
    for t in [3usize, 5, 10, 15, 20, 25, 30] {
        let p = predict(&obs[..=t], &ttc, &[Ooi { pos: ooi }], &cfg);
        let x = 12.0 * t as f64;
        let off = (ooi[1]).atan2(ooi[0] - x).to_degrees();
        let dist = ((ooi[0] - x).powi(2) + ooi[1].powi(2)).sqrt();
        println!("{:>5} {:>8.0} {:>11.0}° {:>9.0} {:>8.2}", t, x, off, dist, p.goal_probs[0]);
    }
}
