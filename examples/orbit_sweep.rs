//! orbit_sweep — 20-seed sweep of a drifting orbit with the OOI ahead of /
//! behind the drift. Counts seeds where P(goal) ends on the right side of the
//! prior (0.5) and prints each seed's posterior.
//!
//! Run: cargo run --release --example orbit_sweep -- [drift_mps] [speed_noise]
//! (env N=<particles> overrides n_samples). With drift 0 the "toward" row is
//! a loiter in place and should be low.
#[path = "../tests/common/intent.rs"]
mod intent;
use intent::*;
use predictor::particle::predict;
use predictor::simulator::{generate_curved_track, Trajectory};
fn main() {
    let drift: f64 = std::env::args().nth(1).map(|s| s.parse().unwrap()).unwrap_or(0.6);
    let traj = Trajectory::Orbit { radius_m: 20.0, omega: 0.4, drift_mps: drift, drift_dir: 0.0 };
    let (mut ok_to, mut ok_away, mut st, mut sa) = (0, 0, 0.0, 0.0);
    let mut line_to = String::new();
    let mut line_aw = String::new();
    for seed in 1..=20u64 {
        let tr = generate_curved_track(&traj, 12.0, 40.0, 1.0, [1.0, 0.0, 1.0], seed * 7 + 3, &ttc());
        let mut c = cfg(seed); c.n_samples = std::env::var("N").map(|s| s.parse().unwrap()).unwrap_or(2000); c.speed_noise_mps = std::env::args().nth(2).map(|s| s.parse().unwrap()).unwrap_or(0.5);
        let pt = predict(&tr, &ttc(), &ooi(300.0, 20.0), &c).goal_probs[0];
        let pa = predict(&tr, &ttc(), &ooi(-300.0, 20.0), &c).goal_probs[0];
        if pt > 0.5 { ok_to += 1 }
        if pa < 0.5 { ok_away += 1 }
        st += pt; sa += pa;
        line_to += &format!(" {:.2}", pt);
        line_aw += &format!(" {:.2}", pa);
    }
    println!("toward: {ok_to}/20 mean {:.2} |{line_to}", st / 20.0);
    println!("away:   {ok_away}/20 mean {:.2} |{line_aw}", sa / 20.0);
}
