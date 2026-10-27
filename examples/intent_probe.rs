//! intent_probe — prints P(goal | track prefix) over time for the intent
//! scenarios, averaged over a few seeds. Used to tune the goal model.
//!
//! Run: cargo run --release --example intent_probe

#[path = "../tests/common/intent.rs"]
mod intent;

use intent::*;
use predictor::particle::predict;
use predictor::types::{Observation, Ooi};

fn report(name: &str, duration: f64, oois: &[Ooi], truth: &dyn Fn(f64) -> [f64; 2]) {
    let seeds = [1u64, 2, 3, 4, 5];
    let tracks: Vec<Vec<Observation>> =
        seeds.iter().map(|&s| observe(duration, s, |t| truth(t))).collect();
    print!("{:<28}", name);
    let mut t_cut = 5.0;
    while t_cut <= duration + 1e-9 {
        let n = (t_cut / DT) as usize + 1;
        let mut pg = 0.0;
        let mut pt = 0.0;
        for (i, tr) in tracks.iter().enumerate() {
            let p = predict(&tr[..n], &ttc(), oois, &cfg(seeds[i]));
            pg += p.goal_probs.iter().sum::<f64>();
            pt += p.turn_prob;
        }
        let k = seeds.len() as f64;
        print!(" t{:>2}:{:.2}/{:.2}", t_cut as u32, pg / k, pt / k);
        t_cut += 5.0;
    }
    println!();
}

/// Fraction of seeds whose true 10 s displacement bearing lies in the 50% / 90%
/// sector, and the mean distance from the true horizon point to the nearest
/// predicted point.
fn coverage(name: &str, duration: f64, oois: &[Ooi], truth: &dyn Fn(f64) -> [f64; 2]) {
    let n_seeds = 40u64;
    let (mut in50, mut in90, mut miss) = (0, 0, 0.0);
    for s in 1..=n_seeds {
        let tr = observe(duration, s, |t| truth(t));
        let c = cfg(s);
        let p = predict(&tr, &ttc(), oois, &c);
        let a = truth(duration);
        let b = truth(duration + c.horizon_s);
        let brg = (b[1] - a[1]).atan2(b[0] - a[0]).rem_euclid(2.0 * std::f64::consts::PI);
        let inside = |start: f64, w: f64| (brg - start).rem_euclid(2.0 * std::f64::consts::PI) <= w;
        if inside(p.sector50_start, p.sector50) { in50 += 1; }
        if inside(p.sector90_start, p.sector90) { in90 += 1; }
        miss += p.point_cloud.iter()
            .map(|q| ((q[0] - b[0]).powi(2) + (q[1] - b[1]).powi(2)).sqrt())
            .fold(f64::INFINITY, f64::min);
    }
    let k = n_seeds as f64;
    println!("{:<28} in50 {:.2}  in90 {:.2}  nearest-pt {:.1} m", name, in50 as f64 / k, in90 as f64 / k, miss / k);
}

fn main() {
    println!("P(goal)/P(turn) by track length; prior P(goal) = 0.5");
    report("straight at OOI", 40.0, &ooi(600.0, 0.0), &straight(8.0, 0.0));
    report("straight, OOI behind", 40.0, &ooi(-300.0, 0.0), &straight(8.0, 0.0));
    report("straight past OOI (t~19)", 40.0, &ooi(150.0, 30.0), &straight(8.0, 0.0));
    report("homing onto OOI", 40.0, &ooi(200.0, 250.0), &homing(8.0, 0.0, [200.0, 250.0], 0.1));
    report("circle in place, OOI side", 40.0, &ooi(0.0, 300.0), &drifting_circle(20.0, 0.4, [0.0, 20.0], 0.0, 0.0));
    report("circle drifting to OOI", 40.0, &ooi(300.0, 20.0), &drifting_circle(20.0, 0.4, [0.0, 20.0], 0.6, 0.0));
    report("circle drifting away", 40.0, &ooi(-300.0, 20.0), &drifting_circle(20.0, 0.4, [0.0, 20.0], 0.6, 0.0));

    println!("\nCoverage of the true 10 s future (40 seeds, full 40 s track)");
    coverage("straight at OOI", 40.0, &ooi(600.0, 0.0), &straight(8.0, 0.0));
    coverage("straight past OOI", 40.0, &ooi(150.0, 30.0), &straight(8.0, 0.0));
    coverage("homing onto OOI", 40.0, &ooi(200.0, 250.0), &homing(8.0, 0.0, [200.0, 250.0], 0.1));
    coverage("circle in place, OOI side", 40.0, &ooi(0.0, 300.0), &drifting_circle(20.0, 0.4, [0.0, 20.0], 0.0, 0.0));
    coverage("circle drifting to OOI", 40.0, &ooi(300.0, 20.0), &drifting_circle(20.0, 0.4, [0.0, 20.0], 0.6, 0.0));
}
