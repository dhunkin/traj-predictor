//! run_scenario — CLI entry point for Stage 1.
//!
//! Reads a scenario file (see `scenarios/simple.txt`), runs the
//! Monte-Carlo predictor, and prints the five numbers per §5:
//!
//!   sector50_deg
//!   sector90_deg
//!   point_cloud_count
//!   bin_count
//!   heading_mean_deg
//!
//! Manual parsing with std::env::args + split_whitespace. No new deps.

use predictor::mc::predict;
use predictor::types::{Observation, Ooi, PredictConfig, Ttc};
use std::env;
use std::f64::consts::PI;
use std::process;

/// Reduce an angle into [0, 2π) for display.
fn psi_rem_euclid(psi: f64) -> f64 {
    let two_pi = 2.0 * PI;
    let rem = psi % two_pi;
    if rem < 0.0 {
        rem + two_pi
    } else {
        rem
    }
}

fn parse_f64(s: &str, ctx: &str) -> f64 {
    s.parse::<f64>().unwrap_or_else(|_| {
        eprintln!("error: invalid number for {}: {}", ctx, s);
        process::exit(1);
    })
}

fn parse_u32(s: &str, ctx: &str) -> u32 {
    s.parse::<u32>()
        .unwrap_or_else(|_| {
            eprintln!("error: invalid integer for {}: {}", ctx, s);
            process::exit(1);
        })
}

fn parse_u64(s: &str, ctx: &str) -> u64 {
    s.parse::<u64>()
        .unwrap_or_else(|_| {
            eprintln!("error: invalid integer for {}: {}", ctx, s);
            process::exit(1);
        })
}

fn parse_scenario(path: &str) -> (Ttc, Vec<Observation>, Vec<Ooi>, PredictConfig) {
    let contents = std::fs::read_to_string(path).unwrap_or_else(|e| {
        eprintln!("error: cannot read scenario file {}: {}", path, e);
        process::exit(1);
    });

    let mut ttc: Option<Ttc> = None;
    let mut observations: Vec<Observation> = Vec::new();
    let mut oois: Vec<Ooi> = Vec::new();
    let mut config: Option<PredictConfig> = None;

    for raw in contents.lines() {
        let line = raw.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let mut it = line.split_whitespace();
        let directive = it.next().unwrap_or("");
        match directive {
            "ttc" => {
                let v_min = parse_f64(it.next().unwrap_or("0"), "ttc v_min");
                let v_max = parse_f64(it.next().unwrap_or("0"), "ttc v_max");
                let omega_max = parse_f64(it.next().unwrap_or("0"), "ttc omega_max");
                ttc = Some(Ttc {
                    v_min,
                    v_max,
                    omega_max,
                });
            }
            "obs" => {
                let t = parse_f64(it.next().unwrap_or("0"), "obs t");
                let x = parse_f64(it.next().unwrap_or("0"), "obs x");
                let y = parse_f64(it.next().unwrap_or("0"), "obs y");
                let cov_xx = parse_f64(it.next().unwrap_or("0"), "obs cov_xx");
                let cov_xy = parse_f64(it.next().unwrap_or("0"), "obs cov_xy");
                let cov_yy = parse_f64(it.next().unwrap_or("0"), "obs cov_yy");
                observations.push(Observation {
                    t,
                    pos: [x, y],
                    v: 0.0, // filled below from the last heading/speed
                    psi: 0.0,
                    cov: [cov_xx, cov_xy, cov_yy],
                });
            }
            "ooi" => {
                let x = parse_f64(it.next().unwrap_or("0"), "ooi x");
                let y = parse_f64(it.next().unwrap_or("0"), "ooi y");
                oois.push(Ooi { pos: [x, y] });
            }
            "config" => {
                let horizon_s = parse_f64(it.next().unwrap_or("0"), "config horizon_s");
                let n_samples = parse_u32(it.next().unwrap_or("0"), "config n_samples");
                let dt_s = parse_f64(it.next().unwrap_or("0"), "config dt_s");
                let p_turn = parse_f64(it.next().unwrap_or("0"), "config p_turn");
                let azimuth_bins = parse_u32(it.next().unwrap_or("0"), "config azimuth_bins");
                let speed_noise = parse_f64(it.next().unwrap_or("0"), "config speed_noise");
                let seed = parse_u64(it.next().unwrap_or("0"), "config seed");
                config = Some(PredictConfig {
                    horizon_s,
                    n_samples,
                    dt_s,
                    p_turn,
                    azimuth_bins,
                    speed_noise_mps: speed_noise,
                    seed,
                });
            }
            other => {
                eprintln!("error: unknown directive: {}", other);
                process::exit(1);
            }
        }
    }

    let ttc = ttc.unwrap_or_else(|| {
        eprintln!("error: scenario is missing a 'ttc' directive");
        process::exit(1);
    });
    let config = config.unwrap_or_else(|| {
        eprintln!("error: scenario is missing a 'config' directive");
        process::exit(1);
    });

    // Stage 1 MC uses the last observation's heading and speed as-is.
    // The scenario provides position + cov per observation but not v/psi,
    // so derive them from the last two observations' positions:
    //   psi = atan2(dy, dx), v = |displacement| / dt.
    // With a single observation, leave v = 0 and psi = 0.
    let n = observations.len();
    if n == 0 {
        eprintln!("error: scenario has no observations");
        process::exit(1);
    }
    if n >= 2 {
        let (px, py, pt) = (
            observations[n - 2].pos[0],
            observations[n - 2].pos[1],
            observations[n - 2].t,
        );
        let last = &mut observations[n - 1];
        let dx = last.pos[0] - px;
        let dy = last.pos[1] - py;
        let dt = (last.t - pt).abs();
        last.psi = dy.atan2(dx);
        last.v = if dt > 0.0 {
            (dx * dx + dy * dy).sqrt() / dt
        } else {
            0.0
        };
    }

    (ttc, observations, oois, config)
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("usage: {} <scenario.txt>", args[0]);
        process::exit(1);
    }

    let (ttc, observations, oois, config) = parse_scenario(&args[1]);

    // Stage 1: predict from the last observation.
    let last = &observations[observations.len() - 1];

    let t_start = std::time::Instant::now();
    let prediction = predict(last, &ttc, &oois, &config);
    let elapsed_ms = t_start.elapsed().as_millis();

    let sector50_deg = prediction.sector50 * 180.0 / PI;
    let sector90_deg = prediction.sector90 * 180.0 / PI;
    let heading_mean_deg =
        psi_rem_euclid(prediction.final_heading_mean) * 180.0 / PI;
    let n_bins_nonzero = prediction
        .bin_probs
        .iter()
        .filter(|&&p| p > 0.0)
        .count();

    println!("sector50_deg = {}", sector50_deg);
    println!("sector90_deg = {}", sector90_deg);
    println!("heading_mean_deg = {}", heading_mean_deg);
    println!("n_bins_nonzero = {}", n_bins_nonzero);
    println!("elapsed_ms = {}", elapsed_ms);
}
