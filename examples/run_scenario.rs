//! run_scenario — CLI entry point for Stage 1.
//!
//! Reads a scenario file (see `scenarios/simple.txt`), runs the
//! Monte-Carlo predictor, and prints five lines in 'key = value'
//! format per §5, in this order:
//!
//!   sector50_deg      width of the 50% sector, degrees
//!   sector90_deg      width of the 90% sector, degrees
//!   heading_mean_deg  circular mean final heading, [0, 360)
//!   n_bins_nonzero    number of azimuth bins with p > 0
//!   elapsed_ms        time spent inside predict(), milliseconds
//!
//! Manual parsing with std::env::args + split_whitespace. No new deps.

use predictor::particle::predict;
use predictor::simulator::{generate_curved_track, Trajectory};
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

fn parse_scenario(path: &str) -> (Ttc, Vec<Observation>, Vec<Ooi>, PredictConfig, Option<Trajectory>, f64) {
    let contents = std::fs::read_to_string(path).unwrap_or_else(|e| {
        eprintln!("error: cannot read scenario file {}: {}", path, e);
        process::exit(1);
    });

    let mut ttc: Option<Ttc> = None;
    let mut observations: Vec<Observation> = Vec::new();
    let mut oois: Vec<Ooi> = Vec::new();
    let mut config: Option<PredictConfig> = None;
    let mut trajectory: Option<Trajectory> = None;
    let mut duration_s: f64 = 60.0;

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
                let a_lat = parse_f64(it.next().unwrap_or("9.0"), "ttc a_lat");
                let a_long = parse_f64(it.next().unwrap_or("3.0"), "ttc a_long");
                let omega_cap = parse_f64(it.next().unwrap_or("1.0"), "ttc omega_cap");
                ttc = Some(Ttc {
                    v_min,
                    v_max,
                    a_lat,
                    a_long,
                    omega_cap,
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
            "duration_s" => {
                duration_s = parse_f64(it.next().unwrap_or("60.0"), "duration_s");
            }
            "trajectory" => {
                let kind = it.next().unwrap_or("");
                let traj = match kind {
                    "straight" => Trajectory::Straight {
                        psi0: parse_f64(it.next().unwrap_or("0"), "trajectory psi0"),
                    },
                    "spiral" => Trajectory::Spiral {
                        r0: parse_f64(it.next().unwrap_or("0"), "trajectory r0"),
                        r_growth: parse_f64(it.next().unwrap_or("0"), "trajectory r_growth"),
                        turns: parse_f64(it.next().unwrap_or("0"), "trajectory turns"),
                    },
                    "sine" => Trajectory::Sine {
                        wavelength_m: parse_f64(it.next().unwrap_or("0"), "trajectory wavelength_m"),
                        amplitude_m: parse_f64(it.next().unwrap_or("0"), "trajectory amplitude_m"),
                        direction: parse_f64(it.next().unwrap_or("0"), "trajectory direction"),
                    },
                    "zigzag" => Trajectory::Zigzag {
                        segment_m: parse_f64(it.next().unwrap_or("0"), "trajectory segment_m"),
                        angle_rad: parse_f64(it.next().unwrap_or("0"), "trajectory angle_rad"),
                        baseline_m: parse_f64(it.next().unwrap_or("0"), "trajectory baseline"),
                    },
                    other => {
                        eprintln!("error: unknown trajectory kind: {}", other);
                        process::exit(1);
                    }
                };
                trajectory = Some(traj);
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

    if let Some(traj) = &trajectory {
        // Curved path: generate observations directly from the trajectory.
        // Ignore any 'obs' directives.
        let obs = generate_curved_track(
            traj,
            /* speed */ 5.0,
            duration_s,
            /* dt_obs */ 1.0,
            /* cov */ [1.0, 0.0, 1.0],
            /* seed */ 42,
            /* ttc */ &ttc,
        );
        return (ttc, obs, oois, config, Some(traj.clone()), duration_s);
    }

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

    (ttc, observations, oois, config, trajectory, duration_s)
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("usage: {} <scenario.txt>", args[0]);
        process::exit(1);
    }

    let (ttc, observations, oois, config, trajectory, duration_s) = parse_scenario(&args[1]);

    // Stage 1: predict from the last observation.
    let last = &observations[observations.len() - 1];

    // If a trajectory was given, print the observation count + last obs info
    // alongside the prediction (curved-path scenarios).
    let mut obs_info = String::new();
    if let Some(traj) = &trajectory {
        let n = observations.len();
        let (x, y, v, psi) = (
            last.pos[0],
            last.pos[1],
            last.v,
            psi_rem_euclid(last.psi) * 180.0 / PI,
        );
        let kind = match traj {
            Trajectory::Straight { .. } => "straight",
            Trajectory::Spiral { .. } => "spiral",
            Trajectory::Sine { .. } => "sine",
            Trajectory::Zigzag { .. } => "zigzag",
        };
        obs_info = format!(
            "\n# trajectory = {kind}, duration_s = {duration_s:.1}, obs = {n}\n# last obs: x = {x:.3}, y = {y:.3}, v = {v:.3}, psi = {psi:.2} deg",
        );
    }

    let t_start = std::time::Instant::now();
    eprintln!("CLI_DEBUG track_len={} o0={:?} oN={:?}", observations.len(), observations[0], observations[observations.len()-1]);
    let prediction = predict(&observations, &ttc, &oois, &config);
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
    print!("{}", obs_info);
}
