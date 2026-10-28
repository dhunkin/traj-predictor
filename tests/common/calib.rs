//! Calibration harness: randomised scenario families, and coverage scoring of
//! the predicted heading fan against the unobserved true future.
//! Shared by examples/calibrate.rs and tests/calibration.rs.

#![allow(dead_code)]

use super::intent::homing;
use predictor::particle::{predict_with, GoalPrior, Mode, ModelParams, Particle};
use predictor::prng::Xoshiro256ss;
use predictor::simulator::{sample_point, Trajectory};
use predictor::types::{Observation, Ooi, PredictConfig, Ttc};
use std::f64::consts::PI;

/// Observed track length (s), prediction horizon (s), observation noise (m).
pub const T_OBS: f64 = 30.0;
pub const HORIZON: f64 = 10.0;
pub const SIGMA: f64 = 1.0;

/// Scenario families. The first `N_CORE` (through "model") are the tuning
/// suite; the rest are platform stress tests (multicopter, fixed-wing) that
/// probe the physics and are reported separately, not tuned on.
pub const FAMILIES: [&str; 17] = [
    "straight",
    "straight_to_ooi",
    "homing",
    "orbit",
    "orbit_drift",
    "weave",
    "weave_fast",
    "turn_recent",
    "turn_future",
    "model",
    // Multicopter: can brake, hover, reverse and change velocity direction
    // without turning (acceleration 3-4 m/s^2).
    "mc_stop_go",
    "mc_depart",
    "mc_reverse",
    "mc_strafe",
    // Fixed-wing (10-30 m/s, 30 deg bank): never hovers, turns only by banking.
    "fw_orbit_wind",
    "fw_racetrack",
    "fw_turn",
];

/// Number of tuning families (indices 0..N_CORE).
pub const N_CORE: usize = 10;

/// Indices of the tuning families (including the "model" self-check).
pub fn core_families() -> Vec<usize> {
    (0..N_CORE).collect()
}

/// Indices of the platform stress families.
pub fn stress_families() -> Vec<usize> {
    (N_CORE..FAMILIES.len()).collect()
}

/// Turn-table limits the predictor is configured with for a family: the
/// fixed-wing families get a fixed-wing envelope, everything else the
/// default multicopter-ish one.
pub fn family_ttc(family: usize) -> Ttc {
    if FAMILIES[family].starts_with("fw_") {
        Ttc { v_min: 10.0, v_max: 30.0, a_lat: 9.0, a_long: 2.0, omega_cap: 1.0 }
    } else {
        ttc()
    }
}

/// g * tan(30 deg): lateral acceleration of a 30-degree coordinated turn.
const A_BANK30: f64 = 9.81 * 0.577_350_269;

/// Integrate a velocity profile from the origin (step 0.02 s) and return a
/// position-at-time lookup (linear interpolation) for t in [0, t_end].
fn integrate(vel: impl Fn(f64) -> [f64; 2], t_end: f64) -> Box<dyn Fn(f64) -> [f64; 2]> {
    let h = 0.02;
    let n = (t_end / h).ceil() as usize + 1;
    let mut pts = Vec::with_capacity(n + 1);
    let mut p = [0.0, 0.0];
    pts.push(p);
    for i in 0..n {
        // Midpoint rule.
        let v = vel((i as f64 + 0.5) * h);
        p = [p[0] + v[0] * h, p[1] + v[1] * h];
        pts.push(p);
    }
    Box::new(move |t: f64| {
        let x = (t / h).max(0.0);
        let i = (x.floor() as usize).min(pts.len() - 2);
        let f = (x - i as f64).min(1.0);
        [pts[i][0] + f * (pts[i + 1][0] - pts[i][0]), pts[i][1] + f * (pts[i + 1][1] - pts[i][1])]
    })
}

pub fn ttc() -> Ttc {
    Ttc { v_min: 3.0, v_max: 12.0, a_lat: 9.0, a_long: 3.0, omega_cap: 1.0 }
}

/// UI defaults (except n_samples, which the caller picks).
pub fn cfg(seed: u64, n_samples: u32) -> PredictConfig {
    PredictConfig {
        horizon_s: HORIZON,
        n_samples,
        dt_s: 1.0,
        p_turn: 0.3,
        azimuth_bins: 72,
        speed_noise_mps: 0.5,
        intent_hypothesis_prob: 0.5,
        seed,
    }
}

/// One calibration case: the noisy observed track, the OOIs, and the true
/// positions at the last observation time and at the horizon.
pub struct Case {
    pub family: usize,
    /// Limits the predictor is run with (see `family_ttc`).
    pub ttc: Ttc,
    pub seed: u64,
    pub track: Vec<Observation>,
    pub oois: Vec<Ooi>,
    pub truth_now: [f64; 2],
    pub truth_future: [f64; 2],
}

fn uni(rng: &mut Xoshiro256ss, a: f64, b: f64) -> f64 {
    a + (b - a) * rng.next_f64()
}

fn add(p: [f64; 2], q: [f64; 2]) -> [f64; 2] {
    [p[0] + q[0], p[1] + q[1]]
}

fn polar(r: f64, a: f64) -> [f64; 2] {
    [r * a.cos(), r * a.sin()]
}

/// Build the case for (family, seed): sample the truth at t = 0..=T_OBS every
/// 1 s with N(0, SIGMA^2) noise, plus the true horizon point.
pub fn make_case(family: usize, seed: u64, mp: &ModelParams) -> Case {
    let mut rng = Xoshiro256ss::from_seed(seed.wrapping_mul(0x9E37_79B9) ^ (family as u64 + 1) * 0x1234_5678_9ABC);
    let random_ooi = |rng: &mut Xoshiro256ss| polar(uni(rng, 150.0, 400.0), uni(rng, 0.0, 2.0 * PI));

    // Truth as a dense function of time, or (model family) a sampled path.
    let truth: Box<dyn Fn(f64) -> [f64; 2]>;
    let oois: Vec<Ooi>;
    match FAMILIES[family] {
        "straight" | "straight_to_ooi" => {
            let v = uni(&mut rng, 4.0, 11.0);
            let psi = uni(&mut rng, 0.0, 2.0 * PI);
            let ooi = if FAMILIES[family] == "straight" {
                random_ooi(&mut rng)
            } else {
                polar(v * T_OBS + uni(&mut rng, 100.0, 400.0), psi)
            };
            oois = vec![Ooi { pos: ooi }];
            truth = Box::new(move |t| polar(v * t, psi));
        }
        "homing" => {
            let v = uni(&mut rng, 5.0, 10.0);
            let psi0 = uni(&mut rng, 0.0, 2.0 * PI);
            let side = if rng.next_f64() < 0.5 { -1.0 } else { 1.0 };
            let g = polar(uni(&mut rng, 250.0, 400.0), psi0 + side * uni(&mut rng, PI / 3.0, 2.0 * PI / 3.0));
            let k = uni(&mut rng, 0.05, 0.15);
            oois = vec![Ooi { pos: g }];
            truth = Box::new(homing(v, psi0, g, k));
        }
        "orbit" | "orbit_drift" => {
            let r = uni(&mut rng, 15.0, 40.0);
            let v = uni(&mut rng, 5.0, 10.0);
            let w = if rng.next_f64() < 0.5 { -v / r } else { v / r };
            let ooi = random_ooi(&mut rng);
            let (drift, dir) = if FAMILIES[family] == "orbit" {
                (0.0, 0.0)
            } else {
                (uni(&mut rng, 0.3, 1.0), ooi[1].atan2(ooi[0]))
            };
            oois = vec![Ooi { pos: ooi }];
            let traj = Trajectory::Orbit { radius_m: r, omega: w, drift_mps: drift, drift_dir: dir };
            truth = Box::new(move |t| sample_point(&traj, t, T_OBS + HORIZON, 0.0).0);
        }
        "weave" | "weave_fast" => {
            // weave: gentle; weave_fast: short wavelength and large swings,
            // with the speed capped so the crests stay within a_lat.
            let fast = FAMILIES[family] == "weave_fast";
            let (wl, amp) = if fast {
                (uni(&mut rng, 70.0, 130.0), uni(&mut rng, 12.0, 25.0))
            } else {
                (uni(&mut rng, 100.0, 250.0), uni(&mut rng, 5.0, 20.0))
            };
            let traj = Trajectory::Sine {
                wavelength_m: wl,
                amplitude_m: amp,
                direction: uni(&mut rng, 0.0, 2.0 * PI),
            };
            let kappa = amp * (2.0 * PI / wl).powi(2);
            let v = uni(&mut rng, 5.0, 9.0).min((0.9 * ttc().a_lat / kappa).sqrt()).max(3.5);
            oois = vec![Ooi { pos: random_ooi(&mut rng) }];
            truth = Box::new(move |t| sample_point(&traj, t, T_OBS + HORIZON, v).0);
        }
        "turn_recent" | "turn_future" => {
            // Straight leg, then a coordinated turn that starts up to 8 s
            // before (recent) or after (future) the last observation.
            let v = uni(&mut rng, 5.0, 10.0);
            let psi = uni(&mut rng, 0.0, 2.0 * PI);
            let w = uni(&mut rng, 0.15, 0.4) * if rng.next_f64() < 0.5 { -1.0 } else { 1.0 };
            let t_on = T_OBS
                + if FAMILIES[family] == "turn_recent" { -uni(&mut rng, 0.0, 8.0) } else { uni(&mut rng, 0.0, 8.0) };
            oois = vec![Ooi { pos: random_ooi(&mut rng) }];
            truth = Box::new(move |t| {
                if t <= t_on {
                    polar(v * t, psi)
                } else {
                    let p = polar(v * t_on, psi);
                    let a = psi + w * (t - t_on);
                    [p[0] + v / w * (a.sin() - psi.sin()), p[1] - v / w * (a.cos() - psi.cos())]
                }
            });
        }
        "model" => {
            // Truth drawn from the predictor's own dynamics: a self-consistency
            // check. If the filter and fan are implemented correctly, coverage
            // here matches the nominal level up to Monte-Carlo error.
            let c = cfg(seed, 0);
            let ooi = random_ooi(&mut rng);
            oois = vec![Ooi { pos: ooi }];
            let prior = GoalPrior { n_oois: 1, p_goal: c.intent_hypothesis_prob };
            let mode = if rng.next_f64() < c.p_turn { Mode::Turn } else { Mode::Cruise };
            let psi = uni(&mut rng, 0.0, 2.0 * PI);
            let mut p = Particle {
                x: 0.0,
                y: 0.0,
                v: uni(&mut rng, 4.0, 11.0),
                psi,
                omega: if mode == Mode::Turn { uni(&mut rng, -0.4, 0.4) } else { 0.0 },
                mode,
                goal: prior.sample(&mut rng),
                agility: mp.sample_agility(&mut rng),
                wander: 0.0,
                course_err: 0.0,
                turn_age: 0.0,
                weave_freq: mp.sample_weave_freq(&mut rng),
                log_w: 0.0,
                weight: 1.0,
            };
            let h = 0.5;
            let n = ((T_OBS + HORIZON) / h).round() as usize;
            let mut path = vec![[0.0, 0.0]];
            let oo = [Ooi { pos: ooi }];
            for _ in 0..n {
                p.step(mp, h, c.speed_noise_mps, c.p_turn, &oo, &prior, &mut rng, &ttc());
                path.push([p.x, p.y]);
            }
            truth = Box::new(move |t| path[((t / h).round() as usize).min(n)]);
        }
        "mc_stop_go" => {
            // Cruise, brake to a hover, hold, then leave in ANY direction (a
            // multicopter does not need to turn first). The stop starts
            // between 10 s before and 4 s after the last observation.
            let (a, v0, v1) = (3.0, uni(&mut rng, 5.0, 10.0), uni(&mut rng, 5.0, 10.0));
            let (e0, e1) = (uni(&mut rng, 0.0, 2.0 * PI), uni(&mut rng, 0.0, 2.0 * PI));
            let t_s = T_OBS + uni(&mut rng, -10.0, 4.0);
            let t_h = t_s + v0 / a;
            let t_d = t_h + uni(&mut rng, 2.0, 6.0);
            oois = vec![Ooi { pos: random_ooi(&mut rng) }];
            truth = integrate(
                move |t| {
                    if t < t_s {
                        polar(v0, e0)
                    } else if t < t_h {
                        polar(v0 - a * (t - t_s), e0)
                    } else if t < t_d {
                        [0.0, 0.0]
                    } else {
                        polar((a * (t - t_d)).min(v1), e1)
                    }
                },
                T_OBS + HORIZON,
            );
        }
        "mc_depart" => {
            // Hovering (small gusty drift) for the whole observed track, then
            // departs in a random direction 0-4 s after the last observation.
            let (a, v1, e1) = (3.0, uni(&mut rng, 5.0, 10.0), uni(&mut rng, 0.0, 2.0 * PI));
            let t_d = T_OBS + uni(&mut rng, 0.0, 4.0);
            let (gx, gy) = (uni(&mut rng, 0.0, 2.0 * PI), uni(&mut rng, 0.0, 2.0 * PI));
            oois = vec![Ooi { pos: random_ooi(&mut rng) }];
            truth = integrate(
                move |t| {
                    let gust = [0.3 * (0.4 * t + gx).sin(), 0.3 * (0.3 * t + gy).sin()];
                    if t < t_d {
                        gust
                    } else {
                        add(gust, polar((a * (t - t_d)).min(v1), e1))
                    }
                },
                T_OBS + HORIZON,
            );
        }
        "mc_reverse" => {
            // Brake straight through zero and fly back the way it came.
            let (a, v, e) = (4.0, uni(&mut rng, 5.0, 10.0), uni(&mut rng, 0.0, 2.0 * PI));
            let t_r = T_OBS + uni(&mut rng, -6.0, 3.0);
            oois = vec![Ooi { pos: random_ooi(&mut rng) }];
            truth = integrate(
                move |t| polar(if t < t_r { v } else { (v - a * (t - t_r)).max(-v) }, e),
                T_OBS + HORIZON,
            );
        }
        "mc_strafe" => {
            // Velocity swings 90 degrees by accelerating sideways, without a
            // coordinated turn (the speed dips to v / sqrt(2) mid-manoeuvre).
            let (a, v, e0) = (4.0, uni(&mut rng, 5.0, 10.0), uni(&mut rng, 0.0, 2.0 * PI));
            let e1 = e0 + if rng.next_f64() < 0.5 { PI / 2.0 } else { -PI / 2.0 };
            let t_c = T_OBS + uni(&mut rng, -6.0, 4.0);
            let dur = v * 2f64.sqrt() / a;
            oois = vec![Ooi { pos: random_ooi(&mut rng) }];
            truth = integrate(
                move |t| {
                    let f = ((t - t_c) / dur).clamp(0.0, 1.0);
                    let (p0, p1) = (polar(v, e0), polar(v, e1));
                    [p0[0] + f * (p1[0] - p0[0]), p0[1] + f * (p1[1] - p0[1])]
                },
                T_OBS + HORIZON,
            );
        }
        "fw_orbit_wind" => {
            // Constant-airspeed 30-degree-bank loiter in a steady wind: the
            // ground track is a trochoid (ground speed varies by +-wind).
            let va = uni(&mut rng, 15.0, 22.0);
            let w = A_BANK30 / va * if rng.next_f64() < 0.5 { -1.0 } else { 1.0 };
            let psi0 = uni(&mut rng, 0.0, 2.0 * PI);
            let wind = polar(uni(&mut rng, 3.0, 7.0), uni(&mut rng, 0.0, 2.0 * PI));
            oois = vec![Ooi { pos: random_ooi(&mut rng) }];
            truth = integrate(move |t| add(polar(va, psi0 + w * t), wind), T_OBS + HORIZON);
        }
        "fw_racetrack" => {
            // Racetrack holding pattern: straight legs joined by 180-degree
            // turns at 30 degrees of bank, entered at a random phase.
            let va = uni(&mut rng, 15.0, 22.0);
            let r = va * va / A_BANK30;
            let leg = uni(&mut rng, 200.0, 400.0);
            let per = 2.0 * leg + 2.0 * PI * r;
            let s0 = uni(&mut rng, 0.0, per);
            let rot = uni(&mut rng, 0.0, 2.0 * PI);
            let heading = move |s: f64| {
                let s = s.rem_euclid(per);
                if s < leg {
                    0.0
                } else if s < leg + PI * r {
                    (s - leg) / r
                } else if s < 2.0 * leg + PI * r {
                    PI
                } else {
                    PI + (s - 2.0 * leg - PI * r) / r
                }
            };
            oois = vec![Ooi { pos: random_ooi(&mut rng) }];
            truth = integrate(move |t| polar(va, rot + heading(s0 + va * t)), T_OBS + HORIZON);
        }
        "fw_turn" => {
            // Straight flight, then a 30-degree-bank turn through 45-180
            // degrees starting within 6 s of the last observation.
            let va = uni(&mut rng, 15.0, 22.0);
            let w = A_BANK30 / va * if rng.next_f64() < 0.5 { -1.0 } else { 1.0 };
            let psi0 = uni(&mut rng, 0.0, 2.0 * PI);
            let turn = uni(&mut rng, PI / 4.0, PI);
            let t_on = T_OBS + uni(&mut rng, -6.0, 6.0);
            let t_off = t_on + turn / w.abs();
            oois = vec![Ooi { pos: random_ooi(&mut rng) }];
            truth = integrate(
                move |t| polar(va, psi0 + w * (t.clamp(t_on, t_off) - t_on)),
                T_OBS + HORIZON,
            );
        }
        _ => unreachable!(),
    }

    let n_obs = T_OBS as usize;
    let track = (0..=n_obs)
        .map(|k| {
            let t = k as f64;
            let p = truth(t);
            Observation {
                t,
                pos: [p[0] + rng.gauss(0.0, SIGMA), p[1] + rng.gauss(0.0, SIGMA)],
                v: 0.0,
                psi: 0.0,
                cov: [SIGMA * SIGMA, 0.0, SIGMA * SIGMA],
            }
        })
        .collect();
    Case {
        family,
        ttc: family_ttc(family),
        seed,
        track,
        oois,
        truth_now: truth(T_OBS),
        truth_future: truth(T_OBS + HORIZON),
    }
}

/// Outcome of one prediction against its truth.
#[derive(Clone, Copy, Debug, Default)]
pub struct Outcome {
    /// Truth bearing inside the reported 50% / 90% sector.
    pub in50: bool,
    pub in90: bool,
    /// HPD level of the truth bin: fan mass in bins more probable than the
    /// truth's bin (+ half the ties, incl. the truth bin itself). Uniform on
    /// [0, 1] for a calibrated fan; `level < a` is coverage at level `a`.
    pub hpd: f64,
    /// Sector widths (radians).
    pub w50: f64,
    pub w90: f64,
    /// Log score: -ln of the fan's probability density (per radian) at the
    /// true bearing, with 1% of the mass spread uniformly so a miss costs a
    /// bounded amount. A proper scoring rule: lower is better, and it rewards
    /// a fan that is both sharp and right, which coverage alone does not.
    pub log_score: f64,
}

pub fn score(case: &Case, mp: &ModelParams, n_samples: u32) -> Outcome {
    let c = cfg(case.seed, n_samples);
    let p = predict_with(&case.track, &case.ttc, &case.oois, &c, mp);
    let (a, b) = (case.truth_now, case.truth_future);
    let brg = (b[1] - a[1]).atan2(b[0] - a[0]).rem_euclid(2.0 * PI);
    let inside = |start: f64, w: f64| (brg - start).rem_euclid(2.0 * PI) < w;

    let n_bins = p.bin_probs.len();
    let j = ((brg / (2.0 * PI) * n_bins as f64) as usize).min(n_bins - 1);
    let pj = p.bin_probs[j];
    let above: f64 = p.bin_probs.iter().filter(|&&q| q > pj).sum();
    let ties: f64 = p.bin_probs.iter().filter(|&&q| q == pj).sum();
    let bin_w = 2.0 * PI / n_bins as f64;
    let density = (0.99 * pj + 0.01 / n_bins as f64) / bin_w;
    Outcome {
        log_score: -density.ln(),
        in50: inside(p.sector50_start, p.sector50),
        in90: inside(p.sector90_start, p.sector90),
        hpd: above + 0.5 * ties,
        w50: p.sector50,
        w90: p.sector90,
    }
}

/// Summary of a family (or of all cases).
#[derive(Clone, Copy, Debug, Default)]
pub struct Summary {
    pub n: usize,
    pub cov50: f64,
    pub cov90: f64,
    pub hpd50: f64,
    pub hpd90: f64,
    pub w50_deg: f64,
    pub w90_deg: f64,
    pub log_score: f64,
}

pub fn summarise(outs: &[Outcome]) -> Summary {
    let n = outs.len().max(1) as f64;
    let frac = |f: &dyn Fn(&Outcome) -> bool| outs.iter().filter(|o| f(o)).count() as f64 / n;
    Summary {
        n: outs.len(),
        cov50: frac(&|o| o.in50),
        cov90: frac(&|o| o.in90),
        hpd50: frac(&|o| o.hpd < 0.5),
        hpd90: frac(&|o| o.hpd < 0.9),
        w50_deg: outs.iter().map(|o| o.w50).sum::<f64>() / n * 180.0 / PI,
        w90_deg: outs.iter().map(|o| o.w90).sum::<f64>() / n * 180.0 / PI,
        log_score: outs.iter().map(|o| o.log_score).sum::<f64>() / n,
    }
}

/// Run every family x seed in parallel; returns per-family outcomes.
pub fn run(mp: &ModelParams, seeds: u64, n_samples: u32, families: &[usize]) -> Vec<Vec<Outcome>> {
    let jobs: Vec<(usize, u64)> =
        families.iter().flat_map(|&f| (1..=seeds).map(move |s| (f, s))).collect();
    let n_threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4);
    let chunk = (jobs.len() + n_threads - 1) / n_threads;
    let results: Vec<(usize, Outcome)> = std::thread::scope(|sc| {
        let handles: Vec<_> = jobs
            .chunks(chunk.max(1))
            .map(|js| {
                sc.spawn(move || {
                    js.iter()
                        .map(|&(f, s)| (f, score(&make_case(f, s, mp), mp, n_samples)))
                        .collect::<Vec<_>>()
                })
            })
            .collect();
        handles.into_iter().flat_map(|h| h.join().unwrap()).collect()
    });
    let mut per = vec![Vec::new(); FAMILIES.len()];
    for (f, o) in results {
        per[f].push(o);
    }
    per
}

/// Calibration error: mean over families of |cov50 - 0.5| + |cov90 - 0.9|
/// using the reported sectors (what the user sees).
pub fn objective(per: &[Vec<Outcome>], families: &[usize]) -> f64 {
    families
        .iter()
        .map(|&f| {
            let s = summarise(&per[f]);
            (s.cov50 - 0.5).abs() + (s.cov90 - 0.9).abs()
        })
        .sum::<f64>()
        / families.len() as f64
}

/// Index of the self-consistency family.
pub fn model_family() -> usize {
    FAMILIES.iter().position(|&f| f == "model").unwrap()
}

/// Mean log score over families (equal weight per family; lower is better).
pub fn mean_log_score(per: &[Vec<Outcome>], families: &[usize]) -> f64 {
    families.iter().map(|&f| summarise(&per[f]).log_score).sum::<f64>() / families.len() as f64
}
