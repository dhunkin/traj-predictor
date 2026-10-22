//! scenarios.rs — end-to-end Monte-Carlo prediction across the reference
//! scenarios from spec §7.
//!
//! Each test builds a scenario (same parser logic the CLI uses), runs
//! `mc::predict` from the last observation, and checks the published Stage 1
//! behaviour. Fixed seeds are used throughout; assertions use tolerances,
//! never exact floats (per spec §7).
//!
//! Stage 1 note: `oois` is an extension point and is NOT used for inference
//! (see `mc::predict`), so the "two OOIs" scenario is exercised as a
//! symmetric two-sided heading spread, and heading-sector width is driven by
//! turn-rate uncertainty (`p_turn`), not by position (`cov`) noise.

use predictor::histogram::psi_rem_euclid;
use predictor::mc::predict;
use predictor::prng::Xoshiro256ss;
use predictor::simulator::generate_track;
use predictor::types::{KinState, Observation, PredictConfig, Ttc};
use std::f64::consts::PI;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn ttc(v_min: f64, v_max: f64, omega_max: f64) -> Ttc {
    Ttc {
        v_min,
        v_max,
        omega_max,
    }
}

fn cfg(
    horizon: f64,
    n_samples: u32,
    p_turn: f64,
    azimuth_bins: u32,
    seed: u64,
    speed_noise: f64,
) -> PredictConfig {
    PredictConfig {
        horizon_s: horizon,
        n_samples,
        dt_s: 1.0, // reserved for Stage 2, unused in Stage 1
        p_turn,
        azimuth_bins,
        speed_noise_mps: speed_noise,
        seed,
    }
}

/// Last observation of a straight +X track ending at (x, y): heading 0,
/// speed 5 m/s. `predict` uses last.v / last.psi as-is (Stage 1).
fn straight_last_obs(x: f64, y: f64, cov: [f64; 3]) -> Observation {
    Observation {
        t: 0.0,
        pos: [x, y],
        v: 5.0,
        psi: 0.0,
        cov,
    }
}

/// Smallest circular sector window (start bin, width in bins) whose cumulative
/// density reaches `threshold`. Mirrors histogram::sector_width's algorithm so
/// callers can test membership of a heading in the sector.
fn smallest_window(
    bin_probs: &[f64],
    threshold: f64,
    n_bins: usize,
) -> (usize, usize) {
    let doubled: Vec<f64> =
        bin_probs.iter().cloned().chain(bin_probs.iter().cloned()).collect();
    for k in 1..=n_bins {
        let mut sum: f64 = doubled[0..k].iter().sum::<f64>();
        if sum >= threshold {
            return (0, k);
        }
        for start in 1..n_bins {
            sum += doubled[start + k - 1] - doubled[start - 1];
            if sum >= threshold {
                return (start, k);
            }
        }
    }
    (0, n_bins)
}

/// True (heading 0 -> bin 0) inside a sector window, with 0/2*pi seam wrap.
fn truth_in_sector(window: (usize, usize), n_bins: usize) -> bool {
    let (start, width) = window;
    for k in 0..width {
        if (start + k) % n_bins == 0 {
            return true;
        }
    }
    false
}

/// Sample a bin index from a probability mass function (inverse-CDF).
fn sample_bin(bin_probs: &[f64], rng: &mut Xoshiro256ss) -> usize {
    let u = rng.next_f64();
    let mut cum = 0.0;
    for (i, &p) in bin_probs.iter().enumerate() {
        cum += p;
        if u <= cum {
            return i;
        }
    }
    bin_probs.len() - 1
}

// ---------------------------------------------------------------------------
// 1. straight_1_ooi_no_noise
// ---------------------------------------------------------------------------

#[test]
fn straight_1_ooi_no_noise() {
    // Zero sensor noise (cov = [0,0,0]) and no turn probability -> the point
    // cloud collapses to a single arc, so the 90% sector is a single bin.
    let obs = straight_last_obs(20.0, 0.0, [0.0, 0.0, 0.0]);
    let t = ttc(2.0, 8.0, 0.5);
    let c = cfg(10.0, 10000, 0.0, 72, 42, 0.0);
    let prediction = predict(&obs, &t, &[], &c);

    let sector90_deg = prediction.sector90 * 180.0 / PI;
    assert!(
        sector90_deg < 10.0,
        "sector90_deg = {}, expected < 10 (no noise)",
        sector90_deg
    );

    // Ground truth (heading 0) must be inside the 90% sector.
    let window = smallest_window(&prediction.bin_probs, 0.9, 72);
    assert!(
        truth_in_sector(window, 72),
        "ground truth not inside the 90% sector"
    );
}

// ---------------------------------------------------------------------------
// 2. two_oois_between
// ---------------------------------------------------------------------------

#[test]
fn two_oois_between() {
    // Two symmetric OOIs with the UAV between them -> in Stage 1 (oois unused
    // for inference) this shows up as a symmetric two-sided heading spread:
    // both the + and - heading regions carry significant probability mass.
    let obs = straight_last_obs(20.0, 0.0, [1.0, 0.0, 1.0]);
    let t = ttc(2.0, 8.0, 0.5);
    let c = cfg(10.0, 20000, 1.0, 72, 7, 0.0);
    let prediction = predict(&obs, &t, &[], &c);

    let n = prediction.bin_probs.len();
    let bin_width = 2.0 * PI / n as f64;
    let mut right: f64 = 0.0; // heading in (0, pi)
    let mut left: f64 = 0.0; // heading in (-pi, 0)
    for i in 0..n {
        let heading = (i as f64 + 0.5) * bin_width; // bin center in [0, 2*pi)
        let h = if heading > PI { heading - 2.0 * PI } else { heading };
        let p = prediction.bin_probs[i];
        if h > 0.0 {
            right = right.max(p);
        } else {
            left = left.max(p);
        }
    }

    assert!(
        right > 0.01 && left > 0.01,
        "expected mass on both sides: right max = {right}, left max = {left}"
    );
    // Two-sided spread: the 90% sector is wide (well over a quarter circle).
    let sector90_deg = prediction.sector90 * 180.0 / PI;
    assert!(
        sector90_deg > 90.0,
        "sector90_deg = {}, expected wide two-sided spread",
        sector90_deg
    );
}

// ---------------------------------------------------------------------------
// 3. noisy_track_wide
// ---------------------------------------------------------------------------

#[test]
fn noisy_track_wide() {
    // Large sensor noise (cov = [9,0,9], sigma = 3 m/axis) plus turn
    // uncertainty -> wide point cloud and wide heading sector, but the ground
    // truth (heading 0) stays inside the 90% sector.
    let obs = straight_last_obs(20.0, 0.0, [9.0, 0.0, 9.0]);
    let t = ttc(2.0, 8.0, 0.5);
    let c = cfg(10.0, 20000, 1.0, 72, 13, 0.0);
    let prediction = predict(&obs, &t, &[], &c);

    // Wide heading sector.
    let sector90_deg = prediction.sector90 * 180.0 / PI;
    assert!(
        sector90_deg > 60.0,
        "sector90_deg = {}, expected wide (> 60)",
        sector90_deg
    );

    // Wide point cloud: max displacement of a final point from the start.
    let start = [20.0, 0.0];
    let max_dev: f64 = prediction
        .point_cloud
        .iter()
        .map(|&p| (p[0] - start[0]).powi(2) + (p[1] - start[1]).powi(2))
        .fold(0.0, f64::max)
        .sqrt();
    assert!(
        max_dev > 20.0,
        "point cloud max dev = {}, expected wide (> 20)",
        max_dev
    );

    // Ground truth still inside the 90% sector.
    let window = smallest_window(&prediction.bin_probs, 0.9, 72);
    assert!(
        truth_in_sector(window, 72),
        "ground truth not inside the 90% sector despite wide noise"
    );
}

// ---------------------------------------------------------------------------
// 4. low_vs_high_turn_rate
// ---------------------------------------------------------------------------

#[test]
fn low_vs_high_turn_rate() {
    // omega_max = 0.05 vs 1.0, all else equal. The high-turn-rate run must
    // produce a strictly wider 90% sector.
    let obs = straight_last_obs(20.0, 0.0, [1.0, 0.0, 1.0]);

    let t_low = ttc(2.0, 8.0, 0.05);
    let c_low = cfg(10.0, 20000, 1.0, 72, 42, 0.0);
    let p_low = predict(&obs, &t_low, &[], &c_low);
    let sector90_low = p_low.sector90 * 180.0 / PI;

    let t_high = ttc(2.0, 8.0, 1.0);
    let c_high = cfg(10.0, 20000, 1.0, 72, 42, 0.0);
    let p_high = predict(&obs, &t_high, &[], &c_high);
    let sector90_high = p_high.sector90 * 180.0 / PI;

    assert!(
        sector90_high > sector90_low,
        "high-turn sector90 ({}) must be strictly wider than low-turn ({})",
        sector90_high,
        sector90_low
    );
}

// ---------------------------------------------------------------------------
// 5. coverage_90  (#[ignore] by default; run with --ignored)
// ---------------------------------------------------------------------------

#[test]
#[ignore]
fn coverage_90() {
    // 1000 simulated tracks (different seeds). Fraction where a fresh heading
    // drawn from the predicted distribution falls inside the 90% sector must
    // be in [0.85, 0.95].
    let obs = straight_last_obs(20.0, 0.0, [1.0, 0.0, 1.0]);
    let t = ttc(2.0, 8.0, 0.5);
    let n_tracks = 1000u64;
    let mut inside = 0u64;

    for seed in 1..=n_tracks {
        let c = cfg(10.0, 20000, 1.0, 72, seed, 0.0);
        let prediction = predict(&obs, &t, &[], &c);
        let window = smallest_window(&prediction.bin_probs, 0.9, 72);
        let mut rng = Xoshiro256ss::from_seed(seed.wrapping_mul(2654435761));
        let j = sample_bin(&prediction.bin_probs, &mut rng);
        let (start, width) = window;
        let mut in_win = false;
        for k in 0..width {
            if (start + k) % 72 == j {
                in_win = true;
                break;
            }
        }
        if in_win {
            inside += 1;
        }
    }

    let frac = inside as f64 / n_tracks as f64;
    assert!(
        frac >= 0.85 && frac <= 0.95,
        "coverage_90 = {}, expected in [0.85, 0.95]",
        frac
    );
}

// ---------------------------------------------------------------------------
// 6. coverage_50  (#[ignore] by default; run with --ignored)
// ---------------------------------------------------------------------------

#[test]
#[ignore]
fn coverage_50() {
    // Same as coverage_90 for the 50% sector; expectation [0.45, 0.55].
    let obs = straight_last_obs(20.0, 0.0, [1.0, 0.0, 1.0]);
    let t = ttc(2.0, 8.0, 0.5);
    let n_tracks = 1000u64;
    let mut inside = 0u64;

    for seed in 1..=n_tracks {
        let c = cfg(10.0, 20000, 1.0, 72, seed, 0.0);
        let prediction = predict(&obs, &t, &[], &c);
        let window = smallest_window(&prediction.bin_probs, 0.5, 72);
        let mut rng = Xoshiro256ss::from_seed(seed.wrapping_mul(2654435761));
        let j = sample_bin(&prediction.bin_probs, &mut rng);
        let (start, width) = window;
        let mut in_win = false;
        for k in 0..width {
            if (start + k) % 72 == j {
                in_win = true;
                break;
            }
        }
        if in_win {
            inside += 1;
        }
    }

    let frac = inside as f64 / n_tracks as f64;
    assert!(
        frac >= 0.45 && frac <= 0.55,
        "coverage_50 = {}, expected in [0.45, 0.55]",
        frac
    );
}

// ---------------------------------------------------------------------------
// Cross-check against the committed CLI reference output (scenarios/simple.txt)
// ---------------------------------------------------------------------------

#[test]
fn reference_scenario_numbers() {
    // Last obs of simple.txt: straight +X, v = 5, psi = 0, cov = [1,0,1].
    let obs = straight_last_obs(20.0, 0.0, [1.0, 0.0, 1.0]);
    let t = ttc(2.0, 8.0, 0.5);
    let c = cfg(10.0, 10000, 0.3, 72, 42, 0.5);
    let prediction = predict(&obs, &t, &[], &c);

    let sector50_deg = prediction.sector50 * 180.0 / PI;
    let sector90_deg = prediction.sector90 * 180.0 / PI;
    let heading_mean_deg = psi_rem_euclid(prediction.final_heading_mean) * 180.0 / PI;
    let n_bins_nonzero = prediction
        .bin_probs
        .iter()
        .filter(|&&p| p > 0.0)
        .count();

    assert!(
        (sector50_deg - 5.0).abs() < 1.0,
        "sector50_deg = {}, expected ~5",
        sector50_deg
    );
    assert!(
        (sector90_deg - 180.0).abs() < 3.0,
        "sector90_deg = {}, expected ~180",
        sector90_deg
    );
    let h = if heading_mean_deg > 180.0 {
        heading_mean_deg - 360.0
    } else {
        heading_mean_deg
    };
    assert!(h.abs() < 2.0, "heading_mean_deg = {}, expected near 0", heading_mean_deg);
    assert!(
        n_bins_nonzero >= 70,
        "n_bins_nonzero = {}, expected ~72",
        n_bins_nonzero
    );
}

// ---------------------------------------------------------------------------
// Simulator + predictor integration (exercises generate_track end to end)
// ---------------------------------------------------------------------------

#[test]
fn simulator_then_predict() {
    // Build a straight +X truth track with the simulator, then predict from
    // its last observation.
    let start = KinState {
        pos: [0.0, 0.0],
        v: 5.0,
        psi: 0.0,
    };
    let t = ttc(2.0, 8.0, 0.5);
    let track = generate_track(&[start], &t, 1.0, [1.0, 0.0, 1.0], 42);
    assert_eq!(track.len(), 1, "one waypoint -> one observation");

    let last = &track[0];
    // First observation carries v = 0, psi = 0 (start of track).
    assert_eq!(last.v, 5.0, "simulator sets v from start");
    assert_eq!(last.psi, 0.0, "simulator sets psi from start");

    let c = cfg(10.0, 10000, 0.3, 72, 42, 0.5);
    let prediction = predict(last, &t, &[], &c);
    let sum: f64 = prediction.bin_probs.iter().sum();
    assert!((sum - 1.0).abs() < 1e-6, "bin_probs sum = {}, expected ~1", sum);
    assert_eq!(
        prediction.bin_probs.len(),
        c.azimuth_bins as usize,
        "bin count == azimuth_bins"
    );
}
