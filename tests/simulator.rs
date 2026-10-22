//! Determinism + basic correctness tests for `simulator::generate_track`.
//!
//! Stage 1 truth is straight-line (omega = 0, speed_noise = 0) starting at
//! truth_waypoints[0]; each observation carries Gaussian noise of covariance
//! obs_cov (via Mat2::from_cov(obs_cov).sample(&mut rng)). With obs_cov =
//! [0, 0, 0] the noise is zero and observations lie exactly on the truth line.

use predictor::simulator::generate_track;
use predictor::types::{KinState, Ttc};

fn ttc() -> Ttc {
    Ttc {
        v_min: 2.0,
        v_max: 8.0,
        omega_max: 0.5,
    }
}

/// Two runs with the same seed produce bit-identical observations.
#[test]
fn determinism_same_seed() {
    let waypoints = vec![
        KinState {
            pos: [0.0, 0.0],
            v: 5.0,
            psi: 0.0,
        },
        KinState {
            pos: [5.0, 0.0],
            v: 5.0,
            psi: 0.0,
        },
        KinState {
            pos: [10.0, 0.0],
            v: 5.0,
            psi: 0.0,
        },
    ];
    let ttc = ttc();
    let obs_cov = [1.0, 0.0, 1.0];
    let dt = 1.0;
    let seed = 42;

    let a = generate_track(&waypoints, &ttc, dt, obs_cov, seed);
    let b = generate_track(&waypoints, &ttc, dt, obs_cov, seed);

    assert_eq!(a.len(), b.len(), "observation counts must match");
    for (x, y) in a.iter().zip(b.iter()) {
        assert_eq!(x.t, y.t, "time mismatch at t={}", x.t);
        assert_eq!(x.pos, y.pos, "position mismatch at t={}", x.t);
        assert_eq!(x.cov, y.cov, "cov mismatch at t={}", x.t);
    }
}

/// Straight-line truth with omega = 0: position advances along psi by v*dt per
/// leg. With obs_cov = [0, 0, 0] the noise is zero and observations lie
/// exactly on the truth line.
#[test]
fn straight_line_truth_geometry() {
    // Three legs, each 1s, v = 4, psi = 0 -> advance 4 m along +x per leg.
    let waypoints = vec![
        KinState {
            pos: [0.0, 0.0],
            v: 4.0,
            psi: 0.0,
        },
        KinState {
            pos: [4.0, 0.0],
            v: 4.0,
            psi: 0.0,
        },
        KinState {
            pos: [8.0, 0.0],
            v: 4.0,
            psi: 0.0,
        },
    ];
    let ttc = ttc();
    let obs_cov = [0.0, 0.0, 0.0]; // zero noise -> exact truth line
    let dt = 1.0;

    let obs = generate_track(&waypoints, &ttc, dt, obs_cov, 1);

    assert_eq!(obs.len(), 3, "3 waypoints -> 3 observations");
    assert_eq!(obs[0].pos, [0.0, 0.0]);
    assert_eq!(obs[1].pos, [4.0, 0.0]);
    assert_eq!(obs[2].pos, [8.0, 0.0]);
    // cov field carries obs_cov.
    for o in &obs {
        assert_eq!(o.cov, [0.0, 0.0, 0.0]);
    }
}

/// Observation count equals waypoints.len().
#[test]
fn observation_count_matches_legs() {
    let waypoints = vec![
        KinState {
            pos: [0.0, 0.0],
            v: 5.0,
            psi: 0.0,
        },
        KinState {
            pos: [5.0, 0.0],
            v: 5.0,
            psi: 0.0,
        },
        KinState {
            pos: [10.0, 0.0],
            v: 5.0,
            psi: 0.0,
        },
        KinState {
            pos: [15.0, 0.0],
            v: 5.0,
            psi: 0.0,
        },
    ];
    let ttc = ttc();
    let obs_cov = [1.0, 0.0, 1.0];
    let dt = 1.0;

    let obs = generate_track(&waypoints, &ttc, dt, obs_cov, 3);
    assert_eq!(obs.len(), 4, "4 waypoints -> 4 observations");
}

/// Single waypoint -> single observation at t = 0, exactly on the truth point
/// (obs_cov = [0, 0, 0]).
#[test]
fn single_waypoint() {
    let waypoints = vec![KinState {
        pos: [2.0, 3.0],
        v: 5.0,
        psi: 0.0,
    }];
    let ttc = ttc();
    let obs_cov = [0.0, 0.0, 0.0]; // zero noise -> exact truth point
    let dt = 1.0;

    let obs = generate_track(&waypoints, &ttc, dt, obs_cov, 99);
    assert_eq!(obs.len(), 1);
    assert_eq!(obs[0].t, 0.0);
    assert_eq!(obs[0].pos, [2.0, 3.0]);
}

/// With obs_cov = [1, 0, 1] along a straight +X line, the per-observation noise
/// variance in y should be close to 1.0 (not 0). This test fails on the
/// noiseless (pre-fix) simulator and passes after BUG 1 is fixed.
#[test]
fn noise_variance_matches_obs_cov() {
    // 500 waypoints -> 500 observations (one per waypoint), all on the +X line
    // since v is along psi = 0 for every leg.
    let waypoints: Vec<KinState> = (0..500)
        .map(|i| KinState {
            pos: [i as f64 * 10.0, 0.0],
            v: 10.0,
            psi: 0.0,
        })
        .collect();
    let ttc = ttc();
    let obs_cov = [1.0, 0.0, 1.0];
    let dt = 1.0;
    let n = 500;

    let obs = generate_track(&waypoints, &ttc, dt, obs_cov, 123);
    assert_eq!(obs.len(), n);

    // truth.y = 0 for all obs (psi = 0), so Var(obs.y) ~= obs_cov[2] = 1.0.
    let y: Vec<f64> = obs.iter().map(|o| o.pos[1]).collect();
    let mean = y.iter().sum::<f64>() / n as f64;
    let var: f64 = y.iter().map(|yi| (yi - mean) * (yi - mean)).sum::<f64>() / n as f64;
    assert!(
        (var - 1.0).abs() < 0.15,
        "Var(y) = {} should be close to 1.0",
        var
    );
}

/// RNG order: two runs with the same seed produce identical noise; a different
/// seed produces different noise. Verifies BUG 3 (no spurious gauss(0,0) draw)
/// keeps the RNG stream deterministic and seed-dependent.
#[test]
fn rng_order_is_seed_dependent() {
    let waypoints = vec![KinState {
        pos: [0.0, 0.0],
        v: 10.0,
        psi: 0.0,
    }];
    let ttc = ttc();
    let obs_cov = [1.0, 0.0, 1.0];
    let dt = 1.0;

    let a = generate_track(&waypoints, &ttc, dt, obs_cov, 7);
    let b = generate_track(&waypoints, &ttc, dt, obs_cov, 7);
    let c = generate_track(&waypoints, &ttc, dt, obs_cov, 8);

    // Same seed -> identical noise on every observation.
    for (x, y) in a.iter().zip(b.iter()) {
        assert_eq!(x.pos, y.pos, "same seed must give identical noise");
    }

    // Different seed -> at least one observation's noise differs.
    let mut any_diff = false;
    for (x, y) in a.iter().zip(c.iter()) {
        if x.pos != y.pos {
            any_diff = true;
        }
    }
    assert!(any_diff, "different seed should produce different noise");
}
