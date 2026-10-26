//! Determinism + basic correctness tests for `simulator::generate_track`.
//!
//! Stage 1 truth is straight-line (omega = 0, speed_noise = 0) starting at
//! truth_waypoints[0]; each observation carries Gaussian noise of covariance
//! obs_cov (via Mat2::from_cov(obs_cov).sample(&mut rng)). With obs_cov =
//! [0, 0, 0] the noise is zero and observations lie exactly on the truth line.

use predictor::simulator::{generate_curved_track, generate_track, Trajectory};
use predictor::types::{KinState, Ttc};

fn ttc() -> Ttc {
    Ttc {
        v_min: 2.0,
        v_max: 8.0,
        a_lat: 9.0,
        a_long: 3.0,
        omega_cap: 1.0,
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

/// Curved straight-line with psi0 = 0 and zero covariance: every observation
/// lies on the +X axis (|y| < 1e-9).
#[test]
fn curved_straight_equals_linear() {
    let traj = Trajectory::Straight { psi0: 0.0 };
    let obs_cov = [0.0, 0.0, 0.0];
    let speed = 5.0;
    let duration_s = 60.0;
    let dt_obs_s = 1.0;
    let seed = 42;

    let obs = generate_curved_track(&traj, speed, duration_s, dt_obs_s, obs_cov, seed, &ttc());
    assert!(obs.len() >= 2, "expected multiple observations");

    for o in &obs {
        assert!(
            o.pos[1].abs() < 1e-9,
            "obs at t={} off the +X axis: y={}",
            o.t,
            o.pos[1]
        );
    }
    // Sanity: x advances ~ speed*dt per leg.
    assert!(obs[1].pos[0] > 4.0 && obs[1].pos[0] < 6.0, "x at t=1 = {}", obs[1].pos[0]);
}

/// Curved Sine with amplitude 5: |y_k| <= 5 + 1e-9 for all k (zero cov).
#[test]
fn curved_sine_y_bounded() {
    let traj = Trajectory::Sine {
        wavelength_m: 100.0,
        amplitude_m: 5.0,
        direction: 0.0,
    };
    let obs_cov = [0.0, 0.0, 0.0];
    let speed = 5.0;
    let duration_s = 200.0; // several wavelengths
    let dt_obs_s = 1.0;
    let seed = 7;

    let obs = generate_curved_track(&traj, speed, duration_s, dt_obs_s, obs_cov, seed, &ttc());
    let mut max_y: f64 = 0.0;
    for o in &obs {
        max_y = max_y.max(o.pos[1].abs());
    }
    assert!(
        max_y <= 5.0 + 1e-9,
        "max |y| = {} exceeds amplitude + eps",
        max_y
    );
}

/// Curved Zigzag: consecutive psi values alternate sign; at least one sign
/// flip within the first 5 observations (zero cov).
#[test]
fn curved_zigzag_alternates() {
    let traj = Trajectory::Zigzag {
        segment_m: 100.0,
        angle_rad: 0.3,
        baseline_m: 0.0,
    };
    let obs_cov = [0.0, 0.0, 0.0];
    let speed = 5.0;
    let duration_s = 60.0;
    let dt_obs_s = 1.0;
    let seed = 3;

    let obs = generate_curved_track(&traj, speed, duration_s, dt_obs_s, obs_cov, seed, &ttc());
    assert!(obs.len() >= 5, "expected at least 5 observations");

    // Each leg spans segment_m/speed = 100/5 = 20 observation intervals, so the
    // first 5 observations all lie in leg 0 (same sign). Check a wider window
    // to catch the sign flip at the leg 0 -> leg 1 boundary (obs[21]).
    let mut flips = 0usize;
    for i in 1..25 {
        let (s0, s1) = (obs[i - 1].psi.signum(), obs[i].psi.signum());
        if s0 != s1 && s0 != 0.0 && s1 != 0.0 {
            flips += 1;
        }
    }
    assert!(flips >= 1, "expected at least one psi sign flip in first 25 obs");
}

/// Curved Spiral with r0 = 10, r_growth = 5, turns = 2: radius grows, so
/// |pos[last]| - |pos[0]| > 2.0 (zero cov).
#[test]
fn curved_spiral_radius_grows() {
    let traj = Trajectory::Spiral {
        r0: 10.0,
        r_growth: 5.0,
        turns: 2.0,
    };
    let obs_cov = [0.0, 0.0, 0.0];
    let speed = 5.0;
    let duration_s = 60.0;
    let dt_obs_s = 1.0;
    let seed = 9;

    let obs = generate_curved_track(&traj, speed, duration_s, dt_obs_s, obs_cov, seed, &ttc());
    let pos0 = obs[0].pos;
    let last = &obs[obs.len() - 1];
    let r0 = (pos0[0] * pos0[0] + pos0[1] * pos0[1]).sqrt();
    let rlast = (last.pos[0] * last.pos[0] + last.pos[1] * last.pos[1]).sqrt();
    assert!(
        rlast - r0 > 2.0,
        "radius growth = {} should be > 2.0 (r0={}, rlast={})",
        rlast - r0,
        r0,
        rlast
    );
}

/// Curved generate_curved_track: same seed + input -> bit-identical positions.
#[test]
fn curved_deterministic_from_seed() {
    let traj = Trajectory::Spiral {
        r0: 10.0,
        r_growth: 5.0,
        turns: 2.0,
    };
    let obs_cov = [1.0, 0.0, 1.0];
    let speed = 5.0;
    let duration_s = 60.0;
    let dt_obs_s = 1.0;

    let a = generate_curved_track(&traj, speed, duration_s, dt_obs_s, obs_cov, 12345, &ttc());
    let b = generate_curved_track(&traj, speed, duration_s, dt_obs_s, obs_cov, 12345, &ttc());

    assert_eq!(a.len(), b.len(), "observation counts must match");
    for (x, y) in a.iter().zip(b.iter()) {
        assert_eq!(x.pos, y.pos, "position mismatch at t={}", x.t);
        assert_eq!(x.psi, y.psi, "psi mismatch at t={}", x.t);
    }
}

/// Sanity: legacy generate_track still works (unchanged behaviour).
#[test]
fn legacy_generate_track_still_runs() {
    let waypoints = vec![KinState {
        pos: [0.0, 0.0],
        v: 5.0,
        psi: 0.0,
    }];
    let ttc = ttc();
    let obs = generate_track(&waypoints, &ttc, 1.0, [0.0, 0.0, 0.0], 5);
    assert_eq!(obs.len(), 1);
    assert_eq!(obs[0].pos, [0.0, 0.0]);
}

/// Zigzag psi sign matches geometry: leg 0 descends (vertex 0 at +h, vertex 1
/// at -h), so obs[0] and obs[1] have negative psi; leg 1 ascends, so obs[21]
/// has positive psi. Regression for the inverted-sign fix.
#[test]
fn zigzag_psi_sign_matches_geometry() {
    let traj = Trajectory::Zigzag {
        segment_m: 100.0,
        angle_rad: 0.3,
        baseline_m: 0.0,
    };
    let obs_cov = [0.0, 0.0, 0.0];
    let speed = 5.0;
    let duration_s = 60.0;
    let dt_obs_s = 1.0;
    let seed = 1;

    let obs = generate_curved_track(&traj, speed, duration_s, dt_obs_s, obs_cov, seed, &ttc());
    assert!(obs.len() >= 22, "expected at least 22 observations");

    // obs[0] and obs[1] are in leg 0 (descends) -> psi < 0.
    assert!(
        obs[0].psi < 0.0,
        "obs[0].psi = {} should be negative (leg 0 descends)",
        obs[0].psi
    );
    assert!(
        obs[1].psi < 0.0,
        "obs[1].psi = {} should be negative (still in leg 0)",
        obs[1].psi
    );
    // obs[21] is in leg 1 (ascends) -> psi > 0.
    assert!(
        obs[21].psi > 0.0,
        "obs[21].psi = {} should be positive (leg 1 ascends)",
        obs[21].psi
    );
}

/// Spiral psi at t = 0 equals atan2(r0, r_growth) (the tangent direction),
/// within 1e-6. Regression for the analytic-spiral-heading fix.
#[test]
fn spiral_heading_at_t0_is_tangent() {
    let traj = Trajectory::Spiral {
        r0: 10.0,
        r_growth: 5.0,
        turns: 2.0,
    };
    let obs_cov = [0.0, 0.0, 0.0];
    let speed = 5.0;
    let duration_s = 60.0;
    let dt_obs_s = 1.0;
    let seed = 1;

    let obs = generate_curved_track(&traj, speed, duration_s, dt_obs_s, obs_cov, seed, &ttc());
    let expected = (10.0_f64).atan2(5.0);
    assert!(
        (obs[0].psi - expected).abs() < 1e-6,
        "obs[0].psi = {} should equal atan2(10, 5) = {} within 1e-6",
        obs[0].psi,
        expected
    );
}
