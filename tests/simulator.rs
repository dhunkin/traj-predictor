//! Determinism + basic correctness tests for `simulator::generate_track`.

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
/// leg. With obs_cov = [1, 0, 1] and speed_noise = 0 the track is exact.
#[test]
fn straight_line_truth_geometry() {
    // Two legs, each 1s, v = 4, psi = 0 -> advance 4 m along +x per leg.
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
    let obs_cov = [1.0, 0.0, 1.0];
    let dt = 1.0;

    let obs = generate_track(&waypoints, &ttc, dt, obs_cov, 1);

    assert_eq!(obs.len(), 3, "2 legs -> 3 observations");
    assert_eq!(obs[0].pos, [0.0, 0.0]);
    assert_eq!(obs[1].pos, [4.0, 0.0]);
    assert_eq!(obs[2].pos, [8.0, 0.0]);
    // cov field carries obs_cov.
    for o in &obs {
        assert_eq!(o.cov, [1.0, 0.0, 1.0]);
    }
}

/// Observation count equals (waypoints.len() - 1) + 1 = waypoints.len().
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

/// Single waypoint -> single observation at t=0.
#[test]
fn single_waypoint() {
    let waypoints = vec![KinState {
        pos: [2.0, 3.0],
        v: 5.0,
        psi: 0.0,
    }];
    let ttc = ttc();
    let obs_cov = [1.0, 0.0, 1.0];
    let dt = 1.0;

    let obs = generate_track(&waypoints, &ttc, dt, obs_cov, 99);
    assert_eq!(obs.len(), 1);
    assert_eq!(obs[0].t, 0.0);
    assert_eq!(obs[0].pos, [2.0, 3.0]);
}
