//! Tests for motion.rs — analytic arc propagation, straight line, circle
//! radius, and speed/omega clamping.

use predictor::motion::propagate;
use predictor::prng::Xoshiro256ss;
use predictor::types::Ttc;

/// Straight line: omega == 0 -> pos advances by v*t along the heading.
#[test]
fn omega_zero_is_straight_line() {
    let ttc = Ttc {
        v_min: 0.0,
        v_max: 100.0,
        a_lat: 9.0,
        a_long: 3.0,
        omega_cap: 1.0,
    };
    let mut rng = Xoshiro256ss::from_seed(1);
    // psi = 0 -> move along +X. No speed noise so it's exact.
    let (pos, psi, v, omega) = propagate(
        [0.0, 0.0], 10.0, 0.0, 0.0, 5.0, 0.0, &mut rng, &ttc,
    );
    assert!((pos[0] - 50.0).abs() < 1e-9, "x: {}", pos[0]);
    assert!((pos[1] - 0.0).abs() < 1e-9, "y: {}", pos[1]);
    assert!((psi - 0.0).abs() < 1e-9, "psi: {}", psi);
    assert!((v - 10.0).abs() < 1e-9, "v: {}", v);
    assert!((omega - 0.0).abs() < 1e-9, "omega: {}", omega);
}

/// omega == omega_max -> after one full turn (t = 2*pi/omega) the UAV should
/// return to the start, tracing a circle of radius r = v/omega_max.
#[test]
fn omega_max_traces_circle_of_radius_v_over_omega() {
    let omega_max = 0.5;
    let v = 10.0;
    let ttc = Ttc {
        v_min: 0.0,
        v_max: 100.0,
        omega_cap: omega_max,
        a_lat: 9.0,
        a_long: 3.0,
    };
    let mut rng = Xoshiro256ss::from_seed(2);
    let t = 2.0 * std::f64::consts::PI / omega_max; // one full revolution
    let (pos, psi, _v, omega) = propagate(
        [0.0, 0.0],
        v,
        std::f64::consts::PI / 2.0,
        omega_max,
        t,
        0.0,
        &mut rng,
        &ttc,
    );
    // Back to the start after a full circle.
    assert!(pos[0].abs() < 1e-6, "x: {}", pos[0]);
    assert!(pos[1].abs() < 1e-6, "y: {}", pos[1]);
    // Heading advanced by exactly 2*pi (returned reduced to [0, 2*pi) via
    // rem_euclid, so PI/2 + 2*pi reduces back to PI/2).
    assert!((psi - (std::f64::consts::PI / 2.0)).abs() < 1e-6,
        "psi: {}",
        psi
    );
    // Radius check via the half-turn diameter: after omega*t = pi the UAV
    // is diametrically opposite the start, so the distance from start is
    // 2r where r = v / omega_max.
    let r = v / omega_max;
    let (half_pos, _half_psi, _half_v, _half_omega) = propagate(
        [0.0, 0.0],
        v,
        std::f64::consts::PI / 2.0,
        omega_max,
        t / 2.0,
        0.0,
        &mut rng,
        &ttc,
    );
    let chord = (half_pos[0] * half_pos[0] + half_pos[1] * half_pos[1]).sqrt();
    assert!((chord - 2.0 * r).abs() < 1e-6,
        "diameter: {}, expected 2r = {}",
        chord,
        2.0 * r
    );
    assert!((omega - omega_max).abs() < 1e-9, "omega: {}", omega);
}

/// Speed clamping: v + noise is clamped to [v_min, v_max].
#[test]
fn speed_is_clamped() {
    let ttc = Ttc {
        v_min: 2.0,
        v_max: 8.0,
        a_lat: 9.0,
        a_long: 3.0,
        omega_cap: 1.0,
    };
    let mut rng = Xoshiro256ss::from_seed(3);
    // v=100, huge noise, omega=0 -> straight line. Per BUG 1 fix, position
    // uses the clamped v_new (sampled velocity), not the input v.
    let (pos, _psi, v, _omega) = propagate(
        [0.0, 0.0], 100.0, 0.0, 0.0, 1.0, 1000.0, &mut rng, &ttc,
    );
    assert!((v - 8.0).abs() < 1e-9, "v clamped high: {}", v);
    // omega=0, t=1, psi=0 => pos[0] = v_new * t * cos(psi) = v_new.
    assert!((pos[0] - v).abs() < 1e-6, "x: {}", pos[0]);

    // v = -100 -> v_new clamped to v_min.
    let (_pos2, _psi2, v2, _omega2) = propagate(
        [0.0, 0.0], -100.0, 0.0, 0.0, 1.0, 0.0, &mut rng, &ttc,
    );
    assert!((v2 - 2.0).abs() < 1e-9, "v clamped low: {}", v2);
}

/// Omega clamping: omega beyond omega_cap is returned clamped in omega_new.
#[test]
fn omega_is_clamped_in_return_value() {
    let ttc = Ttc {
        v_min: 0.0,
        v_max: 100.0,
        a_lat: 9.0,
        a_long: 3.0,
        omega_cap: 1.0,
    };
    let mut rng = Xoshiro256ss::from_seed(4);
    // omega = 5.0 -> omega_new clamped to omega_cap (1.0).
    let (_pos, _psi, _v, omega) = propagate(
        [0.0, 0.0], 10.0, 0.0, 5.0, 1.0, 0.0, &mut rng, &ttc,
    );
    assert!((omega - 1.0).abs() < 1e-9, "omega clamped high: {}", omega);

    // omega = -5.0 -> omega_new clamped to -omega_cap (-1.0).
    let (_pos2, _psi2, _v2, omega2) = propagate(
        [0.0, 0.0], 10.0, 0.0, -5.0, 1.0, 0.0, &mut rng, &ttc,
    );
    assert!((omega2 - (-1.0)).abs() < 1e-9, "omega clamped low: {}", omega2);
}

/// Determinism: same seed + same input => bit-identical output.
#[test]
fn propagation_is_deterministic() {
    let ttc = Ttc {
        v_min: 0.0,
        v_max: 100.0,
        a_lat: 9.0,
        a_long: 3.0,
        omega_cap: 1.0,
    };
    let mut a = Xoshiro256ss::from_seed(7);
    let mut b = Xoshiro256ss::from_seed(7);
    let (pa, psi_a, va, oa) = propagate(
        [1.0, 2.0], 12.0, 0.7, 0.25, 3.0, 1.5, &mut a, &ttc,
    );
    let (pb, psi_b, vb, ob) = propagate(
        [1.0, 2.0], 12.0, 0.7, 0.25, 3.0, 1.5, &mut b, &ttc,
    );
    assert_eq!(pa, pb);
    assert_eq!(psi_a.to_bits(), psi_b.to_bits());
    assert_eq!(va.to_bits(), vb.to_bits());
    assert_eq!(oa.to_bits(), ob.to_bits());
}
