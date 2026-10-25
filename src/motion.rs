//! motion.rs — propagate: CV + rate-limited turn, analytic arc.
//!
//! Analytic constant-turn-rate propagation over a single time step `t`.
//! The input turn rate `omega` drives the arc and heading update as-is;
//! the returned `omega_new` is that rate clamped to `[-omega_max,
//! omega_max]` (computed per §4.3). `omega_new` is returned so callers and
//! tests can inspect the rate-limit clamp; the arc geometry is driven by the
//! un-clamped input `omega`.
//!
//! TODO(Stage2): (reserved)

use crate::prng::Xoshiro256ss;
use crate::types::Ttc;

/// Propagate a kinematic state over one analytic arc.
///
/// Returns `(pos_new, psi_new, v_new, omega_new)`:
///   - `v_new = clamp(v + gauss(0, speed_noise), v_min, v_max)`
///   - arc via the chord identity: `half = omega * t / 2`,
///     `sinc(x) = sin(x)/x` (Taylor `1 - x²/6` near 0),
///     `chord = v_new * t * sinc(half)`, `mid = psi + half`,
///     `pos_new = pos + chord * [cos(mid), sin(mid)]` — exact for all `omega`,
///     including the straight-line limit `omega -> 0`.
///   - `psi_new = (psi + omega * t).rem_euclid(2*pi)`
///   - `omega_new = clamp(omega, -omega_max, omega_max)`
pub fn propagate(
    pos: [f64; 2],
    v: f64,
    psi: f64,
    omega: f64,
    t: f64,
    speed_noise: f64,
    rng: &mut Xoshiro256ss,
    ttc: &Ttc,
) -> ([f64; 2], f64, f64, f64) {
    let omega_clamped = omega.clamp(-ttc.omega_cap, ttc.omega_cap);
    let v_new = (v + rng.gauss(0.0, speed_noise)).clamp(ttc.v_min, ttc.v_max);
    
    let half = 0.5 * omega_clamped * t;
    let sinc = if half.abs() < 1e-4 { 1.0 - half * half / 6.0 } else { half.sin() / half };
    let chord = v_new * t * sinc;
    let mid = psi + half;
    
    let pos_new = [
        pos[0] + chord * mid.cos(),
        pos[1] + chord * mid.sin(),
    ];
    
    let psi_new = (psi + omega_clamped * t).rem_euclid(2.0 * std::f64::consts::PI);
    
    (pos_new, psi_new, v_new, omega_clamped)
}

/// Propagate the state one step forward using a time-correlated turn-rate
/// (Ornstein-Uhlenbeck) model with speed-dependent lateral-acceleration limits.
///
/// Returns `(pos_new, psi_new, v_new, omega_new)`:
///   1. `v` is perturbed by longitudinal Gaussian noise (scaled by `sqrt(dt)`)
///      and clamped to `[v_min, v_max]`.
///   2. `omega` is updated via an OU process (mean-reversion time `tau = 2.0s`),
///      then clamped to a speed-dependent limit `min(a_lat / v_new, omega_cap)`.
///   3. `psi` and `pos` are updated via the chord/sinc identity.
///
/// `turn_noise_sigma` is the OU stationary std-dev of `omega`; `speed_noise`
/// is the per-second longitudinal noise std-dev (applied per unit time).
pub fn propagate_step(
    pos: [f64; 2],
    v: f64,
    psi: f64,
    omega: f64,
    dt: f64,
    speed_noise: f64,
    turn_noise_sigma: f64,
    rng: &mut Xoshiro256ss,
    ttc: &Ttc,
) -> ([f64; 2], f64, f64, f64) {
    // 1. Update speed (longitudinal)
    let v_noise = rng.gauss(0.0, speed_noise) * dt.sqrt();
    let v_new = (v + v_noise).clamp(ttc.v_min, ttc.v_max);

    // 2. Update omega using Ornstein-Uhlenbeck process (tau = 2.0s)
    let tau = 2.0;
    let alpha = (-dt / tau).exp();
    let noise = (1.0 - alpha * alpha).sqrt() * turn_noise_sigma * rng.gauss(0.0, 1.0);
    let mut omega_new = alpha * omega + noise;

    // 3. Clamp omega using speed-dependent limit (uses updated speed)
    let max_turn = ttc.a_lat / v_new.max(0.1);
    let limit = max_turn.min(ttc.omega_cap);
    omega_new = omega_new.clamp(-limit, limit);

    // 4. Update heading and position (chord identity)
    let half = 0.5 * omega_new * dt;
    let sinc = if half.abs() < 1e-4 { 1.0 - half * half / 6.0 } else { half.sin() / half };
    let chord = v_new * dt * sinc;
    let mid = psi + half;

    let pos_new = [
        pos[0] + chord * mid.cos(),
        pos[1] + chord * mid.sin(),
    ];
    let psi_new = (psi + omega_new * dt).rem_euclid(2.0 * std::f64::consts::PI);

    (pos_new, psi_new, v_new, omega_new)
}
