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
    let omega_clamped = omega.clamp(-ttc.omega_max, ttc.omega_max);
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
