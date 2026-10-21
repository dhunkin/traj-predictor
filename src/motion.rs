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
///   - `psi_new = psi + omega * t`
///   - straight line if `|omega| < 1e-9`, else the analytic constant-turn arc
///   - `v_new = clamp(v + gauss(0, speed_noise), v_min, v_max)`
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
    let omega = omega.clamp(-ttc.omega_max, ttc.omega_max);
    let v_new = (v + rng.gauss(0.0, speed_noise))
        .clamp(ttc.v_min, ttc.v_max);
    let psi_new = psi + omega * t;
    let pos_new = if omega.abs() < 1e-9 {
        [
            pos[0] + v_new * t * psi.cos(),
            pos[1] + v_new * t * psi.sin(),
        ]
    } else {
        let ratio = v_new / omega;
        [
            pos[0] + ratio * (psi_new.sin() - psi.sin()),
            pos[1] + ratio * (-psi_new.cos() + psi.cos()),
        ]
    };
    (pos_new, psi_new, v_new, omega)
}
