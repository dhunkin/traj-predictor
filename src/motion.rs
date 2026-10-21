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
    let psi_new = psi + omega * t;
    let pos_new = if omega.abs() < 1e-9 {
        // Constant-velocity (straight-line) motion.
        let dx = v * t * psi.cos();
        let dy = v * t * psi.sin();
        [pos[0] + dx, pos[1] + dy]
    } else {
        // Analytic arc for a constant turn rate.
        let ratio = v / omega;
        let dx = ratio * (psi_new.sin() - psi.sin());
        let dy = ratio * (-psi_new.cos() + psi.cos());
        [pos[0] + dx, pos[1] + dy]
    };
    let v_new = (v + rng.gauss(0.0, speed_noise)).clamp(ttc.v_min, ttc.v_max);
    let omega_new = omega.clamp(-ttc.omega_max, ttc.omega_max);
    (pos_new, psi_new, v_new, omega_new)
}
