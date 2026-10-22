//! mc.rs — Monte-Carlo engine.
//!
//! Monte-Carlo propagation of the UAV heading over `horizon_s`. For each of
//! `n_samples`:
//!   1. Sample initial position from `last.pos + Cholesky(last.cov)`.
//!      Last heading and speed are used as-is (Stage 1).
//!      // TODO(Stage2): replace with state estimate from filter over track.
//!   2. With probability `p_turn`, draw omega ~ gauss(0, omega_max/3),
//!      clamped to [-omega_max, omega_max]; otherwise omega = 0.
//!   3. Propagate analytically for `horizon_s` via `motion::propagate`.
//!   4. Store the final (pos, psi).
//!
//! Returns the final point cloud, the azimuth histogram built from the final
//! headings, and the 50%/90% sectors. See §4.4.
//!
//! TODO(Stage2): replace initial-position sampling with filter state estimate.

use crate::histogram::{bin_index, sector_width};
use crate::motion::propagate;
use crate::prng::Xoshiro256ss;
use crate::types::{Observation, Ooi, PredictConfig, Prediction, Ttc};
use crate::vec2::Mat2;

pub fn predict(
    last: &Observation,
    ttc: &Ttc,
    oois: &[Ooi],
    cfg: &PredictConfig,
) -> Prediction {
    let _ = oois; // Stage 1: OOIs are an extension point, not used for inference.

    let n_samples = cfg.n_samples as usize;
    let n_bins = cfg.azimuth_bins as usize;

    let mut rng = Xoshiro256ss::from_seed(cfg.seed);
    let cov_factor = Mat2::from_cov(last.cov);

    // Accumulate final headings for the histogram and circular mean.
    let mut bin_counts = vec![0.0; n_bins];
    let mut sum_cos = 0.0;
    let mut sum_sin = 0.0;
    let mut point_cloud = Vec::with_capacity(n_samples);

    let omega_std = ttc.omega_max / 3.0;

    for _ in 0..n_samples {
        // 1. Sample initial position from last.pos + Cholesky(last.cov).
        let noise = cov_factor.sample(&mut rng);
        let pos0 = [
            last.pos[0] + noise.x,
            last.pos[1] + noise.y,
        ];

        // 2. Sample turn rate.
        let omega = if rng.next_f64() < cfg.p_turn {
            let w = rng.gauss(0.0, omega_std);
            w.clamp(-ttc.omega_max, ttc.omega_max)
        } else {
            0.0
        };

        // 3. Propagate analytically for horizon_s.
        let (pos_new, psi_new, _v_new, _omega_new) = propagate(
            pos0,
            last.v,
            last.psi,
            omega,
            cfg.horizon_s,
            cfg.speed_noise_mps,
            &mut rng,
            ttc,
        );

        // 4. Store final (pos, psi).
        point_cloud.push(pos_new);
        let idx = bin_index(psi_new, n_bins);
        bin_counts[idx] += 1.0;

        sum_cos += psi_new.cos();
        sum_sin += psi_new.sin();
    }

    // Histogram density (sums to 1).
    let bin_probs: Vec<f64> = bin_counts
        .iter()
        .map(|&c| c / n_samples as f64)
        .collect();

    // Bin centers: center of bin i = (i + 0.5) * bin_width.
    let bin_width = 2.0 * std::f64::consts::PI / n_bins as f64;
    let bin_centers: Vec<f64> = (0..n_bins)
        .map(|i| (i as f64 + 0.5) * bin_width)
        .collect();

    let sector50 = sector_width(&bin_counts, n_samples, 0.5, n_bins);
    let sector90 = sector_width(&bin_counts, n_samples, 0.9, n_bins);

    // Circular mean of final headings (robust to 2*pi wrap).
    let final_heading_mean = sum_sin.atan2(sum_cos);

    Prediction {
        bin_centers,
        bin_probs,
        sector50,
        sector90,
        point_cloud,
        final_heading_mean,
    }
}
