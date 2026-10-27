//! histogram.rs — azimuth bins, 50/90% sector extraction.
//!
//! Azimuth histogram built from a set of final headings (radians), and the
//! extraction of the smallest circular sector whose cumulative density reaches
//! a given threshold. See §4.5.
//!
//! TODO(Stage2): (reserved)

use std::f64::consts::PI;

/// Reduce an angle to `[0, 2*pi)` (Euclidean/positive remainder).
pub fn psi_rem_euclid(angle: f64) -> f64 {
    let twopi = 2.0 * PI;
    let r = angle.rem_euclid(twopi);
    if r >= twopi { 0.0 } else { r }
}

/// Azimuth bin index for a heading, given the number of azimuth bins.
///
/// `floor(psi_rem_euclid(2*pi) / (2*pi / N)) % N`.
pub fn bin_index(psi: f64, n_bins: usize) -> usize {
    if n_bins == 0 {
        return 0;
    }
    let bin_width = 2.0 * PI / n_bins as f64;
    let rem = psi_rem_euclid(psi);
    let idx = (rem / bin_width) as usize;
    idx % n_bins
}

/// Smallest circular window of bins whose cumulative count reaches
/// `threshold * n_samples`, as `(start_bin, width_in_bins)`.
///
/// The counts are treated cyclically (doubled), so a window may wrap around the
/// 0/2*pi seam. The window starts at the beginning of `start_bin`, i.e. at angle
/// `start_bin * 2*pi / n_bins`. Integer counts avoid float drift in the
/// threshold. Because the window is found from the counts, not centred on a mean,
/// it stays on the dominant mode of a multi-modal distribution.
pub fn sector_window(counts: &[u64], n_samples: u64, threshold: f64, n_bins: usize) -> (usize, usize) {
    if n_bins == 0 || n_samples == 0 { return (0, 0); }
    assert!(counts.len() == n_bins, "counts length must match n_bins");

    let need = ((threshold * n_samples as f64) - 1e-9).ceil() as u64;

    let mut doubled = Vec::with_capacity(2 * n_bins);
    doubled.extend_from_slice(counts);
    doubled.extend_from_slice(counts);

    for k in 1..=n_bins {
        let mut sum: u64 = doubled[0..k].iter().sum();
        if sum >= need { return (0, k); }
        for start in 1..n_bins {
            sum = sum + doubled[start + k - 1] - doubled[start - 1];
            if sum >= need { return (start, k); }
        }
    }
    (0, n_bins)
}

/// Smallest circular sector (in radians) whose cumulative density reaches
/// `threshold`, over `n_bins` azimuth bins (§4.5).
///
/// Discrete approximation: the returned width is an integer multiple of one
/// bin width (2*pi / n_bins). See `sector_window` for the window search.
pub fn sector_width(counts: &[u64], n_samples: u64, threshold: f64, n_bins: usize) -> f64 {
    if n_bins == 0 || n_samples == 0 { return 0.0; }
    let (_, k) = sector_window(counts, n_samples, threshold, n_bins);
    k as f64 * 2.0 * PI / n_bins as f64
}
