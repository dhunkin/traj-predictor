//! Tests for histogram.rs — azimuth bins and sector extraction (§4.5).

use predictor::histogram::{bin_index, psi_rem_euclid, sector_width};
use std::f64::consts::PI;

#[test]
fn psi_rem_euclid_wraps_into_zero_to_twopi() {
    assert!((psi_rem_euclid(0.5) - 0.5).abs() < 1e-9);
    assert!((psi_rem_euclid(2.0 * PI)).abs() < 1e-9);
    assert!((psi_rem_euclid(-0.5) - (2.0 * PI - 0.5)).abs() < 1e-9);
    assert!((psi_rem_euclid(3.0 * PI) - PI).abs() < 1e-9);
}

#[test]
fn bin_index_maps_heading_into_range() {
    let n_bins = 72;
    assert_eq!(bin_index(0.0, n_bins), 0);
    assert_eq!(bin_index(PI, n_bins), 36);
    assert_eq!(bin_index(2.0 * PI, n_bins), 0);
    // Wraps correctly.
    assert_eq!(bin_index(2.0 * PI + 0.001, n_bins), bin_index(0.001, n_bins));
    // Out of range is reduced.
    assert!(bin_index(-0.001, n_bins) == n_bins - 1);
}

#[test]
fn delta_function_sector_is_one_bin() {
    // All samples in bin 0.
    let counts = [100u64, 0, 0, 0];
    let n_samples = 100u64;
    let n_bins = 4usize;
    let bin_width = 2.0 * PI / n_bins as f64;

    let s50 = sector_width(&counts, n_samples, 0.5, n_bins);
    let s90 = sector_width(&counts, n_samples, 0.9, n_bins);

    assert!((s50 - bin_width).abs() < 1e-9, "s50 = {}", s50);
    assert!((s90 - bin_width).abs() < 1e-9, "s90 = {}", s90);
}

#[test]
fn uniform_sector_is_half_and_ninety() {
    // 72 bins, uniform density.
    let n_bins = 72usize;
    let n_samples = 7200u64;
    let counts = vec![100u64; n_bins];

    let s50 = sector_width(&counts, n_samples, 0.5, n_bins);
    let s90 = sector_width(&counts, n_samples, 0.9, n_bins);

    // sector50 ~ 180 deg, sector90 ~ 324 deg (within 1 bin).
    let bin_width_deg = 360.0 / n_bins as f64;
    let one_bin_rad = bin_width_deg * PI / 180.0;
    assert!(
        (s50 - PI).abs() < one_bin_rad,
        "s50 = {} rad (want ~{})",
        s50,
        PI
    );
    assert!(
        (s90 - 324.0 * PI / 180.0).abs() < one_bin_rad,
        "s90 = {} rad (want ~324 deg)",
        s90
    );
}

#[test]
fn symmetric_bimodal_sector() {
    // Two modes at bin 0 and bin 36, each with 50 samples (72 bins).
    let n_bins = 72usize;
    let n_samples = 100u64;
    let mut counts = vec![0u64; n_bins];
    counts[0] = 50;
    counts[36] = 50;

    let s50 = sector_width(&counts, n_samples, 0.5, n_bins);
    let s90 = sector_width(&counts, n_samples, 0.9, n_bins);

    let bin_width_deg = 360.0 / n_bins as f64;
    let one_bin_rad = bin_width_deg * PI / 180.0;

    // sector50 = 1 bin width (one mode).
    assert!(
        (s50 - one_bin_rad).abs() < 1e-9,
        "s50 = {} rad (want 1 bin)",
        s50
    );
    // sector90 must span both modes: 37 bins (185 deg).
    assert!(
        (s90 - 37.0 * one_bin_rad).abs() < 1e-9,
        "s90 = {} rad (want 37 bins)",
        s90
    );
}

#[test]
fn sector50_leq_sector90() {
    let counts = vec![10u64, 20, 30, 40];
    let n_samples = 100u64;
    let n_bins = 4usize;
    let s50 = sector_width(&counts, n_samples, 0.5, n_bins);
    let s90 = sector_width(&counts, n_samples, 0.9, n_bins);
    assert!(s50 <= s90, "s50 = {} should be <= s90 = {}", s50, s90);
}
