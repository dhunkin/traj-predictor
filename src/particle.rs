//! particle.rs — particle filter with goal (intent) inference.
//!
//! This module is the main entry point for heading prediction: `predict()`
//! filters the whole observation track, then fans every particle forward to
//! the horizon and histograms the direction of travel.
//!
//! Each particle is a joint hypothesis over
//!
//!   * the kinematic state `(x, y, v, psi, omega)`,
//!   * a manoeuvre mode — `Cruise` (turn rate mean-reverts) or `Turn`
//!     (turn rate is held, so loops / orbits persist), and
//!   * a goal — one of the OOIs, or `None`.
//!
//! The mode and the goal are hidden variables the filter *tests against the
//! data*. Both evolve as Markov chains (small switching hazards), and the goal
//! only biases the dynamics, it never overrides them:
//!
//!   * Cruise: omega is an Ornstein-Uhlenbeck process around a goal-biased mean
//!     `mu = K_GOAL * (bearing_to_goal - psi)` (`mu = 0` without a goal), so a
//!     goal particle bends toward its OOI with a time constant of ~1/K_GOAL.
//!   * Turn: omega is held (slow random walk), which *is* the manoeuvre. A goal
//!     particle's loop instead drifts toward the goal: its ground velocity is
//!     the circling velocity plus up to `GOAL_DRIFT_FRAC * v` along the goal
//!     bearing, so the loop keeps its shape while its centre migrates. The
//!     drift fades out for |omega| < OMEGA_LOOP: a near-straight "turn" plus a
//!     drift would otherwise explain any straight track, whatever the goal. (Modulating the
//!     speed by heading gives the same mean drift only with twice the speed
//!     oscillation of a real drifting loop, which fits the track worse.)
//!
//! The SAME dynamics (`Particle::step`) run in the filter and in prediction.
//! A goal hypothesis that does not fit the track loses weight through the
//! position likelihood alone; nothing is penalised by hand. The posterior
//! P(goal | track) is the summed weight of the particles carrying that goal and
//! is reported in `Prediction::goal_probs`.
//!
//! Pipeline:
//!   1. INITIALISE N particles around the first observation, heading and speed
//!      from the first displacement, mode drawn with P(Turn) = `p_turn`, goal
//!      drawn from the prior (`intent_hypothesis_prob`, the rest is `None`).
//!   2. FILTER over `track[1..]`: propagate, add the 2D Gaussian position
//!      log-likelihood to the log weight, normalise (log-sum-exp), and
//!      resample (per mode, see `resample_by_mode`) when the effective sample
//!      size drops below N/2.
//!   3. PREDICT: resample once, then propagate each particle from its own state
//!      to `horizon_s` and histogram the displacement bearing.
//!
//! Dependency-free: all math is manual, using `crate::histogram`,
//! `crate::vec2`, `crate::prng`, `crate::types`.

use crate::histogram::{bin_index, sector_window};
use crate::prng::Xoshiro256ss;
use crate::types::{Observation, Ooi, PredictConfig, Prediction, Ttc};
use crate::vec2::Mat2;
use std::f64::consts::PI;

// ---- Model constants -------------------------------------------------------

/// Integration sub-step (s), used in the filter and in prediction.
const SUBSTEP_S: f64 = 0.5;
/// Cruise mode: OU mean-reversion time (s) and stationary std-dev (rad/s) of omega.
const CRUISE_TAU_S: f64 = 2.0;
const CRUISE_OMEGA_SIGMA: f64 = 0.1;
/// Turn mode: random-walk intensity of the held omega (rad/s per sqrt(s)).
const TURN_OMEGA_WALK: f64 = 0.03;
/// Turn mode: minimum |omega| (rad/s). A "turn" that does not turn would be
/// a second, goal-blind way to fly straight; straight flight belongs to Cruise,
/// where the goal bias applies. 0.1 rad/s is a radius of 80 m at 8 m/s.
const OMEGA_TURN_MIN: f64 = 0.1;
/// Turn mode: std-dev of the turn-rate jump when a turn starts (rad/s).
const TURN_ENTRY_SIGMA: f64 = 0.3;
/// Total mode-switching rate (1/s). Split so the stationary P(Turn) = p_turn.
const MODE_SWITCH_RATE: f64 = 0.1;
/// Goal-switching hazard (1/s): the goal is redrawn from the prior, so a
/// retargeting UAV can be followed and the posterior relaxes toward the prior.
const GOAL_SWITCH_RATE: f64 = 0.02;
/// Cruise mode: goal-steering gain (1/s); heading closes on the goal bearing
/// with a time constant of ~1/K_GOAL = 10 s.
const K_GOAL: f64 = 0.1;
/// Turn mode: drift speed of a goal particle's loop toward the goal, as a
/// fraction of its airspeed (0.08 -> 0.64 m/s at 8 m/s).
const GOAL_DRIFT_FRAC: f64 = 0.08;
/// Turn rate (rad/s) above which a turn counts as a full loop for the drift.
const OMEGA_LOOP: f64 = 0.2;
/// Speed-noise floor (m/s per sqrt(s)) during filtering, so resampled copies
/// of a particle can diversify in speed even when `speed_noise_mps` is 0.
const FILTER_SPEED_NOISE_FLOOR: f64 = 0.1;
/// Minimum share of a bank's particles kept in each mode at resampling, so a
/// mode that is locally worse (e.g. Turn during half a loop) is not wiped out
/// and can win back when the data favour it again.
const MIN_MODE_SHARE: f64 = 0.1;
/// Measurement-noise floor (m^2) added to the observation covariance.
const MEAS_FLOOR_M2: f64 = 0.1;

/// Signed angle wrap into `[-pi, pi)`.
fn wrap(a: f64) -> f64 {
    (a + PI).rem_euclid(2.0 * PI) - PI
}

/// Manoeuvre mode of a particle.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    /// Turn rate mean-reverts (toward 0, or toward the goal-biased mean).
    Cruise,
    /// Coordinated turn: the turn rate is held, so loops persist.
    Turn,
}

/// Prior over the goal label: `None` with probability `1 - p_goal`, otherwise
/// uniform over the OOIs.
#[derive(Clone, Copy, Debug)]
struct GoalPrior {
    n_oois: usize,
    p_goal: f64,
}

impl GoalPrior {
    fn sample(&self, rng: &mut Xoshiro256ss) -> Option<usize> {
        if self.n_oois == 0 || rng.next_f64() >= self.p_goal {
            None
        } else {
            Some(((rng.next_f64() * self.n_oois as f64) as usize).min(self.n_oois - 1))
        }
    }
}

/// A single particle: kinematic state, mode, goal, and weight.
#[derive(Clone, Copy, Debug)]
pub struct Particle {
    pub x: f64,
    pub y: f64,
    pub v: f64,
    pub psi: f64,
    pub omega: f64,
    pub mode: Mode,
    /// Goal hypothesis: index into `oois`, or `None`.
    pub goal: Option<usize>,
    /// Normalised log weight (the filter's working representation).
    pub log_w: f64,
    /// Linear copy of `exp(log_w)`, used for N_eff, resampling and summaries.
    pub weight: f64,
}

impl Particle {
    /// Propagate the particle over `dt` seconds with the goal-conditioned,
    /// mode-switching dynamics (see the module docstring). Used unchanged in
    /// the filter and in prediction.
    fn step(
        &mut self,
        dt: f64,
        speed_noise: f64,
        p_turn: f64,
        oois: &[Ooi],
        prior: &GoalPrior,
        rng: &mut Xoshiro256ss,
        ttc: &Ttc,
    ) {
        // 1. Goal switching (redraw from the prior).
        if prior.n_oois > 0 && rng.next_f64() < 1.0 - (-GOAL_SWITCH_RATE * dt).exp() {
            self.goal = prior.sample(rng);
        }

        // 2. Mode switching. A new turn perturbs the current turn rate (rather
        //    than redrawing it) so a particle that was already curving keeps
        //    its curvature when it is reclassified as turning.
        let rate = match self.mode {
            Mode::Cruise => MODE_SWITCH_RATE * p_turn,
            Mode::Turn => MODE_SWITCH_RATE * (1.0 - p_turn),
        };
        if rng.next_f64() < 1.0 - (-rate * dt).exp() {
            self.mode = match self.mode {
                Mode::Cruise => {
                    self.omega += rng.gauss(0.0, TURN_ENTRY_SIGMA);
                    Mode::Turn
                }
                Mode::Turn => Mode::Cruise,
            };
        }

        // 3. Speed: random walk, clamped to the turn-table limits.
        self.v = (self.v + rng.gauss(0.0, speed_noise) * dt.sqrt()).clamp(ttc.v_min, ttc.v_max);
        let limit = (ttc.a_lat / self.v.max(0.1)).min(ttc.omega_cap);

        let bearing = self
            .goal
            .map(|g| (oois[g].pos[1] - self.y).atan2(oois[g].pos[0] - self.x));

        // 4. Turn rate.
        match self.mode {
            Mode::Cruise => {
                let mu = match bearing {
                    Some(b) => (K_GOAL * wrap(b - self.psi)).clamp(-limit, limit),
                    None => 0.0,
                };
                let alpha = (-dt / CRUISE_TAU_S).exp();
                self.omega = mu
                    + alpha * (self.omega - mu)
                    + (1.0 - alpha * alpha).sqrt() * CRUISE_OMEGA_SIGMA * rng.gauss(0.0, 1.0);
            }
            Mode::Turn => {
                self.omega += rng.gauss(0.0, TURN_OMEGA_WALK) * dt.sqrt();
                if self.omega.abs() < OMEGA_TURN_MIN {
                    let sign = if self.omega < 0.0 { -1.0 } else { 1.0 };
                    self.omega = sign * OMEGA_TURN_MIN;
                }
            }
        }
        self.omega = self.omega.clamp(-limit, limit);

        // 5. Heading and position (exact arc via the chord identity).
        let half = 0.5 * self.omega * dt;
        let sinc = if half.abs() < 1e-4 { 1.0 - half * half / 6.0 } else { half.sin() / half };
        let chord = self.v * dt * sinc;
        let mid = self.psi + half;
        self.x += chord * mid.cos();
        self.y += chord * mid.sin();

        // 6. In a turn, a goal particle's loop drifts toward the goal.
        if let (Mode::Turn, Some(b)) = (self.mode, bearing) {
            let looping = (self.omega.abs() / OMEGA_LOOP).min(1.0);
            let drift = GOAL_DRIFT_FRAC * looping * self.v * dt;
            self.x += drift * b.cos();
            self.y += drift * b.sin();
        }
        self.psi = (self.psi + self.omega * dt).rem_euclid(2.0 * PI);
    }
}

/// Systematic resampling: draw `n_out` equally weighted copies from the
/// weighted set `particles`.
///
///   1. Build the cumulative-weight array.
///   2. Draw a single random start `u0 ~ U[0, 1/n_out)`.
///   3. Advance one particle index across the cumulative weights while the
///      k-th stratified position `u0 + k/n_out` lies inside it.
///   4. Reset every weight to `1 / n_out` (and every log weight to its log).
fn resample_n(particles: &[Particle], n_out: usize, rng: &mut Xoshiro256ss) -> Vec<Particle> {
    let n = particles.len();
    if n == 0 || n_out == 0 {
        return Vec::new();
    }

    let mut cum = Vec::with_capacity(n);
    let mut total = 0.0;
    for p in particles.iter() {
        total += p.weight;
        cum.push(total);
    }

    let step = rng.next_f64() / n_out as f64;

    let mut chosen = Vec::with_capacity(n_out);
    let mut idx = 0usize;
    for k in 0..n_out {
        let u = (step + (k as f64) / n_out as f64) * total;
        while idx < n - 1 && u > cum[idx] {
            idx += 1;
        }
        chosen.push(particles[idx]);
    }

    let log_uniform = -(n_out as f64).ln();
    for p in chosen.iter_mut() {
        p.weight = 1.0 / n_out as f64;
        p.log_w = log_uniform;
    }
    chosen
}

/// Systematic resampling of the particle set in place (same size).
pub fn resample(particles: &mut Vec<Particle>, rng: &mut Xoshiro256ss) {
    let n = particles.len();
    *particles = resample_n(particles, n, rng);
}

/// Mode-stratified resampling, in place (same size).
///
/// Each mode class is resampled on its own to `n_m` particles, where `n_m` is
/// its weight share of the set, but at least `MIN_MODE_SHARE` of the set while
/// the class has any particles. Every copy then carries weight `W_m / n_m`, so
/// the weighted mixture (and each class's total weight) is unchanged; only the
/// particle budget per class is protected.
fn resample_by_mode(particles: &mut Vec<Particle>, rng: &mut Xoshiro256ss) {
    let n = particles.len();
    let (turn, cruise): (Vec<Particle>, Vec<Particle>) =
        particles.iter().partition(|p| p.mode == Mode::Turn);
    let w_turn: f64 = turn.iter().map(|p| p.weight).sum();
    let w_cruise: f64 = cruise.iter().map(|p| p.weight).sum();
    if turn.is_empty() || cruise.is_empty() || w_turn <= 0.0 || w_cruise <= 0.0 {
        resample(particles, rng);
        return;
    }
    let n_min = ((MIN_MODE_SHARE * n as f64).ceil() as usize).max(1);
    let share = w_turn / (w_turn + w_cruise);
    let n_turn = ((share * n as f64).round() as usize).clamp(n_min, n - n_min);
    let mut out = Vec::with_capacity(n);
    for (group, w, n_g) in [(&turn, w_turn, n_turn), (&cruise, w_cruise, n - n_turn)] {
        let mut copies = resample_n(group, n_g, rng);
        let log_w = (w / (w_turn + w_cruise) / n_g as f64).ln();
        for p in copies.iter_mut() {
            p.log_w = log_w;
            p.weight = log_w.exp();
        }
        out.extend(copies);
    }
    *particles = out;
}

/// Normalise the log weights (log-sum-exp) and refresh the linear copies.
///
/// Returns `log_z = ln(sum_i exp(log_w_i))`. When the weights were normalised
/// before the latest likelihood was added, `log_z` is the log predictive
/// likelihood of that observation under this particle set. If every weight is
/// non-finite (track lost), the set is reset to uniform and `-inf` returned.
fn normalise(particles: &mut [Particle]) -> f64 {
    let n = particles.len() as f64;
    let max = particles.iter().map(|p| p.log_w).fold(f64::NEG_INFINITY, f64::max);
    if !max.is_finite() {
        for p in particles.iter_mut() {
            p.log_w = -n.ln();
            p.weight = 1.0 / n;
        }
        return f64::NEG_INFINITY;
    }
    let sum: f64 = particles.iter().map(|p| (p.log_w - max).exp()).sum();
    let log_z = max + sum.ln();
    for p in particles.iter_mut() {
        p.log_w -= log_z;
        p.weight = p.log_w.exp();
    }
    log_z
}

/// One sub-filter of the multiple-model bank: all its particles share the
/// goal hypothesis `goal`, and `prob` is the posterior P(goal | track so far).
struct Bank {
    goal: Option<usize>,
    prior: f64,
    prob: f64,
    particles: Vec<Particle>,
}

/// Number of integration sub-steps for an interval of `dt` seconds.
fn substeps(dt: f64) -> (usize, f64) {
    let n = (dt / SUBSTEP_S).ceil().max(1.0) as usize;
    (n, dt / n as f64)
}

/// Particle filter over the full track, then a forward prediction fan.
///
/// See the module docstring for the model. `track` is the ordered list of
/// observations; only positions and covariances are used (`Observation::psi`
/// and `Observation::v` are ignored — a real radar does not report them).
pub fn predict(
    track: &[Observation],
    ttc: &Ttc,
    oois: &[Ooi],
    cfg: &PredictConfig,
) -> Prediction {
    let n_samples = cfg.n_samples as usize;
    let n_bins = cfg.azimuth_bins as usize;

    assert!(track.len() >= 2, "Needs at least 2 observations");
    assert!(n_samples >= 2, "Needs at least 2 particles");

    let mut rng = Xoshiro256ss::from_seed(cfg.seed);
    let prior = GoalPrior {
        n_oois: oois.len(),
        p_goal: cfg.intent_hypothesis_prob.clamp(0.0, 1.0),
    };
    // Inside a bank the goal is fixed; goal switching is handled at the
    // bank-probability level instead (see the filter loop).
    let no_switch = GoalPrior { n_oois: 0, p_goal: 0.0 };
    let p_turn = cfg.p_turn.clamp(0.0, 1.0);

    // ---- 1. INITIALISATION -------------------------------------------------
    // One bank per goal hypothesis with non-zero prior: `None`, then each OOI.
    let mut hypotheses: Vec<(Option<usize>, f64)> = Vec::new();
    if oois.is_empty() {
        hypotheses.push((None, 1.0));
    } else {
        if prior.p_goal < 1.0 {
            hypotheses.push((None, 1.0 - prior.p_goal));
        }
        if prior.p_goal > 0.0 {
            for g in 0..oois.len() {
                hypotheses.push((Some(g), prior.p_goal / oois.len() as f64));
            }
        }
    }
    let per_bank = (n_samples / hypotheses.len()).max(2);

    // Heading and speed come from the first displacement, with a spread that
    // reflects the position noise of the two observations. The goal does NOT
    // change the initial physical state.
    let cov_factor = Mat2::from_cov(track[0].cov);
    let dx = track[1].pos[0] - track[0].pos[0];
    let dy = track[1].pos[1] - track[0].pos[1];
    let dt0 = (track[1].t - track[0].t).max(1e-3);
    let dist = (dx * dx + dy * dy).sqrt();
    let sigma_pos = (0.5 * (track[0].cov[0] + track[0].cov[2]) + MEAS_FLOOR_M2).sqrt();
    let base_psi = dy.atan2(dx);
    let sigma_psi = if dist > 1e-6 {
        (std::f64::consts::SQRT_2 * sigma_pos / dist + 0.1).min(PI)
    } else {
        PI
    };
    let v0 = dist / dt0;
    let sigma_v = std::f64::consts::SQRT_2 * sigma_pos / dt0 + 0.5;

    let mut banks: Vec<Bank> = Vec::with_capacity(hypotheses.len());
    for &(goal, pr) in hypotheses.iter() {
        let mut particles = Vec::with_capacity(per_bank);
        for _ in 0..per_bank {
            let noise = cov_factor.sample(&mut rng);
            let mode = if rng.next_f64() < p_turn { Mode::Turn } else { Mode::Cruise };
            let omega = match mode {
                Mode::Turn => rng.gauss(0.0, TURN_ENTRY_SIGMA),
                Mode::Cruise => rng.gauss(0.0, CRUISE_OMEGA_SIGMA),
            };
            particles.push(Particle {
                x: track[0].pos[0] + noise.x,
                y: track[0].pos[1] + noise.y,
                v: rng.gauss(v0, sigma_v).clamp(ttc.v_min, ttc.v_max),
                psi: (base_psi + rng.gauss(0.0, sigma_psi)).rem_euclid(2.0 * PI),
                omega,
                mode,
                goal,
                log_w: -(per_bank as f64).ln(),
                weight: 1.0 / per_bank as f64,
            });
        }
        banks.push(Bank { goal, prior: pr, prob: pr, particles });
    }

    // ---- 2. FILTERING LOOP (track[1..]) ------------------------------------
    // Each bank is an ordinary particle filter under its own goal-conditioned
    // dynamics. Its predictive likelihood of each observation updates the bank
    // probability, P(g) <- P_pred(g) * L_g / Z, where P_pred mixes toward the
    // prior with the goal-switching hazard (so a retargeting UAV is followed).
    let filter_speed_noise = cfg.speed_noise_mps.max(FILTER_SPEED_NOISE_FLOOR);
    for i in 1..track.len() {
        let obs = &track[i];
        let dt = obs.t - track[i - 1].t;
        if dt <= 0.0 {
            continue;
        }
        let (n_sub, h) = substeps(dt);

        // Gaussian position likelihood with a noise floor. The normalising
        // constant is the same for every particle and bank, so it is dropped.
        let a = obs.cov[0] + MEAS_FLOOR_M2;
        let b = obs.cov[1];
        let c = obs.cov[2] + MEAS_FLOOR_M2;
        let det = (a * c - b * b).max(1e-9);

        let mut log_l = Vec::with_capacity(banks.len());
        for bank in banks.iter_mut() {
            for p in bank.particles.iter_mut() {
                for _ in 0..n_sub {
                    p.step(h, filter_speed_noise, p_turn, oois, &no_switch, &mut rng, ttc);
                }
                let ex = p.x - obs.pos[0];
                let ey = p.y - obs.pos[1];
                let d2 = (c * ex * ex - 2.0 * b * ex * ey + a * ey * ey) / det;
                p.log_w += -0.5 * d2;
            }
            log_l.push(normalise(&mut bank.particles));

            let sum_sq: f64 = bank.particles.iter().map(|p| p.weight * p.weight).sum();
            if 1.0 / sum_sq < bank.particles.len() as f64 / 2.0 {
                resample_by_mode(&mut bank.particles, &mut rng);
            }
        }

        let hazard = 1.0 - (-GOAL_SWITCH_RATE * dt).exp();
        let max_l = log_l.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        if max_l.is_finite() {
            let mut z = 0.0;
            for (bank, &ll) in banks.iter_mut().zip(log_l.iter()) {
                let p_pred = (1.0 - hazard) * bank.prob + hazard * bank.prior;
                bank.prob = p_pred * (ll - max_l).exp();
                z += bank.prob;
            }
            for bank in banks.iter_mut() {
                bank.prob /= z;
            }
        }
    }

    // ---- 3. POSTERIOR SUMMARIES -------------------------------------------
    let mut goal_probs = vec![0.0; oois.len()];
    let mut turn_prob = 0.0;
    let mut v_est = 0.0;
    let mut omega_turn_sum = 0.0;
    for bank in banks.iter() {
        if let Some(g) = bank.goal {
            goal_probs[g] += bank.prob;
        }
        for p in bank.particles.iter() {
            let w = bank.prob * p.weight;
            v_est += w * p.v;
            if p.mode == Mode::Turn {
                turn_prob += w;
                omega_turn_sum += w * p.omega;
            }
        }
    }
    let omega_est = if turn_prob > 0.0 { omega_turn_sum / turn_prob } else { 0.0 };

    // ---- 4. PREDICTION FAN ------------------------------------------------
    // Draw n_samples equally weighted particles across the banks in proportion
    // to their posterior (largest-remainder split), then let each continue
    // from its OWN state under the same dynamics as the filter — now with goal
    // switching from the prior — with no steering override and no anchoring.
    let mut alloc: Vec<usize> = banks.iter().map(|b| (b.prob * n_samples as f64).floor() as usize).collect();
    let mut rest = n_samples - alloc.iter().sum::<usize>();
    let mut order: Vec<usize> = (0..banks.len()).collect();
    order.sort_by(|&x, &y| {
        let fx = banks[x].prob * n_samples as f64 - alloc[x] as f64;
        let fy = banks[y].prob * n_samples as f64 - alloc[y] as f64;
        fy.partial_cmp(&fx).unwrap_or(std::cmp::Ordering::Equal)
    });
    for &k in order.iter().cycle() {
        if rest == 0 {
            break;
        }
        alloc[k] += 1;
        rest -= 1;
    }
    let mut fan: Vec<Particle> = Vec::with_capacity(n_samples);
    for (bank, &n_k) in banks.iter().zip(alloc.iter()) {
        fan.extend(resample_n(&bank.particles, n_k, &mut rng));
    }

    let mut counts = vec![0u64; n_bins];
    let bin_width = 2.0 * PI / n_bins as f64;
    let mut point_cloud = Vec::with_capacity(n_samples);
    let mut path_cloud = Vec::with_capacity(n_samples);
    let mut path_goal = Vec::with_capacity(n_samples);
    let mut sum_sin = 0.0;
    let mut sum_cos = 0.0;
    let (n_steps, h) = substeps(cfg.horizon_s.max(SUBSTEP_S));

    for p0 in fan.iter() {
        let mut p = *p0;
        let start = [p.x, p.y];
        let mut path: Vec<[f64; 2]> = Vec::with_capacity(n_steps + 1);
        path.push(start);
        for _ in 0..n_steps {
            p.step(h, cfg.speed_noise_mps, p_turn, oois, &prior, &mut rng, ttc);
            path.push([p.x, p.y]);
        }
        // Bin the net direction of travel (start -> horizon end), which matches
        // the drawn point cloud even when the path curves.
        let bearing = (p.y - start[1]).atan2(p.x - start[0]);
        counts[bin_index(bearing, n_bins)] += 1;
        sum_sin += bearing.sin();
        sum_cos += bearing.cos();
        point_cloud.push([p.x, p.y]);
        path_cloud.push(path);
        path_goal.push(p0.goal);
    }

    // ---- 5. RETURN ---------------------------------------------------------
    let bin_probs: Vec<f64> = counts.iter().map(|&c| c as f64 / n_samples as f64).collect();
    let bin_centers: Vec<f64> = (0..n_bins).map(|i| (i as f64 + 0.5) * bin_width).collect();

    let (s50_start, s50_k) = sector_window(&counts, n_samples as u64, 0.5, n_bins);
    let (s90_start, s90_k) = sector_window(&counts, n_samples as u64, 0.9, n_bins);

    Prediction {
        bin_centers,
        bin_probs,
        sector50: s50_k as f64 * bin_width,
        sector90: s90_k as f64 * bin_width,
        sector50_start: s50_start as f64 * bin_width,
        sector90_start: s90_start as f64 * bin_width,
        point_cloud,
        path_cloud,
        path_goal,
        final_heading_mean: sum_sin.atan2(sum_cos),
        v_est,
        omega_est,
        goal_probs,
        turn_prob,
    }
}
