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
//!   * Cruise: omega = wander + `k_goal * (bearing_to_goal - psi)`, or without
//!     a goal `k_course * (course - psi)`: cruise is course-keeping flight
//!     around a slowly drifting course (to the goal, if any). The wander is an Ornstein-Uhlenbeck process
//!     around 0 whose size (the particle's `agility`) the filter learns, and the
//!     goal term closes the heading error with a time constant of ~1/`k_goal`.
//!   * Turn: omega is held (slow random walk), which *is* the manoeuvre. A goal
//!     particle's loop instead drifts toward the goal: its ground velocity is
//!     the circling velocity plus up to `goal_drift_frac * v` along the goal
//!     bearing, so the loop keeps its shape while its centre migrates. The
//!     drift fades out for |omega| < `omega_loop`: a near-straight "turn" plus a
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
/// Tunable model parameters. `Default` holds the calibrated values used by
/// `predict()`; `predict_with()` takes an explicit set (calibration harness).
#[derive(Clone, Copy, Debug)]
pub struct ModelParams {
    /// Cruise mode: OU mean-reversion time (s) of omega.
    pub cruise_tau_s: f64,
    /// Cruise mode: stationary std-dev (rad/s) of omega ("agility"). Each
    /// particle draws its own value log-uniformly in [min, max] and keeps it,
    /// so the filter learns how much this UAV wanders: a steady flyer gets a
    /// sharp fan, a weaving one a wide fan. min == max gives a fixed value.
    pub cruise_omega_sigma_min: f64,
    pub cruise_omega_sigma_max: f64,
    /// Turn mode: random-walk intensity of the held omega (rad/s per sqrt(s)).
    pub turn_omega_walk: f64,
    /// Turn mode: minimum |omega| (rad/s). A "turn" that does not turn would
    /// be a second, goal-blind way to fly straight; straight flight belongs to
    /// Cruise, where the goal bias applies. 0.1 rad/s is a radius of 80 m at
    /// 8 m/s.
    pub omega_turn_min: f64,
    /// Turn mode: std-dev of the turn-rate jump when a turn starts (rad/s).
    pub turn_entry_sigma: f64,
    /// Total mode-switching rate (1/s). Split so that, without turn
    /// persistence, the stationary P(Turn) = p_turn.
    pub mode_switch_rate: f64,
    /// Turn persistence (s): the hazard of a turn ending decays with its age
    /// as `t_p / (t_p + age)`, so a loiter that has lasted a while is expected
    /// to go on, while a fresh turn may still end soon. `f64::INFINITY` gives a
    /// constant hazard.
    pub turn_persistence_s: f64,
    /// Turn reversal / re-turn rate (1/s): while turning, the turn rate may
    /// jump to a freshly drawn value (often the other way), starting a new
    /// turn. S-turns and waypoint changes need it; without it a "no goal"
    /// particle cannot follow a reversal and a goal hypothesis whose steering
    /// happens to point the new way wins by accident. Like the end of a turn,
    /// it fades with the turn's age (turn_persistence_s), so a long loiter is
    /// not expected to reverse.
    pub turn_change_rate: f64,
    /// Goal-switching hazard (1/s): the goal is redrawn from the prior, so a
    /// retargeting UAV can be followed and the posterior relaxes toward the
    /// prior.
    pub goal_switch_rate: f64,
    /// Cruise mode: goal-steering gain (1/s); heading closes on the goal
    /// bearing with a time constant of ~1/k_goal.
    pub k_goal: f64,
    /// Cruise mode without a goal: gain (1/s) pulling the heading back onto
    /// the particle's course, so cruise means course-keeping flight. A weave
    /// then predicts back toward its mean course, and sustained circling can
    /// only be explained by the Turn mode.
    pub k_course: f64,
    /// Prior share of "weaver" particles. A weaver's cruise heading follows a
    /// bounded sinusoid about its course (or goal bearing),
    /// `heading - target = amp * sin(phase)`, so a weaving UAV is tracked and
    /// predicted to keep weaving about its mean course, never to loop. The
    /// filter learns which kind fits, and the weave's frequency, amplitude
    /// and phase.
    pub p_weave: f64,
    /// Weaver angular frequency (rad/s) of the heading swing, drawn
    /// log-uniformly per particle in [min, max]. A sine of wavelength L flown
    /// at baseline speed u swings at 2*pi*u/L (100 m at 9 m/s: 0.57 rad/s).
    pub weave_freq_min: f64,
    pub weave_freq_max: f64,
    /// Largest weave amplitude (rad): the heading's swing either side of the
    /// course.
    pub weave_amp_max: f64,
    /// Weave phase jitter (rad per sqrt(s)) and amplitude random walk (rad per
    /// sqrt(s)): how fast a weave may drift out of rhythm or change size.
    pub weave_phase_noise: f64,
    pub weave_amp_noise: f64,
    /// Steady <-> weaver switching rate (1/s), split so the stationary weaver
    /// share is p_weave. A new weaver is born in step with the current heading
    /// error, so weavers keep appearing while the track weaves.
    pub weave_switch_rate: f64,
    /// Newborn weaver: amplitude (rad) drawn uniformly between the current
    /// heading error and this (capped by weave_amp_max). A small value makes
    /// a weave start gently and grow through `weave_amp_noise` only if the
    /// track keeps weaving.
    pub weave_birth_amp: f64,
    /// Restlessness: each particle draws a multiplier on its manoeuvre-onset
    /// rates (cruise -> turn, steady -> weaver), log-uniform in [min, max],
    /// and keeps it. The filter learns it from the track: a long steady track
    /// weeds out restless particles, so its fan narrows, while a track that
    /// keeps manoeuvring keeps them. min == max == 1 disables it.
    pub restless_min: f64,
    pub restless_max: f64,
    /// Cruise mode: random-walk intensity of the course (rad per sqrt(s)), so
    /// a slow course change is still cruise.
    pub course_walk: f64,
    /// Turn mode: drift speed of a goal particle's loop toward the goal, as a
    /// fraction of its airspeed (0.08 -> 0.64 m/s at 8 m/s).
    pub goal_drift_frac: f64,
    /// Turn rate (rad/s) above which a turn counts as a full loop for the drift.
    pub omega_loop: f64,
    /// Speed-noise floor (m/s per sqrt(s)) during filtering, so resampled
    /// copies of a particle can diversify in speed even when
    /// `speed_noise_mps` is 0.
    pub filter_speed_noise_floor: f64,
    /// Minimum share of a bank's particles kept in each mode at resampling, so
    /// a mode that is locally worse (e.g. Turn during half a loop) is not
    /// wiped out and can win back when the data favour it again.
    pub min_mode_share: f64,
    /// Measurement-noise floor (m^2) added to the observation covariance.
    pub meas_floor_m2: f64,
    /// Track-loss threshold: if no particle in any bank is within this
    /// squared Mahalanobis distance of an observation (9 = 3 sigma), the
    /// target has out-manoeuvred the model (e.g. it breaks the turn-table
    /// limits) and the kinematics are re-seeded from the latest observations.
    pub recovery_d2: f64,
    /// Course memory (s): a cruise particle's course relaxes toward its
    /// current heading with this time constant, so a heading held for a while
    /// becomes the new course and old manoeuvres are forgotten. Without it, a
    /// "no goal" particle that turned in cruise keeps being pulled back to the
    /// course it had before, fits a new straight line badly, and the goal
    /// hypotheses win by default. `f64::INFINITY` keeps the course forever.
    pub course_tau_s: f64,
    /// Outlier-tolerant measurement model: each observation is Gaussian
    /// (its covariance) with probability 1 - outlier_prob, otherwise from a
    /// wide Gaussian with outlier_sigma_m added in quadrature. Bounds how
    /// much a single awkward step can count against a hypothesis.
    pub outlier_prob: f64,
    pub outlier_sigma_m: f64,
    /// Common random numbers: every goal sub-filter starts from the same
    /// particles, draws the same random numbers and resamples at the same
    /// steps, so the difference between their likelihoods reflects the goal
    /// and not Monte-Carlo luck (P(goal) is far less noisy).
    pub common_random_numbers: bool,
    /// Tempering of the goal evidence (0..1]: each step's likelihood ratio
    /// between goal hypotheses is raised to this power before P(goal) is
    /// updated (a "power posterior"). The motion model is only an
    /// approximation, so its likelihoods are over-sharp; tempering keeps
    /// P(goal) from swinging to 0 or 1 on a few lucky or unlucky steps. The
    /// particles inside each hypothesis are weighted normally. 1 = plain Bayes.
    pub goal_evidence_power: f64,
    /// Two-stage intent inference. One goal-free filter (all particles)
    /// tracks the UAV; P(goal) is then scored once per observation from that
    /// filter's estimates instead of from one particle filter per goal:
    ///   * cruise: did the heading change as steering toward the OOI predicts
    ///     (k_goal * bearing error), or as course-keeping predicts?
    ///   * turn: does the loop centre drift toward the OOI
    ///     (goal_drift_frac * v), or stay put?
    /// The two are mixed by the filter's P(turn), so a sharp manoeuvre leaves
    /// the steering evidence uninformative instead of deciding the goal.
    /// Far less Monte-Carlo noise than per-goal filters, at the cost of
    /// scoring point estimates rather than whole particle paths.
    pub intent_two_stage: bool,
    /// Hybrid: keep one particle filter per goal for tracking and for the
    /// prediction fan (goal-consistent states), but take P(goal) from the
    /// two-stage score computed on the "no goal" filter (stable), and mix the
    /// fan by it. Ignored when `intent_two_stage` is set.
    pub intent_hybrid: bool,
    /// Two-stage: intent is scored over non-overlapping windows of this
    /// length (s). Per-second heading changes are dominated by estimation
    /// noise; over a few seconds a steering turn accumulates well above it.
    pub intent_window_s: f64,
    /// Two-stage: std-dev (rad) of the heading change over one window around
    /// the steering prediction (estimation noise at both ends plus wander).
    pub intent_sigma_heading: f64,
    /// Two-stage: std-dev (m/s) of the measured loop-centre drift velocity.
    pub intent_sigma_drift: f64,
    /// Two-stage: share of each intent likelihood that is uninformative
    /// (outlier tolerance).
    pub intent_outlier: f64,
    /// Two-stage: tempering of the per-observation intent evidence (the
    /// features are correlated between observations).
    pub intent_power: f64,
    /// Two-stage: degrees of freedom of the Student-t score for both
    /// features. Heavy tails make one badly estimated heading (e.g. mid-
    /// manoeuvre) count far less than a Gaussian would; `f64::INFINITY` is
    /// Gaussian.
    pub intent_dof: f64,
}

impl ModelParams {
    /// Draw a particle's agility log-uniformly in
    /// [cruise_omega_sigma_min, cruise_omega_sigma_max].
    /// Draw a particle's weave frequency: 0 (steady flyer) with probability
    /// 1 - p_weave, else a weaver frequency.
    pub fn sample_weave_freq(&self, rng: &mut Xoshiro256ss) -> f64 {
        if rng.next_f64() >= self.p_weave {
            0.0
        } else {
            self.sample_weave_freq_given_weaver(rng)
        }
    }

    /// A weaver's frequency, log-uniform in [weave_freq_min, weave_freq_max]
    /// (0 when weaving is disabled).
    pub fn sample_weave_freq_given_weaver(&self, rng: &mut Xoshiro256ss) -> f64 {
        let (lo, hi) = (self.weave_freq_min.max(1e-3), self.weave_freq_max);
        if hi <= 0.0 {
            0.0
        } else if hi <= lo {
            hi
        } else {
            lo * (hi / lo).powf(rng.next_f64())
        }
    }

    /// Set a parameter by field name (booleans: 0 = false, anything else =
    /// true). Used by the calibration harness and the flight replay to sweep
    /// or override parameters from the command line.
    pub fn set(&mut self, key: &str, v: f64) -> Result<(), String> {
        match key {
            "cruise_tau_s" => self.cruise_tau_s = v,
            "cruise_omega_sigma" => {
                self.cruise_omega_sigma_min = v;
                self.cruise_omega_sigma_max = v;
            }
            "cruise_omega_sigma_min" => self.cruise_omega_sigma_min = v,
            "cruise_omega_sigma_max" => self.cruise_omega_sigma_max = v,
            "turn_omega_walk" => self.turn_omega_walk = v,
            "omega_turn_min" => self.omega_turn_min = v,
            "turn_entry_sigma" => self.turn_entry_sigma = v,
            "mode_switch_rate" => self.mode_switch_rate = v,
            "turn_persistence_s" => self.turn_persistence_s = v,
            "turn_change_rate" => self.turn_change_rate = v,
            "goal_evidence_power" => self.goal_evidence_power = v,
            "intent_two_stage" => self.intent_two_stage = v != 0.0,
            "intent_hybrid" => self.intent_hybrid = v != 0.0,
            "intent_sigma_heading" => self.intent_sigma_heading = v,
            "intent_sigma_drift" => self.intent_sigma_drift = v,
            "intent_outlier" => self.intent_outlier = v,
            "intent_window_s" => self.intent_window_s = v,
            "intent_power" => self.intent_power = v,
            "intent_dof" => self.intent_dof = v,
            "goal_switch_rate" => self.goal_switch_rate = v,
            "k_goal" => self.k_goal = v,
            "k_course" => self.k_course = v,
            "course_walk" => self.course_walk = v,
            "p_weave" => self.p_weave = v,
            "weave_amp_max" => self.weave_amp_max = v,
            "weave_switch_rate" => self.weave_switch_rate = v,
            "weave_birth_amp" => self.weave_birth_amp = v,
            "restless_min" => self.restless_min = v,
            "restless_max" => self.restless_max = v,
            "weave_phase_noise" => self.weave_phase_noise = v,
            "weave_amp_noise" => self.weave_amp_noise = v,
            "weave_freq_min" => self.weave_freq_min = v,
            "weave_freq_max" => self.weave_freq_max = v,
            "recovery_d2" => self.recovery_d2 = v,
            "course_tau_s" => self.course_tau_s = v,
            "outlier_prob" => self.outlier_prob = v,
            "outlier_sigma_m" => self.outlier_sigma_m = v,
            "common_random_numbers" => self.common_random_numbers = v != 0.0,
            "goal_drift_frac" => self.goal_drift_frac = v,
            "omega_loop" => self.omega_loop = v,
            "filter_speed_noise_floor" => self.filter_speed_noise_floor = v,
            "min_mode_share" => self.min_mode_share = v,
            "meas_floor_m2" => self.meas_floor_m2 = v,
            
            _ => return Err(format!("unknown ModelParams field: {key}")),
        }
        Ok(())
    }

    /// Draw a particle's restlessness multiplier (see `restless_min`).
    pub fn sample_restlessness(&self, rng: &mut Xoshiro256ss) -> f64 {
        let (lo, hi) = (self.restless_min.max(1e-6), self.restless_max);
        if hi <= lo {
            lo
        } else {
            lo * (hi / lo).powf(rng.next_f64())
        }
    }

    pub fn sample_agility(&self, rng: &mut Xoshiro256ss) -> f64 {
        let (lo, hi) = (self.cruise_omega_sigma_min, self.cruise_omega_sigma_max);
        if hi <= lo {
            return lo;
        }
        lo * (hi / lo).powf(rng.next_f64())
    }
}

/// Defaults calibrated with `examples/calibrate.rs` (see its docs for the
/// scenario suite and the per-family coverage they achieve).
impl Default for ModelParams {
    fn default() -> Self {
        ModelParams {
            cruise_tau_s: 4.0,
            cruise_omega_sigma_min: 0.01,
            cruise_omega_sigma_max: 0.15,
            turn_omega_walk: 0.03,
            omega_turn_min: 0.1,
            turn_entry_sigma: 0.45,
            mode_switch_rate: 0.1,
            turn_persistence_s: 5.0,
            turn_change_rate: 0.1,
            goal_switch_rate: 0.02,
            k_goal: 0.1,
            k_course: 0.2,
            course_walk: 0.03,
            p_weave: 0.3,
            weave_freq_min: 0.2,
            weave_freq_max: 1.5,
            weave_amp_max: 1.2,
            weave_phase_noise: 0.3,
            weave_amp_noise: 0.1,
            weave_switch_rate: 0.1,
            weave_birth_amp: 1.2,
            restless_min: 1.0,
            restless_max: 1.0,
            goal_drift_frac: 0.08,
            omega_loop: 0.2,
            filter_speed_noise_floor: 0.1,
            min_mode_share: 0.1,
            meas_floor_m2: 0.1,
            recovery_d2: 9.0,
            course_tau_s: 6.0,
            outlier_prob: 0.05,
            outlier_sigma_m: 10.0,
            common_random_numbers: true,
            goal_evidence_power: 1.0,
            intent_two_stage: false,
            intent_hybrid: true,
            intent_window_s: 5.0,
            intent_sigma_heading: 0.2,
            intent_sigma_drift: 0.4,
            intent_outlier: 0.1,
            intent_power: 1.0,
            intent_dof: 2.0,
        }
    }
}

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
pub struct GoalPrior {
    pub n_oois: usize,
    pub p_goal: f64,
}

impl GoalPrior {
    pub fn sample(&self, rng: &mut Xoshiro256ss) -> Option<usize> {
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
    /// Cruise-mode turn-rate std-dev of this particle (rad/s); see
    /// `ModelParams::cruise_omega_sigma_min`.
    pub agility: f64,
    /// Cruise mode: the wandering part of the turn rate (rad/s), an OU process
    /// around 0. The applied rate is `wander + goal steering`, so the goal
    /// closes the heading error first-order (time constant 1/k_goal) instead
    /// of through the OU lag, which would make the steering oscillate.
    pub wander: f64,
    /// Cruise mode: heading minus course (rad), NOT wrapped: it accumulates
    /// the total turn since the course was set, so a cruise particle that
    /// circles feels an ever stronger pull back instead of one that flips
    /// sign every half loop. Reset to 0 when a turn ends.
    pub course_err: f64,
    /// Turn mode: time (s) since the current turn started.
    pub turn_age: f64,
    /// Cruise mode: 0 for a steady flyer, else the angular frequency (rad/s)
    /// of this weaver's heading swing; see `ModelParams::p_weave`.
    pub weave_freq: f64,
    /// Weaver: amplitude (rad) and phase (rad) of the heading swing.
    pub weave_amp: f64,
    pub weave_phase: f64,
    /// Multiplier on this particle's manoeuvre-onset rates; see
    /// `ModelParams::restless_min`.
    pub restlessness: f64,
    /// Normalised log weight (the filter's working representation).
    pub log_w: f64,
    /// Linear copy of `exp(log_w)`, used for N_eff, resampling and summaries.
    pub weight: f64,
}

impl Particle {
    /// Propagate the particle over `dt` seconds with the goal-conditioned,
    /// mode-switching dynamics (see the module docstring). Used unchanged in
    /// the filter and in prediction.
    pub fn step(
        &mut self,
        mp: &ModelParams,
        dt: f64,
        speed_noise: f64,
        p_turn: f64,
        oois: &[Ooi],
        prior: &GoalPrior,
        rng: &mut Xoshiro256ss,
        ttc: &Ttc,
    ) {
        // 1. Goal switching (redraw from the prior).
        if prior.n_oois > 0 && rng.next_f64() < 1.0 - (-mp.goal_switch_rate * dt).exp() {
            self.goal = prior.sample(rng);
        }

        // 2. Mode switching. A new turn perturbs the current turn rate (rather
        //    than redrawing it) so a particle that was already curving keeps
        //    its curvature when it is reclassified as turning.
        let rate = match self.mode {
            Mode::Cruise => mp.mode_switch_rate * p_turn * self.restlessness,
            Mode::Turn => {
                let persist = if mp.turn_persistence_s.is_finite() {
                    mp.turn_persistence_s / (mp.turn_persistence_s + self.turn_age)
                } else {
                    1.0
                };
                mp.mode_switch_rate * (1.0 - p_turn) * persist
            }
        };
        if self.mode == Mode::Turn {
            let persist = if mp.turn_persistence_s.is_finite() {
                mp.turn_persistence_s / (mp.turn_persistence_s + self.turn_age)
            } else {
                1.0
            };
            self.turn_age += dt;
            if rng.next_f64() < 1.0 - (-mp.turn_change_rate * self.restlessness * persist * dt).exp() {
                self.omega = rng.gauss(0.0, mp.turn_entry_sigma);
                self.turn_age = 0.0;
            }
        }
        if rng.next_f64() < 1.0 - (-rate * dt).exp() {
            self.mode = match self.mode {
                Mode::Cruise => {
                    self.omega += rng.gauss(0.0, mp.turn_entry_sigma);
                    self.turn_age = 0.0;
                    Mode::Turn
                }
                Mode::Turn => {
                    // The turn rate becomes cruise wander and decays with
                    // cruise_tau_s: a released turn eases out.
                    self.wander = self.omega;
                    self.course_err = 0.0;
                    // A weave resumes from the new course, mid-swing at 0.
                    self.weave_phase = 0.0;
                    Mode::Cruise
                }
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
                if mp.course_tau_s.is_finite() {
                    self.course_err *= (-dt / mp.course_tau_s.max(1e-6)).exp();
                }
                self.course_err += rng.gauss(0.0, mp.course_walk) * dt.sqrt();
                // Heading error to the target direction (goal bearing, else
                // the course; the course error is unwrapped, see `course_err`).
                let (err, k) = match bearing {
                    Some(b) => (-wrap(b - self.psi), mp.k_goal),
                    None => (self.course_err, mp.k_course),
                };
                // Steady <-> weaver switching. A newborn weaver continues the
                // current heading: amplitude >= |err|, phase with
                // amp * sin(phase) = err, swinging either way.
                let weaving = self.weave_freq > 0.0;
                let switch_rate = mp.weave_switch_rate
                    * if weaving { 1.0 - mp.p_weave } else { mp.p_weave * self.restlessness };
                if rng.next_f64() < 1.0 - (-switch_rate * dt).exp() {
                    if weaving {
                        self.weave_freq = 0.0;
                        self.wander = self.omega;
                    } else {
                        self.weave_freq = mp.sample_weave_freq_given_weaver(rng);
                        let e = err.clamp(-mp.weave_amp_max, mp.weave_amp_max);
                        let top = mp.weave_birth_amp.min(mp.weave_amp_max).max(e.abs());
                        self.weave_amp = e.abs() + (top - e.abs()) * rng.next_f64();
                        let base = (e / self.weave_amp.max(1e-9)).clamp(-1.0, 1.0).asin();
                        self.weave_phase = if rng.next_f64() < 0.5 { base } else { PI - base };
                    }
                }
                if self.weave_freq > 0.0 {
                    // Weaver: steer the heading error onto the swing
                    // amp * sin(phase); the rhythm and size jitter slowly. The
                    // turn-rate limit below may make it lag a sharp swing.
                    self.weave_phase += self.weave_freq * dt
                        + rng.gauss(0.0, mp.weave_phase_noise) * dt.sqrt();
                    self.weave_amp = (self.weave_amp + rng.gauss(0.0, mp.weave_amp_noise) * dt.sqrt())
                        .clamp(0.0, mp.weave_amp_max);
                    let desired = self.weave_amp * self.weave_phase.sin();
                    self.omega = (desired - err) / dt;
                    self.wander = self.omega;
                } else {
                    // Steady flyer: OU wander plus a first-order pull onto the
                    // target with gain k.
                    let mu = (-k * err).clamp(-limit, limit);
                    let alpha = (-dt / mp.cruise_tau_s).exp();
                    self.wander = alpha * self.wander
                        + (1.0 - alpha * alpha).sqrt() * self.agility * rng.gauss(0.0, 1.0);
                    self.omega = self.wander + mu;
                }
            }
            Mode::Turn => {
                self.omega += rng.gauss(0.0, mp.turn_omega_walk) * dt.sqrt();
                if self.omega.abs() < mp.omega_turn_min {
                    let sign = if self.omega < 0.0 { -1.0 } else { 1.0 };
                    self.omega = sign * mp.omega_turn_min;
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
            let looping = (self.omega.abs() / mp.omega_loop).min(1.0);
            let drift = mp.goal_drift_frac * looping * self.v * dt;
            self.x += drift * b.cos();
            self.y += drift * b.sin();
        }
        self.psi = (self.psi + self.omega * dt).rem_euclid(2.0 * PI);
        self.course_err += self.omega * dt;
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
/// its weight share of the set, but at least `min_mode_share` of the set while
/// the class has any particles. Every copy then carries weight `W_m / n_m`, so
/// the weighted mixture (and each class's total weight) is unchanged; only the
/// particle budget per class is protected.
fn resample_by_mode(particles: &mut Vec<Particle>, mp: &ModelParams, rng: &mut Xoshiro256ss) {
    let n = particles.len();
    let (turn, cruise): (Vec<Particle>, Vec<Particle>) =
        particles.iter().partition(|p| p.mode == Mode::Turn);
    let w_turn: f64 = turn.iter().map(|p| p.weight).sum();
    let w_cruise: f64 = cruise.iter().map(|p| p.weight).sum();
    if turn.is_empty() || cruise.is_empty() || w_turn <= 0.0 || w_cruise <= 0.0 {
        resample(particles, rng);
        return;
    }
    let n_min = ((mp.min_mode_share * n as f64).ceil() as usize).max(1);
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

/// Kinematic prior from two consecutive observations: heading and speed of
/// the displacement, with spreads that reflect the position noise.
struct KinPrior {
    psi: f64,
    sigma_psi: f64,
    v: f64,
    sigma_v: f64,
}

fn kin_prior(o0: &Observation, o1: &Observation, mp: &ModelParams) -> KinPrior {
    let dx = o1.pos[0] - o0.pos[0];
    let dy = o1.pos[1] - o0.pos[1];
    let dt = (o1.t - o0.t).max(1e-3);
    let dist = (dx * dx + dy * dy).sqrt();
    let sigma_pos = (0.5 * (o0.cov[0] + o0.cov[2]) + mp.meas_floor_m2).sqrt();
    KinPrior {
        psi: dy.atan2(dx),
        sigma_psi: if dist > 1e-6 {
            (std::f64::consts::SQRT_2 * sigma_pos / dist + 0.1).min(PI)
        } else {
            PI
        },
        v: dist / dt,
        sigma_v: std::f64::consts::SQRT_2 * sigma_pos / dt + 0.5,
    }
}

/// Track-loss recovery: put every particle back on the latest observation
/// with heading and speed from the last displacement. Mode, goal, agility,
/// turn rate and the absolute course are kept, so the hypotheses survive and
/// a cruise particle still steers back to the course it held before (a zigzag
/// corner does not erase the mean direction); weights reset to uniform.
fn reseed(particles: &mut [Particle], o0: &Observation, o1: &Observation, mp: &ModelParams, rng: &mut Xoshiro256ss, ttc: &Ttc) {
    let k = kin_prior(o0, o1, mp);
    let cov_factor = Mat2::from_cov(o1.cov);
    let n = particles.len() as f64;
    for p in particles.iter_mut() {
        let noise = cov_factor.sample(rng);
        p.x = o1.pos[0] + noise.x;
        p.y = o1.pos[1] + noise.y;
        let course = p.psi - p.course_err;
        p.psi = (k.psi + rng.gauss(0.0, k.sigma_psi)).rem_euclid(2.0 * PI);
        p.v = rng.gauss(k.v, k.sigma_v).clamp(ttc.v_min, ttc.v_max);
        p.course_err = wrap(p.psi - course);
        p.log_w = -n.ln();
        p.weight = 1.0 / n;
    }
}

/// Number of integration sub-steps for an interval of `dt` seconds.
fn substeps(dt: f64) -> (usize, f64) {
    let n = (dt / SUBSTEP_S).ceil().max(1.0) as usize;
    (n, dt / n as f64)
}

/// Weighted summary of a goal-free particle set at one observation, used by
/// the two-stage intent scoring.
struct Summary {
    t: f64,
    pos: [f64; 2],
    psi: f64,
    omega: f64,
    v: f64,
    turn: f64,
}

impl Summary {
    /// Centre of the current loop (position + radius to the left of the
    /// heading for a CCW turn), or `None` when the turn is too gentle for the
    /// centre to be meaningful.
    fn centre(&self, mp: &ModelParams) -> Option<[f64; 2]> {
        if self.omega.abs() < 0.5 * mp.omega_loop || self.turn < 0.5 {
            return None;
        }
        let r = self.v / self.omega;
        Some([self.pos[0] - r * self.psi.sin(), self.pos[1] + r * self.psi.cos()])
    }
}

fn summarise(particles: &[Particle], t: f64) -> Summary {
    let (mut x, mut y, mut s, mut c, mut w_om, mut v, mut turn) =
        (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    for p in particles {
        let w = p.weight;
        x += w * p.x;
        y += w * p.y;
        s += w * p.psi.sin();
        c += w * p.psi.cos();
        w_om += w * p.omega;
        v += w * p.v;
        if p.mode == Mode::Turn {
            turn += w;
        }
    }
    Summary {
        t,
        pos: [x, y],
        psi: s.atan2(c),
        omega: w_om,
        v,
        turn,
    }
}

/// Density of a standardised residual `r` (radius for `dim` = 2) under an
/// isotropic Student-t with `dof` degrees of freedom; Gaussian for infinite
/// `dof`. Normalised over the line (dim 1) or the plane (dim 2).
fn student_t(r: f64, dof: f64, dim: u32) -> f64 {
    let r2 = r * r;
    if !dof.is_finite() {
        return if dim == 1 {
            (-0.5 * r2).exp() / (2.0 * PI).sqrt()
        } else {
            (-0.5 * r2).exp() / (2.0 * PI)
        };
    }
    let nu = dof.max(0.1);
    if dim == 1 {
        // Gamma((nu+1)/2) / (sqrt(nu pi) Gamma(nu/2)) * (1 + r^2/nu)^(-(nu+1)/2)
        let c = (ln_gamma(0.5 * (nu + 1.0)) - ln_gamma(0.5 * nu)).exp() / (nu * PI).sqrt();
        c * (1.0 + r2 / nu).powf(-0.5 * (nu + 1.0))
    } else {
        // Bivariate: 1 / (2 pi) * (1 + r^2/nu)^(-(nu+2)/2)
        (1.0 + r2 / nu).powf(-0.5 * (nu + 2.0)) / (2.0 * PI)
    }
}

/// ln Gamma(x) for x > 0 (Lanczos approximation, g = 7).
fn ln_gamma(x: f64) -> f64 {
    const C: [f64; 9] = [
        0.999_999_999_999_809_9, 676.520_368_121_885_1, -1_259.139_216_722_402_8,
        771.323_428_777_653_1, -176.615_029_162_140_6, 12.507_343_278_686_905,
        -0.138_571_095_265_720_12, 9.984_369_578_019_572e-6, 1.505_632_735_149_311_6e-7,
    ];
    if x < 0.5 {
        return (PI / (PI * x).sin()).ln() - ln_gamma(1.0 - x);
    }
    let x = x - 1.0;
    let mut a = C[0];
    let t = x + 7.5;
    for (i, &c) in C.iter().enumerate().skip(1) {
        a += c / (x + i as f64);
    }
    0.5 * (2.0 * PI).ln() + (x + 0.5) * t.ln() - t + a.ln()
}

/// Log-likelihood of one intent window (filter summaries from its start to
/// its end) under one hypothesis (`goal`), mixing a cruise (steering) and a
/// turn (loop-drift) term by the window's mean P(turn). Each term keeps an
/// `intent_outlier` share uninformative, and the "other" feature of each term
/// is uninformative too, so both are densities over the same space.
fn intent_log_lik(
    win: &[Summary],
    goal: Option<usize>,
    oois: &[Ooi],
    mp: &ModelParams,
    ttc: &Ttc,
) -> f64 {
    let (a, b) = (&win[0], &win[win.len() - 1]);
    let span = (b.t - a.t).max(1e-6);
    let eps = mp.intent_outlier.clamp(0.0, 1.0);
    let limit_of = |v: f64| (ttc.a_lat / v.max(0.1)).min(ttc.omega_cap);
    // Uniform densities of the two features over their plausible ranges.
    let u_psi = 1.0 / (2.0 * limit_of(a.v) * span).clamp(1e-3, 2.0 * PI);
    let vmax = ttc.v_max.max(1.0);
    let u_drift = 1.0 / (4.0 * vmax * vmax);

    // Cruise: heading change over the window vs the steering each hypothesis
    // predicts, summed over the window's steps. "No goal" means "not steered
    // toward any OOI": no systematic change. A goal is scored averaged over
    // steering gains 0.5x, 1x and 2x k_goal (UAVs differ in how hard they
    // steer toward a target).
    let sh = mp.intent_sigma_heading.max(1e-6);
    let dpsi = wrap(b.psi - a.psi);
    let score = |mean: f64| student_t((dpsi - mean) / sh, mp.intent_dof, 1) / sh;
    let steer_density = match goal {
        Some(g) => {
            let predicted = |f: f64| -> f64 {
                win.windows(2)
                    .map(|w| {
                        let bearing = (oois[g].pos[1] - w[0].pos[1]).atan2(oois[g].pos[0] - w[0].pos[0]);
                        let lim = limit_of(w[0].v);
                        (f * mp.k_goal * wrap(bearing - w[0].psi)).clamp(-lim, lim) * (w[1].t - w[0].t)
                    })
                    .sum()
            };
            let gains = [0.5, 1.0, 2.0];
            gains.iter().map(|&f| score(predicted(f))).sum::<f64>() / gains.len() as f64
        }
        None => score(0.0),
    };
    let steer = (1.0 - eps) * steer_density + eps * u_psi;

    // Turn: loop-centre drift velocity over the window.
    let drift = match (a.centre(mp), b.centre(mp)) {
        (Some(c0), Some(c1)) => {
            let u = [(c1[0] - c0[0]) / span, (c1[1] - c0[1]) / span];
            let expect = match goal {
                Some(g) => {
                    let bearing = (oois[g].pos[1] - c1[1]).atan2(oois[g].pos[0] - c1[0]);
                    let looping = (b.omega.abs() / mp.omega_loop).min(1.0);
                    let sp = mp.goal_drift_frac * looping * b.v;
                    [sp * bearing.cos(), sp * bearing.sin()]
                }
                None => [0.0, 0.0],
            };
            let sd = mp.intent_sigma_drift.max(1e-6);
            let d2 = ((u[0] - expect[0]).powi(2) + (u[1] - expect[1]).powi(2)) / (sd * sd);
            (1.0 - eps) * student_t(d2.sqrt(), mp.intent_dof, 2) / (sd * sd) + eps * u_drift
        }
        _ => u_drift,
    };

    let m = (win.iter().map(|s| s.turn).sum::<f64>() / win.len() as f64).clamp(0.0, 1.0);
    ((1.0 - m) * steer * u_drift + m * u_psi * drift).max(1e-300).ln()
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
    predict_with(track, ttc, oois, cfg, &ModelParams::default())
}

/// `predict()` with explicit model parameters (used by the calibration
/// harness to sweep them).
pub fn predict_with(
    track: &[Observation],
    ttc: &Ttc,
    oois: &[Ooi],
    cfg: &PredictConfig,
    mp: &ModelParams,
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
    // Two-stage: the filter tracks with one goal-free bank; `hypotheses` is
    // kept for scoring intent from that bank's estimates.
    let filter_hyps: Vec<(Option<usize>, f64)> =
        if mp.intent_two_stage { vec![(None, 1.0)] } else { hypotheses.clone() };
    let per_bank = (n_samples / filter_hyps.len()).max(2);

    // Heading and speed come from the first displacement, with a spread that
    // reflects the position noise of the two observations. The goal does NOT
    // change the initial physical state.
    let cov_factor = Mat2::from_cov(track[0].cov);
    let k0 = kin_prior(&track[0], &track[1], mp);

    // With common random numbers every bank starts from the same particles
    // (only the goal label differs) and gets an identical random stream;
    // otherwise each bank draws its own particles and stream.
    let make_particles = |rng: &mut Xoshiro256ss, goal: Option<usize>| -> Vec<Particle> {
        let mut particles = Vec::with_capacity(per_bank);
        for _ in 0..per_bank {
            let noise = cov_factor.sample(rng);
            let mode = if rng.next_f64() < p_turn { Mode::Turn } else { Mode::Cruise };
            let agility = mp.sample_agility(rng);
            let omega = match mode {
                Mode::Turn => rng.gauss(0.0, mp.turn_entry_sigma),
                Mode::Cruise => rng.gauss(0.0, agility),
            };
            let psi = (k0.psi + rng.gauss(0.0, k0.sigma_psi)).rem_euclid(2.0 * PI);
            particles.push(Particle {
                x: track[0].pos[0] + noise.x,
                y: track[0].pos[1] + noise.y,
                v: rng.gauss(k0.v, k0.sigma_v).clamp(ttc.v_min, ttc.v_max),
                psi,
                omega,
                mode,
                goal,
                agility,
                wander: omega,
                course_err: 0.0,
                turn_age: 0.0,
                weave_freq: mp.sample_weave_freq(rng),
                weave_amp: mp.weave_amp_max * rng.next_f64(),
                weave_phase: 2.0 * PI * rng.next_f64(),
                restlessness: mp.sample_restlessness(rng),
                log_w: -(per_bank as f64).ln(),
                weight: 1.0 / per_bank as f64,
            });
        }
        particles
    };
    let mut banks: Vec<Bank> = Vec::with_capacity(filter_hyps.len());
    let base = make_particles(&mut rng, None);
    for &(goal, pr) in filter_hyps.iter() {
        let particles = if mp.common_random_numbers {
            base.iter().map(|p| Particle { goal, ..*p }).collect()
        } else {
            make_particles(&mut rng, goal)
        };
        banks.push(Bank { goal, prior: pr, prob: pr, particles });
    }
    let filter_seed = cfg.seed ^ 0x9E37_79B9_7F4A_7C15;
    let mut bank_rngs: Vec<Xoshiro256ss> = (0..banks.len())
        .map(|b| {
            let k = if mp.common_random_numbers { 0 } else { b as u64 + 1 };
            Xoshiro256ss::from_seed(filter_seed.wrapping_add(k.wrapping_mul(0xD1B5_4A32_D192_ED03)))
        })
        .collect();

    // ---- 2. FILTERING LOOP (track[1..]) ------------------------------------
    // Each bank is an ordinary particle filter under its own goal-conditioned
    // dynamics. Its predictive likelihood of each observation updates the bank
    // probability, P(g) <- P_pred(g) * L_g / Z, where P_pred mixes toward the
    // prior with the goal-switching hazard (so a retargeting UAV is followed).
    let filter_speed_noise = cfg.speed_noise_mps.max(mp.filter_speed_noise_floor);
    let mut recoveries = 0u32;
    // Two-stage intent state: P(hypothesis), the previous filter summary and
    // the history of loop-centre estimates.
    let mut intent_probs: Vec<f64> = hypotheses.iter().map(|h| h.1).collect();
    let mut summaries: Vec<Summary> = Vec::new();
    let mut win_start = 0usize; // index into `summaries` of the open window
    for i in 1..track.len() {
        let obs = &track[i];
        let dt = obs.t - track[i - 1].t;
        if dt <= 0.0 {
            continue;
        }
        let (n_sub, h) = substeps(dt);

        // Gaussian position likelihood with a noise floor. The normalising
        // constant is the same for every particle and bank, so it is dropped.
        let a = obs.cov[0] + mp.meas_floor_m2;
        let b = obs.cov[1];
        let c = obs.cov[2] + mp.meas_floor_m2;
        let det = (a * c - b * b).max(1e-9);

        // Outlier component: the same, widened by outlier_sigma_m. The log
        // likelihood is the log of the two-component mixture (the 2*pi is
        // common to every particle and bank and is dropped).
        let so2 = mp.outlier_sigma_m * mp.outlier_sigma_m;
        let (ao, co) = (a + so2, c + so2);
        let det_o = (ao * co - b * b).max(1e-9);
        let eps = mp.outlier_prob.clamp(0.0, 0.999);
        let ln_in = (1.0 - eps).ln() - 0.5 * det.ln();
        let ln_out = if eps > 0.0 { eps.ln() - 0.5 * det_o.ln() } else { f64::NEG_INFINITY };

        let mut min_d2 = f64::INFINITY;
        for (bank, brng) in banks.iter_mut().zip(bank_rngs.iter_mut()) {
            for p in bank.particles.iter_mut() {
                for _ in 0..n_sub {
                    p.step(mp, h, filter_speed_noise, p_turn, oois, &no_switch, brng, ttc);
                }
                let ex = p.x - obs.pos[0];
                let ey = p.y - obs.pos[1];
                let d2 = (c * ex * ex - 2.0 * b * ex * ey + a * ey * ey) / det;
                let d2o = (co * ex * ex - 2.0 * b * ex * ey + ao * ey * ey) / det_o;
                min_d2 = min_d2.min(d2);
                let (l1, l2) = (ln_in - 0.5 * d2, ln_out - 0.5 * d2o);
                let m = l1.max(l2);
                p.log_w += m + ((l1 - m).exp() + (l2 - m).exp()).ln();
            }
        }

        // Track lost: no hypothesis explains the observation. Re-seed the
        // kinematics; the bank probabilities are left as they were, since this
        // observation carries no usable evidence between the goals.
        if min_d2 > mp.recovery_d2 {
            recoveries += 1;
            for (bank, brng) in banks.iter_mut().zip(bank_rngs.iter_mut()) {
                reseed(&mut bank.particles, &track[i - 1], obs, mp, brng, ttc);
            }
            continue;
        }

        let mut log_l = Vec::with_capacity(banks.len());
        let mut need_resample = false;
        for bank in banks.iter_mut() {
            log_l.push(normalise(&mut bank.particles));
            let sum_sq: f64 = bank.particles.iter().map(|p| p.weight * p.weight).sum();
            need_resample |= 1.0 / sum_sq < bank.particles.len() as f64 / 2.0;
        }
        let score_bank = if mp.intent_two_stage {
            Some(0)
        } else if mp.intent_hybrid {
            banks.iter().position(|b| b.goal.is_none())
        } else {
            None
        };
        if let Some(sb) = score_bank {
            summaries.push(summarise(&banks[sb].particles, obs.t));
            // Retargeting hazard every step; evidence once per closed window.
            let hazard = 1.0 - (-mp.goal_switch_rate * dt).exp();
            for (k, p) in intent_probs.iter_mut().enumerate() {
                *p = (1.0 - hazard) * *p + hazard * hypotheses[k].1;
            }
            let last = summaries.len() - 1;
            if summaries[last].t - summaries[win_start].t >= mp.intent_window_s - 1e-6 {
                let win = &summaries[win_start..=last];
                let ll: Vec<f64> =
                    hypotheses.iter().map(|&(goal, _)| intent_log_lik(win, goal, oois, mp, ttc)).collect();
                let max_l = ll.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                if max_l.is_finite() {
                    let mut z = 0.0;
                    for (k, p) in intent_probs.iter_mut().enumerate() {
                        *p *= (mp.intent_power * (ll[k] - max_l)).exp();
                        z += *p;
                    }
                    for p in intent_probs.iter_mut() {
                        *p /= z;
                    }
                }
                win_start = last;
            }
        }

        // Resample every bank at the same steps (keeps the common random
        // streams aligned).
        if need_resample {
            for (bank, brng) in banks.iter_mut().zip(bank_rngs.iter_mut()) {
                resample_by_mode(&mut bank.particles, mp, brng);
            }
        }

        let hazard = 1.0 - (-mp.goal_switch_rate * dt).exp();
        let max_l = log_l.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        if max_l.is_finite() {
            let mut z = 0.0;
            for (bank, &ll) in banks.iter_mut().zip(log_l.iter()) {
                let p_pred = (1.0 - hazard) * bank.prob + hazard * bank.prior;
                bank.prob = p_pred * (mp.goal_evidence_power * (ll - max_l)).exp();
                z += bank.prob;
            }
            for bank in banks.iter_mut() {
                bank.prob /= z;
            }
        }
    }

    let hybrid = !mp.intent_two_stage && mp.intent_hybrid && banks.iter().any(|b| b.goal.is_none());
    if hybrid {
        // Mix the per-goal filters by the two-stage P(goal).
        for bank in banks.iter_mut() {
            let k = hypotheses.iter().position(|h| h.0 == bank.goal).unwrap();
            bank.prob = intent_probs[k];
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
    if mp.intent_two_stage || hybrid {
        goal_probs = vec![0.0; oois.len()];
        for (k, &(goal, _)) in hypotheses.iter().enumerate() {
            if let Some(g) = goal {
                goal_probs[g] += intent_probs[k];
            }
        }
    }

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
    if mp.intent_two_stage {
        // Draw from the goal-free filter, then give each particle a goal
        // label drawn from the intent posterior.
        fan = resample_n(&banks[0].particles, n_samples, &mut rng);
        for p in fan.iter_mut() {
            let u = rng.next_f64();
            let mut acc = 0.0;
            p.goal = None;
            for (k, &(goal, _)) in hypotheses.iter().enumerate() {
                acc += intent_probs[k];
                if u < acc {
                    p.goal = goal;
                    break;
                }
            }
        }
    } else {
        for (bank, &n_k) in banks.iter().zip(alloc.iter()) {
            fan.extend(resample_n(&bank.particles, n_k, &mut rng));
        }
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
            p.step(mp, h, cfg.speed_noise_mps, p_turn, oois, &prior, &mut rng, ttc);
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
        recoveries,
    }
}
