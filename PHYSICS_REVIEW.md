# Physics review: fixed-wing and multicopter UAVs

Status: review + measurements, 2026-10-28. Nothing in this document is
implemented yet beyond the stress families in the calibration harness.

Reproduce the numbers:

```
cargo run --release --example calibrate -- --seeds 300 --n 5000 --stress
```

## 1. What the predictor assumes today

One kinematic model for every target, written in terms of **heading**:

| State | Meaning |
|---|---|
| `x, y` | ground position |
| `v` | ground speed, random walk (`speed_noise_mps`), clamped to `[v_min, v_max]` |
| `psi, omega` | heading and turn rate; `omega` limited to `min(omega_cap, a_lat / v)` |
| mode | **Cruise** (course-keeping, optionally weaving) or **Turn** (held turn rate, persistent) |
| goal | none / OOI, tested by one sub-filter per hypothesis |

Every change of direction is a *coordinated turn*: the velocity vector rotates
at `omega` while the speed barely changes. That is the physics of an aircraft.
It is not how a multicopter usually changes direction.

Gaps visible straight from the code:

- **No hover.** `v >= v_min` (UI default 3 m/s), so a stopped multicopter has
  to be "flying" in some direction.
- **`a_long` is unused.** Speed only changes through a random walk
  (0.5 m/s per sqrt(s)), so braking or a dash, at 3-5 m/s^2, is 6-10 sigma per
  second and is either missed or triggers a track-loss reset.
- **No wind.** A fixed-wing in a 5 m/s wind loitering at 18 m/s airspeed has a
  ground speed that swings between 13 and 23 m/s and a ground turn rate that
  varies around the loop. The model treats ground speed and turn rate as
  near-constant.
- **No roll dynamics.** A turn starts with an instantaneous jump in turn rate
  (`turn_entry_sigma`). Real aircraft roll in over 1-2 s, which is exactly the
  early evidence that a turn has begun.
- **One envelope for everything.** `v_min, v_max, a_lat` are shared, while the
  two platform classes have different, sometimes disjoint, envelopes.

## 2. The two platform classes

### Fixed-wing

- **Airspeed band.** Between stall plus margin and maximum (small UAVs
  typically 12-30 m/s). It **never hovers** and rarely varies airspeed by more
  than about 15% in level flight.
- **Turns only by banking.** `a_lat = g tan(bank)`, so `omega = g tan(bank) / V_air`.
  Typical bank limits are 30-45 degrees (`a_lat` 5.7-9.8 m/s^2). Turn rates are
  therefore low at speed: 0.25-0.45 rad/s at 15-22 m/s.
- **Roll rate.** It takes 1-2 s to establish a bank, so turn rate ramps.
- **Wind matters.** Wind is often 20-40% of airspeed. Ground track = air track
  + wind, so loiter circles become trochoids drifting downwind, and crabbing
  means heading is not the ground course.
- **Typical patterns.** Transit legs, loiter circles, racetracks, lawnmower
  survey lines. Turns are finite (90 / 180 / 360 degrees), not open-ended.
- **Predictability.** Good over 10 s, since inertia and the bank limit
  constrain it strongly. The hard part is *when* the next turn starts.

### Multicopter

- **Speed 0 to v_max** (consumer 15-20 m/s, sport / FPV 30+). **Hover is a
  normal state.**
- **Accelerates in any direction**, up to roughly 3-5 m/s^2 (consumer) or
  10+ m/s^2 (sport), **independent of where it points**. The natural state is
  the 2D velocity vector, not a heading. Yaw is decoupled from the direction of
  travel.
- **At cruise speed it banks into turns like an aircraft** (orbit-a-point,
  smooth waypoint turns), so the current Turn mode still applies there.
- **Typical patterns.** Hover and loiter, stop-and-go waypoint missions (stop at
  each corner, then accelerate toward the next), orbit a point, and erratic
  manual flight: dashes, reversals, sidesteps.
- **Predictability.** Fundamentally poor when it is hovering or about to
  manoeuvre: the next direction can be anything. The best a predictor can do
  there is a **calibrated wide fan**, and **intent** (OOI hypotheses) is the
  main lever to narrow it.

## 3. Measured gaps

300 seeds per family, 5000 particles, 30 s observed + 10 s horizon, 1 m
noise. Coverage is the fraction of runs whose true 10 s bearing falls inside the
reported sector (targets 0.50 / 0.90). The log score is -ln of the fan's density
at the truth; lower is better, and **a uniform fan scores 1.84**, so anything
above that is confidently wrong.

| Family | What the truth does | cov50 | cov90 | logS |
|---|---|---|---|---|
| tuning suite (pooled) | straight, homing, orbits, weaves, turns | 0.68 | 0.92 | 0.27 |
| mc_stop_go | brakes at 3 m/s^2, hovers 2-6 s, leaves in any direction | 0.26 | 0.65 | 1.97 |
| mc_depart | hovers the whole track, departs 0-4 s after | 0.16 | 0.47 | **2.82** |
| mc_reverse | brakes through zero and flies back | 0.04 | **0.22** | **4.27** |
| mc_strafe | velocity swings 90 deg by sideways acceleration | 0.07 | **0.39** | **2.76** |
| fw_orbit_wind | 30 deg bank loiter in a 3-7 m/s wind | 0.38 | 0.81 | 1.18 |
| fw_racetrack | legs + 180 deg turns at 30 deg bank | 0.22 | 0.74 | 1.55 |
| fw_turn | 45-180 deg banked turn starting within 6 s of now | 0.13 | 0.63 | 2.07 |

Experiment, with the multicopter families run at a lower `v_min`:

| v_min | mc_depart cov90 / logS | others |
|---|---|---|
| 3.0 | 0.47 / 2.82 | unchanged |
| 0.5 | 0.75 / 2.05 | unchanged |
| 0.1 | 0.82 / 1.93 | unchanged |

Allowing hover fixes the depart case. The fan becomes almost uniform (about
290 degrees), which is the honest answer. Reversal, strafe and stop-and-go do not
move: a heading/turn-rate model cannot represent braking through zero or
accelerating sideways. **That is a missing model, not a tuning problem.**

Fixed-wing is in better shape, because the current model *is* aircraft
physics. Its gaps are wind (orbit_wind), finite turn angles (racetrack), and
turn onset without roll-in evidence (fw_turn, which is also partly
unpredictable by nature).

## 4. Proposed model

Treat the platform class as another hidden variable, exactly like the goal:
sub-filters per class, each with its own physics, and report P(class). With a
known platform, the user fixes it; with `unknown`, the filter infers it.

### 4.1 Fixed-wing sub-model (extends today's model)

- **Air-relative kinematics plus a wind state.** Particles carry
  `(x, y, V_air, psi_air, omega)` and a wind vector `(w_x, w_y)` that is a slow
  random walk (about 0.05 m/s per sqrt(s)), estimated by the filter. Ground
  velocity = `V_air * [cos psi, sin psi] + w`. Loiters in wind become trochoids
  automatically, and the "goal drift" term in the Turn mode becomes unnecessary
  for wind-driven drift.
- **Airspeed** is an OU process around a per-particle cruise airspeed within
  `[v_stall, v_max]`, with `a_long` as the bound on dV/dt.
- **Bank-limited turns with roll-in:** `omega -> g tan(bank) / V_air`, with bank
  following a first-order lag (time constant about 0.7 s) toward a commanded
  bank, `|bank| <= bank_max`.
- **Finite turns:** the Turn mode's end hazard depends on the heading turned
  so far (peaks near 90 / 180 / 360 degrees) as well as on time. This captures
  racetracks and waypoint turns, while long loiters stay persistent.

### 4.2 Multicopter sub-model (new, velocity space)

State `(x, y, vx, vy)`, with `|v| <= v_max` and `|dv/dt| <= a_max` (`a_long`),
and modes:

| Mode | Dynamics |
|---|---|
| **Hold** | near-constant velocity, small acceleration noise (includes hover when v is about 0) |
| **Maneuver** | accelerate at up to `a_max` toward a target velocity drawn from a prior: stop (v = 0), reverse, sidestep, a new random direction, or toward the OOI if it has a goal; switch to Hold on arrival |
| **Orbit** | coordinated turn (today's Turn mode), for orbit-a-point and smooth waypoint turns |

The Maneuver mode is what represents stop-and-go, reversals and strafes. Its
target-velocity prior is where intent enters: a goal particle's maneuver
targets point at the OOI.

### 4.3 Shared changes

- `v_min` becomes platform-specific (0 for multicopter, stall + margin for
  fixed-wing). The UI gets a platform selector: fixed-wing / multicopter /
  unknown.
- `a_long` is actually used.
- Report predictability honestly: P(hover), and the fan's entropy or width.
  A wide fan for a hovering multicopter is correct output, not a failure.
- The stress families join the tuning suite, each evaluated with its own
  platform's envelope.

Expected outcome. Fixed-wing families should approach the tuning suite's
calibration, since most of their error is wind and turn shape. Multicopter
families should become *calibrated* (log score at or below a uniform fan,
coverage near nominal), but their fans will stay wide when the target is
hovering or between manoeuvres. That is the physics, and intent is the only
real lever.

Effort: roughly one session per sub-model, plus recalibration with the
harness. The multicopter model is the bigger change, as it adds a new state
representation.

## 5. Bearing-only detection: is it possible?

**Yes, but what can be predicted depends on the sensor geometry.** The
particle filter is the right tool, because the posterior is strongly
non-Gaussian (a long "banana" along the line of sight) and a Kalman filter
handles that badly.

### What changes in the filter

- **Likelihood.** Replace the 2D position Gaussian with a wrapped Gaussian (or
  von Mises) on bearing, `sigma_bearing` from the sensor (typically 1-5 deg for
  RF direction-finding, under 0.5 deg for optical).
- **Initialisation.** A range-parameterised prior: spread particles along the
  first bearing ray between `r_min` and `r_max` (e.g. 50 m - 3 km,
  log-uniform), with velocities from the platform's speed prior.
- **Display.** The fan's apex is no longer a known point. Show the
  predicted position cloud, and a predicted *bearing* sector as seen from the
  sensor, which is well determined even when range is not.

### Observability, by sensor setup

| Setup | Range | Direction of travel | Prediction quality |
|---|---|---|---|
| One fixed sensor, azimuth only, target at constant velocity | **unobservable**: a target at range r, speed v looks identical to one at k*r, speed k*v | ambiguous (radial component unknown) | future *bearing* good; position and heading fan wide, bounded only by the speed prior `[v_min, v_max]` |
| + speed prior (platform class) | bounded: range is proportional to speed / bearing-rate | partly | fixed-wing (narrow airspeed band) gives a useful range; multicopter much less |
| + target manoeuvres (turns) | weakly observable over time | improves | depends on the manoeuvre |
| + **OOI intent hypotheses** | the goal hypothesis pins the geometry: only ranges where the track heads at the OOI fit | good, when P(goal) is high | the intent machinery we already have becomes a range estimator |
| + **elevation angle** + altitude prior (drones mostly fly under 120 m) | `r = h / tan(elev)`, observable (coarse) | good | big win if the sensor reports elevation |
| **Two or more sensors** (triangulation) | observable | good | close to today's position-based quality, depending on baseline and geometry |
| **Moving sensor** (own platform manoeuvres) | observable after an own-ship manoeuvre | good | classic bearings-only tracking setup |

So, from a single fixed azimuth-only sensor, you can reliably predict **where
the target will appear** (future bearing, and its rate). You can predict its
heading and position only as well as the speed prior and intent allow. With
elevation, a second sensor, or a moving sensor, it becomes a well-posed
tracking problem and the existing machinery carries over almost unchanged.

### Questions that decide the design

1. How many sensors, and are they fixed or moving?
2. Azimuth only, or azimuth + elevation? What bearing accuracy and update rate?
3. Is any range-like signal available (RSSI, time-of-arrival)? Even a crude one
   collapses the banana.
4. Which platform classes do you expect (this sets the speed prior)?
