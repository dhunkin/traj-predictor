// web/demo.js — client for the heading-prediction server.
//
// Maps the predictor core (Rust backend) onto a MapLibre GL JS map with a
// canvas overlay ("fan") that draws the track, point cloud, histogram fan,
// and 50%/90% sector contours. The prediction and simulation logic runs in
// the Rust server over HTTP/JSON; the browser talks to it via fetch.
//
// Two ways to get a track: "Simulated track" (an analytic test trajectory
// from the server) or "Live flight" (a fixed-wing drone flown with the arrow
// keys, predicted continuously and scored against where it really went).
//
// Anchoring: scenario-local metres are converted to lng/lat relative to an
// arbitrary anchor point, then map.project() turns lng/lat into screen px.

// API server: same host, port 8081 by default. Override with ?api=<port> or
// ?api=<full URL>, e.g. to compare two server builds side by side.
const API_BASE = (() => {
  const api = new URLSearchParams(window.location.search).get('api');
  if (!api) return `${window.location.protocol}//${window.location.hostname}:8081`;
  return /^\d+$/.test(api) ? `${window.location.protocol}//${window.location.hostname}:${api}` : api;
})();

// ---- Anchoring: arbitrary origin of the scenario-local metre grid ----------
const ANCHOR_LNG = -79.4;
const ANCHOR_LAT = 43.7;
const METER_PER_DEG_LAT = 111320.0;

// Inverse of metersToLngLat: map clicks -> scenario metres.
function lngLatToMeters(lng, lat) {
  return [
    (lng - ANCHOR_LNG) * METER_PER_DEG_LAT * Math.cos(ANCHOR_LAT * Math.PI / 180),
    (lat - ANCHOR_LAT) * METER_PER_DEG_LAT,
  ];
}

function metersToLngLat(x, y) {
  const lat = ANCHOR_LAT + y / METER_PER_DEG_LAT;
  const lng = ANCHOR_LNG + x / (METER_PER_DEG_LAT * Math.cos(ANCHOR_LAT * Math.PI / 180));
  return [lng, lat];
}
// Scenario-local metres (x, y) -> screen pixel [px, py].
function metersToScreen(x, y) {
  const p = map.project(metersToLngLat(x, y));
  return [p.x, p.y];
}

// ---- Map setup -------------------------------------------------------------
const map = new maplibregl.Map({
  container: 'map',
  style: {
    version: 8,
    sources: {
      osm: {
        type: 'raster',
        tiles: ['https://tile.openstreetmap.org/{z}/{x}/{y}.png'],
        tileSize: 256,
        attribution: '© OpenStreetMap contributors',
      },
    },
    layers: [{ id: 'osm', type: 'raster', source: 'osm' }],
  },
  center: [ANCHOR_LNG, ANCHOR_LAT],
  zoom: 18,
});

// ---- Zoom controls (FIX 1) -----------------------------------------------
map.addControl(new maplibregl.NavigationControl(), 'top-left');

// Metric scale bar (updates with zoom). Scenario metres are placed on the map
// around the anchor with a local flat-earth conversion, so at these distances
// the bar measures the same metres as the predictor works in.
map.addControl(new maplibregl.ScaleControl({ maxWidth: 160, unit: 'metric' }), 'bottom-left');

// Keep the bottom-left controls (the scale) just above the stats bar, whose
// height changes when it wraps onto two lines.
const statsBar = document.getElementById('stats');
function liftBottomControls() {
  const corner = document.querySelector('.maplibregl-ctrl-bottom-left');
  if (corner) corner.style.bottom = statsBar.offsetHeight + 'px';
}
new ResizeObserver(liftBottomControls).observe(statsBar);
liftBottomControls();

// ---- Canvas overlay --------------------------------------------------------
const fan = document.getElementById('fan');
const ctx = fan.getContext('2d');

function resizeCanvas() {
  const c = map.getCanvas();
  const rect = c.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  fan.style.left = rect.left + 'px';
  fan.style.top = rect.top + 'px';
  fan.width = Math.round(rect.width * dpr);
  fan.height = Math.round(rect.height * dpr);
  fan.style.width = rect.width + 'px';
  fan.style.height = rect.height + 'px';
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
}

// ---- UI refs ---------------------------------------------------------------
const el = (id) => document.getElementById(id);
const inputs = {
  seed: 'seed', v_min: 'v_min', v_max: 'v_max', a_lat: 'a_lat', a_long: 'a_long', omega_cap: 'omega_cap',
  horizon_s: 'horizon_s', duration_s: 'duration_s', n_samples: 'n_samples',
  p_turn: 'p_turn', speed_noise_mps: 'speed_noise_mps', azimuth_bins: 'azimuth_bins', sim_speed: 'sim_speed',
  intent_hypothesis_prob: 'intent_hypothesis_prob',
  fly_x: 'fly_x', fly_y: 'fly_y', fly_heading: 'fly_heading', fly_noise: 'fly_noise',
};
function val(id) { return parseFloat(el(inputs[id]).value); }
function status(msg) { el('status').textContent = msg; }

// ---- State -----------------------------------------------------------------
let observations = []; // {t, x, y, cov_xx, cov_xy, cov_yy, v, psi}
let true_future = []; // the analytic truth past duration_s (for the dashed line)
let pred = null;
// OOIs (objects of interest): each one is a goal hypothesis the filter tests.
// Edited in the panel list or placed by clicking the map ("+ Add on map").
// Each OOI gets its own sub-filter, sharing n_samples, so keep it to a few.
const MAX_OOIS = 6;
let oois = [{ x: 350, y: 0 }];
let addingOoi = false;
// The OOIs the last prediction was made with, so their P(goal) labels stay
// attached to the right points even while the list is being edited.
let predOois = [];
function currentOois() { return oois.map((o) => ({ x: o.x, y: o.y })); }

// P(goal) of OOI i from the last prediction, or null if the list changed.
function goalProbOf(i) {
  if (!pred || !pred.goal_probs || predOois.length !== oois.length) return null;
  const o = oois[i], q = predOois[i];
  return o.x === q.x && o.y === q.y ? pred.goal_probs[i] : null;
}

function renderOoiList() {
  const list = el('ooi-list');
  list.innerHTML = '';
  oois.forEach((o, i) => {
    const row = document.createElement('div');
    row.className = 'row';
    const name = document.createElement('label');
    name.textContent = 'OOI ' + (i + 1);
    name.dataset.tip = 'Position (m) of this object of interest: x east, y north. Each OOI is a separate goal hypothesis; the red number is its P(goal).';
    const mk = (key) => {
      const inp = document.createElement('input');
      inp.type = 'number';
      inp.step = '10';
      inp.value = o[key];
      inp.addEventListener('input', () => {
        const v = Number(inp.value);
        if (Number.isFinite(v)) { oois[i][key] = v; draw(); }
      });
      return inp;
    };
    const pg = document.createElement('span');
    pg.className = 'pg';
    const p = goalProbOf(i);
    pg.textContent = p === null ? '' : p.toFixed(2);
    const del = document.createElement('button');
    del.textContent = '×';
    del.title = 'Remove this OOI';
    del.addEventListener('click', () => { oois.splice(i, 1); renderOoiList(); draw(); });
    row.append(name, mk('x'), mk('y'), pg, del);
    list.appendChild(row);
  });
  if (!oois.length) {
    const none = document.createElement('div');
    none.className = 'hint';
    none.textContent = 'No OOIs: the filter only tests "no goal".';
    list.appendChild(none);
  }
  el('btn_add_ooi').disabled = oois.length >= MAX_OOIS;
}

function setAddingOoi(on) {
  addingOoi = on && oois.length < MAX_OOIS;
  el('btn_add_ooi').classList.toggle('active', addingOoi);
  el('btn_add_ooi').textContent = addingOoi ? 'Click the map… (Esc)' : '+ Add on map';
  map.getCanvas().style.cursor = addingOoi ? 'crosshair' : '';
}
// Last observation, hoisted to module scope so predict() can use it
// regardless of handler scope (fixes "last is not defined" on Predict).
let lastObservation = null;

// ---- HTTP helpers ----------------------------------------------------------
async function apiPost(path, payload) {
  const res = await fetch(API_BASE + path, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  if (!res.ok) {
    let detail = '';
    try { detail = await res.text(); } catch (_) { /* ignore */ }
    throw new Error('HTTP ' + res.status + (detail ? ': ' + detail : ''));
  }
  return res.json();
}

// ---- Rendering -------------------------------------------------------------
function draw() {
  ctx.clearRect(0, 0, fan.width, fan.height);
  drawScene();
  drawOois();
  drawFlight();
}

// OOIs as red circles, labelled with their P(goal) from the last prediction.
function drawOois() {
  oois.forEach((o, i) => {
    const [px, py] = metersToScreen(o.x, o.y);
    ctx.beginPath();
    ctx.fillStyle = 'red';
    ctx.arc(px, py, 8, 0, Math.PI * 2);
    ctx.fill();
    const p = goalProbOf(i);
    const label = 'OOI ' + (i + 1) + (p === null ? '' : '  ' + p.toFixed(2));
    ctx.font = 'bold 12px sans-serif';
    ctx.lineWidth = 3;
    ctx.strokeStyle = 'rgba(255,255,255,0.9)';
    ctx.strokeText(label, px + 11, py + 4);
    ctx.fillStyle = '#b00';
    ctx.fillText(label, px + 11, py + 4);
  });
}

function drawScene() {
  const showTrack = el('show_track').checked;
  const showCloud = el('show_cloud').checked;
  const showRibbon = el('show_paths').checked;

  // Track polyline + observation dots.
  if (showTrack && observations.length) {
    const pts = observations.map((o) => metersToScreen(o.x, o.y));
    ctx.strokeStyle = '#1a5276';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(pts[0][0], pts[0][1]);
    for (let i = 1; i < pts.length; i++) ctx.lineTo(pts[i][0], pts[i][1]);
    ctx.stroke();
    ctx.fillStyle = '#154360';
    for (const p of pts) {
      ctx.beginPath();
      ctx.arc(p[0], p[1], 3, 0, Math.PI * 2);
      ctx.fill();
    }
  }

  if (!lastObservation) return;
  const last = lastObservation;

  // True Future: the analytic truth past duration_s, drawn as a dashed black
  // line from the last observation to the end of truth. Drawn first so the fan
  // (drawn below) overlays it.
  if (true_future.length) {
    const pts = true_future.map((o) => metersToScreen(o.x, o.y));
    // Start exactly at the last observation's screen position.
    const start = metersToScreen(last.x, last.y);
    ctx.save();
    ctx.setLineDash([5, 5]);
    ctx.strokeStyle = '#000';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(start[0], start[1]);
    for (let i = 0; i < pts.length; i++) ctx.lineTo(pts[i][0], pts[i][1]);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.restore();
  }

  // Point cloud.
  if (showCloud && pred) {
    const cloud = pred.point_cloud;
    ctx.fillStyle = 'rgba(90,90,90,0.5)';
    for (let i = 0; i + 1 < cloud.length; i += 2) {
      const [px, py] = metersToScreen(cloud[i], cloud[i + 1]);
      ctx.beginPath();
      ctx.arc(px, py, 1.5, 0, Math.PI * 2);
      ctx.fill();
    }
  }

  // Trajectory ribbon: draw a subset of per-particle polylines (every Nth
  // particle) with higher alpha so the distinct curved lines are visible
  // instead of a solid cone. Drawn below the fan so the fan colors show.
  if (showRibbon && pred && pred.path_cloud) {
    drawPathRibbon(pred.path_cloud, pred.path_goal);
  }

  if (!pred) return;

  // Apex = last observation's screen position.
  const [ax, ay] = metersToScreen(last.x, last.y);

  // Fan: filled thin sectors, one per non-zero bin.
  const binCenters = pred.bin_centers;
  const binProbs = pred.bin_probs;
  const azimuthBins = val('azimuth_bins');
  const binWidth = (2 * Math.PI) / azimuthBins;
  const length = fanLength();

  let maxP = 0;
  for (const p of binProbs) if (p > maxP) maxP = p;

  // FIX 4a: skip low-prob bins so the fan tail is cut, not a full disc.
  const REL_THRESHOLD = 0.05;

  ctx.save();
  ctx.lineWidth = 1;
  for (let i = 0; i < binCenters.length; i++) {
    const p = binProbs[i];
    if (p <= 0) continue;
    if (p < REL_THRESHOLD * maxP) continue;
    const a = binCenters[i];
    const a0 = a - binWidth / 2;
    const a1 = a + binWidth / 2;
    // Screen y is flipped vs. scenario y, so negate sin.
    const [x0, y0] = metersToScreen(
      last.x + length * Math.cos(a0), last.y + length * Math.sin(a0));
    const [x1, y1] = metersToScreen(
      last.x + length * Math.cos(a1), last.y + length * Math.sin(a1));

    const norm = maxP > 0 ? p / maxP : 0;
    const hue = 240 * (1 - norm); // blue (low) -> red (high)
    // FIX 4b: steepen alpha so low-prob bins fade toward 0.15.
    const alpha = 0.15 + 0.75 * Math.pow(norm, 1.5);
    ctx.beginPath();
    ctx.moveTo(ax, ay);
    ctx.lineTo(x0, y0);
    ctx.lineTo(x1, y1);
    ctx.closePath();
    ctx.fillStyle = `hsla(${hue}, 90%, 55%, ${alpha})`;
    ctx.fill();
    ctx.strokeStyle = `hsla(${hue}, 90%, 40%, ${alpha})`;
    ctx.stroke();
  }
  ctx.restore();

  // Sector contours: 50% and 90% as solid bright radial lines. The backend
  // returns each sector's start angle, so the lines bound the sector it
  // actually found (centring on the mean is wrong for a two-mode fan).
  drawSectorContour(ax, ay, pred.sector50_start, pred.sector50, '#00ff00', last);
  drawSectorContour(ax, ay, pred.sector90_start, pred.sector90, '#ff00ff', last);
}

// Radial length (m) of the fan and sector lines: the filtered speed times
// the horizon, falling back to the mid speed before the first prediction.
function fanLength() {
  const v = pred && Number.isFinite(pred.v_est) && pred.v_est > 0
    ? pred.v_est : (val('v_min') + val('v_max')) / 2;
  return val('horizon_s') * v;
}

function drawSectorContour(ax, ay, start, width, color, last) {
  const length = fanLength();
  const [x0, y0] = metersToScreen(
    last.x + length * Math.cos(start),
    last.y + length * Math.sin(start));
  const [x1, y1] = metersToScreen(
    last.x + length * Math.cos(start + width),
    last.y + length * Math.sin(start + width));
  ctx.strokeStyle = color;
  ctx.lineWidth = 3;
  ctx.beginPath();
  ctx.moveTo(ax, ay);
  ctx.lineTo(x0, y0);
  ctx.moveTo(ax, ay);
  ctx.lineTo(x1, y1);
  ctx.stroke();
}

// Draw the trajectory ribbon: a subset of per-particle polylines so the
// distinct curved lines show instead of a solid cone. Paths of particles that
// carry a goal hypothesis are red, goal-free ones dark grey, so you can see
// which hypothesis each branch of the fan comes from.
function drawPathRibbon(path_cloud, path_goal) {
  if (!path_cloud || path_cloud.length === 0) return;
  // Draw every Nth particle; fewer lines for large particle counts.
  const n = path_cloud.length;
  const step = n > 4000 ? 40 : n > 1000 ? 20 : 10;
  ctx.save();
  ctx.lineWidth = 1.5;
  for (let i = 0; i < n; i += step) {
    const path = path_cloud[i];
    if (!path || path.length < 2) continue;
    const hasGoal = path_goal && path_goal[i] >= 0;
    ctx.strokeStyle = hasGoal ? 'rgba(200, 0, 0, 0.45)' : 'rgba(0, 0, 0, 0.3)';
    ctx.beginPath();
    for (let j = 0; j < path.length; j++) {
      const [px, py] = metersToScreen(path[j][0], path[j][1]);
      if (j === 0) ctx.moveTo(px, py);
      else ctx.lineTo(px, py);
    }
    ctx.stroke();
  }
  ctx.restore();
}

// ---- Trajectory parameter UI -----------------------------------------------
// Build number inputs for the selected trajectory's parameters.
const TRAJ_DEFS = {
  straight: { fields: [
    { id: 'traj-psi0', label: 'psi0', value: 0, tip: 'Heading of the straight line, in radians: 0 = east, π/2 = north (counter-clockwise).' },
  ]},
  spiral: { fields: [
    { id: 'traj-r0', label: 'r0', value: 10, tip: 'Starting radius of the spiral (m). Flown at sim_speed, so a small r0 means a tight, high-g turn.' },
    { id: 'traj-r-growth', label: 'r_growth', value: 1, tip: 'How much the radius grows per radian turned (m/rad). 2π·r_growth is the gap between loops.' },
    { id: 'traj-turns', label: 'turns', value: 3, tip: 'Number of spiral loops; after that the UAV keeps orbiting at the final radius.' },
  ]},
  orbit: { fields: [
    { id: 'traj-radius', label: 'radius (m)', value: 20, tip: 'Radius of the circle (m). Airspeed = radius × |omega|; sim_speed is not used.' },
    { id: 'traj-omega', label: 'omega (rad/s)', value: 0.4, tip: 'Turn rate (rad/s); positive = counter-clockwise. Lateral acceleration = radius × omega².' },
    { id: 'traj-drift', label: 'drift (m/s)', value: 0.6, tip: 'Speed at which the circle centre moves (m/s), e.g. a loiter drifting toward a target or with the wind. 0 = loiter in place.' },
    { id: 'traj-drift-dir', label: 'drift dir (rad)', value: 0, tip: 'Direction the circle centre drifts, in radians: 0 = east, π/2 = north.' },
  ]},
  sine: { fields: [
    { id: 'traj-wavelength', label: 'wavelength', value: 100, tip: 'Distance along the baseline for one full weave (m). Shorter = sharper swings.' },
    { id: 'traj-amplitude', label: 'amplitude', value: 20, tip: 'Sideways swing either side of the baseline (m). Peak lateral acceleration ≈ sim_speed² × amplitude × (2π/wavelength)².' },
    { id: 'traj-direction', label: 'direction', value: 0, tip: 'Mean direction of travel (baseline), in radians: 0 = east, π/2 = north.' },
  ]},
  zigzag: { fields: [
    { id: 'traj-segment', label: 'segment', value: 30, tip: 'Length of each straight leg (m). Corners are instantaneous, which no real UAV can fly; the simulator warns.' },
    { id: 'traj-angle', label: 'angle', value: 0.5, tip: 'Tilt of each leg from the +x axis (rad), alternating left/right. The heading jumps by 2 × angle at every corner.' },
    { id: 'traj-baseline', label: 'baseline', value: 0, tip: 'Sideways (y) offset of the zigzag centre line (m).' },
  ]},
};

function populateTrajParams() {
  const type = el('traj-type').value;
  const def = TRAJ_DEFS[type];
  const container = el('traj-params');
  container.innerHTML = '';
  if (!def) return;
  for (const f of def.fields) {
    const row = document.createElement('div');
    row.className = 'row';
    const label = document.createElement('label');
    label.setAttribute('for', f.id);
    label.textContent = f.label;
    if (f.tip) label.dataset.tip = f.tip;
    const input = document.createElement('input');
    input.id = f.id;
    input.type = 'number';
    input.value = f.value;
    input.step = 'any';
    row.appendChild(label);
    row.appendChild(input);
    container.appendChild(row);
  }
}

// ---- Actions ---------------------------------------------------------------
async function simulate() {
  if (flight.on) stopFlight();
  await W(async () => {
    const seed = BigInt(val('seed') | 0);
    const speed = val('sim_speed');
    const duration = val('duration_s');
    const horizon = val('horizon_s');
    const dt = 1.0;
    const cov_xx = 1.0;
    const cov_xy = 0.0;
    const cov_yy = 1.0;
    const trajType = el('traj-type').value;

    // Read trajectory parameters by id from the UI (only the ones for the
    // selected type exist; read all but tolerate missing inputs).
    const gp = (id) => { const n = el(id); return n ? parseFloat(n.value) : 0; };
    // Send only the params relevant to the selected trajectory type.
    // (The backend reads params[0..3] per type; a flat 10-element array
    // would misalign sine/zigzag params.)
    let params;
    switch (trajType) {
      case 'straight':
        params = [gp('traj-psi0')];
        break;
      case 'spiral':
        params = [gp('traj-r0'), gp('traj-r-growth'), gp('traj-turns')];
        break;
      case 'orbit':
        params = [gp('traj-radius'), gp('traj-omega'), gp('traj-drift'), gp('traj-drift-dir')];
        break;
      case 'sine':
        params = [gp('traj-wavelength'), gp('traj-amplitude'), gp('traj-direction')];
        break;
      case 'zigzag':
        params = [gp('traj-segment'), gp('traj-angle'), gp('traj-baseline')];
        break;
      default:
        params = [0];
    }

    // POST to the server and parse the returned observations.
    const res = await apiPost('/api/simulate', {
      trajectory_type: trajType,
      params,
      speed,
      duration_s: duration,
      dt_obs_s: dt,
      cov_xx, cov_xy, cov_yy,
      seed: Number(seed),
      horizon_s: horizon,
      ttc: {
        v_min: val('v_min'),
        v_max: val('v_max'),
        a_lat: val('a_lat'),
        a_long: val('a_long'),
        omega_cap: val('omega_cap'),
      },
    });

    // Split the returned observations: the first part (t <= duration_s) is the
    // observed track fed to the filter; the remainder is the analytic truth
    // past duration_s, drawn as the dashed "True Future" line.
    const dur = duration;
    observations = res.observations
      .filter((o) => o.t <= dur)
      .map((o) => ({ t: o.t, x: o.x, y: o.y, v: o.v, psi: o.psi, cov_xx: o.cov_xx, cov_xy: o.cov_xy, cov_yy: o.cov_yy }));
    true_future = res.observations
      .filter((o) => o.t > dur)
      .map((o) => ({ t: o.t, x: o.x, y: o.y, v: o.v, psi: o.psi, cov_xx: o.cov_xx, cov_xy: o.cov_xy, cov_yy: o.cov_yy }));
    lastObservation = observations[observations.length - 1];
    pred = null;
    draw();
    // The truth may break the turn-table limits (e.g. zigzag corners); the
    // filter is told such motion is impossible and will lag, so say so.
    const warn = res.warnings && res.warnings.length
      ? ' — ⚠ exceeds limits: ' + res.warnings.join('; ') : '';
    status('track simulated: ' + observations.length + ' observations' + warn);
    el('status').style.color = warn ? '#b00' : '';

    // FIX 2: fit map to track + fan extents after Simulate.
    if (map.isStyleLoaded() && observations.length) {
      const vMean = (val('v_min') + val('v_max')) / 2;
      const fanLen = val('horizon_s') * vMean;
      const last = lastObservation;
      const pts = observations.map(o => metersToLngLat(o.x, o.y));
      for (const o of currentOois()) pts.push(metersToLngLat(o.x, o.y));
      pts.push(
        metersToLngLat(last.x + fanLen, last.y),
        metersToLngLat(last.x - fanLen, last.y),
        metersToLngLat(last.x, last.y + fanLen),
        metersToLngLat(last.x, last.y - fanLen),
      );
      const bounds = new maplibregl.LngLatBounds();
      pts.forEach(p => bounds.extend(p));
      map.fitBounds(bounds, { padding: 80, duration: 800 });
    }
  });
}

// POST the track to /api/predict with the current settings and OOIs.
async function requestPrediction(track, ooiList) {
  const last = track[track.length - 1];
  const obsDto = (o) => ({
    t: o.t, x: o.x, y: o.y, v: o.v, psi: o.psi,
    cov_xx: o.cov_xx, cov_xy: o.cov_xy, cov_yy: o.cov_yy,
  });
  // Send the full observation track; the server runs the particle filter
  // over it (positions + covariances only) and fans forward from the result.
  return apiPost('/api/predict', {
    ttc: {
      v_min: val('v_min'),
      v_max: val('v_max'),
      a_lat: val('a_lat'),
      a_long: val('a_long'),
      omega_cap: val('omega_cap'),
    },
    track: track.map(obsDto),
    last_observation: obsDto(last),
    oois: ooiList,
    config: {
      horizon_s: val('horizon_s'),
      n_samples: val('n_samples'),
      dt_s: 1.0,
      p_turn: val('p_turn'),
      azimuth_bins: Math.round(val('azimuth_bins')),
      speed_noise_mps: val('speed_noise_mps'),
      intent_hypothesis_prob: val('intent_hypothesis_prob'),
      seed: Number(val('seed') | 0),
    },
  });
}

// Store a prediction response, fill the stats bar and redraw. `apex` is the
// last observation the prediction was made from (the fan's origin).
function applyPrediction(res, apex, ooiList, t0) {
  pred = {
    bin_centers: res.bin_centers,
    bin_probs: res.bin_probs,
    point_cloud: res.point_cloud,
    path_cloud: res.path_cloud, // [[ [x,y], ... ], ...] one polyline per particle
    path_goal: res.path_goal, // per path: OOI index, or -1 for no goal
    goal_probs: res.goal_probs, // per OOI
    final_heading_mean: res.heading_mean,
    sector50: res.sector50,
    sector90: res.sector90,
    sector50_start: res.sector50_start,
    sector90_start: res.sector90_start,
    v_est: res.v_est,
  };
  predOois = ooiList;
  lastObservation = apex;

  // Stats (radians -> degrees).
  el('s50').textContent = (pred.sector50 * 180 / Math.PI).toFixed(1);
  el('s90').textContent = (pred.sector90 * 180 / Math.PI).toFixed(1);
  el('smean').textContent = (pred.final_heading_mean * 180 / Math.PI).toFixed(1);
  el('selapsed').textContent = Math.round(performance.now() - t0);
  // Posterior intent, summed over OOIs (per-OOI values label the map).
  const pGoal = res.goal_probs.reduce((a, p) => a + p, 0);
  el('sgoal').textContent = ooiList.length ? pGoal.toFixed(2) : '—';
  el('sturn').textContent = res.turn_prob.toFixed(2);
  el('svest').textContent = res.v_est.toFixed(1);
  const nz = pred.bin_probs.reduce((a, p) => a + (p > 0 ? 1 : 0), 0);
  el('snz').textContent = nz;
  renderOoiList();
  draw();
}

async function predict() {
  await W(async () => {
    if (!observations.length) {
      status('simulate a track (or start a flight) first');
      return;
    }
    if (observations.length < 2) {
      status('need at least 2 observations');
      return;
    }
    const t0 = performance.now();
    const ooiList = currentOois();
    const res = await requestPrediction(observations, ooiList);
    applyPrediction(res, observations[observations.length - 1], ooiList, t0);
    status('prediction complete');
  });
}

// ---- Live flight -------------------------------------------------------------
// A fixed-wing drone flown with the arrow keys. Physics: the bank angle rolls
// toward the commanded bank at ROLL_RATE; the bank limit is atan(a_lat / g),
// so the turn rate g*tan(bank)/v respects a_lat (and is capped by omega_cap).
// Speed changes at a_long while Up/Down is held, within [v_min, v_max]; it
// never hovers. A sensor reports a noisy position every OBS_DT seconds; the
// last duration_s seconds are sent to the predictor after every report.
//
// Live scoring: every prediction made at time t is checked at t + horizon_s:
// is the bearing from the drone's true position at t to its true position
// now inside the predicted 50% / 90% sector? A calibrated predictor hits
// about 50% and 90% of the time. Each check also gets a log score (-ln of the
// fan's density at the true bearing; a uniform fan scores ln(2 pi) = 1.84),
// which rewards a fan for being both sharp and right.
//
// Honest sample size: predictions one second apart with a 10 s horizon
// overlap, so consecutive checks are strongly correlated. The effective
// number of independent checks is about one per horizon of flight, and the
// hit rates carry a 90% Wilson interval computed from that number.
const G = 9.81;
const ROLL_RATE = 60 * Math.PI / 180; // rad/s
const OBS_DT = 1.0; // s between sensor reports
const PHYS_DT = 0.02; // s, integration step
const TRUTH_DT = 0.1; // s between recorded true states

const flight = {
  on: false,
  paused: false,
  t: 0, x: 0, y: 0, psi: 0, v: 0, bank: 0, omega: 0,
  keys: { left: false, right: false, up: false, down: false },
  truth: [], // {t, x, y, psi, v, bank} every TRUTH_DT
  obs: [], // every sensor report (the predictor gets the recent window)
  nextObs: 0,
  inflight: false, // a prediction request is running
  checks: [], // predictions waiting to be scored
  scored: [], // scored predictions (exported with the flight)
  hits: { n: 0, h50: 0, h90: 0, sumLog: 0, tFirst: 0, tLast: 0 },
  lastFrame: 0,
};

const LOG_UNIFORM = Math.log(2 * Math.PI); // log score of a uniform fan

// Log score of a prediction: -ln(density per radian) of its direction
// histogram at the true bearing, with 1% of the mass spread uniformly so a
// miss costs a bounded amount (the harness in tests/common/calib.rs does the
// same).
function logScore(binProbs, bearing) {
  const n = binProbs.length;
  const width = 2 * Math.PI / n;
  const a = ((bearing % (2 * Math.PI)) + 2 * Math.PI) % (2 * Math.PI);
  const j = Math.min(n - 1, Math.floor(a / width));
  return -Math.log((0.99 * binProbs[j] + 0.01 / n) / width);
}

// Effective number of independent checks: about one per horizon of flight
// spanned by the scored predictions.
function effectiveN(h) {
  if (!h.n) return 0;
  return Math.min(h.n, (h.tLast - h.tFirst) / Math.max(val('horizon_s'), 1e-9) + 1);
}

// 90% Wilson interval of a hit rate k/n, evaluated at the effective sample
// size (so correlated checks do not pass for more evidence than they are).
function wilson90(k, n, nEff) {
  if (!n || nEff <= 0) return [0, 1];
  const z = 1.645, p = k / n, m = nEff;
  const c = (p + z * z / (2 * m)) / (1 + z * z / m);
  const h = z * Math.sqrt(p * (1 - p) / m + z * z / (4 * m * m)) / (1 + z * z / m);
  return [Math.max(0, c - h), Math.min(1, c + h)];
}

function gauss() {
  // Box-Muller.
  const u = 1 - Math.random(), v = Math.random();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

function startFlight() {
  if (flight.on) stopFlight();
  const vMin = val('v_min'), vMax = val('v_max');
  Object.assign(flight, {
    on: true, paused: false,
    t: 0, x: val('fly_x'), y: val('fly_y'), psi: val('fly_heading') * Math.PI / 180,
    v: (vMin + vMax) / 2, bank: 0, omega: 0,
    truth: [], obs: [], nextObs: 0, inflight: false,
    checks: [], scored: [], hits: { n: 0, h50: 0, h90: 0, sumLog: 0, tFirst: 0, tLast: 0 }, lastFrame: 0,
  });
  flight.truth.push(trueState());
  observations = [];
  true_future = [];
  pred = null;
  lastObservation = null;
  predOois = [];
  setAddingOoi(false);
  map.keyboard.disable(); // the arrow keys fly the drone, not the map
  map.jumpTo({ center: metersToLngLat(flight.x, flight.y) });
  el('btn_fly').textContent = 'Stop flight';
  el('btn_download').disabled = true;
  for (const id of ['slive50', 'slive90', 'slogs', 'sneff']) el(id).textContent = '—';
  status('flying — use the arrow keys');
  el('status').style.color = '';
  requestAnimationFrame(frame);
}

function stopFlight() {
  flight.on = false;
  map.keyboard.enable();
  el('btn_fly').textContent = 'Start flight';
  el('btn_download').disabled = flight.truth.length < 2;
  status('flight stopped' + (flight.hits.n ? ' — ' + hitSummary() : ''));
}

function trueState() {
  const f = flight;
  return { t: f.t, x: f.x, y: f.y, psi: f.psi, v: f.v, bank: f.bank };
}

function stepPhysics(dt) {
  const f = flight;
  const vMin = val('v_min'), vMax = val('v_max');
  const bankMax = Math.atan(Math.max(val('a_lat'), 0) / G);
  const target = f.keys.left && !f.keys.right ? bankMax
    : f.keys.right && !f.keys.left ? -bankMax : 0;
  const db = target - f.bank;
  f.bank += Math.max(-ROLL_RATE * dt, Math.min(ROLL_RATE * dt, db));
  const aLong = Math.max(val('a_long'), 0);
  if (f.keys.up) f.v += aLong * dt;
  if (f.keys.down) f.v -= aLong * dt;
  f.v = Math.max(vMin, Math.min(vMax, f.v));
  const cap = Math.max(val('omega_cap'), 0);
  f.omega = Math.max(-cap, Math.min(cap, G * Math.tan(f.bank) / Math.max(f.v, 0.1)));
  // Exact arc over the step (chord identity).
  const half = 0.5 * f.omega * dt;
  const sinc = Math.abs(half) < 1e-6 ? 1 : Math.sin(half) / half;
  const chord = f.v * dt * sinc;
  f.x += chord * Math.cos(f.psi + half);
  f.y += chord * Math.sin(f.psi + half);
  f.psi += f.omega * dt;
  f.t += dt;
}

// Sensor report: noisy position; then predict on the recent window.
function addObservation() {
  const f = flight;
  const sigma = Math.max(val('fly_noise'), 0);
  const var_ = Math.max(sigma * sigma, 0.01);
  const o = {
    t: f.t, x: f.x + sigma * gauss(), y: f.y + sigma * gauss(), v: 0, psi: 0,
    cov_xx: var_, cov_xy: 0, cov_yy: var_,
  };
  f.obs.push(o);
  const window = val('duration_s');
  observations = f.obs.filter((q) => q.t >= f.t - window - 1e-9);
  if (observations.length >= 3 && !f.inflight) livePredict(observations.slice(), { x: f.x, y: f.y });
}

async function livePredict(track, truthAtObs) {
  const f = flight;
  f.inflight = true;
  const t0 = performance.now();
  const ooiList = currentOois();
  const apex = track[track.length - 1];
  try {
    const res = await requestPrediction(track, ooiList);
    if (!f.on) return;
    applyPrediction(res, apex, ooiList, t0);
    f.checks.push({
      t: apex.t,
      due: apex.t + val('horizon_s'),
      from: truthAtObs,
      s50: res.sector50_start, w50: res.sector50,
      s90: res.sector90_start, w90: res.sector90,
      p_goal: res.goal_probs.slice(),
      p_turn: res.turn_prob,
      bin_probs: res.bin_probs.slice(),
    });
  } catch (e) {
    status('prediction error: ' + (e && e.message ? e.message : e));
  } finally {
    f.inflight = false;
  }
}

// Score predictions whose horizon has elapsed against the true position.
function scoreChecks() {
  const f = flight;
  while (f.checks.length && f.checks[0].due <= f.t) {
    const c = f.checks.shift();
    const brg = Math.atan2(f.y - c.from.y, f.x - c.from.x);
    const inside = (start, w) => ((brg - start) % (2 * Math.PI) + 2 * Math.PI) % (2 * Math.PI) < w;
    const in50 = inside(c.s50, c.w50), in90 = inside(c.s90, c.w90);
    const ls = logScore(c.bin_probs, brg);
    const h = f.hits;
    if (!h.n) h.tFirst = c.t;
    h.tLast = c.t;
    h.n += 1;
    if (in50) h.h50 += 1;
    if (in90) h.h90 += 1;
    h.sumLog += ls;
    const { bin_probs, ...rest } = c; // keep exports small
    f.scored.push({ ...rest, true_bearing: brg, in50, in90, log_score: ls });
  }
  if (f.hits.n) {
    const h = f.hits, nEff = effectiveN(h);
    const fmt = (k) => {
      const [lo, hi] = wilson90(k, h.n, nEff);
      return `${Math.round(100 * k / h.n)}% (${Math.round(100 * lo)}–${Math.round(100 * hi)}%)`;
    };
    el('slive50').textContent = fmt(h.h50);
    el('slive90').textContent = fmt(h.h90);
    el('slogs').textContent = (h.sumLog / h.n).toFixed(2);
    el('sneff').textContent = `${h.n} ≈ ${nEff.toFixed(0)} indep.`;
  }
}

function hitSummary() {
  const h = flight.hits;
  const nEff = effectiveN(h);
  const pct = (k) => Math.round(100 * k / h.n) + '%';
  return `live: 50% sector ${pct(h.h50)}, 90% sector ${pct(h.h90)}, log score `
    + `${(h.sumLog / h.n).toFixed(2)} (uniform ${LOG_UNIFORM.toFixed(2)}) over ${h.n} predictions `
    + `≈ ${nEff.toFixed(0)} independent`;
}

function frame(now) {
  const f = flight;
  if (!f.on) return;
  const dtReal = f.lastFrame ? Math.min((now - f.lastFrame) / 1000, 0.1) : 0;
  f.lastFrame = now;
  if (!f.paused) {
    let rem = dtReal;
    while (rem > 1e-9) {
      const h = Math.min(PHYS_DT, rem);
      stepPhysics(h);
      rem -= h;
      if (f.t - f.truth[f.truth.length - 1].t >= TRUTH_DT - 1e-9) f.truth.push(trueState());
      if (f.t >= f.nextObs) {
        addObservation();
        f.nextObs += OBS_DT;
      }
    }
    scoreChecks();
  }
  el('fly_readout').textContent =
    `t ${f.t.toFixed(0)} s · v ${f.v.toFixed(1)} m/s · bank ${(f.bank * 180 / Math.PI).toFixed(0)}°`
    + ` · turn ${(f.omega * 180 / Math.PI).toFixed(1)}°/s${f.paused ? ' · PAUSED' : ''}`;
  if (el('fly_follow').checked) map.jumpTo({ center: metersToLngLat(f.x, f.y) }); // redraws
  else draw();
  requestAnimationFrame(frame);
}

// The drone's true trail and an arrow at its true position.
function drawFlight() {
  const f = flight;
  if (!f.truth.length) return;
  ctx.save();
  ctx.strokeStyle = 'rgba(0, 140, 60, 0.75)';
  ctx.lineWidth = 2;
  ctx.beginPath();
  f.truth.forEach((s, i) => {
    const [px, py] = metersToScreen(s.x, s.y);
    if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
  });
  const [cx, cy] = metersToScreen(f.x, f.y);
  ctx.lineTo(cx, cy);
  ctx.stroke();
  // Arrow: screen y points down, so the heading's y component flips.
  const a = -f.psi;
  const pt = (r, da) => [cx + r * Math.cos(a + da), cy + r * Math.sin(a + da)];
  const [nx, ny] = pt(13, 0), [lx, ly] = pt(9, 2.5), [rx, ry] = pt(9, -2.5);
  ctx.beginPath();
  ctx.moveTo(nx, ny);
  ctx.lineTo(lx, ly);
  ctx.lineTo(cx, cy);
  ctx.lineTo(rx, ry);
  ctx.closePath();
  ctx.fillStyle = '#0a8a3a';
  ctx.strokeStyle = '#fff';
  ctx.lineWidth = 1.5;
  ctx.fill();
  ctx.stroke();
  ctx.restore();
}

function onKey(e, down) {
  if (!flight.on) return;
  const tag = e.target && e.target.tagName;
  if (tag === 'INPUT' || tag === 'SELECT' || tag === 'TEXTAREA') return;
  const map_ = { ArrowLeft: 'left', ArrowRight: 'right', ArrowUp: 'up', ArrowDown: 'down' };
  if (map_[e.key]) {
    flight.keys[map_[e.key]] = down;
    e.preventDefault();
  } else if (down && (e.key === 'p' || e.key === 'P')) {
    flight.paused = !flight.paused;
    e.preventDefault();
  }
}

// Download the flight as JSON: settings, OOIs, the true path, the sensor
// reports and every scored prediction — a recorded track for later analysis.
function downloadFlight() {
  const f = flight;
  const settings = {};
  for (const id of Object.keys(inputs)) settings[id] = val(id);
  const data = {
    format: 'traj-predictor flight v1',
    created: new Date().toISOString(),
    settings,
    oois: currentOois(),
    truth: f.truth,
    observations: f.obs,
    predictions: f.scored,
    hits: { ...f.hits, effective_n: effectiveN(f.hits), mean_log_score: f.hits.n ? f.hits.sumLog / f.hits.n : null },
  };
  const blob = new Blob([JSON.stringify(data)], { type: 'application/json' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'flight-' + data.created.replace(/[:.]/g, '-') + '.json';
  a.click();
  URL.revokeObjectURL(a.href);
}

// ---- Tooltips --------------------------------------------------------------
// One explanation per control and stat, keyed by element id. Angles are in
// radians, 0 = east (+x), counter-clockwise; positions are scenario metres
// with the origin at the start of the simulated track (+x east, +y north).
const TIPS = {
  seed: 'Random seed for the simulated noise and for the particle filter. The same seed gives identical results.',
  v_min: 'Slowest speed the model allows (m/s). Particles never fly slower, so the model cannot hover; a multicopter that stops will be mis-tracked. Fixed-wing: stall speed plus a margin.',
  v_max: 'Fastest speed the model allows (m/s). The simulator warns when the simulated truth goes faster.',
  a_lat: 'Maximum sideways (centripetal) acceleration, m/s². Caps the turn rate at a_lat / speed. Fixed-wing: g·tan(max bank) — 30° ≈ 5.7, 45° ≈ 9.8.',
  a_long: 'Maximum speed-up / slow-down, m/s². Currently NOT used by the predictor (speed changes come from speed_noise_mps); kept for the planned physics model.',
  omega_cap: 'Absolute maximum turn rate (rad/s). The effective limit is the smaller of this and a_lat / speed.',
  horizon_s: 'How far ahead to predict (s). The fan, point cloud and path ribbon show where the UAV may be this long after the last observation.',
  duration_s: 'Length of the simulated, observed track (s), one noisy position per second. The truth continues for horizon_s more as the dashed line.',
  sim_speed: 'Ground speed of the simulated truth (m/s) for straight, spiral, sine and zigzag. The orbit uses radius × omega instead.',
  n_samples: 'Number of particles (hypotheses), shared across the goal hypotheses. More = smoother fan and steadier P(goal), but slower (5000 ≈ 0.1 s).',
  p_turn: 'Long-run share of time the model expects the UAV to spend in a sustained turn (orbit / loiter), and the starting P(turn). 1 forbids straight flight.',
  speed_noise_mps: 'How quickly the model lets speed wander, m/s per √s. Larger = more spread along the direction of travel. The filter uses at least 0.1.',
  azimuth_bins: 'Number of direction bins for the fan (72 = 5° each). Sector widths are whole bins.',
  intent_hypothesis_prob: 'Prior probability that the UAV is heading for one of the OOIs (split evenly between them; the rest is "no goal"). The filter updates it from the track.',
  btn_add_ooi: 'Click, then click the map to place an object of interest. Each OOI is a destination the filter tests as a goal hypothesis. Esc cancels.',
  btn_clear_ooi: 'Remove all OOIs.',
  fly_x: 'Where the live flight starts: east (m).',
  fly_y: 'Where the live flight starts: north (m).',
  fly_heading: 'Starting heading in degrees: 0 = east, 90 = north (counter-clockwise).',
  fly_noise: 'Standard deviation (m) of the simulated sensor’s position error. One report per second goes to the predictor.',
  fly_follow: 'Keep the map centred on the drone.',
  btn_fly: 'Start a live flight: a fixed-wing drone flown with the arrow keys (← → bank, ↑ ↓ speed, P pause). The predictor runs after every sensor report and each prediction is scored 10 s later.',
  btn_download: 'Save the flight as JSON: the true path, sensor reports, OOIs, settings and every scored prediction. Recorded flights can become test data.',
  slive50: 'Live flight: share of predictions whose 50% sector contained where the drone actually went, horizon_s later (target about 50%). In brackets: the 90% uncertainty range given the effective number of independent checks; if it includes 50%, the result is consistent with calibration.',
  slive90: 'Live flight: share of predictions whose 90% sector contained where the drone actually went (target about 90%), with its 90% uncertainty range from the effective number of independent checks.',
  slogs: 'Live flight: mean log score of the predictions (-ln of the fan density at the true direction). Lower is better; a uniform "no idea" fan scores 1.84, so anything above that is confidently wrong. Rewards sharp and right together.',
  sneff: 'Live flight: predictions scored, and about how many independent checks they amount to. Predictions one second apart with a 10 s horizon overlap, so roughly one per horizon of flight counts as independent evidence.',
  show_track: 'Draw the observed positions (dots) and the line through them.',
  show_cloud: 'Draw where each particle ends up at the horizon (grey dots).',
  show_paths: 'Draw a sample of predicted particle paths. Red paths belong to the OOI goal hypothesis, dark grey to "no goal".',
  'traj-type': 'Shape of the simulated truth. These are test tracks: the predictor sees only the noisy positions, never the shape or its parameters.',
  btn_sim: 'Generate a new noisy track (1 m position noise, 1 per second) plus the true future (dashed). Warns if it breaks the speed / turn limits above.',
  btn_predict: 'Run the particle filter over the observed track and draw the predicted fan.',
  s50: 'Narrowest arc containing 50% of the predicted directions of travel (net bearing from now to the horizon). Green lines on the map.',
  s90: 'Narrowest arc containing 90% of the predicted directions of travel. Magenta lines on the map.',
  smean: 'Average predicted direction of travel (degrees, 0 = east, counter-clockwise). Can sit between modes when the fan splits.',
  sgoal: 'Posterior probability that the UAV is heading for an OOI, summed over all OOIs; each OOI’s own value labels it on the map. If it does not respond to the track, intent inference is not working.',
  sturn: 'Posterior probability that the UAV is in a sustained turn (orbit / loiter) right now.',
  svest: 'Filtered speed estimate (m/s) at the last observation.',
  selapsed: 'Round-trip time of the last Predict call (ms).',
  snz: 'Number of direction bins with any predicted mass.',
};

// Attach a tip to the label (controls) or the stat cell (stats bar), marked
// with ⓘ via CSS.
function attachTips() {
  for (const [id, tip] of Object.entries(TIPS)) {
    const node = el(id);
    if (!node) continue;
    let target = document.querySelector(`label[for="${id}"]`);
    if (!target && node.type === 'checkbox') target = node.closest('label');
    if (!target && node.tagName === 'BUTTON') target = node;
    if (!target && node.closest('#stats')) target = node.parentElement;
    if (target) target.dataset.tip = tip;
  }
}

// A single floating box, positioned next to the hovered element and kept on
// screen (the panel sits at the right edge, the stats bar at the bottom).
const tipBox = document.createElement('div');
tipBox.id = 'tipbox';
document.body.appendChild(tipBox);
function showTip(node) {
  tipBox.textContent = node.dataset.tip;
  tipBox.style.display = 'block';
  const r = node.getBoundingClientRect();
  const b = tipBox.getBoundingClientRect();
  const inStats = !!node.closest('#stats');
  let left = inStats ? r.left : r.left - b.width - 10;
  let top = inStats ? r.top - b.height - 8 : r.top;
  if (left < 8) left = Math.min(r.right + 10, window.innerWidth - b.width - 8);
  left = Math.max(8, Math.min(left, window.innerWidth - b.width - 8));
  top = Math.max(8, Math.min(top, window.innerHeight - b.height - 8));
  tipBox.style.left = left + 'px';
  tipBox.style.top = top + 'px';
}
document.addEventListener('mouseover', (e) => {
  const node = e.target.closest && e.target.closest('[data-tip]');
  if (node) showTip(node); else tipBox.style.display = 'none';
});
document.addEventListener('focusin', (e) => {
  const lbl = e.target.id && document.querySelector(`label[for="${e.target.id}"][data-tip]`);
  if (lbl) showTip(lbl);
});
document.addEventListener('focusout', () => { tipBox.style.display = 'none'; });

// ---- Wire up ---------------------------------------------------------------
let buttonsWired = false;
function wireButtons() {
  if (buttonsWired) return;
  buttonsWired = true;
  resizeCanvas();
  // Panning does not change the canvas size, so only redraw on move.
  map.on('move', draw);
  map.on('resize', () => { resizeCanvas(); draw(); });
  window.addEventListener('resize', () => { resizeCanvas(); draw(); });
  el('btn_sim').addEventListener('click', simulate);
  el('btn_predict').addEventListener('click', predict);
  // Populate trajectory parameter inputs on load and when the selector changes.
  populateTrajParams();
  attachTips();
  el('traj-type').addEventListener('change', populateTrajParams);
  renderOoiList();
  el('btn_add_ooi').addEventListener('click', () => setAddingOoi(!addingOoi));
  el('btn_clear_ooi').addEventListener('click', () => { oois = []; renderOoiList(); draw(); });
  map.on('click', (e) => {
    if (!addingOoi) return;
    const [x, y] = lngLatToMeters(e.lngLat.lng, e.lngLat.lat);
    oois.push({ x: Math.round(x), y: Math.round(y) });
    renderOoiList();
    setAddingOoi(false);
    draw();
  });
  el('btn_fly').addEventListener('click', (e) => {
    e.currentTarget.blur(); // keep the arrow keys for flying
    if (flight.on) stopFlight(); else startFlight();
  });
  el('btn_download').addEventListener('click', downloadFlight);
  window.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && addingOoi) setAddingOoi(false);
    onKey(e, true);
  });
  window.addEventListener('keyup', (e) => onKey(e, false));
  draw();
  status('Ready.');
}
map.on('load', wireButtons);
// Fallback: if the 'load' event doesn't fire (e.g., tiles not loading in a
// headless browser), wire up the buttons after a short delay.
setTimeout(wireButtons, 2000);

// Global error wrapper for the async action handlers.
async function W(fn) {
  try { return await fn(); }
  catch (e) { status('error: ' + (e && e.message ? e.message : e)); throw e; }
}
