// web/demo.js — Stage 1 client/server heading-prediction demo.
//
// Maps the predictor core (Rust backend) onto a MapLibre GL JS map with a
// canvas overlay ("fan") that draws the track, point cloud, histogram fan,
// and 50%/90% sector contours. The prediction and simulation logic now runs
// in the Rust server over HTTP/JSON; the browser talks to it via fetch.
//
// Anchoring: scenario-local metres are converted to lng/lat relative to an
// arbitrary anchor point, then map.project() turns lng/lat into screen px.

const API_BASE = `${window.location.protocol}//${window.location.hostname}:8081`;

// ---- Anchoring: arbitrary origin of the scenario-local metre grid ----------
const ANCHOR_LNG = -79.4;
const ANCHOR_LAT = 43.7;
const METER_PER_DEG_LAT = 111320.0;

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
};
function val(id) { return parseFloat(el(inputs[id]).value); }
function status(msg) { el('status').textContent = msg; }

// ---- State -----------------------------------------------------------------
let observations = []; // {t, x, y, cov_xx, cov_xy, cov_yy, v, psi}
let true_future = []; // the analytic truth past duration_s (for the dashed line)
let pred = null;
// OOIs (observable objective intent states) rendered + sent to the backend.
// Updated by predict(); consumed by draw() to render the red circles.
let oois = [];
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

  // OOIs: render the goal points the backend used as large red circles so we
  // can see where the prediction fan is being steered toward.
  if (oois && oois.length > 0) {
    for (const o of oois) {
      const [px, py] = metersToScreen(o.x, o.y);
      ctx.beginPath();
      ctx.fillStyle = 'red';
      ctx.arc(px, py, 8, 0, Math.PI * 2);
      ctx.fill();
      ctx.fillStyle = 'white';
      ctx.font = '12px sans-serif';
      ctx.fillText('OOI', px + 10, py + 5);
    }
  }

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
    drawPathRibbon(pred.path_cloud, showCloud);
  }

  if (!pred) return;

  // Apex = last observation's screen position.
  const [ax, ay] = metersToScreen(last.x, last.y);

  // Fan: filled thin sectors, one per non-zero bin.
  const binCenters = pred.bin_centers;
  const binProbs = pred.bin_probs;
  const azimuthBins = val('azimuth_bins');
  const binWidth = (2 * Math.PI) / azimuthBins;
  const vMean = (val('v_min') + val('v_max')) / 2;
  const horizon = val('horizon_s');
  const length = horizon * vMean;

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

  // Sector contours: 50% and 90% as solid bright radial lines.
  const mean = pred.final_heading_mean;
  drawSectorContour(ax, ay, mean, pred.sector50, '#00ff00', last);
  drawSectorContour(ax, ay, mean, pred.sector90, '#ff00ff', last);
}

function drawSectorContour(ax, ay, mean, width, color, last) {
  const vMean = (val('v_min') + val('v_max')) / 2;
  const horizon = val('horizon_s');
  const length = horizon * vMean;
  const half = width / 2;
  const [x0, y0] = metersToScreen(
    last.x + length * Math.cos(mean - half),
    last.y + length * Math.sin(mean - half));
  const [x1, y1] = metersToScreen(
    last.x + length * Math.cos(mean + half),
    last.y + length * Math.sin(mean + half));
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
// distinct curved lines show instead of a solid cone. Every Nth particle is
// drawn with a higher alpha so the lines stand out against the point cloud.
function drawPathRibbon(path_cloud, showCloud) {
  if (!path_cloud || path_cloud.length === 0) return;
  // Draw every Nth particle; fewer lines for large particle counts.
  const n = path_cloud.length;
  const step = n > 2000 ? 100 : n > 1000 ? 50 : 25;
  ctx.save();
  ctx.lineWidth = 1.5;
  ctx.strokeStyle = 'rgba(0, 0, 0, 0.3)';
  for (let i = 0; i < n; i += step) {
    const path = path_cloud[i];
    if (!path || path.length < 2) continue;
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
    { id: 'traj-psi0', label: 'psi0', value: 0 },
  ]},
  spiral: { fields: [
    { id: 'traj-r0', label: 'r0', value: 10 },
    { id: 'traj-r-growth', label: 'r_growth', value: 1 },
    { id: 'traj-turns', label: 'turns', value: 3 },
  ]},
  sine: { fields: [
    { id: 'traj-wavelength', label: 'wavelength', value: 100 },
    { id: 'traj-amplitude', label: 'amplitude', value: 20 },
    { id: 'traj-direction', label: 'direction', value: 0 },
  ]},
  zigzag: { fields: [
    { id: 'traj-segment', label: 'segment', value: 30 },
    { id: 'traj-angle', label: 'angle', value: 0.5 },
    { id: 'traj-baseline', label: 'baseline', value: 0 },
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
    status('track simulated: ' + observations.length + ' observations');

    // FIX 2: fit map to track + fan extents after Simulate.
    if (map.isStyleLoaded() && observations.length) {
      const vMean = (val('v_min') + val('v_max')) / 2;
      const fanLen = val('horizon_s') * vMean;
      const last = lastObservation;
      const pts = observations.map(o => metersToLngLat(o.x, o.y));
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

async function predict() {
  await W(async () => {
    if (!lastObservation) {
      console.warn('Predict pressed before Simulate; ignoring');
      status('simulate first');
      return;
    }
    const last = lastObservation;
    const azimuthBins = Math.round(val('azimuth_bins'));
    const t0 = performance.now();

    // Stage 2: send the full observations array as `track`. The server runs
    // the Kalman filter over it and feeds the smoothed state to predict().
    // `last_observation` is kept as the last point for backward compat.
    const res = await apiPost('/api/predict', {
      ttc: {
        v_min: val('v_min'),
        v_max: val('v_max'),
        a_lat: val('a_lat'),
        a_long: val('a_long'),
        omega_cap: val('omega_cap'),
      },
      track: observations.map((o) => ({
        t: o.t, x: o.x, y: o.y, v: o.v, psi: o.psi,
        cov_xx: o.cov_xx, cov_xy: o.cov_xy, cov_yy: o.cov_yy,
      })),
      last_observation: {
        t: last.t,
        x: last.x,
        y: last.y,
        v: last.v,
        psi: last.psi,
        cov_xx: last.cov_xx,
        cov_xy: last.cov_xy,
        cov_yy: last.cov_yy,
      },
      oois: [{ x: Number(el('ooi_x').value), y: Number(el('ooi_y').value) }],
      config: {
        horizon_s: val('horizon_s'),
        n_samples: val('n_samples'),
        dt_s: 1.0,
        p_turn: val('p_turn'),
        azimuth_bins: azimuthBins,
        speed_noise_mps: val('speed_noise_mps'),
        intent_hypothesis_prob: val('intent_hypothesis_prob'),
        seed: Number(val('seed') | 0),
      },
    });

    pred = {
      bin_centers: res.bin_centers,
      bin_probs: res.bin_probs,
      point_cloud: res.point_cloud,
      path_cloud: res.path_cloud, // [[ [x,y], ... ], ...] one polyline per particle
      final_heading_mean: res.heading_mean,
      sector50: res.sector50,
      sector90: res.sector90,
    };

    // Keep the OOIs the backend used so draw() can render them.
    oois = [{ x: Number(el('ooi_x').value), y: Number(el('ooi_y').value) }];

    // Stats (radians -> degrees).
    el('s50').textContent = (pred.sector50 * 180 / Math.PI).toFixed(1);
    el('s90').textContent = (pred.sector90 * 180 / Math.PI).toFixed(1);
    el('smean').textContent = (pred.final_heading_mean * 180 / Math.PI).toFixed(1);
    el('selapsed').textContent = Math.round(performance.now() - t0);
    // FIX 4c: n_bins_nonzero stat.
    const nz = pred.bin_probs.reduce((a, p) => a + (p > 0 ? 1 : 0), 0);
    el('snz').textContent = nz;

    draw();
    status('prediction complete');
  });
}

// ---- Wire up ---------------------------------------------------------------
let buttonsWired = false;
function wireButtons() {
  if (buttonsWired) return;
  buttonsWired = true;
  resizeCanvas();
  map.on('move', () => { resizeCanvas(); draw(); });
  map.on('zoom', () => { resizeCanvas(); draw(); });
  map.on('moveend', () => { resizeCanvas(); draw(); });
  window.addEventListener('resize', () => { resizeCanvas(); draw(); });
  el('btn_sim').addEventListener('click', simulate);
  el('btn_predict').addEventListener('click', predict);
  // Populate trajectory parameter inputs on load and when the selector changes.
  populateTrajParams();
  el('traj-type').addEventListener('change', populateTrajParams);
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
