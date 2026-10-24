// web/demo.js — Stage 1 WASM heading-prediction demo.
//
// Maps the predictor core (src/wasm.rs) onto a MapLibre GL JS map with a
// canvas overlay ("fan") that draws the track, point cloud, histogram fan,
// and 50%/90% sector contours.
//
// Anchoring: scenario-local metres are converted to lng/lat relative to an
// arbitrary anchor point, then map.project() turns lng/lat into screen px.

import init, { Simulator, Predictor } from './pkg/predictor.js';

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
  seed: 'seed', v_min: 'v_min', v_max: 'v_max', omega_max: 'omega_max',
  horizon_s: 'horizon_s', duration_s: 'duration_s', n_samples: 'n_samples',
  p_turn: 'p_turn', speed_noise_mps: 'speed_noise_mps', azimuth_bins: 'azimuth_bins',
};
function val(id) { return parseFloat(el(inputs[id]).value); }
function status(msg) { el('status').textContent = msg; }

// ---- State -----------------------------------------------------------------
let observations = []; // {t, x, y, cov_xx, cov_xy, cov_yy, v, psi}
let sim = null;
let pred = null;
// Last observation, hoisted to module scope so predict() can use it
// regardless of handler scope (fixes "last is not defined" on Predict).
let lastObservation = null;

const W = async (fn) => {
  try { return await fn(); }
  catch (e) { status('error: ' + (e && e.message ? e.message : e)); throw e; }
};

// ---- Rendering -------------------------------------------------------------
function draw() {
  ctx.clearRect(0, 0, fan.width, fan.height);
  const showTrack = el('show_track').checked;
  const showCloud = el('show_cloud').checked;

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

  ctx.save();
  ctx.lineWidth = 1;
  for (let i = 0; i < binCenters.length; i++) {
    const p = binProbs[i];
    if (p <= 0) continue;
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
    const alpha = 0.25 + 0.6 * norm;
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
    const speed = val('v_max');
    const duration = val('duration_s');
    const vMin = val('v_min');
    const vMax = val('v_max');
    const omegaMax = val('omega_max');
    const dt = 1.0;
    const cov = [1.0, 0.0, 1.0]; // obs_cov
    const trajType = el('traj-type').value;

    // Read trajectory parameters by id from the UI (only the ones for the
    // selected type exist; read all but tolerate missing inputs).
    const gp = (id) => { const n = el(id); return n ? parseFloat(n.value) : 0; };
    const p = {
      psi0: gp('traj-psi0'),
      r0: gp('traj-r0'),
      rGrowth: gp('traj-r-growth'),
      turns: gp('traj-turns'),
      wavelength: gp('traj-wavelength'),
      amplitude: gp('traj-amplitude'),
      direction: gp('traj-direction'),
      segment: gp('traj-segment'),
      angle: gp('traj-angle'),
      baseline: gp('traj-baseline'),
    };

    // Build the argument list and call the matching wasm method.
    const sim = new Simulator(seed, vMin, vMax, omegaMax, dt, cov[0], cov[1], cov[2]);
    let flat;
    switch (trajType) {
      case 'straight':
        flat = sim.generate_straight(p.psi0, speed, duration);
        break;
      case 'spiral':
        flat = sim.generate_spiral(p.r0, p.rGrowth, p.turns, speed, duration);
        break;
      case 'sine':
        flat = sim.generate_sine(p.wavelength, p.amplitude, p.direction, speed, duration);
        break;
      case 'zigzag':
        flat = sim.generate_zigzag(p.segment, p.angle, p.baseline, speed, duration);
        break;
      default:
        throw new Error('unknown trajectory: ' + trajType);
    }
    sim.free();

    // Parse the 8-field flat output.
    observations = [];
    for (let i = 0; i + 7 < flat.length; i += 8) {
      observations.push({
        t:     flat[i],
        x:     flat[i + 1],
        y:     flat[i + 2],
        v:     flat[i + 3],
        psi:   flat[i + 4],
        cov_xx:flat[i + 5],
        cov_xy:flat[i + 6],
        cov_yy:flat[i + 7],
      });
    }
    lastObservation = observations[observations.length - 1];
    pred = null;
    draw();
    status('track simulated: ' + observations.length + ' observations');

    // Center the scenario start in the left half of the viewport so the
    // prediction fan opens into empty space on the right (BUG 2).
    if (map.isStyleLoaded()) {
      const [lng, lat] = metersToLngLat(lastObservation.x, lastObservation.y);
      map.easeTo({ center: [lng, lat], zoom: 16, duration: 800 });
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

    pred = new Predictor(val('v_min'), val('v_max'), val('omega_max'));
    const res = pred.predict(
      last.t, last.x, last.y,
      last.cov_xx, last.cov_xy, last.cov_yy,
      last.psi, last.v,
      val('horizon_s'), val('n_samples'), val('p_turn'),
      azimuthBins, val('speed_noise_mps'), BigInt(val('seed') | 0));

    // Copy typed arrays out before freeing.
    pred = {
      bin_centers: copy(res.bin_centers()),
      bin_probs: copy(res.bin_probs()),
      point_cloud: copy(res.point_cloud()),
      final_heading_mean: res.final_heading_mean(),
      sector50: res.sector50(),
      sector90: res.sector90(),
    };
    res.free();

    // Stats (radians -> degrees).
    el('s50').textContent = (pred.sector50 * 180 / Math.PI).toFixed(1);
    el('s90').textContent = (pred.sector90 * 180 / Math.PI).toFixed(1);
    el('smean').textContent = (pred.final_heading_mean * 180 / Math.PI).toFixed(1);
    el('selapsed').textContent = Math.round(performance.now() - t0);

    draw();
    status('prediction complete');
  });
}

function copy(arr) {
  const out = new Float64Array(arr.length);
  out.set(arr);
  return out;
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
// headless browser), wire up the buttons after init() completes.
setTimeout(wireButtons, 2000);

// Ensure the WASM module is instantiated before use.
await init();
