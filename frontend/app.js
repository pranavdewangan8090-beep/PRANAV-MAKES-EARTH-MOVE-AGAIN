// OCTO-EARTH viewer.
// The browser does NO physics. It draws states recorded by the Python engine:
//   * Earth: one recorded state (position, velocity before and after the event) per ejection event,
//     joined by a cubic Hermite curve in polar coordinates (accuracy checked by Python check 13).
//     The curve uses the velocity AFTER the event at its start and BEFORE the next event at its end,
//     so the recoil kink is shown, never smoothed away. Gaps larger than 2.5 event spacings are NOT
//     interpolated: Earth is held at the last recorded state and a notice is shown.
//   * Moon (both models): its own recorded stream, 8 states per event (fast model: exact Earth-Moon
//     two-body orbit; high-fidelity: N-body). Its current orbit is drawn as a thin ellipse.
//   * Asteroid: docked to Earth; drawn as a rock on Earth's marker on the side packets leave from.
//   * Packets: positions computed in Python (two-body solar gravity) at fixed times after release.
import * as THREE from './vendor/three.module.js';
import { OrbitControls } from './vendor/OrbitControls.js';

const $ = (id) => document.getElementById(id);
const S = 10;                       // scene units per AU
const MOON_X = 100;                 // Moon distance exaggeration (drawing only)
const AU = 1.495978707e11, YEAR_S = 31557600, DAY_Y = 1 / 365.25;
const KMS2AUY = 1000 * YEAR_S / AU; // km/s -> AU/yr
const M_EARTH = 5.97216787e24, M_MOON = 7.3458e22;
const EARTH_R = 0.3, AST_RMAX = 0.17;   // marker radii [scene units] (not to scale)
const AST_OFFSET_AU = (EARTH_R + AST_RMAX) / S;  // packet paths start at the asteroid marker (visual offset)
const GM_EARTH_KM = 398600.4418;    // km^3/s^2
const PERSIST_WALL_S = 0.6;         // packet paths stay visible at least this long (wall clock)
// row layout (see _viewer_rows in octoearth_rl.py)
const R_T = 0, R_X = 1, R_VPRE = 4, R_VPOST = 7, R_DM = 10, R_U = 11, R_VREL = 12, R_FATE = 13, R_MAST = 14,
  R_EREM = 15, R_NFRAG = 16, R_DV = 17, R_EDRAW = 18, R_EHAT = 19;
const FATE = { 0: 'none', 1: 'leaves the Solar System', 2: 'stays bound to the Sun', 3: 'FELL BACK to Earth', 4: 'hit the Moon' };

// ------------------------------------------------------------------ formatting helpers
const ok = (v) => v !== null && v !== undefined && Number.isFinite(+v);
const pct = (v, d = 1) => ok(v) ? (100 * v).toFixed(d) + ' %' : 'N/A';
const num = (v, d = 4) => ok(v) ? (+v).toFixed(d) : 'N/A';
const sci = (v, d = 2) => ok(v) ? (+v).toExponential(d).replace('e+', 'e') : 'N/A';
const int = (v) => ok(v) ? Math.round(v).toLocaleString('en-US') : 'N/A';
const esc = (s) => String(s ?? '').replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
function fmtSpeed(y) {
  if (y < 1) return (y * 365.25).toFixed(y * 365.25 < 10 ? 1 : 0) + (Math.abs(y * 365.25 - 1) < 0.05 ? ' day' : ' days');
  return (y < 10 ? y.toFixed(1) : Math.round(y)) + (Math.abs(y - 1) < 0.05 ? ' year' : ' years');
}
function fmtYear(t) { return 'Year ' + (t < 100 ? t.toFixed(3) : t.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })); }
function dirWords(d) {
  if (!d) return '—';
  const [r, t, n] = d; const a = [[Math.abs(t), t >= 0 ? 'prograde' : 'retrograde'], [Math.abs(r), r >= 0 ? 'outward' : 'inward'], [Math.abs(n), n >= 0 ? 'north' : 'south']];
  a.sort((x, y) => y[0] - x[0]);
  return a[1][0] > 0.25 ? `${a[0][1]} + ${a[1][1]}` : a[0][1];
}

// ------------------------------------------------------------------ three.js scene
const canvas = $('scene');
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
renderer.setPixelRatio(Math.min(2, window.devicePixelRatio || 1));
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x05070c);
const camera = new THREE.PerspectiveCamera(50, 1, 0.01, 5000);
// default view is shifted so the Sun sits right of centre: the HUD panel covers the left part of the canvas
const HOME_X = -5.5;
camera.position.set(HOME_X, -40, 34);
const controls = new OrbitControls(camera, canvas);
controls.target.set(HOME_X, 0, 0);
controls.enableDamping = true;
scene.add(new THREE.AmbientLight(0xffffff, 0.45));
scene.add(new THREE.PointLight(0xffffff, 3, 0, 0));
const sun = new THREE.Mesh(new THREE.SphereGeometry(0.6, 32, 16), new THREE.MeshBasicMaterial({ color: 0xffcc33 }));
scene.add(sun);

function circlePts(rAU, n = 256) {
  const p = []; for (let i = 0; i <= n; i++) { const a = i / n * Math.PI * 2; p.push(new THREE.Vector3(Math.cos(a) * rAU * S, Math.sin(a) * rAU * S, 0)); } return p;
}
function ring(rAU, color, opacity) {
  const l = new THREE.Line(new THREE.BufferGeometry().setFromPoints(circlePts(rAU)), new THREE.LineBasicMaterial({ color, transparent: true, opacity }));
  scene.add(l); return l;
}
ring(1.0, 0x4fa3ff, 0.18);
let targetRing = ring(2.7, 0x3ddc84, 0.65);
let targetAU = 2.7;
function setTargetRing(au) { if (au === targetAU) return; targetAU = au; scene.remove(targetRing); targetRing = ring(au, 0x3ddc84, 0.65); }

const earth = new THREE.Mesh(new THREE.SphereGeometry(EARTH_R, 24, 12), new THREE.MeshStandardMaterial({ color: 0x4fa3ff, emissive: 0x0a2a55 }));
scene.add(earth);
const moon = new THREE.Mesh(new THREE.SphereGeometry(0.17, 16, 8), new THREE.MeshStandardMaterial({ color: 0xdddddd, emissive: 0x555555 }));
scene.add(moon);
// the Moon's current (osculating) orbit around Earth, from the simulated Earth-Moon state, drawn at the same x100 scale
const moonOrbPos = new Float32Array(129 * 3);
const moonOrbGeom = new THREE.BufferGeometry(); moonOrbGeom.setAttribute('position', new THREE.BufferAttribute(moonOrbPos, 3)); moonOrbGeom.setDrawRange(0, 0);
const moonOrbit = new THREE.Line(moonOrbGeom, new THREE.LineBasicMaterial({ color: 0xbbbbbb, transparent: true, opacity: 0.45 }));
moonOrbit.frustumCulled = false; scene.add(moonOrbit);
const moonLinkGeom = new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(), new THREE.Vector3()]);
const moonLink = new THREE.Line(moonLinkGeom, new THREE.LineBasicMaterial({ color: 0xbbbbbb, transparent: true, opacity: 0.3 }));
moonLink.frustumCulled = false; scene.add(moonLink);
const asteroid = new THREE.Mesh(new THREE.IcosahedronGeometry(1, 1), new THREE.MeshStandardMaterial({ color: 0xc08a58, emissive: 0x3a2412, flatShading: true }));
scene.add(asteroid);
// text labels that follow the bodies (HTML overlay)
const LBL = {};
for (const [k, c] of [['sun', '#ffcc33'], ['earth', '#7cc4ff'], ['moon', '#dddddd'], ['ast', '#e0a46a']]) {
  const d = document.createElement('div'); d.className = 'blabel mono small'; d.style.color = c; d.style.display = 'none';
  $('center').appendChild(d); LBL[k] = d;
}
// awayFrom (optional 3-D point): put the label on the side of the body facing away from that point
function placeLabel(k, pos3, text, dx = 8, dy = -8, awayFrom = null) {
  const el = LBL[k];
  if (!text) { el.style.display = 'none'; return; }
  _lv.copy(pos3).project(camera);
  if (_lv.z > 1 || Math.abs(_lv.x) > 1 || Math.abs(_lv.y) > 1) { el.style.display = 'none'; return; }   // off-screen
  const w = canvas.clientWidth, h = canvas.clientHeight;
  if (el.textContent !== text) el.textContent = text;
  el.style.display = 'block';
  let x = (_lv.x * 0.5 + 0.5) * w, y = (-_lv.y * 0.5 + 0.5) * h;
  const lw = el.offsetWidth, lh = el.offsetHeight;
  if (awayFrom) {
    _lw.copy(awayFrom).project(camera);
    const ax = (_lw.x * 0.5 + 0.5) * w, ay = (-_lw.y * 0.5 + 0.5) * h;
    x += x < ax ? -Math.abs(dx) - lw : Math.abs(dx);
    y += y < ay ? -Math.abs(dy) - lh : Math.abs(dy);
  } else { x += dx; y += dy; }
  el.style.left = `${Math.max(2, Math.min(w - lw - 2, x))}px`; el.style.top = `${Math.max(2, Math.min(h - lh - 2, y))}px`;
}
const _lw = new THREE.Vector3();
function resolveLabels() {        // shift overlapping labels apart vertically (readability only)
  const placed = [];
  const dv = $('dvLabel');            // the recoil label stays where it is; body labels move around it
  if (dv.style.display !== 'none' && dv.style.left) placed.push({ x: parseFloat(dv.style.left), y: parseFloat(dv.style.top), w: dv.offsetWidth, h: dv.offsetHeight });
  for (const k of ['sun', 'earth', 'moon', 'ast']) {
    const el = LBL[k]; if (el.style.display === 'none') continue;
    let x = parseFloat(el.style.left), y = parseFloat(el.style.top);
    const w = el.offsetWidth, h = el.offsetHeight;
    for (let it = 0; it < 6; it++) {
      const hit = placed.find((b) => x < b.x + b.w && x + w > b.x && y < b.y + b.h && y + h > b.y);
      if (!hit) break;
      y = hit.y + hit.h + 2;
    }
    el.style.top = `${y}px`; placed.push({ x, y, w, h });
  }
}
const _lv = new THREE.Vector3();
const recoil = new THREE.ArrowHelper(new THREE.Vector3(0, 1, 0), new THREE.Vector3(), 1.4, 0x7fe08a, 0.35, 0.2);
recoil.visible = false; scene.add(recoil);

// past trail (solid) – row positions up to the display time, plus the live point
const MAXTRAIL = 400000;
const TRAIL_SUB = 4;                // the past path follows the same Hermite curve as Earth (4 points per row interval)
const trailPos = new Float32Array((MAXTRAIL + 1) * 3), trailCol = new Float32Array((MAXTRAIL + 1) * 3);
const trailGeom = new THREE.BufferGeometry(); trailGeom.setAttribute('position', new THREE.BufferAttribute(trailPos, 3));
trailGeom.setAttribute('color', new THREE.BufferAttribute(trailCol, 3)); trailGeom.setDrawRange(0, 0);
// colour encodes simulated time: dark blue at year 0 -> light cyan at the end of the mission
const trailLine = new THREE.Line(trailGeom, new THREE.LineBasicMaterial({ vertexColors: true, transparent: true, opacity: 0.6 }));
const trailDots = new THREE.Points(trailGeom, new THREE.PointsMaterial({ vertexColors: true, size: 0.06, transparent: true, opacity: 0.7 }));
trailLine.frustumCulled = trailDots.frustumCulled = false;
scene.add(trailLine); scene.add(trailDots);
// computed ahead (physics done, not yet displayed) – dim
const AHEADMAX = 4000;
const aheadPos = new Float32Array(AHEADMAX * 3);
const aheadGeom = new THREE.BufferGeometry(); aheadGeom.setAttribute('position', new THREE.BufferAttribute(aheadPos, 3)); aheadGeom.setDrawRange(0, 0);
const ahead = new THREE.Points(aheadGeom, new THREE.PointsMaterial({ color: 0x8090a8, size: 0.035, transparent: true, opacity: 0.3 }));
ahead.frustumCulled = false; scene.add(ahead);
// coasting (osculating) orbit – dashed
const coastPos = new Float32Array(257 * 3);
const coastGeom = new THREE.BufferGeometry(); coastGeom.setAttribute('position', new THREE.BufferAttribute(coastPos, 3)); coastGeom.setDrawRange(0, 0);
const coast = new THREE.Line(coastGeom, new THREE.LineDashedMaterial({ color: 0x7cc4ff, dashSize: 0.35, gapSize: 0.25, transparent: true, opacity: 0.6 }));
coast.frustumCulled = false; scene.add(coast);
// packets: line segments with per-vertex colour (brightness fades with wall-clock age)
const MAXPK = 600, MAXSEG = MAXPK * 25;
const pkPos = new Float32Array(MAXSEG * 6), pkCol = new Float32Array(MAXSEG * 6);
const pkGeom = new THREE.BufferGeometry();
pkGeom.setAttribute('position', new THREE.BufferAttribute(pkPos, 3)); pkGeom.setAttribute('color', new THREE.BufferAttribute(pkCol, 3)); pkGeom.setDrawRange(0, 0);
const pkLines = new THREE.LineSegments(pkGeom, new THREE.LineBasicMaterial({ vertexColors: true }));
pkLines.frustumCulled = false; scene.add(pkLines);
const headPos = new Float32Array(MAXPK * 3), headCol = new Float32Array(MAXPK * 3);
const headGeom = new THREE.BufferGeometry();
headGeom.setAttribute('position', new THREE.BufferAttribute(headPos, 3)); headGeom.setAttribute('color', new THREE.BufferAttribute(headCol, 3)); headGeom.setDrawRange(0, 0);
const heads = new THREE.Points(headGeom, new THREE.PointsMaterial({ size: 0.12, vertexColors: true }));
heads.frustumCulled = false; scene.add(heads);

function resize() {
  const w = canvas.clientWidth, h = canvas.clientHeight, pr = renderer.getPixelRatio();
  if (canvas.width !== Math.floor(w * pr) || canvas.height !== Math.floor(h * pr)) {
    renderer.setSize(w, h, false); camera.aspect = w / Math.max(1, h); camera.updateProjectionMatrix();
  }
}

// ------------------------------------------------------------------ viewer state
const V = {
  run: -1, mode: 'idle', label: '', fidelity: '', policy: null, done: true, final: null, meta: null,
  rows: [], cumPk: [], cumDv: [], since: 0, moon: [], msince: 0, packets: [], psince: 0, pstrideRx: 1,
  decisions: [], trailSrv: null, disp: 0, playing: true, trailUpTo: -1, trailN: 0, gap: false, waiting: false,
  evRow: -1, evDec: 0, lastEj: -1, ejScan: -1, flags: {}, wallStart: performance.now(), produced: 0,
};
let speedY = 2;                        // simulated years per wall-clock second
let DOCS = null;                       // guide texts from /api/results
function speedFromSlider(v) { return DAY_Y * Math.pow(500 * 365.25, v / 1000); }
function updSpeed() { speedY = speedFromSlider(+$('speed').value); $('speedLbl').textContent = fmtSpeed(speedY); }
$('speed').addEventListener('input', updSpeed); updSpeed();

function resetViewer() {
  Object.assign(V, { rows: [], cumPk: [], cumDv: [], since: 0, moon: [], msince: 0, packets: [], psince: 0, decisions: [],
    trailSrv: null, disp: 0, trailUpTo: -1, trailN: 0, gap: false, waiting: false, evRow: -1, evDec: 0, lastEj: -1, ejScan: -1,
    flags: {}, wallStart: performance.now(), meta: null });
  trailGeom.setDrawRange(0, 0); aheadGeom.setDrawRange(0, 0); coastGeom.setDrawRange(0, 0); pkGeom.setDrawRange(0, 0);
  headGeom.setDrawRange(0, 0); recoil.visible = false; $('evList').innerHTML = ''; evLines.length = 0;
}

function bsearch(arr, t) {          // last index with arr[i][0] <= t, or -1
  let lo = 0, hi = arr.length - 1, ans = -1;
  while (lo <= hi) { const m = (lo + hi) >> 1; if (arr[m][0] <= t) { ans = m; lo = m + 1; } else hi = m - 1; }
  return ans;
}

// cubic Hermite in (rho, theta, z); p in length unit L, v in L per time unit, h in the same time unit
function polarHermite(p0, v0, p1, v1, h, s) {
  const pol = (p, v) => { const r2 = p[0] * p[0] + p[1] * p[1], r = Math.sqrt(r2);
    return [r, Math.atan2(p[1], p[0]), p[2], (p[0] * v[0] + p[1] * v[1]) / r, (p[0] * v[1] - p[1] * v[0]) / r2, v[2]]; };
  const a = pol(p0, v0), b = pol(p1, v1);
  let th1 = b[1]; while (th1 - a[1] < -Math.PI) th1 += 2 * Math.PI; while (th1 - a[1] > Math.PI) th1 -= 2 * Math.PI;
  const s2 = s * s, s3 = s2 * s;
  const h00 = 2 * s3 - 3 * s2 + 1, h10 = s3 - 2 * s2 + s, h01 = -2 * s3 + 3 * s2, h11 = s3 - s2;
  const d00 = 6 * s2 - 6 * s, d10 = 3 * s2 - 4 * s + 1, d01 = -6 * s2 + 6 * s, d11 = 3 * s2 - 2 * s;
  const H = (f0, f1, g0, g1) => h00 * f0 + h10 * h * g0 + h01 * f1 + h11 * h * g1;
  const D = (f0, f1, g0, g1) => (d00 * f0 + d10 * h * g0 + d01 * f1 + d11 * h * g1) / h;
  const r = H(a[0], b[0], a[3], b[3]), th = H(a[1], th1, a[4], b[4]), z = H(a[2], b[2], a[5], b[5]);
  const dr = D(a[0], b[0], a[3], b[3]), dth = D(a[1], th1, a[4], b[4]), dz = D(a[2], b[2], a[5], b[5]);
  const c = Math.cos(th), sn = Math.sin(th);
  return { p: [r * c, r * sn, z], v: [dr * c - r * sn * dth, dr * sn + r * c * dth, dz] };
}
const rowP = (q) => [q[R_X], q[R_X + 1], q[R_X + 2]];
const rowV = (q, k) => [q[k] * KMS2AUY, q[k + 1] * KMS2AUY, q[k + 2] * KMS2AUY];

function eventSpacing() { return V.meta ? 1 / V.meta.ejections_per_year : 0.125; }
function earthAt(t) {             // -> {p [AU], v [AU/yr], i, gap, hold}
  const R = V.rows; if (!R.length) return null;
  const i = bsearch(R, t);
  if (i < 0) return { p: rowP(R[0]), v: rowV(R[0], R_VPRE), i: 0, hold: true };
  if (i >= R.length - 1) return { p: rowP(R[i]), v: rowV(R[i], R_VPOST), i, hold: true };
  const a = R[i], b = R[i + 1], h = b[R_T] - a[R_T];
  if (h > 2.5 * eventSpacing() + 1e-9 || h <= 0) return { p: rowP(a), v: rowV(a, R_VPOST), i, gap: true };
  const s = Math.max(0, Math.min(1, (t - a[R_T]) / h));
  const o = polarHermite(rowP(a), rowV(a, R_VPOST), rowP(b), rowV(b, R_VPRE), h, s);
  return { p: o.p, v: o.v, i };
}
function moonAt(t) {              // relative position [km] or null
  const M = V.moon; if (!M.length) return null;
  const i = bsearch(M, t); if (i < 0 || i >= M.length - 1) return null;
  const a = M[i], b = M[i + 1], h = b[0] - a[0];
  if (h <= 0 || h > 2.5 * eventSpacing() / 8 + 1e-9) return null;
  const k = YEAR_S / 1000; // m/s -> km/yr
  return polarHermite([a[1], a[2], a[3]], [a[4] * k, a[5] * k, a[6] * k], [b[1], b[2], b[3]], [b[4] * k, b[5] * k, b[6] * k], h, (t - a[0]) / h).p;
}
function moonStateAt(t) {         // nearest recorded Moon state at or before t: {r:[km], v:[km/s]}
  const M = V.moon; if (!M.length) return null;
  const i = bsearch(M, t); if (i < 0 || i >= M.length) return null;
  const a = M[i]; if (t - a[0] > 2.5 * eventSpacing() / 8 + 1e-9) return null;
  return { r: [a[1], a[2], a[3]], v: [a[4] / 1000, a[5] / 1000, a[6] / 1000] };
}
function drawMoonOrbit(center, ms, mAst) {   // osculating Earth-Moon ellipse (km) -> scene, x MOON_X
  if (!ms) { moonOrbGeom.setDrawRange(0, 0); return; }
  const mu = GM_EARTH_KM * (1 + M_MOON / M_EARTH + (mAst || 0) / M_EARTH);
  const r = ms.r, v = ms.v, rm = Math.hypot(...r), v2 = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
  const h = [r[1] * v[2] - r[2] * v[1], r[2] * v[0] - r[0] * v[2], r[0] * v[1] - r[1] * v[0]], hm = Math.hypot(...h);
  const eps = v2 / 2 - mu / rm; if (eps >= 0) { moonOrbGeom.setDrawRange(0, 0); return; }
  const a = -mu / (2 * eps);
  const ev = [(v[1] * h[2] - v[2] * h[1]) / mu - r[0] / rm, (v[2] * h[0] - v[0] * h[2]) / mu - r[1] / rm, (v[0] * h[1] - v[1] * h[0]) / mu - r[2] / rm];
  const e = Math.hypot(...ev);
  const P = e > 1e-6 ? ev.map((x) => x / e) : r.map((x) => x / rm);
  const hh = h.map((x) => x / hm);
  const Q = [hh[1] * P[2] - hh[2] * P[1], hh[2] * P[0] - hh[0] * P[2], hh[0] * P[1] - hh[1] * P[0]];
  const b = a * Math.sqrt(Math.max(0, 1 - e * e)), k = 1000 * MOON_X / AU * S;
  for (let j = 0; j <= 128; j++) {
    const E = j / 128 * 2 * Math.PI, x = a * (Math.cos(E) - e), y = b * Math.sin(E);
    moonOrbPos[3 * j] = center.x + (x * P[0] + y * Q[0]) * k; moonOrbPos[3 * j + 1] = center.y + (x * P[1] + y * Q[1]) * k;
    moonOrbPos[3 * j + 2] = center.z + (x * P[2] + y * Q[2]) * k;
  }
  moonOrbGeom.attributes.position.needsUpdate = true; moonOrbGeom.setDrawRange(0, 129);
}
function placeMoon(center, rel) {         // rel = Moon minus Earth [km]
  if (!rel) { moon.visible = false; moonLink.visible = false; placeLabel('moon', null, ''); return; }
  const k = 1000 * MOON_X / AU * S;
  moon.position.set(center.x + rel[0] * k, center.y + rel[1] * k, center.z + rel[2] * k); moon.visible = true;
  const lp = moonLinkGeom.attributes.position; lp.setXYZ(0, center.x, center.y, center.z); lp.setXYZ(1, moon.position.x, moon.position.y, moon.position.z);
  lp.needsUpdate = true; moonLink.visible = true;
  placeLabel('moon', moon.position, `Moon · ${int(Math.hypot(...rel))} km`, 8, 4, center);
}
function placeAsteroid(center, dir, frac, t) {
  if (!(frac > 0)) { asteroid.visible = false; placeLabel('ast', center, frac === 0 ? 'asteroid used up' : '', 10, 14); return; }
  const ra = Math.max(0.06, AST_RMAX * Math.cbrt(frac));
  asteroid.scale.setScalar(ra); asteroid.visible = true;
  asteroid.position.set(center.x + dir[0] * (EARTH_R + 0.8 * ra), center.y + dir[1] * (EARTH_R + 0.8 * ra), center.z + dir[2] * (EARTH_R + 0.8 * ra));
  asteroid.rotation.z = t * 3;
  placeLabel('ast', asteroid.position, `asteroid (docked) · ${(100 * frac).toFixed(frac < 0.1 ? 2 : 1)} % left`, 8, 6, center);
}
function muAU() { return (V.meta ? V.meta.mu : 1.3271284e20) * YEAR_S * YEAR_S / (AU * AU * AU); }
function elements(p, v) {
  const mu = muAU(), r = Math.hypot(...p), v2 = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
  const h = [p[1] * v[2] - p[2] * v[1], p[2] * v[0] - p[0] * v[2], p[0] * v[1] - p[1] * v[0]], hm = Math.hypot(...h);
  const eps = v2 / 2 - mu / r;
  const ev = [(v[1] * h[2] - v[2] * h[1]) / mu - p[0] / r, (v[2] * h[0] - v[0] * h[2]) / mu - p[1] / r, (v[0] * h[1] - v[1] * h[0]) / mu - p[2] / r];
  const e = Math.hypot(...ev);
  return { a: eps < 0 ? -mu / (2 * eps) : Infinity, e, inc: Math.acos(Math.max(-1, Math.min(1, h[2] / hm))) * 180 / Math.PI, eps, r, speed: Math.sqrt(v2) / KMS2AUY, ev, h, hm };
}

// ------------------------------------------------------------------ event log
const evLines = [];
function ev(t, text, cls = '') {
  evLines.unshift(`<div class="${cls}"><b>${fmtYear(t)}</b>: ${text}</div>`);
  if (evLines.length > 60) evLines.length = 60;
  evDirty = true;
}
let evDirty = false;
function processEvents(t, i) {
  const m0 = V.meta ? V.meta.m_ast0 : 1, E0 = V.meta ? V.meta.E0 : 1;
  let perPacket = speedY <= 1.0, shown = 0, fb = 0, mh = 0;
  for (let k = V.evRow + 1; k <= i && k < V.rows.length; k++) {
    const q = V.rows[k];
    if (q[R_DM] > 0) {
      if (!V.flags.first) { V.flags.first = true; ev(q[R_T], `first packet launched: ${sci(q[R_DM])} kg at ${int(q[R_U] / 1000)} km/s. Earth recoils by ${num(q[R_DV], 3)} m/s.`, 'sp'); }
      if (q[R_FATE] === 3) fb++;
      if (q[R_FATE] === 4) mh++;
      if (perPacket && shown < 12) {
        shown++;
        ev(q[R_T], `ejected ${sci(q[R_DM])} kg at ${int(q[R_U] / 1000)} km/s → ${FATE[q[R_FATE]]}; Earth Δv ${num(q[R_DV], 3)} m/s; asteroid left ${sci(q[R_MAST], 4)} kg (${pct(q[R_MAST] / m0, 2)})`, 'pk');
      }
      if (q[R_MAST] <= 0 && !V.flags.mass0) { V.flags.mass0 = true; ev(q[R_T], 'asteroid completely used up – no reaction mass left (resources exhausted).', 'bad'); }
      if (q[R_EREM] <= 1e-9 * E0 && !V.flags.e0) { V.flags.e0 = true; ev(q[R_T], 'propulsion energy budget used up (resources exhausted).', 'bad'); }
    }
    V.evRow = k;
  }
  if (perPacket && shown >= 12) ev(t, `(more packets this frame not listed – slow down to see each one)`, 'dim');
  if (fb) ev(t, `${fb} packet(s) FELL BACK to Earth: launched slower than escape speed (11.2 km/s), so no net push and energy wasted.`, 'bad');
  if (mh) ev(t, `${mh} packet(s) hit the Moon (momentum transferred to the Moon).`, 'sp');
  while (V.evDec < V.decisions.length && V.decisions[V.evDec].t_years <= t + 1e-9) {
    const d = V.decisions[V.evDec], prev = V.evDec > 0 ? V.decisions[V.evDec - 1] : null;
    const t0 = d.t_years - d.years;
    if (d.dm_kg > 0 && !perPacket) {
      ev(d.t_years, `years ${int(t0)}–${int(d.t_years)}: ${int(d.n_packets)} packets of ${sci(d.dm_kg)} kg at ${int(d.u_ms / 1000)} km/s, ` +
        `thrust ${dirWords(d.d_rtn)} (prograde cosine ${num(d.d_rtn[1], 2)}); Δv ${num(d.dv_ms, 1)} m/s; a = ${num(d.a_au)} AU, e = ${num(d.e)}; ` +
        `asteroid left ${pct(d.m_ast_frac)}, energy left ${pct(d.energy_frac)}`);
    }
    if (d.dm_kg <= 0 && (!prev || prev.dm_kg > 0) && V.policy !== 'none') ev(t0, 'agent stops ejecting (coasting).', 'sp');
    if (d.dm_kg > 0 && prev && prev.dm_kg <= 0) ev(t0, 'ejections resume.', 'sp');
    if (d.moon_bound === false && !V.flags.moon) { V.flags.moon = true; ev(d.t_years, 'MOON LOST – no longer bound to Earth.', 'bad'); }
    if (d.reason) {
      const txt = DOCS && DOCS.reason_docs[d.reason] ? DOCS.reason_docs[d.reason] : d.reason;
      ev(d.t_years, `MISSION END – ${d.reason}: ${txt} Final a = ${num(d.a_au)} AU, e = ${num(d.e)}.`, d.reason === 'success' ? 'ok' : 'bad');
    }
    V.evDec++;
  }
  if (V.done && V.final && V.final.validation && !V.flags.valid && V.rows.length && t >= V.rows[V.rows.length - 1][R_T] - 1e-9) {
    V.flags.valid = true; const vv = V.final.validation;
    ev(t, `stability check (${int(vv.years)} years without thrust, high-fidelity): ${vv.passed ? 'PASSED' : 'FAILED'} – worst |a − target| ${num(vv.max_abs_da_au)} AU, worst e ${num(vv.max_e)}, Moon ${vv.moon_bound ? 'bound' : 'LOST'}.`, vv.passed ? 'ok' : 'bad');
  }
  if (evDirty) { evDirty = false; $('evList').innerHTML = evLines.slice(0, 10).join(''); }
}

// ------------------------------------------------------------------ drawing helpers
function setTrailVertex(n, p, t) {
  trailPos[n * 3] = p[0] * S; trailPos[n * 3 + 1] = p[1] * S; trailPos[n * 3 + 2] = p[2] * S;
  const f = Math.max(0, Math.min(1, (t ?? 0) / (V.meta ? V.meta.mission_years : 10000)));
  trailCol[n * 3] = 0.10 + 0.55 * f; trailCol[n * 3 + 1] = 0.22 + 0.66 * f; trailCol[n * 3 + 2] = 0.50 + 0.50 * f;
}
function updateTrail(i, live) {
  if (V.trailN === 0 && V.trailSrv && V.trailSrv.length && V.rows.length) {
    const t0 = V.rows[0][R_T];     // joined mid-run: decimated path before the first received row
    for (const p of V.trailSrv) { if (V.trailN >= MAXTRAIL || p[0] >= t0) break; setTrailVertex(V.trailN++, p.slice(1), p[0]); }
  }
  const sub = V.rows.length * TRAIL_SUB > MAXTRAIL * 0.9 ? 1 : TRAIL_SUB;
  for (let k = V.trailUpTo + 1; k <= i; k++) {
    if (k > 0 && sub > 1) {
      const a = V.rows[k - 1], b = V.rows[k], h = b[R_T] - a[R_T];
      if (h > 0 && h <= 2.5 * eventSpacing() + 1e-9) {
        for (let j = 1; j < sub && V.trailN < MAXTRAIL; j++) setTrailVertex(V.trailN++, polarHermite(rowP(a), rowV(a, R_VPOST), rowP(b), rowV(b, R_VPRE), h, j / sub).p, a[R_T] + h * j / sub);
      }
    }
    if (V.trailN < MAXTRAIL) setTrailVertex(V.trailN++, rowP(V.rows[k]), V.rows[k][R_T]);
    V.trailUpTo = k;
  }
  setTrailVertex(V.trailN, live, V.disp);
  trailGeom.attributes.position.needsUpdate = true; trailGeom.attributes.color.needsUpdate = true; trailGeom.setDrawRange(0, V.trailN + 1);
}
let aheadT = 0;
function updateAhead(i) {
  const now = performance.now(); if (now - aheadT < 300) return; aheadT = now;
  const n = V.rows.length - 1 - i; if (n <= 0) { aheadGeom.setDrawRange(0, 0); return; }
  const st = Math.max(1, Math.ceil(n / AHEADMAX)); let c = 0;
  for (let k = i + 1; k < V.rows.length && c < AHEADMAX; k += st) { const q = V.rows[k]; aheadPos[c * 3] = q[R_X] * S; aheadPos[c * 3 + 1] = q[R_X + 1] * S; aheadPos[c * 3 + 2] = q[R_X + 2] * S; c++; }
  aheadGeom.attributes.position.needsUpdate = true; aheadGeom.setDrawRange(0, c);
}
let coastFrame = 0;
function updateCoast(p, v, el) {
  if ((coastFrame++ % 6) !== 0) return;
  if (!(el.e < 0.99) || !Number.isFinite(el.a)) { coastGeom.setDrawRange(0, 0); return; }
  const P = el.e > 1e-9 ? el.ev.map((x) => x / el.e) : p.map((x) => x / el.r);
  const W = el.h.map((x) => x / el.hm);
  const Q = [W[1] * P[2] - W[2] * P[1], W[2] * P[0] - W[0] * P[2], W[0] * P[1] - W[1] * P[0]];
  const b = el.a * Math.sqrt(1 - el.e * el.e);
  for (let k = 0; k <= 256; k++) {
    const E = k / 256 * 2 * Math.PI, x = el.a * (Math.cos(E) - el.e), y = b * Math.sin(E);
    for (let c = 0; c < 3; c++) coastPos[k * 3 + c] = (x * P[c] + y * Q[c]) * S;
  }
  coastGeom.attributes.position.needsUpdate = true; coastGeom.setDrawRange(0, 257); coast.computeLineDistances();
}
function astOffsetDir(i) {        // direction of the latest ejection at or before row i
  for (let k = Math.min(i, V.rows.length - 1); k >= 0 && k > i - 4000; k--) { const q = V.rows[k]; if (q[R_DM] > 0) return [q[R_EHAT], q[R_EHAT + 1], q[R_EHAT + 2]]; }
  const q = V.rows[Math.max(0, Math.min(i, V.rows.length - 1))]; if (!q) return [1, 0, 0];
  const v = [q[R_VPRE], q[R_VPRE + 1], q[R_VPRE + 2]], m = Math.hypot(...v) || 1; return v.map((x) => -x / m);
}
const PKCOL = { 1: [1.0, 0.6, 0.3], 2: [1.0, 0.88, 0.3], 3: [1.0, 0.35, 0.35], 4: [0.75, 0.75, 0.75] };
let pkNote = '';
function updatePackets(t) {
  const off = V.meta ? V.meta.offsets_days : [0];
  const persist = Math.max(PERSIST_WALL_S * speedY, 2 * DAY_Y);
  // drop records that can no longer be visible
  let cut = 0; while (cut < V.packets.length && V.packets[cut][0] + 1.0 + persist < t) cut++;
  if (cut > 200) V.packets.splice(0, cut);
  const lo = bsearch(V.packets, t - 1.0 - persist) + 1, hi = bsearch(V.packets, t);
  const cand = Math.max(0, hi - lo + 1);
  const step = Math.max(1, Math.ceil(cand / MAXPK));
  pkNote = (step > 1 || V.pstrideRx > 1) ? `Packets: drawing 1 of every ${step * V.pstrideRx} at this speed (all are in the physics).` : '';
  let ns = 0, nh = 0;
  for (let j = lo; j <= hi && j >= 0; j += step) {
    const q = V.packets[j]; const tp = q[0], fate = q[1], n = q[4];
    const ageD = (t - tp) * 365.25; if (ageD < 0) continue;
    const lifeD = off[n - 1] || 0;
    const fade = ageD <= lifeD ? 1 : 1 - (ageD - lifeD) / (persist * 365.25);
    if (fade <= 0.02) continue;
    const col = PKCOL[fate] || [1, 1, 1];
    // visual offset: shift the path so it starts at the drawn asteroid (AST_OFFSET_AU from Earth's centre)
    let ox = 0, oy = 0, oz = 0;
    if (n >= 2) { const dx = q[8] - q[5], dy = q[9] - q[6], dz = q[10] - q[7], m = Math.hypot(dx, dy, dz) || 1; ox = dx / m * AST_OFFSET_AU; oy = dy / m * AST_OFFSET_AU; oz = dz / m * AST_OFFSET_AU; }
    if (n === 1) {             // fell back / hit Moon: marker at Earth
      if (nh < MAXPK) { headPos.set([(q[5] + ox) * S, (q[6] + oy) * S, (q[7] + oz) * S], nh * 3); headCol.set(col.map((c) => c * fade), nh * 3); nh++; }
      continue;
    }
    const aD = Math.min(ageD, lifeD);
    let last = null;
    for (let k = 0; k < n - 1 && ns < MAXSEG; k++) {
      if (off[k] > aD) break;
      const f = off[k + 1] <= aD ? 1 : (aD - off[k]) / (off[k + 1] - off[k]);
      const b0 = 5 + 3 * k, b1 = b0 + 3;
      const x0 = q[b0] + ox, y0 = q[b0 + 1] + oy, z0 = q[b0 + 2] + oz;
      const x1 = q[b0] + (q[b1] - q[b0]) * f + ox, y1 = q[b0 + 1] + (q[b1 + 1] - q[b0 + 1]) * f + oy, z1 = q[b0 + 2] + (q[b1 + 2] - q[b0 + 2]) * f + oz;
      pkPos.set([x0 * S, y0 * S, z0 * S, x1 * S, y1 * S, z1 * S], ns * 6);
      const c = col.map((v) => v * fade * 0.85);
      pkCol.set([...c, ...c], ns * 6); ns++; last = [x1, y1, z1];
    }
    if (last && ageD <= lifeD && nh < MAXPK) { headPos.set(last.map((x) => x * S), nh * 3); headCol.set(col, nh * 3); nh++; }
  }
  pkGeom.attributes.position.needsUpdate = true; pkGeom.attributes.color.needsUpdate = true; pkGeom.setDrawRange(0, ns * 2);
  headGeom.attributes.position.needsUpdate = true; headGeom.attributes.color.needsUpdate = true; headGeom.setDrawRange(0, nh);
}
function trimMoon(t) { let c = 0; while (c < V.moon.length - 2 && V.moon[c + 1][0] < t - 0.05) c++; if (c > 500) V.moon.splice(0, c); }

// ------------------------------------------------------------------ HUD
function tip(key) {
  const o = DOCS && DOCS.outputs[key]; if (!o) return '';
  return esc(`${o.label} [${o.unit}] — ${o.measures}\nHow: ${o.how}\nType: ${o.kind}\nDesirable: ${o.better}\nMeaning for success: ${o.success}`);
}
function hrow(key, label, value, cls = '') { return `<div class="row ${cls}" title="${tip(key)}"><span>${label}</span><span>${value}</span></div>`; }
function decAt(t) { let d = null; for (let k = V.decisions.length - 1; k >= 0; k--) if (V.decisions[k].t_years <= t + 1e-9) { d = V.decisions[k]; break; } return d; }
function decActive(t) { for (const d of V.decisions) if (d.t_years > t + 1e-9) return d; return null; }
function renderHud(t, st, el, i) {
  const wall = (performance.now() - V.wallStart) / 1000;
  let h = '';
  const modeTxt = V.mode === 'idle' ? 'Nothing running – choose a policy and press ▶ Play' :
    V.mode === 'training' ? `LIVE TRAINING · ${esc(V.label)} · path drawn as points, not animated` : `MISSION PLAYBACK · ${esc(V.label)}`;
  h += `<div>${modeTxt}</div>`;
  if (V.mode === 'playback') {
    const dayOfYear = Math.floor((t % 1) * 365.25);
    h += `<div title="Simulated time since the start of the mission. It is NOT wall-clock time."><span class="big">${fmtYear(t)}</span> <span class="dim">(day ${dayOfYear}) simulated</span></div>`;
    h += `<div class="dim" title="Real time since this playback started, and the time multiplier.">wall clock ${wall.toFixed(0)} s · speed ${fmtSpeed(speedY)}/s · ${V.playing ? 'playing' : 'PAUSED'}</div>`;
  }
  if (V.mode === 'training' && V.final) {
    const f = V.final;
    h += `<div class="sec">Result of this exploring episode</div>` + hrow('a', 'final a', `${num(f.a_au)} AU`) + hrow('e', 'final e', num(f.e)) +
      hrow('m_ast', 'asteroid used', pct(f.mass_used_frac)) + hrow('E_used', 'energy used', pct(f.energy_used_frac)) +
      `<div class="row"><span>outcome</span><span>${esc(f.reason)} after ${int(f.t_years)} yr</span></div>`;
  }
  if (V.mode === 'playback' && st && el) {
    const q = V.rows[Math.min(i, V.rows.length - 1)], m0 = V.meta.m_ast0, E0 = V.meta.E0;
    const mast = q[R_MAST], eRem = q[R_EREM];
    const Memb = M_EARTH + M_MOON + mast;
    const epsSI = el.eps * (AU / YEAR_S) ** 2;
    h += `<div class="sec">Orbit (from the simulated state)</div>`;
    h += hrow('a', 'size a', `${num(el.a)} AU <span class="dim">(target ${V.meta.target_au} ± ${V.meta.tol_a_au})</span>`);
    h += hrow('e', 'eccentricity e', `${num(el.e)} <span class="dim">(≤ ${V.meta.tol_e})</span>`);
    h += hrow('inc', 'inclination', `${num(el.inc, 3)}°`);
    h += hrow('r', 'distance from Sun', `${num(el.r)} AU`);
    h += hrow('speed', 'orbital speed', `${num(el.speed, 3)} km/s`);
    h += hrow('E_orb', 'orbital energy', `${sci(epsSI * Memb, 4)} J`);
    h += `<div class="sec">Propulsion</div>`;
    const lastEj = V.lastEj >= 0 ? V.rows[V.lastEj] : null;
    h += hrow('packets', 'packets ejected', int(V.cumPk[i] ?? 0));
    h += hrow('recoil', 'last packet', lastEj ? `${sci(lastEj[R_DM])} kg @ ${int(lastEj[R_U] / 1000)} km/s` : '—');
    h += hrow('recoil', 'recoil per packet', lastEj ? `${num(lastEj[R_DV], 3)} m/s` : '—');
    const dvc = V.cumDv[i] ?? 0;
    h += hrow('dv', 'total Δv delivered', `${dvc < 1000 ? num(dvc, 1) : int(dvc)} m/s <span class="dim">(need ≈ 11,700)</span>`);
    const da = decActive(t);
    h += hrow('thrust', 'thrust (interval avg.)', da ? `${sci(da.thrust_N)} N ${da.dm_kg > 0 ? dirWords(da.d_rtn) : '(coasting)'}` : '—');
    h += `<div class="sec">Resources</div>`;
    h += hrow('m_ast', 'asteroid left', `${sci(mast, 3)} kg (${pct(mast / m0)})`, mast <= 0 ? 'warn' : '');
    h += hrow('E_used', 'energy used', `${sci(E0 - eRem, 3)} J (${pct((E0 - eRem) / E0)})`, eRem <= 0 ? 'warn' : '');
    h += `<div class="sec">Moon</div>`;
    {
      const mr = moonAt(t);
      h += hrow('moon', 'Earth–Moon distance', mr ? `${int(Math.hypot(...mr))} km` : 'no data at this time');
      const d = decAt(t);
      h += hrow('moon', 'Moon bound', d ? (d.moon_bound ? 'yes' : '<span class="warn">NO</span>') : 'yes (start)');
      h += `<div class="dim small">${V.fidelity === 'hifi' ? 'N-body: Sun, Earth(+asteroid), Moon and fragments all pull on each other.' : 'Fast model: exact Earth–Moon two-body orbit; the Sun\'s tide on the Moon is neglected.'}</div>`;
      if (d && d.conservation) {
        h += `<div class="sec">Numerical accuracy (end of last decision)</div>`;
        h += hrow('cons_E', 'energy error', sci(d.conservation.energy_residual_rel, 1));
        h += hrow('cons_P', 'momentum error', sci(d.conservation.momentum_residual_rel, 1));
        h += hrow('cons_M', 'mass error', sci(d.conservation.mass_residual_rel, 1));
      }
    }
    const d = decAt(t);
    h += `<div class="sec">Training score</div>` + hrow('return', 'cumulative reward', d ? num(d.cum_reward, 2) + ' <span class="dim">(not a physical quantity)</span>' : '—');
    if (V.done && V.final && t >= V.rows[V.rows.length - 1][R_T] - 1e-9) {
      h += `<div class="${V.final.success ? 'pass' : 'fail'}">END: ${esc(V.final.reason)}${V.final.validation ? ` · stability check ${V.final.validation.passed ? 'passed' : 'FAILED'}` : ''}</div>`;
    }
  }
  $('hud').innerHTML = h;
}
function renderNotes(st) {
  const n = [];
  if (V.mode === 'playback') {
    if (V.fidelity === 'fast') n.push('Fast model (what the agent trained on): Moon simulated as an exact two-body orbit around Earth; the Sun\'s tidal pull on the Moon is neglected.');
    if (st && st.gap) n.push('Data gap: Earth is held at the last recorded state (no interpolation across a gap).');
    if (V.waiting) n.push('Waiting for the physics engine…');
    if (pkNote) n.push(pkNote);
    n.push(`Not to scale: Earth, Sun and asteroid markers enlarged; Moon distance ×100; packet paths shifted ${AST_OFFSET_AU.toFixed(3)} AU to start at the asteroid marker.`);
    n.push('Packet paths: two-body solar gravity, drawn for ≤ 1 year or 60 AU.');
  }
  $('notes').innerHTML = n.map((x) => `<div>${esc(x)}</div>`).join('');
  $('notes').style.display = n.length ? 'block' : 'none';
}
const _v = new THREE.Vector3();
function placeDvLabel(text) {
  const el = $('dvLabel');
  if (!text) { el.style.display = 'none'; return; }
  _v.copy(recoil.position).add(new THREE.Vector3().copy(recoil.userData.dir).multiplyScalar(1.5)).project(camera);
  const w = canvas.clientWidth, h = canvas.clientHeight;
  if (_v.z > 1) { el.style.display = 'none'; return; }
  el.style.display = 'block'; el.style.left = `${(_v.x * 0.5 + 0.5) * w + 6}px`; el.style.top = `${(-_v.y * 0.5 + 0.5) * h - 8}px`; el.textContent = text;
}

// ------------------------------------------------------------------ frame loop
let lastFrame = performance.now();
function frame() {
  const now = performance.now(); const dt = Math.min(0.1, (now - lastFrame) / 1000); lastFrame = now;
  resize();
  trailLine.visible = V.mode !== 'training'; trailDots.visible = V.mode === 'training';
  let st = null, el = null, i = 0;
  if (V.mode === 'training' && V.rows.length) {
    if (V.trailUpTo !== V.rows.length) {
      V.trailN = 0; V.rows.forEach((q, k) => { if (V.trailN < MAXTRAIL) setTrailVertex(V.trailN++, [q[1], q[2], q[3]], 10000 * k / V.rows.length); });
      trailGeom.attributes.position.needsUpdate = true; trailGeom.attributes.color.needsUpdate = true; trailGeom.setDrawRange(0, V.trailN); V.trailUpTo = V.rows.length;
    }
    const q = V.rows[V.rows.length - 1]; earth.position.set(q[1] * S, q[2] * S, q[3] * S);
    const f = V.final || {};
    placeMoon(earth.position, f.moon_rel_km || null); moonOrbGeom.setDrawRange(0, 0);
    const q0 = V.rows[Math.max(0, V.rows.length - 2)], dd = [q0[1] - q[1], q0[2] - q[2], q0[3] - q[3]], dm = Math.hypot(...dd) || 1;
    placeAsteroid(earth.position, dd.map((x) => x / dm), f.mass_used_frac !== undefined ? Math.max(0, 1 - f.mass_used_frac) : 1, 0);
    recoil.visible = false; placeDvLabel('');
    coastGeom.setDrawRange(0, 0); aheadGeom.setDrawRange(0, 0); pkGeom.setDrawRange(0, 0); headGeom.setDrawRange(0, 0);
  } else if (V.mode === 'playback' && V.rows.length && V.meta) {
    const tEnd = V.rows[V.rows.length - 1][R_T];
    if (V.disp < V.rows[0][R_T]) V.disp = V.rows[0][R_T];
    if (V.playing) V.disp = Math.min(V.disp + speedY * dt, tEnd);
    V.waiting = V.playing && !V.done && V.disp >= tEnd - 1e-12;
    const t = V.disp;
    st = earthAt(t); i = st.i;
    // latest ejection at or before t
    for (let k = V.ejScan + 1; k <= i; k++) if (V.rows[k][R_DM] > 0) V.lastEj = k;
    V.ejScan = Math.max(V.ejScan, i);
    const p = st.p; earth.position.set(p[0] * S, p[1] * S, p[2] * S);
    el = elements(p, st.v);
    updateTrail(i, p); updateAhead(i); updateCoast(p, st.v, el);
    // asteroid: size ~ cbrt(remaining mass)
    const q = V.rows[i], frac = Math.max(0, q[R_MAST] / V.meta.m_ast0);
    placeAsteroid(earth.position, astOffsetDir(i), frac, t);
    // Moon (simulated in both models): position from its own recorded stream + its current orbit
    placeMoon(earth.position, moonAt(t));
    drawMoonOrbit(earth.position, moonStateAt(t), q[R_MAST]);
    trimMoon(t);
    // recoil arrow: direction of v_post - v_pre of the latest ejection, shown while ejections are recent
    const lj = V.lastEj >= 0 ? V.rows[V.lastEj] : null;
    const recent = lj && (t - lj[R_T]) <= Math.max(1.5 * eventSpacing(), PERSIST_WALL_S * speedY);
    if (recent && lj[R_DV] > 0) {
      const d = [lj[R_VPOST] - lj[R_VPRE], lj[R_VPOST + 1] - lj[R_VPRE + 1], lj[R_VPOST + 2] - lj[R_VPRE + 2]], m = Math.hypot(...d) || 1;
      const dir = new THREE.Vector3(d[0] / m, d[1] / m, d[2] / m);
      recoil.position.copy(earth.position); recoil.setDirection(dir); recoil.userData.dir = dir; recoil.visible = true;
      placeDvLabel(`recoil ${num(lj[R_DV], 3)} m/s per packet · total Δv ${V.cumDv[i] < 1000 ? num(V.cumDv[i], 1) : int(V.cumDv[i])} m/s`);
    } else { recoil.visible = false; placeDvLabel(''); }
    updatePackets(t);
    processEvents(t, i);
  } else {
    // nothing running: show the simulated START state (year 0) of the mission, if known
    const I = V.init;
    if (I) earth.position.set(I.earth[0] * S, I.earth[1] * S, I.earth[2] * S); else earth.position.set(S, 0, 0);
    placeMoon(earth.position, I ? I.moon_rel_km : null);
    drawMoonOrbit(earth.position, I ? { r: I.moon_rel_km, v: I.moon_v_kms } : null, I ? I.m_ast0 : 0);
    placeAsteroid(earth.position, I ? I.ast_dir : [-1, 0, 0], 1, 0);
    recoil.visible = false; placeDvLabel('');
  }
  if (V.mode !== 'playback' && V.mode !== 'idle') moonOrbGeom.setDrawRange(0, 0);
  placeLabel('sun', sun.position, 'Sun', 12, -10);
  placeLabel('earth', earth.position, 'Earth', 8, 8, asteroid.visible ? asteroid.position : sun.position);
  resolveLabels();
  if ($('follow').checked) {        // keep the camera's offset to Earth constant
    const d = new THREE.Vector3().subVectors(earth.position, controls.target);
    controls.target.add(d); camera.position.add(d);
  }
  renderHud(V.disp, st, el, i); renderNotes(st);
  controls.update();
  renderer.render(scene, camera);
  requestAnimationFrame(frame);
}
requestAnimationFrame(frame);

// ------------------------------------------------------------------ API
async function api(path, body) {
  const opt = body === undefined ? {} : { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) };
  const r = await fetch(path, opt);
  const j = await r.json();
  if (!r.ok || j.ok === false) throw new Error(j.error || r.statusText);
  return j;
}
function toast(msg) { const l = $('log'); l.textContent = `>> ${msg}\n` + l.textContent; }

function ingest(v) {
  if (v.meta) { V.meta = v.meta; setTargetRing(v.meta.target_au); }
  if (v.trail) V.trailSrv = v.trail;
  if (v.dstart === V.decisions.length) for (const d of v.decisions) V.decisions.push(d);
  if (v.rows && v.rows.length && V.mode !== 'training') {
    if (v.rstart !== V.since && V.rows.length) toast(`data window skipped ${v.rstart - V.since} rows (display was far behind)`);
    if (!V.rows.length && v.rstart > 0) {      // joined mid-run: start counters from the decision records
      const t0 = v.rows[0][R_T]; let pk = 0, dv = 0;
      for (const d of V.decisions) if (d.t_years <= t0) { pk += d.n_packets || 0; dv = d.dv_cum_ms || dv; }
      V.cumPk0 = pk; V.cumDv0 = dv;
    }
    for (const q of v.rows) {
      const k = V.rows.length;
      V.cumPk.push((k ? V.cumPk[k - 1] : (V.cumPk0 || 0)) + (q[R_DM] > 0 ? 1 : 0));
      V.cumDv.push((k ? V.cumDv[k - 1] : (V.cumDv0 || 0)) + (q[R_DV] || 0));
      V.rows.push(q);
    }
    V.since = v.rstart + v.rows.length;
  } else if (v.rows && V.mode === 'training') { V.rows = v.rows; }
  if (v.moon && v.moon.length) { for (const m of v.moon) V.moon.push(m); V.msince = v.mstart + v.moon.length; }
  if (v.pnext !== undefined && v.pnext > V.psince) { for (const p of v.packets) V.packets.push(p); V.psince = v.pnext; V.pstrideRx = v.pstride || 1; }
}

let lastEpisode = -1, lastCompare = '', resultsDirty = true;
async function pollState() {
  try {
    const pstride = speedY > 2 ? Math.ceil(speedY / 2) : 1;
    const q = `run=${V.run}&since=${V.since}&msince=${V.msince}&psince=${V.psince}&dsince=${V.decisions.length}&pstride=${pstride}` +
      (V.mode === 'playback' ? `&disp=${V.disp.toFixed(4)}&ahead=${Math.min(1500, 2.5 * speedY).toFixed(2)}` : '');
    const st = await api('/api/state?' + q);
    $('conn').textContent = 'connected'; $('conn').className = 'pill ok';
    const v = st.viewer;
    if (v.run_id !== V.run || v.reset) { resetViewer(); V.run = v.run_id; }
    if (st.init) V.init = st.init;
    V.mode = v.mode; V.label = v.label; V.done = v.done; V.final = v.final; V.fidelity = v.fidelity; V.policy = v.policy; V.produced = v.produced_years;
    ingest(v);
    renderTrain(st);
    $('log').textContent = st.log.slice().reverse().join('\n');
    if (Math.floor(st.train.episode / 10) !== Math.floor(lastEpisode / 10) || st.compare_state !== lastCompare) resultsDirty = true;
    lastEpisode = st.train.episode; lastCompare = st.compare_state;
  } catch (e) {
    $('conn').textContent = 'disconnected'; $('conn').className = 'pill off';
  }
  setTimeout(pollState, V.mode === 'playback' ? 250 : 700);
}
function renderTrain(st) {
  const t = st.train;
  $('trainStatus').textContent = `${t.state} · episode ${t.episode}/${t.target} · ${t.timesteps} decisions · ${(t.wall_s || 0).toFixed(0)} s wall` +
    (t.eps_per_s ? ` · ${t.eps_per_s.toFixed(1)} episodes/s` : '') + (t.error ? `\nERROR: ${t.error}` : '') +
    (t.state === 'idle' && st.best ? '\n(showing saved results of a previous run)' : '');
  $('trainBar').style.width = `${Math.min(100, 100 * t.episode / Math.max(1, t.target))}%`;
  $('btnTrain').disabled = t.state === 'running' || st.checks_ok === false;
  $('btnStop').disabled = t.state !== 'running';
  $('btnCompare').disabled = st.compare_state === 'running';
  $('btnCompare').textContent = st.compare_state === 'running' ? 'Comparing…' : 'Compare policies';
  const b = st.best;
  $('bestBox').innerHTML = b ? `<b>best_model.zip</b> = episode ${b.episode}: deterministic test success ${pct(b.eval.success_rate, 0)}, a = ${num(b.eval.a_au)} AU, ` +
    `e = ${num(b.eval.e)}, energy ${pct(b.eval.energy_frac)}, ${int(b.eval.years)} yr. <span class="dim">Best of the tested checkpoints – not a global optimum.</span>` : 'best model: N/A (not trained yet)';
  $('feasPill').textContent = st.feasible ? 'FEASIBLE (in this model)' : 'INFEASIBLE';
  $('feasPill').className = 'pill ' + (st.feasible ? 'ok' : 'bad'); $('feasPill').title = st.verdict + '\n(see Guide → Feasibility)';
  $('checksPill').textContent = st.checks_ok === null ? 'checks …' : st.checks_ok ? 'physics checks PASS' : 'physics checks FAIL';
  $('checksPill').className = 'pill ' + (st.checks_ok ? 'ok' : st.checks_ok === false ? 'bad' : '');
  const L = t.losses || {};
  $('losses').textContent = Object.keys(L).length ? Object.entries(L).map(([k, x]) => `${k} = ${(+x).toPrecision(3)}`).join(' · ') + `  (update ${t.ppo_iter})` : 'no PPO update yet';
}
async function pollResults() {
  if (resultsDirty) { resultsDirty = false; try { renderResults(await api('/api/results')); } catch (e) { resultsDirty = true; } }
  setTimeout(pollResults, 1500);
}

// ------------------------------------------------------------------ charts (plain canvas)
function chart(cv, series, opt = {}) {
  const dpr = window.devicePixelRatio || 1, W = cv.clientWidth, H = cv.clientHeight;
  if (!W) return;
  cv.width = W * dpr; cv.height = H * dpr;
  const g = cv.getContext('2d'); g.scale(dpr, dpr); g.clearRect(0, 0, W, H);
  const L = 46, R = 8, T = 30, B = 18;
  const xs = [], ys = [];
  for (const s of series) for (const p of s.data) if (p[1] !== null && Number.isFinite(p[1])) { xs.push(p[0]); ys.push(p[1]); }
  if (opt.hline !== undefined) ys.push(opt.hline);
  g.fillStyle = '#9fb6d6'; g.font = '11px system-ui'; g.fillText(opt.title || '', L, 11);
  if (!xs.length) { g.fillStyle = '#7d8aa0'; g.fillText('N/A (no data yet)', L + 10, H / 2); return; }
  let x0 = Math.min(...xs), x1 = Math.max(...xs), y0 = opt.ymin ?? Math.min(...ys), y1 = opt.ymax ?? Math.max(...ys);
  if (x1 === x0) x1 = x0 + 1; if (y1 === y0) { y1 += 1; y0 -= 1; }
  const X = (x) => L + (x - x0) / (x1 - x0) * (W - L - R), Y = (y) => T + (1 - (y - y0) / (y1 - y0)) * (H - T - B);
  g.strokeStyle = '#1f2a3b'; g.beginPath(); g.moveTo(L, T); g.lineTo(L, H - B); g.lineTo(W - R, H - B); g.stroke();
  g.fillStyle = '#7d8aa0'; g.font = '10px system-ui';
  const fmt = (v) => Math.abs(v) >= 1e4 || (Math.abs(v) < 0.01 && v !== 0) ? v.toExponential(1) : (+v.toFixed(3)).toString();
  g.fillText(fmt(y1), 2, T + 4); g.fillText(fmt(y0), 2, H - B); g.fillText(fmt(x0), L, H - 4);
  g.fillText(fmt(x1) + (opt.xlabel ? ' ' + opt.xlabel : ''), W - R - 80, H - 4);
  if (opt.hline !== undefined) { g.strokeStyle = '#3ddc84'; g.setLineDash([4, 3]); g.beginPath(); g.moveTo(L, Y(opt.hline)); g.lineTo(W - R, Y(opt.hline)); g.stroke(); g.setLineDash([]); }
  let lx = L;
  for (const s of series) {
    g.strokeStyle = g.fillStyle = s.color; g.globalAlpha = s.alpha ?? 1;
    if (s.dash) g.setLineDash(s.dash);
    if (s.dots) { for (const p of s.data) if (p[1] !== null) g.fillRect(X(p[0]) - 1, Y(p[1]) - 1, s.size || 2, s.size || 2); }
    else if (s.bars) { for (const p of s.data) if (p[1] !== null) g.fillRect(X(p[0]) - 1, Y(Math.max(p[1], y0)), 2.5, Y(y0) - Y(Math.max(p[1], y0))); }
    else { g.beginPath(); let on = false; for (const p of s.data) { if (p[1] === null) { on = false; continue; } on ? g.lineTo(X(p[0]), Y(p[1])) : g.moveTo(X(p[0]), Y(p[1])); on = true; } g.stroke(); }
    g.setLineDash([]); g.globalAlpha = 1;
    if (s.label) { g.fillText(s.label, lx, 24); lx += g.measureText(s.label).width + 12; }
  }
}
const movavg = (a, n) => a.map((_, i) => { const w = a.slice(Math.max(0, i - n + 1), i + 1).filter((v) => v !== null); return w.length ? w.reduce((x, y) => x + y, 0) / w.length : null; });

// ------------------------------------------------------------------ results rendering
const TH = (label, title, cls = '') => `<th class="${cls}" title="${esc(title)}">${label}</th>`;
let lastR = null;
function renderResults(r) {
  lastR = r;
  DOCS = { outputs: Object.fromEntries(r.output_docs.map((o) => [o.key, o])), reason_docs: r.reason_docs };
  setTargetRing(r.config.target_au);
  buildParams(r);
  $('speedInfo').title = docText(r.speed_doc);
  // training curves
  const cfg = r.config, h = r.history, ep = h.map((x) => x.ep), rew = h.map((x) => x.reward);
  chart($('chReward'), [
    { data: h.map((x, i) => [ep[i], rew[i]]), color: '#4fa3ff', dots: true, alpha: 0.35, label: 'episode return (exploring)' },
    { data: movavg(rew, 50).map((v, i) => [ep[i], v]), color: '#ffffff', label: 'avg of 50' },
    { data: r.evals.map((e) => [e.episode, e.reward]), color: '#ffcc33', dots: true, size: 4, label: 'deterministic test' },
  ], { title: 'Training score: return per episode [points]', xlabel: 'episode' });
  chart($('chA'), [
    { data: h.map((x, i) => [ep[i], x.a]), color: '#7cc4ff', dots: true, alpha: 0.35, label: 'final a (exploring)' },
    { data: r.evals.map((e) => [e.episode, e.a_au]), color: '#ffcc33', label: 'deterministic test' },
  ], { title: 'Physics: final orbit size a [AU] (green = target)', hline: cfg.target_au, xlabel: 'episode' });
  chart($('chRes'), [
    { data: movavg(h.map((x) => x.mass), 50).map((v, i) => [ep[i], v]), color: '#ff7a45', label: 'asteroid used' },
    { data: movavg(h.map((x) => x.energy), 50).map((v, i) => [ep[i], v]), color: '#c084fc', label: 'energy used' },
    { data: movavg(h.map((x) => (x.success ? 1 : 0)), 50).map((v, i) => [ep[i], v]), color: '#3ddc84', label: 'success rate' },
  ], { title: 'Physics: resources used and success (fraction, avg of 50)', ymin: 0, ymax: 1, xlabel: 'episode' });
  const oc = r.training_outcomes || {}, tot = Object.values(oc).reduce((a, b) => a + b, 0);
  $('outcomes').innerHTML = tot ? '<table>' + Object.entries(oc).map(([k, n]) => `<tr><td title="${esc(r.reason_docs[k] || '')}">${esc(k)}</td><td>${n}</td><td>${pct(n / tot)}</td><td class="l dim">${esc(r.reason_docs[k] || '')}</td></tr>`).join('') + '</table>' : '<span class="dim">N/A – no training episodes in this session yet.</span>';
  renderComparison(r);
  renderGuide(r);
}

function renderComparison(r) {
  const c = r.comparison, cfg = r.config;
  if (!c) { $('verdict').className = 'verdict'; $('verdict').textContent = 'No comparison yet (runs automatically after training, or press “Compare policies”).'; $('cmpTable').innerHTML = ''; $('bestPanel').innerHTML = ''; chart($('chTraj'), [], { title: 'a(t)' }); return; }
  const rows = c.rows, ppo = rows.find((x) => x.policy === 'ppo');
  const vd = $('verdict');
  if (!ppo) { vd.className = 'verdict'; vd.textContent = 'No trained model – baselines only.'; }
  else if (ppo.mission_success) {
    vd.className = 'verdict ok';
    vd.textContent = `Target reached by the best PPO model in the high-fidelity N-body run, and the orbit stayed stable for ${cfg.validate_years} years without thrust: a = ${num(ppo.hifi.a_au)} AU, e = ${num(ppo.hifi.e)} after ${int(ppo.hifi.t_years)} simulated years.`;
  } else {
    vd.className = 'verdict bad';
    vd.textContent = `Target NOT reached by the best PPO model in the high-fidelity run (${ppo.hifi.reason}): a = ${num(ppo.hifi.a_au)} AU (short by ${num(ppo.hifi.shortfall_au)} AU), e = ${num(ppo.hifi.e)}.`;
  }
  let t = '<table><tr>' + TH('policy', 'Decision rule that controls the ejections.') + TH('fast<br>success', 'Success rate on the fast model, 3 starting seeds.') +
    TH('hi-fi<br>outcome', 'How the high-fidelity N-body run ended.') + TH('a [AU]', 'Final semi-major axis (orbit size), high-fidelity.') + TH('e', 'Final eccentricity, high-fidelity.') +
    TH('years', 'Simulated years until the mission ended.') + TH('asteroid<br>used', 'Fraction of the asteroid thrown away.') + TH('energy<br>[J]', 'Propulsion energy drawn.') +
    TH('sunlight', 'Insolation at the final orbit relative to today (estimate).') + TH('stable', 'Passed the ballistic stability check (high-fidelity).') + '</tr>';
  for (const x of rows) {
    t += `<tr><td>${esc(x.label)}</td><td>${pct(x.fast.success_rate, 0)}</td><td>${esc(x.hifi.reason)}</td><td>${num(x.hifi.a_au)}</td><td>${num(x.hifi.e)}</td><td>${int(x.hifi.t_years)}</td>` +
      `<td>${pct(x.hifi.mass_used_frac)}</td><td>${sci(x.hifi.energy_used_J)}</td><td>${num(x.hifi.insolation_rel, 3)}×</td><td class="${x.mission_success ? 'pass' : 'fail'}">${x.mission_success ? '✔' : '✘'}</td></tr>`;
  }
  $('cmpTable').innerHTML = t + '</table><div class="small dim">Physical results only (reward is a training score and is listed separately in the best-model section). High-fidelity = N-body Sun, Earth, Moon, fragments; seed 2001.</div>';
  const col = { none: '#888', heuristic: '#ffcc33', ppo: '#4fa3ff' }, ser = [];
  for (const [k, tr] of Object.entries(c.traces)) {
    ser.push({ data: [[0, 1]].concat(tr.fast.map((d) => [d.t, d.a])), color: col[k], label: k });
    ser.push({ data: [[0, 1]].concat(tr.hifi.map((d) => [d.t, d.a])), color: col[k], dash: [3, 3], alpha: 0.7 });
  }
  chart($('chTraj'), ser, { title: 'Orbit size a(t) [AU] — solid fast model, dashed high-fidelity', hline: cfg.target_au, xlabel: 'yr' });
  renderBest(c, r);
}

function renderBest(c, r) {
  const bp = c.best_panel, cfg = r.config;
  if (!bp) { $('bestPanel').innerHTML = '<div class="note">This comparison was produced by an older version and has no best-model details. Press “Compare policies” to regenerate it.</div>'; return; }
  let h = '<h3>Best model</h3>';
  const f = bp.file, sel = bp.selection;
  h += `<div class="card small"><b>File:</b> ${esc(f.name)} ${f.exists ? '' : '<span class="fail">(missing)</span>'}<br><span class="dim">${esc(f.path)}</span><br>` +
    (f.exists ? `${f.size_kb} kB · saved ${esc(f.modified)}<br>` : '') +
    (sel ? `<b>Chosen at</b> training episode ${sel.episode} (${int(sel.timesteps)} decisions).<br><b>Selection rule:</b> ${esc(sel.selection)}.<br>` +
      `<b>Test at selection:</b> success ${pct(sel.eval.success_rate, 0)}, a = ${num(sel.eval.a_au)} AU, e = ${num(sel.eval.e)}, energy ${pct(sel.eval.energy_frac)}, ${int(sel.eval.years)} yr.<br>` +
      `<b>Network &amp; settings:</b> <span class="mono">${esc(Object.entries(sel.hyperparameters || {}).map(([k, v]) => `${k}=${Array.isArray(v) ? v.join('x') : v}`).join(', '))}</span>` : 'Selection info: N/A') + '</div>';
  h += bp.disclaimer.map((d) => `<div class="note">${esc(d)}</div>`).join('');
  // evals
  if (r.evals && r.evals.length) {
    h += '<h4>Deterministic tests during training <i class="info" title="Every few episodes the current policy is tested without random exploration on seeds 1001-1002. The best test result becomes best_model.zip.">ⓘ</i></h4><div class="scroll"><table><tr>' +
      TH('episode', 'Training episode at which the test was run.') + TH('success', 'Fraction of test episodes that reached the target.') + TH('a [AU]', 'Mean final orbit size.') +
      TH('e', 'Mean final eccentricity.') + TH('energy', 'Fraction of energy budget used.') + TH('years', 'Mean simulated years to finish.') + TH('score', 'Mean training reward (score, not physics).') + '</tr>';
    for (const e of r.evals) {
      const best = sel && e.episode === sel.episode;
      h += `<tr style="${best ? 'background:#1e3a5f' : ''}"><td>${e.episode}${best ? ' ★' : ''}</td><td>${pct(e.success_rate, 0)}</td><td>${num(e.a_au)}</td><td>${num(e.e)}</td><td>${pct(e.energy_frac)}</td><td>${int(e.years)}</td><td>${num(e.reward, 1)}</td></tr>`;
    }
    h += '</table></div>';
  }
  if (!bp.available) { $('bestPanel').innerHTML = h + '<div class="note">No PPO model in this comparison.</div>'; return; }
  // comparison
  h += '<h4>Compared with the baselines (high-fidelity run)</h4><table><tr>' + TH('policy', 'Decision rule.') + TH('success', 'Reached the target and passed the stability check.') +
    TH('a [AU]', 'Final orbit size.') + TH('e', 'Final eccentricity.') + TH('years', 'Simulated years.') + TH('asteroid', 'Fraction of asteroid used.') + TH('energy [J]', 'Energy drawn.') +
    TH('score', 'Total training reward of that run (a score, NOT a physical quantity).') + '</tr>';
  for (const x of bp.comparison) h += `<tr><td>${esc(x.policy)}</td><td class="${x.success ? 'pass' : 'fail'}">${x.success ? 'yes' : 'no'}</td><td>${num(x.a)}</td><td>${num(x.e)}</td><td>${int(x.years)}</td><td>${pct(x.mass)}</td><td>${sci(x.energy_J)}</td><td class="dim">${num(x.reward, 1)}</td></tr>`;
  h += '</table>';
  // criteria
  h += '<h4>Final orbit vs. target criteria</h4><table><tr>' + TH('criterion', 'What is tested.') + TH('value', 'Value reached in the high-fidelity run.') + TH('required', 'Requirement.') + TH('', 'Pass or fail.') + '</tr>';
  for (const k of bp.criteria) {
    const v = typeof k.value === 'boolean' ? (k.value ? 'yes' : 'no') : (typeof k.value === 'number' ? (Math.abs(k.value) < 0.01 && k.value !== 0 ? sci(k.value) : num(k.value, k.unit === 'yr' ? 0 : 4)) : 'N/A');
    h += `<tr><td>${esc(k.name)}</td><td>${v} ${esc(k.unit)}</td><td>${esc(k.required)}</td><td class="${k.passed ? 'pass' : 'fail'}">${k.passed ? 'PASS' : 'FAIL'}</td></tr>`;
  }
  h += '</table>';
  // actions
  const cls = bp.action_classes || {};
  h += `<h4>What the model actually did, decision by decision <i class="info" title="Recorded from the high-fidelity run of best_model.zip. 'Helped' = the orbit grew and e stayed within tolerance or decreased; 'mixed' = grew but e rose above tolerance; 'hurt' = orbit shrank or overshot; 'coast' = no ejection.">ⓘ</i></h4>`;
  h += `<div class="small">${Object.entries(cls).map(([k, n]) => `<span class="c-${k}">${k}: ${n}</span>`).join(' · ')} (of ${bp.actions.length} decisions of ${cfg.decision_interval_years} years)</div>`;
  h += '<canvas id="chAct" class="chart"></canvas>';
  h += '<div class="scroll"><table><tr>' + TH('years', 'Decision interval (simulated years).') + TH('throttle', 'Fraction of the maximum packet mass chosen.') + TH('packets', 'Number of packets in this interval.') +
    TH('packet [kg]', 'Mass of each packet.') + TH('speed [km/s]', 'Launch speed of the packets.') + TH('thrust dir.', 'Direction of the push on Earth (packets fly the opposite way). RTN = radial, along-track, normal cosines.') +
    TH('Δv [m/s]', 'Speed change given to Earth-Moon in this interval.') + TH('Δa [AU]', 'Change of orbit size in this interval.') + TH('e after', 'Eccentricity at the end of the interval.') +
    TH('effect', 'Classification from the recorded change of a and e.') + TH('why (from recorded data)', 'Explanation built only from the recorded numbers.', 'l') + '</tr>';
  for (const a of bp.actions) {
    h += `<tr><td>${int(a.t - a.dt)}–${int(a.t)}</td><td>${num(a.thr, 2)}</td><td>${int(a.npk)}</td><td>${a.dm > 0 ? sci(a.dm) : '—'}</td><td>${a.dm > 0 ? int(a.u / 1000) : '—'}</td>` +
      `<td title="RTN ${a.d_rtn.map((x) => x.toFixed(2)).join(', ')}">${a.dm > 0 ? dirWords(a.d_rtn) : '—'}</td><td>${num(a.dv, 1)}</td><td>${a.da >= 0 ? '+' : ''}${num(a.da)}</td><td>${num(a.e)}</td>` +
      `<td class="c-${a.cls}">${a.cls}</td><td class="l small">${esc(a.why)}</td></tr>`;
  }
  h += '</table></div>';
  if (bp.top_helpful && bp.top_helpful.length) {
    h += '<h4>Decisions that helped most</h4><ul class="small">' + bp.top_helpful.map((a) => `<li>Years ${int(a.t - a.dt)}–${int(a.t)}: Δa ${a.da >= 0 ? '+' : ''}${num(a.da)} AU (${pct(a.share)} of all orbit growth) — ${esc(a.why)}; ${int(a.npk)} packets at ${int(a.u / 1000)} km/s, ${sci(a.da_per_1e19kg)} AU per 1e19 kg.</li>`).join('') + '</ul>';
  }
  h += '<h4>Why it behaves this way</h4><ul class="small">' + (c.explanation.summary || []).map((s) => `<li>${esc(s)}</li>`).join('') + '</ul>';
  // failures
  h += '<h4>Failed attempts, exhausted resources, constraint violations</h4>';
  const oc = bp.training_outcomes || {}, tot = Object.values(oc).reduce((a, b) => a + b, 0);
  h += tot ? `<div class="small">Training episodes (with random exploration): ${Object.entries(oc).map(([k, n]) => `<span title="${esc(r.reason_docs[k] || '')}">${esc(k)} ${n} (${pct(n / tot, 0)})</span>`).join(' · ')}</div>` : '<div class="small dim">Training outcome counts: N/A (training history not available in this session).</div>';
  h += bp.failed_evals && bp.failed_evals.length ? `<div class="small">Deterministic tests that did not fully succeed: ${bp.failed_evals.length} (episodes ${bp.failed_evals.slice(0, 12).map((e) => e.episode).join(', ')}${bp.failed_evals.length > 12 ? ', …' : ''}).</div>` : '<div class="small">All deterministic tests of this run succeeded (or none recorded).</div>';
  h += '<table>' + bp.constraints.map((k) => `<tr><td>${esc(k.name)}</td><td>${typeof k.value === 'number' ? (k.unit === 'fraction' ? pct(k.value) : int(k.value)) : esc(k.value)}</td><td class="l dim">${esc(k.note)}</td></tr>`).join('') + '</table>';
  $('bestPanel').innerHTML = h;
  const A = bp.actions;
  chart($('chAct'), [
    { data: A.map((a) => [a.t, a.thr]), color: '#ffcc33', label: 'throttle (0-1)' },
    { data: A.map((a) => [a.t, a.dm > 0 ? a.u / 1e7 : null]), color: '#c084fc', label: 'speed / 10,000 km/s' },
    { data: A.map((a) => [a.t, a.dm > 0 ? a.d_rtn[1] : null]), color: '#7fe08a', label: 'prograde cosine' },
    { data: A.map((a) => [a.t, Math.max(0, a.da) * 10]), color: '#4fa3ff', bars: true, label: 'Δa×10 [AU]' },
  ], { title: 'Actions over time (high-fidelity run)', xlabel: 'yr', ymin: -0.2, ymax: 1.1 });
}

// ------------------------------------------------------------------ guide
function docText(d) {
  if (!d) return '';
  const rng = d.min === null || d.min === undefined ? '' : `\nAllowed range: ${d.min} .. ${d.max} ${d.unit}`;
  return `${d.label} [${d.unit}] — ${d.category}\n${d.meaning}\nWhy it matters: ${d.why}\nIncrease: ${d.increase}\nDecrease: ${d.decrease}${rng}${d.constraint ? '\nConstraint: ' + d.constraint : ''}`;
}
function fmtVal(v) { return typeof v === 'number' && (Math.abs(v) >= 1e5 || (Math.abs(v) < 1e-3 && v !== 0)) ? v.toExponential(4).replace('e+', 'e') : String(v); }
let guideBuilt = '';
function renderGuide(r) {
  const key = JSON.stringify([r.config, r.checks && r.checks.length, r.feasibility.verdict]);
  if (key === guideBuilt) return; guideBuilt = key;
  const g = [];
  const det = (sum, body, tags = '') => `<details><summary>${sum}${tags}</summary><div class="gd">${body}</div></details>`;
  g.push('<div class="note">How to use: 1) check the feasibility pill (green = the target is physically reachable in this model); 2) train; 3) compare policies; 4) watch the best model at a slow speed to see single packets, or fast to see the whole spiral. Hover any ⓘ, table header or telemetry line for an explanation.</div>');
  g.push('<h4>Feasibility in plain English</h4><ul class="small">' + (r.feasibility.plain || []).map((x) => `<li>${esc(x)}</li>`).join('') + '</ul>');
  g.push('<h4>Configurable parameters</h4>');
  const cats = [...new Set(r.param_docs.map((d) => d.category))];
  for (const c of cats) {
    g.push(`<div class="small dim" style="margin-top:6px">${esc(c)}</div>`);
    for (const d of r.param_docs.filter((x) => x.category === c)) {
      const val = r.config[d.key], def = r.defaults[d.key];
      g.push(det(`${esc(d.label)} <span class="u">= ${esc(fmtVal(val))} ${esc(d.unit)}${val !== def ? ` (default ${esc(fmtVal(def))})` : ''}</span>`,
        `<b>Meaning:</b> ${esc(d.meaning)}<br><b>Why it matters:</b> ${esc(d.why)}<br><b>Increase:</b> ${esc(d.increase)}<br><b>Decrease:</b> ${esc(d.decrease)}<br>` +
        (d.min !== null ? `<b>Allowed:</b> ${d.min} .. ${d.max} ${esc(d.unit)}<br>` : '') + (d.constraint ? `<b>Constraint:</b> ${esc(d.constraint)}<br>` : '') + `<b>Config key:</b> <span class="mono">${d.key}</span>`,
        `<span class="tag">${esc(d.category)}</span>`));
    }
  }
  const sd = r.speed_doc;
  g.push(det(`${esc(sd.label)} <span class="u">= ${fmtSpeed(speedY)}/s</span>`, `<b>Meaning:</b> ${esc(sd.meaning)}<br><b>Why:</b> ${esc(sd.why)}<br><b>Increase:</b> ${esc(sd.increase)}<br><b>Decrease:</b> ${esc(sd.decrease)}<br><b>Range:</b> ${esc(sd.constraint)}`, `<span class="tag">${esc(sd.category)}</span>`));
  g.push('<h4>Physical constants (fixed)</h4><table>' + r.constants.map((k) => `<tr><td title="${esc(k.meaning)}">${esc(k.name)}</td><td>${sci(k.value, 6)}</td><td class="l">${esc(k.unit)}</td></tr>`).join('') + '</table>');
  g.push('<h4>Outputs and results</h4>');
  for (const o of r.output_docs) g.push(det(`${esc(o.label)} <span class="u">[${esc(o.unit)}]</span>`, `<b>Measures:</b> ${esc(o.measures)}<br><b>How calculated:</b> ${esc(o.how)}<br><b>Type:</b> ${esc(o.kind)}<br><b>Desirable:</b> ${esc(o.better)}<br><b>Meaning for success:</b> ${esc(o.success)}`, `<span class="tag">${esc(o.kind)}</span>`));
  g.push('<h4>How an episode can end</h4><table>' + Object.entries(r.reason_docs).map(([k, v]) => `<tr><td>${esc(k)}</td><td class="l">${esc(v)}</td></tr>`).join('') + '</table>');
  g.push('<h4>Physics checks (run at every start)</h4>');
  for (const k of r.checks || []) {
    const id = (k.name.match(/^(\d+)/) || [])[1];
    g.push(det(`<span class="${k.passed ? 'pass' : 'fail'}">${k.passed ? 'PASS' : 'FAIL'}</span> ${esc(k.name)}`, `<b>What it tests:</b> ${esc(r.check_docs[id] || '')}<br><b>Measured:</b> <span class="dim">${esc(k.detail)}</span>`));
  }
  g.push('<h4>Assumptions, approximations and limitations</h4>');
  for (const [a, b] of r.assumptions) g.push(det(`⚠ ${esc(a)}`, esc(b)));
  g.push('<h4>Glossary</h4>');
  for (const [a, b] of r.glossary) g.push(det(esc(a), esc(b)));
  g.push(`<div class="small dim" style="margin-top:8px">Results folder: ${esc(r.out_dir)}</div>`);
  $('guide').innerHTML = g.join('');
  filterGuide();
}
function filterGuide() {
  const q = $('guideSearch').value.trim().toLowerCase();
  for (const d of document.querySelectorAll('#guide details')) {
    const hit = !q || d.textContent.toLowerCase().includes(q);
    d.style.display = hit ? '' : 'none'; if (q && hit) d.open = true;
  }
}
$('guideSearch').addEventListener('input', filterGuide);

// ------------------------------------------------------------------ parameters panel
let paramsKey = '';
function buildParams(r, force = false) {
  const key = JSON.stringify(r.config);
  if (key === paramsKey && !force) return; paramsKey = key;
  const sel = $('preset');
  sel.innerHTML = r.presets.map((p) => `<option ${p === r.config.preset ? 'selected' : ''}>${p}</option>`).join('') +
    (r.presets.includes(r.config.preset) ? '' : `<option selected>${r.config.preset}</option>`);
  let h = '', cat = '';
  const order = [...new Set(r.param_docs.map((x) => x.category))];
  const docs = r.param_docs.slice().sort((a, b) => order.indexOf(a.category) - order.indexOf(b.category));
  for (const d of docs) {
    if (d.category !== cat) { cat = d.category; h += `<div class="cat">${esc(cat)}</div>`; }
    const v = r.config[d.key];
    const inp = typeof v === 'boolean' ? `<select data-k="${d.key}" data-bool="1"><option ${v ? 'selected' : ''}>true</option><option ${!v ? 'selected' : ''}>false</option></select>`
      : `<input data-k="${d.key}" data-orig="${esc(fmtVal(v))}" value="${esc(fmtVal(v))}" title="${esc(`allowed ${d.min} .. ${d.max} ${d.unit}; default ${fmtVal(r.defaults[d.key])}`)}">`;
    h += `<div class="row"><span>${esc(d.label)} <span class="dim">[${esc(d.unit)}]</span></span><i class="info" title="${esc(docText(d) + `\nDefault: ${fmtVal(r.defaults[d.key])}`)}">ⓘ</i>${inp}</div>`;
  }
  $('params').innerHTML = h;
  for (const inp of document.querySelectorAll('#params input')) {
    inp.addEventListener('input', () => {
      const d = r.param_docs.find((x) => x.key === inp.dataset.k), x = Number(inp.value);
      inp.classList.toggle('bad', !(Number.isFinite(x) && x >= d.min && x <= d.max));
      inp.classList.toggle('changed', inp.value !== inp.dataset.orig);
    });
  }
}
async function applyParams(body) {
  try {
    const res = await api('/api/config', body);
    $('paramMsg').className = 'small ok'; $('paramMsg').textContent = `Applied (preset ${res.config.preset}). Feasibility: ${res.feasibility.verdict}`;
    resultsDirty = true; paramsKey = '';
  } catch (e) { $('paramMsg').className = 'small err'; $('paramMsg').textContent = 'Rejected: ' + e.message; }
}
$('btnApply').onclick = () => {
  const body = {};
  const preset = $('preset').value;
  if (lastR && preset !== lastR.config.preset) body.preset = preset;     // preset change: preset values win
  else for (const el of document.querySelectorAll('#params [data-k]')) if (el.dataset.bool || el.value !== el.dataset.orig) body[el.dataset.k] = el.value;
  applyParams(body);
};
$('btnDefaults').onclick = () => applyParams({ reset_defaults: true });

// ------------------------------------------------------------------ buttons
for (const b of document.querySelectorAll('.tab')) b.onclick = () => {
  for (const x of document.querySelectorAll('.tab')) x.classList.toggle('active', x === b);
  for (const p of document.querySelectorAll('.tabpane')) p.classList.toggle('active', p.id === b.dataset.tab);
  resultsDirty = true;
};
$('btnTrain').onclick = () => api('/api/train', { action: 'start' }).then(() => toast('training started')).catch((e) => toast(e.message));
$('btnStop').onclick = () => api('/api/train', { action: 'stop' }).then(() => toast('stopping – saving model')).catch((e) => toast(e.message));
$('btnCompare').onclick = () => api('/api/compare', {}).then(() => toast('comparison started')).catch((e) => toast(e.message));
function startRun() {
  return api('/api/play', { policy: $('playPolicy').value, fidelity: $('playFid').value })
    .then(() => { V.playing = true; toast(`mission started: ${$('playPolicy').value}, ${$('playFid').value}`); }).catch((e) => toast(e.message));
}
$('btnPlay').onclick = () => {
  const same = V.mode === 'playback' && V.policy === $('playPolicy').value && V.fidelity === $('playFid').value;
  const finished = V.done && V.rows.length && V.disp >= V.rows[V.rows.length - 1][R_T] - 1e-9;
  if (same && !finished) { V.playing = true; return; }
  startRun();
};
$('btnPause').onclick = () => { V.playing = false; };
$('follow').addEventListener('change', () => {
  if ($('follow').checked) { controls.target.copy(earth.position); camera.position.copy(earth.position).add(new THREE.Vector3(0, -4.5, 3.6)); }
  else { controls.target.set(HOME_X, 0, 0); camera.position.set(HOME_X, -40, 34); }
});
$('btnReset').onclick = () => {
  if (V.mode === 'playback' && V.policy) { $('playPolicy').value = V.policy; $('playFid').value = V.fidelity; }
  startRun();
};
setInterval(() => { $('btnPlay').classList.toggle('on', V.mode === 'playback' && V.playing); $('btnPause').classList.toggle('on', V.mode === 'playback' && !V.playing); }, 300);

window.__oe = { V, get speed() { return speedY; } };
pollState(); pollResults();
window.addEventListener('resize', () => { resultsDirty = true; });
