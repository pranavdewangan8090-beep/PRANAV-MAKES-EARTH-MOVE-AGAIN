// SIM 2 · HYDROGEN COLLECTING DISK — full 3D (three.js r160 from frontend/vendor).
// Earth's eight octo-hands (shared rig) spread into a net; the ram flow of interstellar
// hydrogen is caught BY THE HANDS THEMSELVES — atoms stick where they hit the net and stay
// there (they are never sent to Earth). Two POVs only: solar-system and Earth.
import * as THREE from '../frontend/vendor/three.module.js';
import { OrbitControls } from '../frontend/vendor/OrbitControls.js';
import { buildOctoEarth, glowTexture } from '../sim-shared/octoearth3d.js';

const $ = (id) => document.getElementById(id);
const sci = (v, d = 2) => (v > 0 ? v.toExponential(d).replace('e+', 'e') : '0');

// ------------------------------------------------------------------ notes' numbers
const N_D = 1e6, R_E = 6.371e6, DV = 2.3e4, M_H = 1.67e-27, YR = 3.15576e7, T_END = 1e4;
const A_NET = 8 * Math.PI * R_E * R_E;
const RATE = N_D * A_NET * DV;
const YR_PER_S = 500;                              // simulated years per wall second
const DOT_ATOMS = RATE * T_END * YR / 240;

// ------------------------------------------------------------------ renderer / scene
const cv = $('cv');
const renderer = new THREE.WebGLRenderer({ canvas: cv, antialias: true });
renderer.setPixelRatio(Math.min(2, window.devicePixelRatio || 1));
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x05070c);
const camera = new THREE.PerspectiveCamera(50, 1, .01, 3000);
const controls = new OrbitControls(camera, cv);
controls.enableDamping = true;
scene.add(new THREE.AmbientLight(0xffffff, .5));

// stars
{
  const n = 900, pos = new Float32Array(n * 3);
  for (let i = 0; i < n; i++) {
    const r = 500 + Math.random() * 400, th = Math.random() * Math.PI * 2, ph = Math.acos(2 * Math.random() - 1);
    pos[i * 3] = r * Math.sin(ph) * Math.cos(th); pos[i * 3 + 1] = r * Math.cos(ph); pos[i * 3 + 2] = r * Math.sin(ph) * Math.sin(th);
  }
  const g = new THREE.BufferGeometry(); g.setAttribute('position', new THREE.BufferAttribute(pos, 3));
  scene.add(new THREE.Points(g, new THREE.PointsMaterial({ color: 0xcfe4ff, size: 1.6, sizeAttenuation: false, transparent: true, opacity: .8 })));
}

// Sun + light + Earth orbit ring
const R_ORB = 26;
const sun = new THREE.Mesh(new THREE.SphereGeometry(3, 32, 16), new THREE.MeshBasicMaterial({ color: 0xffcc33 }));
scene.add(sun);
const sunLight = new THREE.PointLight(0xffffff, 2.6, 0, 0);
scene.add(sunLight);
const glow = new THREE.Sprite(new THREE.SpriteMaterial({ map: glowTexture(), blending: THREE.AdditiveBlending, depthWrite: false, transparent: true }));
glow.scale.setScalar(26); scene.add(glow);
{
  const p = []; for (let i = 0; i <= 128; i++) { const a = i / 128 * Math.PI * 2; p.push(new THREE.Vector3(Math.cos(a) * R_ORB, 0, Math.sin(a) * R_ORB)); }
  scene.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(p), new THREE.LineBasicMaterial({ color: 0x4fa3ff, transparent: true, opacity: .22 })));
}

// ------------------------------------------------------------------ octo-Earth + net
const earthGroup = new THREE.Group();
scene.add(earthGroup);
const rig = buildOctoEarth({ hand: 'pad', face: +1 });     // hands open toward the ram flow
earthGroup.add(rig.group);

// net membrane + strands between the hand tips (the "net" the hands form)
const membrane = new THREE.Mesh(
  new THREE.CircleGeometry(rig.tipSpan, 44).rotateY(Math.PI / 2),
  new THREE.MeshBasicMaterial({ color: 0x7fd0ff, transparent: true, opacity: .08, side: THREE.DoubleSide, depthWrite: false }));
membrane.position.x = rig.tipX;
rig.group.add(membrane);
const STRAND_SEGS = 24;                                    // 8 rim + 8 spokes + 8 inner
const strandPos = new Float32Array(STRAND_SEGS * 6);
const strandGeo = new THREE.BufferGeometry();
strandGeo.setAttribute('position', new THREE.BufferAttribute(strandPos, 3).setUsage(THREE.DynamicDrawUsage));
const strands = new THREE.LineSegments(strandGeo, new THREE.LineBasicMaterial({ color: 0x7fd0ff, transparent: true, opacity: .5 }));
rig.group.add(strands);
const tipsL = Array.from({ length: 8 }, () => new THREE.Vector3());

// ------------------------------------------------------------------ H gas: free stream + stuck-on-hands
const N_FREE = 1500;
const freePos = new Float32Array(N_FREE * 3);
const freeJit = new Float32Array(N_FREE * 3);
const WIND = new THREE.Vector3(.9, .12, .35);              // ISM wind (solar frame)
const STUCK_MAX = 6000;
const stuckPos = new Float32Array(STUCK_MAX * 3);
const stuckCol = new Float32Array(STUCK_MAX * 3);
let nStuck = 0;
const freeGeo = new THREE.BufferGeometry();
freeGeo.setAttribute('position', new THREE.BufferAttribute(freePos, 3).setUsage(THREE.DynamicDrawUsage));
const freePts = new THREE.Points(freeGeo, new THREE.PointsMaterial({ color: 0xbfe8ff, size: .14, transparent: true, opacity: .9 }));
scene.add(freePts);
const stuckGeo = new THREE.BufferGeometry();
stuckGeo.setAttribute('position', new THREE.BufferAttribute(stuckPos, 3).setUsage(THREE.DynamicDrawUsage));
stuckGeo.setAttribute('color', new THREE.BufferAttribute(stuckCol, 3).setUsage(THREE.DynamicDrawUsage));
stuckGeo.setDrawRange(0, 0);
const stuckPts = new THREE.Points(stuckGeo, new THREE.PointsMaterial({ size: .16, vertexColors: true, transparent: true, opacity: 1 }));
earthGroup.add(stuckPts);                                  // stuck atoms ride with Earth (on the hands)

const earthPos = new THREE.Vector3(R_ORB, 0, 0);
const prevEarth = earthPos.clone();
function respawn(i, initial) {
  // upstream of the net so the flow always feeds the hands
  const f = faceDir;
  const d = 14 + Math.random() * 30, rad = Math.random() * rig.tipSpan * 1.6;
  const a = Math.random() * Math.PI * 2;
  const u = perp1, w = perp2;
  freePos[i * 3] = earthPos.x + f.x * d + (u.x * Math.cos(a) + w.x * Math.sin(a)) * rad;
  freePos[i * 3 + 1] = earthPos.y + f.y * d + (u.y * Math.cos(a) + w.y * Math.sin(a)) * rad;
  freePos[i * 3 + 2] = earthPos.z + f.z * d + (u.z * Math.cos(a) + w.z * Math.sin(a)) * rad;
  if (initial) { freeJit[i * 3] = (Math.random() - .5) * .1; freeJit[i * 3 + 1] = (Math.random() - .5) * .1; freeJit[i * 3 + 2] = (Math.random() - .5) * .1; }
}
const perp1 = new THREE.Vector3(), perp2 = new THREE.Vector3(), faceDir = new THREE.Vector3(1, 0, 0);
const flowDir = new THREE.Vector3(), vE = new THREE.Vector3(), inv = new THREE.Matrix4(), pL = new THREE.Vector3(), pW = new THREE.Vector3();
function setPerps() {
  perp1.set(0, 1, 0); if (Math.abs(faceDir.y) > .9) perp1.set(1, 0, 0);
  perp1.crossVectors(faceDir, perp1).normalize();
  perp2.crossVectors(faceDir, perp1).normalize();
}
setPerps();
for (let i = 0; i < N_FREE; i++) respawn(i, true);

// ------------------------------------------------------------------ state / controls
let pov = 'earth', playing = true, tYr = 0, theta = 0, captured = 0;
const handCount = new Array(8).fill(0);
const W_ORB = Math.PI * 2 / 26;                            // visual orbit rate (drawing only)
function syncPlay() { $('btnPlay').textContent = playing ? '❚❚ Pause' : '▶ Play'; }
function reset() {
  tYr = 0; theta = 0; captured = 0; nStuck = 0; handCount.fill(0);
  stuckGeo.setDrawRange(0, 0); playing = true; syncPlay(); $('stamp').classList.remove('show');
  earthPos.set(R_ORB, 0, 0); prevEarth.copy(earthPos);
  for (let i = 0; i < N_FREE; i++) respawn(i, true);
  freeGeo.attributes.position.needsUpdate = true;
}
$('povSolar').onclick = () => { pov = 'solar'; $('povSolar').classList.add('active'); $('povEarth').classList.remove('active'); setPreset(); };
$('povEarth').onclick = () => { pov = 'earth'; $('povEarth').classList.add('active'); $('povSolar').classList.remove('active'); setPreset(); };
$('btnPlay').onclick = () => { if (tYr >= T_END) reset(); else { playing = !playing; syncPlay(); } };
$('btnReset').onclick = reset;
function setPreset() {
  if (pov === 'solar') { camera.position.set(0, -58, 44); controls.target.set(0, 0, 0); }
  else { camera.position.copy(earthPos).add(new THREE.Vector3(9, 4, 7)); controls.target.copy(earthPos); }
  prevEarth.copy(earthPos);
}
setPreset();

// ------------------------------------------------------------------ update
const handTipsW = Array.from({ length: 8 }, () => new THREE.Vector3());
function update(dt, t) {
  const adt = playing ? dt : 0;
  if (playing) tYr = Math.min(T_END, tYr + dt * YR_PER_S);
  if (tYr >= T_END && playing) {
    playing = false; syncPlay();
    const s = $('stamp');
    s.innerHTML = '10,000 yr of sweeping: Vol = 7.4e30 m^3 · N = 7.4e36 atoms · M = 1.2e10 kg<br>far too little H to feed a fusion cycle → INFEASIBLE (notes)';
    s.classList.add('show');
  }
  // Earth on its (visual-speed) orbit; gas wind fixed in the solar frame
  theta += W_ORB * adt;
  earthPos.set(Math.cos(theta) * R_ORB, 0, Math.sin(theta) * R_ORB);
  vE.set(-Math.sin(theta), 0, Math.cos(theta)).multiplyScalar(R_ORB * W_ORB * (playing ? 1 : 0));
  flowDir.copy(WIND).sub(vE).normalize();                  // gas motion relative to Earth
  faceDir.copy(flowDir).multiplyScalar(-1);                // net faces upstream
  setPerps();
  earthGroup.position.copy(earthPos);
  const q = new THREE.Quaternion().setFromUnitVectors(new THREE.Vector3(1, 0, 0), faceDir);
  earthGroup.quaternion.slerp(q, .08);
  rig.update(t, dt);

  // strands follow the swaying hand tips
  scene.updateMatrixWorld(true);
  rig.handTipsLocal(tipsL);
  let s = 0;
  const put = (a, b) => { strandPos[s++] = a.x; strandPos[s++] = a.y; strandPos[s++] = a.z; strandPos[s++] = b.x; strandPos[s++] = b.y; strandPos[s++] = b.z; };
  for (let k = 0; k < 8; k++) {
    put(tipsL[k], tipsL[(k + 1) % 8]);                                  // rim
    put(tipsL[k], tipsL[k].clone().multiplyScalar(.45));               // spokes
    put(tipsL[k].clone().multiplyScalar(.45), new THREE.Vector3(rig.tipX * .8, 0, 0)); // inner to hub
  }
  strandGeo.attributes.position.needsUpdate = true;

  // gas: drift with the wind; capture in Earth-local frame on the net plane
  inv.copy(earthGroup.matrixWorld).invert();
  const attr = freeGeo.attributes.position;
  for (let i = 0; i < N_FREE; i++) {
    pW.set(freePos[i * 3] += (WIND.x + freeJit[i * 3]) * adt,
      freePos[i * 3 + 1] += (WIND.y + freeJit[i * 3 + 1]) * adt,
      freePos[i * 3 + 2] += (WIND.z + freeJit[i * 3 + 2]) * adt);
    pL.copy(pW).applyMatrix4(inv);
    const rad = Math.hypot(pL.y, pL.z);
    if (Math.abs(pL.x - rig.tipX) < .55 && rad < rig.tipSpan && nStuck < STUCK_MAX) {
      // caught BY A HAND: stick on the net, never sent to Earth
      const j = nStuck++;
      stuckPos[j * 3] = pL.x; stuckPos[j * 3 + 1] = pL.y; stuckPos[j * 3 + 2] = pL.z;
      const g = .75 + Math.random() * .25;
      stuckCol[j * 3] = .45 * g; stuckCol[j * 3 + 1] = 1 * g; stuckCol[j * 3 + 2] = .7 * g;
      let bk = 0, bd = 1e9;
      for (let k = 0; k < 8; k++) { const d = tipsL[k].distanceToSquared(pL); if (d < bd) { bd = d; bk = k; } }
      handCount[bk]++; captured++;
      stuckGeo.setDrawRange(0, nStuck);
      stuckGeo.attributes.color.needsUpdate = true;
      respawn(i, false);
    } else if (pL.x < -14 || pW.distanceTo(earthPos) > 70) respawn(i, false);
  }
  attr.needsUpdate = true;
  stuckGeo.attributes.position.needsUpdate = true;

  // cameras
  if (pov === 'earth') {
    const delta = pL.copy(earthPos).sub(prevEarth);        // reuse temp
    camera.position.add(delta);
    controls.target.copy(earthPos);
  }
  prevEarth.copy(earthPos);
  controls.update();
}

// ------------------------------------------------------------------ HUD + calc panel
function panel() {
  const tSec = tYr * YR, vol = DV * tSec * A_NET, n = N_D * vol, m = n * M_H;
  $('cA').textContent = sci(A_NET); $('cVol').textContent = sci(vol);
  $('cN').textContent = sci(n); $('cM').textContent = sci(m);
  $('cDot').textContent = sci(DOT_ATOMS);
  const busiest = handCount.indexOf(Math.max(...handCount));
  $('hud').textContent =
    (pov === 'earth' ? 'EARTH POV' : 'SOLAR SYSTEM POV') + ' · 3D\n' +
    't = ' + Math.round(tYr).toLocaleString('en-US') + ' yr of 10,000 yr\n' +
    'notes: N = ' + sci(n) + ' atoms, M = ' + sci(m) + ' kg\n' +
    'hands caught ' + captured + ' dots (≈ ' + sci(captured * DOT_ATOMS) + ' atoms)\n' +
    'busiest hand: #' + (busiest + 1) + ' with ' + handCount[busiest] + ' dots';
}

// ------------------------------------------------------------------ loop
function resize() {
  const w = cv.clientWidth, h = cv.clientHeight;
  renderer.setSize(w, h, false);
  camera.aspect = w / h; camera.updateProjectionMatrix();
}
addEventListener('resize', resize);
let last = 0;
function frame(ts) {
  if (!last) last = ts;
  const dt = Math.min(.05, (ts - last) / 1000); last = ts;
  resize();
  update(dt, ts / 1000);
  panel();
  renderer.render(scene, camera);
  requestAnimationFrame(frame);
}
resize(); syncPlay(); requestAnimationFrame(frame);
