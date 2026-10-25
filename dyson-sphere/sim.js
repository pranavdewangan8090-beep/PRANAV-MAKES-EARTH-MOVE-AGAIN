// SIM 3 · DYSON SPHERE BEAM — full 3D (three.js r160 from frontend/vendor).
// A Dyson sphere around the Sun concentrates the star's energy into one beam aimed at Earth.
// Earth's eight octo-hands each hold a HUGE concave solar absorber (open spherical cap,
// cave side on the Sun side). Reflected photon momentum gives F = 2P/c on the arms: that pushes
// Earth away from the Sun, but at the notes' power (4.59e22 N vs F_max 4e21 N) the arms snap
// off in 3D and tumble away. Two POVs only: solar-system and Earth.
import * as THREE from '../frontend/vendor/three.module.js';
import { OrbitControls } from '../frontend/vendor/OrbitControls.js';
import { buildOctoEarth, glowTexture } from '../sim-shared/octoearth3d.js';

const $ = (id) => document.getElementById(id);
const sci = (v, d = 2) => (v > 0 ? v.toExponential(d).replace('e+', 'e') : '0');

// ------------------------------------------------------------------ notes' numbers
const C = 3e8, F_MAX = 4e21, M_E = 5.972e24;
const BREAK_T = 2.2, CRACK_T = .25;
let P = 10 ** 30.838;                                   // 6.885e30 W (notes)
const F = () => 2 * P / C;
const ratio = () => F() / F_MAX;

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

{ // stars
  const n = 900, pos = new Float32Array(n * 3);
  for (let i = 0; i < n; i++) {
    const r = 500 + Math.random() * 400, th = Math.random() * Math.PI * 2, ph = Math.acos(2 * Math.random() - 1);
    pos[i * 3] = r * Math.sin(ph) * Math.cos(th); pos[i * 3 + 1] = r * Math.cos(ph); pos[i * 3 + 2] = r * Math.sin(ph) * Math.sin(th);
  }
  const g = new THREE.BufferGeometry(); g.setAttribute('position', new THREE.BufferAttribute(pos, 3));
  scene.add(new THREE.Points(g, new THREE.PointsMaterial({ color: 0xcfe4ff, size: 1.6, sizeAttenuation: false, transparent: true, opacity: .8 })));
}

// Sun + Dyson sphere shell
const sun = new THREE.Mesh(new THREE.SphereGeometry(3, 32, 16), new THREE.MeshBasicMaterial({ color: 0xffcc33 }));
scene.add(sun);
scene.add(new THREE.PointLight(0xffffff, 2.6, 0, 0));
const glow = new THREE.Sprite(new THREE.SpriteMaterial({ map: glowTexture(), blending: THREE.AdditiveBlending, depthWrite: false, transparent: true }));
glow.scale.setScalar(24); scene.add(glow);
const shell = new THREE.Mesh(new THREE.IcosahedronGeometry(5.4, 1),
  new THREE.MeshBasicMaterial({ color: 0xffd27f, wireframe: true, transparent: true, opacity: .75 }));
scene.add(shell);

// ------------------------------------------------------------------ octo-Earth with concave absorbers
const D0 = 34;
const earthGroup = new THREE.Group();
earthGroup.position.set(D0, 0, 0);
scene.add(earthGroup);
const rig0 = buildOctoEarth({ hand: 'mirror', mirrorR: 1.7, face: -1 });   // cave side toward the Sun
let rig = rig0;
earthGroup.add(rig.group);
const earthPos = earthGroup.position;
const prevEarth = earthPos.clone();

// beam: outer glow cylinder + hot core, along +X from the shell to the mirror curtain
function beamCyl(r, color, op) {
  const m = new THREE.Mesh(new THREE.CylinderGeometry(r, r, 1, 14, 1, true).rotateZ(Math.PI / 2),
    new THREE.MeshBasicMaterial({ color, transparent: true, opacity: op, blending: THREE.AdditiveBlending, depthWrite: false, side: THREE.DoubleSide }));
  scene.add(m); return m;
}
const beamOuter = beamCyl(.62, 0xffe14d, .30);
const beamCore = beamCyl(.20, 0xfff3b0, .85);
function setBeam(x0, x1) {
  for (const b of [beamOuter, beamCore]) { b.scale.x = Math.max(.1, x1 - x0); b.position.x = (x0 + x1) / 2; }
}

// photon pulses racing down the beam
const N_PH = 46;
const photons = [];
{
  const g = new THREE.SphereGeometry(.17, 8, 6);
  const m = new THREE.MeshBasicMaterial({ color: 0xffe14d, blending: THREE.AdditiveBlending, depthWrite: false, transparent: true });
  for (let i = 0; i < N_PH; i++) {
    const mesh = new THREE.Mesh(g, m);
    const p = { mesh, x: 5.4 + Math.random() * (D0 - 8), r: Math.random() * .5, a: Math.random() * Math.PI * 2, v: 26 + Math.random() * 8 };
    photons.push(p); scene.add(mesh);
  }
}

// impact flash pool (rings facing the Sun on the absorbers)
const flashes = [];
{
  const g = new THREE.RingGeometry(.4, .55, 24);
  const q = new THREE.Quaternion().setFromUnitVectors(new THREE.Vector3(0, 0, 1), new THREE.Vector3(-1, 0, 0));
  for (let i = 0; i < 20; i++) {
    const mesh = new THREE.Mesh(g, new THREE.MeshBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0, blending: THREE.AdditiveBlending, depthWrite: false, side: THREE.DoubleSide }));
    mesh.quaternion.copy(q); mesh.visible = false;
    flashes.push({ mesh, t: 9 }); scene.add(mesh);
  }
}
function flashAt(pos) {
  const f = flashes.reduce((a, b) => (a.t > b.t ? a : b));
  f.t = 0; f.mesh.visible = true; f.mesh.position.copy(pos);
}

// ------------------------------------------------------------------ state / controls
let pov = 'earth', playing = true, tRun = 0, cracked = false, broken = false, breakT = 0, shake = 0;
const shakeOff = new THREE.Vector3();
function syncPlay() { $('btnPlay').textContent = playing ? '❚❚ Pause' : '▶ Play'; }
function resetRun() {
  tRun = 0; cracked = false; broken = false; shake = 0; playing = true; syncPlay();
  $('stamp').classList.remove('show');
  rig.setStress(0);
  rig.bodyMat.emissive.setHex(0x0a2a55);
  if (armsWereBroken) rebuildRig();                        // fresh octo-hands + absorbers
}
let armsWereBroken = false;
function rebuildRig() {
  for (const a of rig.arms) if (a.broken) scene.remove(a.shoulder);   // clear debris
  earthGroup.remove(rig.group);
  rig = buildOctoEarth({ hand: 'mirror', mirrorR: 1.7, face: -1 });
  earthGroup.add(rig.group);
  armsWereBroken = false;
}
$('povSolar').onclick = () => { pov = 'solar'; $('povSolar').classList.add('active'); $('povEarth').classList.remove('active'); setPreset(); };
$('povEarth').onclick = () => { pov = 'earth'; $('povEarth').classList.add('active'); $('povSolar').classList.remove('active'); setPreset(); };
$('btnPlay').onclick = () => { playing = !playing; syncPlay(); };
$('btnReset').onclick = resetRun;
$('pow').oninput = (e) => { P = 10 ** parseFloat(e.target.value); resetRun(); };
function setPreset() {
  if (pov === 'solar') { camera.position.set(17, -46, 34); controls.target.set(17, 0, 0); }
  else { camera.position.copy(earthPos).add(new THREE.Vector3(-11, 4.5, 12)); controls.target.copy(earthPos); }
  prevEarth.copy(earthPos);
}
setPreset();

// ------------------------------------------------------------------ update
const tmp = new THREE.Vector3();
function update(dt, t) {
  if (playing) tRun += dt;
  shell.rotation.y += dt * .12; shell.rotation.x += dt * .03;
  rig.update(t, dt);

  const r = ratio();
  if (r > 1 && tRun > BREAK_T) cracked = true;
  if (cracked && !broken && tRun > BREAK_T + CRACK_T) {
    broken = true; armsWereBroken = true; breakT = tRun; shake = 1;
    rig.breakArms(scene, new THREE.Vector3(1, 0, 0));       // blown back along the beam
    for (const a of rig.arms) flashAt(a.shoulder.position);
    const s = $('stamp');
    s.innerHTML = 'ARM FAILURE: F = 2P/c = ' + sci(F()) + ' N vs F_max = 4e21 N (' + r.toFixed(1) + 'x overload)<br>the octo-hands + absorbers snap under the incident beam → NOT POSSIBLE (notes)';
    s.classList.add('show');
  }
  const heat = Math.min(1, tRun / 1.2) * Math.min(1, r / 11.5);
  if (!broken) rig.setStress(heat);
  if (broken) {                                            // sun-facing side cooks
    const b = Math.min(1, (tRun - breakT) / 1.2);
    rig.bodyMat.emissive.setRGB(.04 + .5 * b, .16 - .1 * b, .33 - .28 * b);
  }
  // Earth pushed away from the Sun
  earthGroup.position.x = D0 + Math.min(10, tRun * .9);
  earthGroup.rotation.y = Math.sin(t * .2) * .04;

  // beam extent: to the mirror curtain, or to the bare body after the break
  const mirrorX = earthPos.x + rig.tipX;                   // tipX < 0 (hands face the Sun)
  setBeam(5.4, broken ? earthPos.x - 1.1 : mirrorX);

  // photons
  for (const p of photons) {
    if (playing) p.x += p.v * dt;
    const stop = broken ? earthPos.x - 1.1 : mirrorX;
    if (p.x >= stop) {
      flashAt(tmp.set(stop, Math.sin(p.a) * p.r * 2.2, Math.cos(p.a) * p.r * 2.2));
      p.x = 5.4; p.a = Math.random() * Math.PI * 2; p.r = Math.random() * .5;
    }
    p.mesh.position.set(p.x, Math.sin(p.a) * p.r, Math.cos(p.a) * p.r);
  }
  for (const f of flashes) {
    f.t += dt;
    const k = f.t / .45;
    if (k >= 1) { f.mesh.visible = false; continue; }
    f.mesh.visible = true;
    f.mesh.scale.setScalar(1 + k * 5);
    f.mesh.material.opacity = 1 - k;
  }
  // camera: follow Earth's push, then shake on the break
  if (pov === 'earth') {
    tmp.copy(earthPos).sub(prevEarth);
    camera.position.add(tmp);
    controls.target.copy(earthPos);
  }
  prevEarth.copy(earthPos);
  controls.update();
  if (shake > 0) {
    shake = Math.max(0, shake - dt * 1.6);
    camera.position.sub(shakeOff);
    shakeOff.set((Math.random() - .5) * .5 * shake, (Math.random() - .5) * .5 * shake, (Math.random() - .5) * .5 * shake);
    camera.position.add(shakeOff);
  } else if (shakeOff.lengthSq() > 0) { camera.position.sub(shakeOff); shakeOff.set(0, 0, 0); }

  if (!broken && r <= 1 && tRun > BREAK_T) {
    const s = $('stamp');
    s.innerHTML = 'arms hold: F = ' + sci(F()) + ' N ≤ F_max = 4e21 N<br>Earth pushed away from the Sun at a = ' + sci(F() / M_E) + ' m/s^2';
    s.classList.add('show');
  }
}

// ------------------------------------------------------------------ HUD + calc panel
function panel() {
  $('cP').textContent = sci(P); $('cF').textContent = sci(F());
  $('cR').textContent = ratio().toFixed(1); $('cA').textContent = sci(F() / M_E);
  $('powLbl').textContent = sci(P) + ' W';
  $('hud').textContent =
    (pov === 'earth' ? 'EARTH POV' : 'SOLAR SYSTEM POV') + ' · 3D\n' +
    'beam on for ' + tRun.toFixed(1) + ' s\n' +
    'P = ' + sci(P) + ' W\n' +
    'F = 2P/c = ' + sci(F()) + ' N\n' +
    'F/F_max = ' + ratio().toFixed(1) + ' x\n' +
    (broken ? '*** ARMS BROKEN — absorbers adrift ***' : cracked ? '!! cracks at the arm roots !!' : 'absorbers holding, stress rising');
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
