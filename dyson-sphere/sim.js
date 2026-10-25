// SIM 3 · DYSON SPHERE BEAM
// A Dyson sphere around the Sun concentrates the star's energy into one beam aimed at Earth.
// Reflectors on Earth's eight octo-hands catch the beam; photon momentum p = E/c reflected
// gives F = 2P/c on the arms. That pushes Earth away from the Sun - but at the notes' beam
// power the force (4.59e22 N) is ~11x what the arms can take (4e21 N), so the arms snap.
// Two POVs only: solar-system POV and Earth POV. Browser-only drawing + the notes' arithmetic.
'use strict';

// ------------------------------------------------------------------ helpers
const $ = (id) => document.getElementById(id);
const cv = $('cv'), ctx = cv.getContext('2d');
let W = 0, H = 0;
function resize() {
  const d = Math.min(2, window.devicePixelRatio || 1);
  W = cv.clientWidth; H = cv.clientHeight;
  cv.width = Math.round(W * d); cv.height = Math.round(H * d);
  ctx.setTransform(d, 0, 0, d, 0, 0);
}
addEventListener('resize', resize);
const sci = (v, d = 2) => (v > 0 ? v.toExponential(d).replace('e+', 'e') : '0');
const TAU = Math.PI * 2;

// ------------------------------------------------------------------ the notes' numbers
const C = 3e8;                 // speed of light [m/s]
const F_MAX = 4e21;            // N, force an arm can take (notes)
const M_E = 5.972e24;          // Earth mass [kg]
const L_SUN = 3.828e26;        // Sun's whole output [W] (reference)
const BREAK_T = 2.2, CRACK_T = .25;   // wall-clock seconds of beam before the arms fail
let P = 10 ** 30.838;          // beam power [W], notes value 6.885e30
const F = () => 2 * P / C;
const ratio = () => F() / F_MAX;

// ------------------------------------------------------------------ state
let pov = 'earth', playing = true, tRun = 0, last = 0;
let cracked = false, broken = false, debris = [], photons = [], flashes = [];
const stars = [];
for (let i = 0; i < 150; i++) stars.push({ x: Math.random(), y: Math.random(), r: Math.random() * 1.2 + .3, a: Math.random() * .45 + .15 });
const ARM_A = [-78.75, -56.25, -33.75, -11.25, 11.25, 33.75, 56.25, 78.75].map(d => (180 + d) * Math.PI / 180);

function resetRun() { tRun = 0; cracked = false; broken = false; debris = []; photons = []; flashes = []; playing = true; syncPlay(); }
function syncPlay() { $('btnPlay').textContent = playing ? '❚❚ Pause' : '▶ Play'; }

// ------------------------------------------------------------------ controls
$('povSolar').onclick = () => { pov = 'solar'; $('povSolar').classList.add('active'); $('povEarth').classList.remove('active'); };
$('povEarth').onclick = () => { pov = 'earth'; $('povEarth').classList.add('active'); $('povSolar').classList.remove('active'); };
$('btnPlay').onclick = () => { playing = !playing; syncPlay(); };
$('btnReset').onclick = resetRun;
$('pow').oninput = (e) => { P = 10 ** parseFloat(e.target.value); resetRun(); };

// ------------------------------------------------------------------ geometry
function earthGeom() {
  const Re = Math.min(W, H) * .15;
  return { cx: W * .66, cy: H * .52, Re, L: Re * 2.1, pad: Re * .55 };
}
function solarGeom() {
  return { sx: W * .24, sy: H * .5, shell: 42, ex0: W * .62, ey: H * .5 };
}
const push = () => Math.min(70, tRun * 6);   // Earth drifts away from the Sun [px]

// ------------------------------------------------------------------ update
function update(dt) {
  if (!playing) return;
  tRun += dt;
  if (ratio() > 1 && tRun > BREAK_T) cracked = true;
  if (cracked && tRun > BREAK_T + CRACK_T && !broken) {
    broken = true;
    const g = earthGeom();
    for (const a of ARM_A) debris.push({
      a, x: g.cx + Math.cos(a) * g.Re, y: g.cy + Math.sin(a) * g.Re,
      vx: 60 + Math.random() * 90, vy: (Math.random() - .5) * 120,
      rot: 0, vr: (Math.random() - .5) * 5, t: 0,
    });
  }
  // photons streaming down the beam
  const rate = pov === 'earth' ? 70 : 40;
  if (photons.length < rate) for (let i = 0; i < 2; i++) photons.push(spawnPhoton());
  const g = pov === 'earth' ? earthGeom() : solarGeom();
  for (const p of photons) {
    p.x += p.v * dt;
    if (pov === 'earth') {
      const stop = broken ? g.cx - g.Re : g.cx - g.L * .92;
      if (p.x >= stop) { p.dead = true; flashes.push({ x: p.x, y: p.y, t: 0 }); }
    } else {
      const stop = broken ? g.ex0 + push() - 11 : g.ex0 + push() - g.shell * .55;
      if (p.x >= stop) { p.dead = true; flashes.push({ x: p.x, y: p.y, t: 0 }); }
    }
  }
  photons = photons.filter(p => !p.dead);
  for (const f of flashes) f.t += dt;
  flashes = flashes.filter(f => f.t < .4);
  for (const d of debris) { d.t += dt; d.x += d.vx * dt; d.y += d.vy * dt; d.rot += d.vr * dt; }
  debris = debris.filter(d => d.t < 2.6);
}
function spawnPhoton() {
  if (pov === 'earth') {
    const g = earthGeom();
    const y = g.cy + (Math.random() - .5) * 2 * g.Re * .8;
    return { x: -10, y, v: 300 + Math.random() * 60 };
  }
  const g = solarGeom();
  return { x: g.sx + g.shell, y: g.ey + (Math.random() - .5) * 8, v: 260 + Math.random() * 50 };
}

// ------------------------------------------------------------------ drawing
function drawStars() {
  for (const s of stars) { ctx.globalAlpha = s.a; ctx.fillStyle = '#cfe4ff'; ctx.fillRect(s.x * W, s.y * H, s.r, s.r); }
  ctx.globalAlpha = 1;
}
function heatColor(heat, l = 65) { return `hsl(${210 * (1 - Math.min(1, heat))},90%,${l}%)`; }
function arrow(x, y, ang, len, color, alpha, w = 1.6) {
  ctx.globalAlpha = alpha; ctx.strokeStyle = color; ctx.lineWidth = w;
  const ux = Math.cos(ang), uy = Math.sin(ang);
  ctx.beginPath(); ctx.moveTo(x - ux * len, y - uy * len); ctx.lineTo(x, y);
  ctx.lineTo(x - ux * 7 - uy * 4.5, y - uy * 7 + ux * 4.5);
  ctx.moveTo(x, y); ctx.lineTo(x - ux * 7 + uy * 4.5, y - uy * 7 - ux * 4.5);
  ctx.stroke(); ctx.globalAlpha = 1;
}
function drawSun(x, y, r) {
  const sg = ctx.createRadialGradient(x, y, 1, x, y, r * 2.4);
  sg.addColorStop(0, '#fff3b0'); sg.addColorStop(.42, '#ffcc33'); sg.addColorStop(1, 'rgba(255,204,51,0)');
  ctx.fillStyle = sg; ctx.beginPath(); ctx.arc(x, y, r * 2.4, 0, TAU); ctx.fill();
}
function drawEarthBody(x, y, r, hot) {
  const gr = ctx.createRadialGradient(x - r * .3, y - r * .3, r * .2, x, y, r);
  gr.addColorStop(0, hot ? '#ffd7a1' : '#7cc4ff'); gr.addColorStop(1, hot ? '#7a2a0a' : '#0a2a55');
  ctx.fillStyle = gr; ctx.beginPath(); ctx.arc(x, y, r, 0, TAU); ctx.fill();
}
function armShape(x, y, a, len, pad, heat, alpha) {
  ctx.globalAlpha = alpha;
  const tx = x + Math.cos(a) * len, ty = y + Math.sin(a) * len;
  ctx.strokeStyle = heatColor(heat); ctx.lineWidth = 3.5;
  ctx.beginPath(); ctx.moveTo(x, y); ctx.lineTo(tx, ty); ctx.stroke();
  ctx.strokeStyle = '#e8f4ff'; ctx.lineWidth = 4;                    // reflector pad
  ctx.beginPath();
  ctx.moveTo(tx - Math.sin(a) * pad, ty + Math.cos(a) * pad);
  ctx.lineTo(tx + Math.sin(a) * pad, ty - Math.cos(a) * pad);
  ctx.stroke();
  ctx.globalAlpha = 1;
  return [tx, ty];
}

function drawEarthPOV() {
  const g = earthGeom();
  const heat = Math.min(1, tRun / 1.2) * Math.min(1, ratio() / 11.5);
  const shake = broken ? Math.max(0, 1 - (tRun - BREAK_T - CRACK_T) / .6) : 0;
  ctx.save();
  if (shake > 0) ctx.translate((Math.random() - .5) * 9 * shake, (Math.random() - .5) * 9 * shake);

  // beam wedge from the Sun side onto the reflector fan
  const h1 = g.L * .95;
  ctx.fillStyle = 'rgba(255,225,77,.07)';
  ctx.beginPath(); ctx.moveTo(0, g.cy - g.Re * .8); ctx.lineTo(g.cx - g.L * .9, g.cy - h1);
  ctx.lineTo(g.cx - g.L * .9, g.cy + h1); ctx.lineTo(0, g.cy + g.Re * .8); ctx.closePath(); ctx.fill();
  ctx.fillStyle = '#ffd27f'; ctx.font = '12px ui-monospace, Menlo, monospace'; ctx.textAlign = 'left';
  ctx.fillText('← from the Dyson sphere: one concentrated beam, power P', 12, g.cy - h1 - 10);

  // Earth + octo-hands with reflectors
  drawEarthBody(g.cx, g.cy, g.Re, broken);
  if (!broken) {
    for (const a of ARM_A) {
      armShape(g.cx + Math.cos(a) * g.Re * .9, g.cy + Math.sin(a) * g.Re * .9, a, g.L - g.Re * .9, g.pad, heat, 1);
      if (cracked) {                                                 // jagged crack at the arm root
        ctx.strokeStyle = '#fff'; ctx.lineWidth = 1.6;
        const bx = g.cx + Math.cos(a) * g.Re * 1.15, by = g.cy + Math.sin(a) * g.Re * 1.15;
        ctx.beginPath(); ctx.moveTo(bx - 5, by - 5); ctx.lineTo(bx + 2, by - 1); ctx.lineTo(bx - 2, by + 3); ctx.lineTo(bx + 5, by + 6); ctx.stroke();
      }
    }
  }
  for (const d of debris) {                                          // snapped arms tumbling away
    ctx.save(); ctx.translate(d.x, d.y); ctx.rotate(d.rot);
    armShape(0, 0, d.a, g.L * .8, g.pad, 1, Math.max(0, 1 - d.t / 2.6));
    ctx.restore();
  }
  ctx.fillStyle = '#dfe7f3'; ctx.font = '12px ui-monospace, Menlo, monospace'; ctx.textAlign = 'center';
  ctx.fillText('Earth', g.cx, g.cy + g.Re + 18);
  ctx.fillStyle = heatColor(heat, 75); ctx.textAlign = 'left';
  ctx.fillText('F on each arm = 2P/c = ' + sci(F()) + ' N', 12, H - 64);
  ctx.fillText('arms take F_max = 4e21 N  ->  ' + (ratio() > 1 ? 'OVERLOAD ' + ratio().toFixed(1) + 'x' : 'within limit'), 12, H - 48);

  for (const p of photons) { ctx.fillStyle = '#ffe14d'; ctx.beginPath(); ctx.arc(p.x, p.y, 2.2, 0, TAU); ctx.fill(); }
  for (const f of flashes) {
    ctx.globalAlpha = 1 - f.t / .4;
    arrow(f.x + 6, f.y, 0, 14, '#ff9a4d', 1 - f.t / .4, 2);          // momentum handed to the arm
    ctx.strokeStyle = '#fff'; ctx.lineWidth = 1.2;
    ctx.beginPath(); ctx.arc(f.x, f.y, 2 + f.t * 30, 0, TAU); ctx.stroke();
    ctx.globalAlpha = 1;
  }
  ctx.restore();
}

function drawSolarPOV() {
  const g = solarGeom();
  const heat = Math.min(1, tRun / 1.2) * Math.min(1, ratio() / 11.5);
  const ex = g.ex0 + push();
  drawSun(g.sx, g.sy, 20);
  ctx.fillStyle = '#dfe7f3'; ctx.font = '12px ui-monospace, Menlo, monospace'; ctx.textAlign = 'center';
  ctx.fillText('Sun', g.sx, g.sy + 70);
  // Dyson shell: panel ring capturing all of L_sun
  ctx.strokeStyle = '#ffd27f'; ctx.lineWidth = 2;
  ctx.beginPath(); ctx.arc(g.sx, g.sy, g.shell, 0, TAU); ctx.stroke();
  for (let i = 0; i < 28; i++) {
    const a = i / 28 * TAU + tRun * .15;
    ctx.strokeStyle = 'rgba(255,210,127,.8)'; ctx.lineWidth = 2.5;
    ctx.beginPath();
    ctx.moveTo(g.sx + Math.cos(a) * (g.shell - 5), g.sy + Math.sin(a) * (g.shell - 5));
    ctx.lineTo(g.sx + Math.cos(a) * (g.shell + 4), g.sy + Math.sin(a) * (g.shell + 4));
    ctx.stroke();
  }
  ctx.fillStyle = '#ffd27f'; ctx.fillText('Dyson sphere: captures all of L☉ = 3.828e26 W,', g.sx + 30, g.sy - g.shell - 26);
  ctx.fillText('re-beams it as one pulse of power P', g.sx + 30, g.sy - g.shell - 12);
  // old orbit + beam
  ctx.strokeStyle = 'rgba(79,163,255,.20)'; ctx.lineWidth = 1;
  ctx.beginPath(); ctx.arc(g.sx, g.sy, g.ex0 - g.sx, 0, TAU); ctx.stroke();
  const bg = ctx.createLinearGradient(g.sx + g.shell, 0, ex, 0);
  bg.addColorStop(0, 'rgba(255,225,77,.95)'); bg.addColorStop(1, 'rgba(255,90,90,.95)');
  ctx.strokeStyle = bg; ctx.lineWidth = 5; ctx.globalAlpha = .8;
  ctx.beginPath(); ctx.moveTo(g.sx + g.shell, g.ey); ctx.lineTo(ex - 10, g.ey); ctx.stroke(); ctx.globalAlpha = 1;
  ctx.fillStyle = '#ffe1b0'; ctx.textAlign = 'center';
  ctx.fillText('high-energy beam → F = 2P/c pushes Earth away from the Sun', (g.sx + ex) / 2, g.ey - 16);
  // Earth + reflector arms (mini)
  drawEarthBody(ex, g.ey, 11, broken);
  if (!broken) for (const a of ARM_A) armShape(ex + Math.cos(a) * 10, g.ey + Math.sin(a) * 10, a, 16, 6, heat, 1);
  for (const d of debris) {
    ctx.save(); ctx.translate(d.x - earthGeom().cx + ex, d.y - earthGeom().cy + g.ey); ctx.rotate(d.rot);
    armShape(0, 0, d.a, 14, 5, 1, Math.max(0, 1 - d.t / 2.6));
    ctx.restore();
  }
  arrow(ex + 26, g.ey, 0, 22, '#7fe08a', .9, 2);
  ctx.fillStyle = '#dfe7f3'; ctx.fillText('Earth (pushed outward)', ex, g.ey + 34);
  for (const p of photons) { ctx.fillStyle = '#ffe14d'; ctx.beginPath(); ctx.arc(p.x, p.y, 2, 0, TAU); ctx.fill(); }
  for (const f of flashes) {
    ctx.globalAlpha = 1 - f.t / .4;
    ctx.strokeStyle = '#fff'; ctx.lineWidth = 1.2;
    ctx.beginPath(); ctx.arc(f.x, f.y, 2 + f.t * 26, 0, TAU); ctx.stroke();
    ctx.globalAlpha = 1;
  }
}

function stamp(lines, color) {
  ctx.font = '13px ui-monospace, Menlo, monospace'; ctx.textAlign = 'center';
  const w = Math.max(...lines.map(l => ctx.measureText(l).width)) + 28;
  const x = W / 2 - w / 2, y = H * .12;
  ctx.fillStyle = 'rgba(5,7,12,.86)'; ctx.strokeStyle = color; ctx.lineWidth = 1.5;
  ctx.beginPath(); ctx.roundRect(x, y, w, lines.length * 20 + 16, 8); ctx.fill(); ctx.stroke();
  ctx.fillStyle = color;
  lines.forEach((l, i) => ctx.fillText(l, W / 2, y + 24 + i * 20));
}

function draw() {
  ctx.clearRect(0, 0, W, H);
  ctx.fillStyle = '#05070c'; ctx.fillRect(0, 0, W, H);
  drawStars();
  if (pov === 'earth') drawEarthPOV(); else drawSolarPOV();
  if (broken) stamp([
    'ARM FAILURE:  F = 2P/c = ' + sci(F()) + ' N   vs   F_max = 4e21 N  (' + ratio().toFixed(1) + 'x overload)',
    'the octo-hands snap under the incident beam  ->  NOT POSSIBLE (notes)',
  ], '#ff5a5a');
  else if (ratio() <= 1 && tRun > BREAK_T) stamp([
    'arms hold:  F = ' + sci(F()) + ' N  <=  F_max = 4e21 N',
    'Earth is pushed away from the Sun at a = ' + sci(F() / M_E) + ' m/s^2',
  ], '#7fe08a');
}

// ------------------------------------------------------------------ HUD + calc panel
function panel() {
  $('cP').textContent = sci(P); $('cF').textContent = sci(F());
  $('cR').textContent = ratio().toFixed(1); $('cA').textContent = sci(F() / M_E);
  $('powLbl').textContent = sci(P) + ' W';
  const bar = $('hudBar'); if (bar) bar.style.width = (Math.min(1, ratio() / 12) * 100).toFixed(0) + '%';
  $('hud').innerHTML =
    (pov === 'earth' ? 'EARTH POV' : 'SOLAR SYSTEM POV') + '\n' +
    'beam on for ' + tRun.toFixed(1) + ' s\n' +
    'P = ' + sci(P) + ' W\n' +
    'F = 2P/c = ' + sci(F()) + ' N\n' +
    'F/F_max = ' + ratio().toFixed(1) + ' x' +
    '<span class="bar"><i id="hudBar"></i></span>' +
    (broken ? '\n*** ARMS BROKEN ***' : '');
  const b2 = $('hudBar'); if (b2) b2.style.width = (Math.min(1, ratio() / 12) * 100).toFixed(0) + '%';
}

// ------------------------------------------------------------------ loop
function frame(ts) {
  if (!last) last = ts;
  const dt = Math.min(.05, (ts - last) / 1000); last = ts;
  if (W !== cv.clientWidth || H !== cv.clientHeight) resize();
  update(dt); draw(); panel();
  requestAnimationFrame(frame);
}
resize(); syncPlay(); requestAnimationFrame(frame);
