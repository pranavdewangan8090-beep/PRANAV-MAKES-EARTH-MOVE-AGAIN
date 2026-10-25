// SIM 2 · HYDROGEN COLLECTING DISK
// Earth's eight octo-hands hold a ring-net around Earth; the ram flow of interstellar
// hydrogen (n = 1e6 atoms/m^3, dv = 23 km/s) is caught by the net and funnelled to Earth.
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
const N_D   = 1e6;                    // H atoms per m^3  (1 m^3 -> 1e6, 1 km^3 -> 1e15)
const R_E   = 6.371e6;                // Earth radius [m]
const DV    = 2.3e4;                  // relative speed [m/s] = 23 km/s
const M_H   = 1.67e-27;               // H atom mass [kg]
const YR    = 3.15576e7;              // s per year
const T_END = 1e4;                    // mission length [yr] (notes)
const A_NET = 8 * Math.PI * R_E * R_E;   // net area 8*pi*R^2 = 1.02e15 m^2
const RATE  = N_D * A_NET * DV;          // atoms captured per second = 2.35e25
const YR_PER_S = 500;                 // simulated years per wall-clock second
const DOT_ATOMS = RATE * T_END * YR / 240;  // atoms represented by one drawn dot

// ------------------------------------------------------------------ state
let pov = 'earth', playing = true, tYr = 0, last = 0, spawnAcc = 0;
let dots = [], flashes = [];
const stars = [];
for (let i = 0; i < 150; i++) stars.push({ x: Math.random(), y: Math.random(), r: Math.random() * 1.2 + .3, a: Math.random() * .45 + .15 });

function reset() { tYr = 0; dots = []; flashes = []; spawnAcc = 0; playing = true; syncPlay(); }
function syncPlay() { $('btnPlay').textContent = playing ? '❚❚ Pause' : '▶ Play'; }

// ------------------------------------------------------------------ controls
$('povSolar').onclick = () => { pov = 'solar'; $('povSolar').classList.add('active'); $('povEarth').classList.remove('active'); };
$('povEarth').onclick = () => { pov = 'earth'; $('povEarth').classList.add('active'); $('povSolar').classList.remove('active'); };
$('btnPlay').onclick = () => { if (tYr >= T_END) reset(); else { playing = !playing; syncPlay(); } };
$('btnReset').onclick = reset;

// ------------------------------------------------------------------ update
function update(dt) {
  if (!playing) return;
  tYr = Math.min(T_END, tYr + dt * YR_PER_S);
  if (tYr >= T_END) { playing = false; syncPlay(); }

  if (pov === 'earth') {
    const g = earthGeom();
    spawnAcc += dt * 26;
    while (spawnAcc > 1) {
      spawnAcc -= 1;
      dots.push({ st: 0, x: g.cx + g.Rn + 40 + Math.random() * 60, y: g.cy - g.Rn + Math.random() * 2 * g.Rn, v: 80 + Math.random() * 40, r: 0 });
    }
    for (const d of dots) {
      if (d.st === 0) {                       // free stream, flowing past Earth (ram side = +x)
        d.x -= d.v * dt;
        const dx = d.x - g.cx, dy = d.y - g.cy;
        if (Math.hypot(dx, dy) <= g.Rn) { d.st = 1; d.r = g.Rn; d.a = Math.atan2(dy, dx); }
      } else if (d.st === 1) {                // caught in the net: funnelled inward along a spoke
        d.r -= 95 * dt;
        d.x = g.cx + Math.cos(d.a) * d.r; d.y = g.cy + Math.sin(d.a) * d.r;
        if (d.r <= g.Re + 4) { d.st = 3; flashes.push({ x: d.x, y: d.y, t: 0 }); }
      }
    }
    dots = dots.filter(d => d.st < 3 && d.x > g.cx - g.Rn - 80);
  } else {
    const g = solarGeom();
    if (dots.length < 110) dots.push({ st: 0, x: W + 10, y: Math.random() * H, v: 26 + Math.random() * 10 });
    for (const d of dots) {
      d.x -= d.v * dt;
      if (Math.abs(d.y - g.ey) < g.lane && Math.abs(d.x - g.ex) < 14) { d.st = 3; flashes.push({ x: g.ex, y: g.ey, t: 0 }); }
    }
    dots = dots.filter(d => d.st === 0 && d.x > -20);
  }
  for (const f of flashes) f.t += dt;
  flashes = flashes.filter(f => f.t < .5);
}

// ------------------------------------------------------------------ geometry
function earthGeom() {
  const Rn = Math.min(W, H) * .34;
  return { cx: W * .5, cy: H * .5, Rn, Re: Rn * .30 };
}
function solarGeom() {
  return { sx: W * .22, sy: H * .5, ex: W * .64, ey: H * .5, lane: 15, orb: W * .42 };
}

// ------------------------------------------------------------------ drawing
function drawStars() {
  for (const s of stars) { ctx.globalAlpha = s.a; ctx.fillStyle = '#cfe4ff'; ctx.fillRect(s.x * W, s.y * H, s.r, s.r); }
  ctx.globalAlpha = 1;
}
function drawFlashes() {
  for (const f of flashes) {
    ctx.globalAlpha = 1 - f.t / .5;
    ctx.strokeStyle = '#9fffcf'; ctx.lineWidth = 1.5;
    ctx.beginPath(); ctx.arc(f.x, f.y, 3 + f.t * 34, 0, TAU); ctx.stroke();
  }
  ctx.globalAlpha = 1;
}
function arrow(x, y, ang, len, color, alpha) {
  ctx.globalAlpha = alpha; ctx.strokeStyle = color; ctx.lineWidth = 1.4;
  const ux = Math.cos(ang), uy = Math.sin(ang);
  ctx.beginPath(); ctx.moveTo(x - ux * len, y - uy * len); ctx.lineTo(x, y);
  ctx.lineTo(x - ux * 6 - uy * 4, y - uy * 6 + ux * 4);
  ctx.moveTo(x, y); ctx.lineTo(x - ux * 6 + uy * 4, y - uy * 6 - ux * 4);
  ctx.stroke(); ctx.globalAlpha = 1;
}

function drawEarthPOV() {
  const g = earthGeom();
  // incoming ram flow (right side): arrows + label
  for (let i = 0; i < 3; i++) arrow(W - 26, g.cy - 60 + i * 60, Math.PI, 26, '#bfe8ff', .8);
  ctx.fillStyle = '#bfe8ff'; ctx.font = '12px ui-monospace, Menlo, monospace'; ctx.textAlign = 'right';
  ctx.fillText('H gas  n = 10^6 /m^3', W - 14, g.cy - 84);
  ctx.fillText('ram speed dv = 23 km/s', W - 14, g.cy - 70);

  // the net ring held by the octo-hands
  ctx.strokeStyle = 'rgba(127,208,255,.85)'; ctx.lineWidth = 2;
  ctx.beginPath(); ctx.arc(g.cx, g.cy, g.Rn, 0, TAU); ctx.stroke();
  ctx.strokeStyle = 'rgba(127,208,255,.30)'; ctx.lineWidth = 1;      // hatching, like the sketch
  for (let i = 0; i < 72; i++) {
    const a = i / 72 * TAU;
    ctx.beginPath();
    ctx.moveTo(g.cx + Math.cos(a) * g.Rn, g.cy + Math.sin(a) * g.Rn);
    ctx.lineTo(g.cx + Math.cos(a) * (g.Rn - 9), g.cy + Math.sin(a) * (g.Rn - 9));
    ctx.stroke();
  }
  // eight octo-hands (spokes) + sagging net threads between the hand tips
  for (let k = 0; k < 8; k++) {
    const a = (k * 45 + 22.5) * Math.PI / 180;
    const x0 = g.cx + Math.cos(a) * g.Re * .95, y0 = g.cy + Math.sin(a) * g.Re * .95;
    const x1 = g.cx + Math.cos(a) * g.Rn, y1 = g.cy + Math.sin(a) * g.Rn;
    ctx.strokeStyle = '#cfd8e3'; ctx.lineWidth = 3;
    ctx.beginPath(); ctx.moveTo(x0, y0); ctx.lineTo(x1, y1); ctx.stroke();
    ctx.fillStyle = '#e8f4ff';
    ctx.beginPath(); ctx.arc(x1, y1, 4, 0, TAU); ctx.fill();
    arrow(g.cx + Math.cos(a) * g.Rn * .62, g.cy + Math.sin(a) * g.Rn * .62, a + Math.PI, 12, '#9fffcf', .55); // collected H funnels inward
    const a2 = ((k + 1) * 45 + 22.5) * Math.PI / 180, am = (a + a2) / 2;
    ctx.strokeStyle = 'rgba(127,208,255,.45)'; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(x1, y1);
    ctx.quadraticCurveTo(g.cx + Math.cos(am) * g.Rn * .93, g.cy + Math.sin(am) * g.Rn * .93,
      g.cx + Math.cos(a2) * g.Rn, g.cy + Math.sin(a2) * g.Rn);
    ctx.stroke();
  }
  // Earth
  const gr = ctx.createRadialGradient(g.cx - g.Re * .3, g.cy - g.Re * .3, g.Re * .2, g.cx, g.cy, g.Re);
  gr.addColorStop(0, '#7cc4ff'); gr.addColorStop(1, '#0a2a55');
  ctx.fillStyle = gr; ctx.beginPath(); ctx.arc(g.cx, g.cy, g.Re, 0, TAU); ctx.fill();
  ctx.fillStyle = '#dfe7f3'; ctx.font = '12px ui-monospace, Menlo, monospace'; ctx.textAlign = 'center';
  ctx.fillText('Earth', g.cx, g.cy + g.Re + 16);

  for (const d of dots) {
    ctx.fillStyle = d.st === 0 ? '#bfe8ff' : '#9fffcf';
    ctx.beginPath(); ctx.arc(d.x, d.y, 2, 0, TAU); ctx.fill();
  }
  drawFlashes();
}

function drawSolarPOV() {
  const g = solarGeom();
  // Sun + Earth's 1 AU orbit for context
  const sg = ctx.createRadialGradient(g.sx, g.sy, 2, g.sx, g.sy, 46);
  sg.addColorStop(0, '#fff3b0'); sg.addColorStop(.4, '#ffcc33'); sg.addColorStop(1, 'rgba(255,204,51,0)');
  ctx.fillStyle = sg; ctx.beginPath(); ctx.arc(g.sx, g.sy, 46, 0, TAU); ctx.fill();
  ctx.fillStyle = '#dfe7f3'; ctx.font = '12px ui-monospace, Menlo, monospace'; ctx.textAlign = 'center';
  ctx.fillText('Sun', g.sx, g.sy + 62);
  ctx.strokeStyle = 'rgba(79,163,255,.22)'; ctx.lineWidth = 1;
  ctx.beginPath(); ctx.arc(g.sx, g.sy, g.orb, 0, TAU); ctx.stroke();
  ctx.fillStyle = '#8090a8'; ctx.fillText("Earth's orbit · 1 AU", g.sx, g.sy - g.orb - 8);

  // swept tube: the lane the net has already cleared (behind) vs pristine gas (ahead)
  ctx.setLineDash([5, 6]); ctx.strokeStyle = 'rgba(127,208,255,.35)';
  ctx.beginPath(); ctx.moveTo(0, g.ey - g.lane); ctx.lineTo(g.ex, g.ey - g.lane);
  ctx.moveTo(0, g.ey + g.lane); ctx.lineTo(g.ex, g.ey + g.lane); ctx.stroke(); ctx.setLineDash([]);
  ctx.fillStyle = '#8090a8'; ctx.textAlign = 'left';
  ctx.fillText('swept tube  Vol = dv · t · 8πR²  (cleared lane)', 12, g.ey - g.lane - 8);

  // Earth with its tiny net ring + 8 hands
  ctx.fillStyle = '#4fa3ff'; ctx.beginPath(); ctx.arc(g.ex, g.ey, 9, 0, TAU); ctx.fill();
  ctx.strokeStyle = 'rgba(127,208,255,.9)'; ctx.lineWidth = 1.4;
  ctx.beginPath(); ctx.arc(g.ex, g.ey, 17, 0, TAU); ctx.stroke();
  for (let k = 0; k < 8; k++) {
    const a = (k * 45 + 22.5) * Math.PI / 180;
    ctx.strokeStyle = '#cfd8e3'; ctx.lineWidth = 1.4;
    ctx.beginPath(); ctx.moveTo(g.ex + Math.cos(a) * 9, g.ey + Math.sin(a) * 9);
    ctx.lineTo(g.ex + Math.cos(a) * 17, g.ey + Math.sin(a) * 17); ctx.stroke();
  }
  ctx.fillStyle = '#dfe7f3'; ctx.textAlign = 'center';
  ctx.fillText('Earth + collecting disk', g.ex, g.ey + 34);

  for (const d of dots) { ctx.fillStyle = '#bfe8ff'; ctx.beginPath(); ctx.arc(d.x, d.y, 1.6, 0, TAU); ctx.fill(); }
  drawFlashes();
}

function stamp(lines, color) {
  ctx.font = '13px ui-monospace, Menlo, monospace'; ctx.textAlign = 'center';
  const w = Math.max(...lines.map(l => ctx.measureText(l).width)) + 28;
  const x = W / 2 - w / 2, y = H * .16;
  ctx.fillStyle = 'rgba(5,7,12,.85)'; ctx.strokeStyle = color; ctx.lineWidth = 1.5;
  ctx.beginPath(); ctx.roundRect(x, y, w, lines.length * 20 + 16, 8); ctx.fill(); ctx.stroke();
  ctx.fillStyle = color;
  lines.forEach((l, i) => ctx.fillText(l, W / 2, y + 24 + i * 20));
}

function draw() {
  ctx.clearRect(0, 0, W, H);
  ctx.fillStyle = '#05070c'; ctx.fillRect(0, 0, W, H);
  drawStars();
  if (pov === 'earth') drawEarthPOV(); else drawSolarPOV();
  if (tYr >= T_END) stamp([
    '10,000 yr of sweeping:  Vol = 7.4e30 m^3,  N = 7.4e36 atoms,  M = 1.2e10 kg',
    'far too little H to feed a fusion cycle  ->  INFEASIBLE (notes)',
  ], '#ff9a4d');
}

// ------------------------------------------------------------------ HUD + calc panel
function panel() {
  const tSec = tYr * YR;
  const vol = DV * tSec * A_NET, n = N_D * vol, m = n * M_H;
  $('cA').textContent = sci(A_NET); $('cVol').textContent = sci(vol);
  $('cN').textContent = sci(n); $('cM').textContent = sci(m);
  $('cDot').textContent = sci(DOT_ATOMS);
  $('hud').textContent =
    (pov === 'earth' ? 'EARTH POV' : 'SOLAR SYSTEM POV') + '\n' +
    't = ' + Math.round(tYr).toLocaleString('en-US') + ' yr of 10,000 yr\n' +
    'captured N = ' + sci(n) + ' atoms\n' +
    'captured M = ' + sci(m) + ' kg';
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
