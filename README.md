# OCTO-EARTH prototype

A reinforcement-learning agent (PPO) ejects small packets of asteroid mass from Earth. The recoil
slowly moves Earth from 1 AU toward a stable, near-circular orbit at **2.7 AU** within a mission time
limit (default 10,000 simulated years). This is a **basic, runnable prototype**: it is meant to be
simple and testable, and can be extended later.

## Start (Windows)

Double-click **`RUN_OCTOEARTH.bat`**. It will:

1. find Python 3.10–3.13 (64-bit) and create `.venv` next to the file (no admin rights);
2. install the missing packages from `requirements.txt`. **The first install needs internet
   (~300 MB, mostly PyTorch CPU).**
3. compile the physics kernels, run the quick physics checks, and print the analytic
   feasibility gate;
4. open the viewer in your browser (`http://127.0.0.1:8800/`, or the next free port), then train PPO
   for 1,000 episodes. When training finishes it automatically compares PPO with the baselines.

On a 2-core test machine, 1,000 episodes took **about 2 minutes** of wall-clock time, plus about
10 s for the comparison. Close the window or press Ctrl+C to stop. If you stop during training,
the model is still saved.

Other systems: `pip install -r requirements.txt`, then `python octoearth_rl.py`.

| command | what it does |
|---|---|
| `python octoearth_rl.py` | checks → viewer → training → comparison |
| `python octoearth_rl.py --smoke` | short pipeline test (30 episodes, 4,000-year missions) |
| `python octoearth_rl.py --check` | all 14 physics checks, then exit (exit code 1 if any fail) |
| `python octoearth_rl.py --headless` | no browser: train + compare + `report.md` |
| `python octoearth_rl.py --no-train` | viewer only (shows results saved in `outputs/`) |
| `--episodes N`, `--preset realistic`, `--set mission_years=5000 target_au=2.5`, `--port`, `--out DIR` | overrides (invalid or unknown values are rejected with a message, exit code 2) |
| `python -m pytest -q tests` | test suite (needs `pip install pytest`) |

## Files

| file | content |
|---|---|
| `octoearth_rl.py` | **all** Python: config, feasibility gate, physics (fast and high-fidelity), Gymnasium env, reward, baselines, PPO training, evaluation, explanation, report, physics checks, HTTP API |
| `frontend/` | Three.js viewer (`index.html`, `app.js`, `style.css`, `vendor/` three.js r160, MIT). It does **no** physics. |
| `outputs/` | written by a run: `best_model.zip`, `last_model.zip`, `best_model_info.json`, `training_log.csv`, `evals.json`, `comparison.json`, `report.md`, `physics_checks.json`, `feasibility.json`, `config.json` |
| `example_results/` | outputs of the tested 1,000-episode run described below |
| `tests/test_octoearth.py` | pytest suite, 15 tests (physics checks, Moon in the training model, gate, env API, energy/mass limits, baselines, recorded-state continuity, packet paths vs Kepler, viewer data, parameter validation, documentation coverage) |

## Parameters

Change them in the viewer (left panel → Parameters → Apply, then Start training), with
`--set key=value`, or in `Config` / `PRESETS` in `octoearth_rl.py`. Every parameter is documented in
one place (`PARAM_DOCS` in `octoearth_rl.py`), and the viewer shows that text in the ⓘ tooltips and
in the **Parameters & Results Guide** tab: name, value, unit, physical meaning, why it matters, the
effect of increasing or decreasing it, the allowed range, constraints and its category.

| category | parameters |
|---|---|
| Engineering assumption | asteroid mass, max packet mass, max/min ejection speed (exhaust velocity), energy budget, launch efficiency, ejection frequency |
| Mission requirement | target orbit radius, mission duration, tolerances on a / e / inclination, stability-check duration |
| Simulation setting | decision interval (the RL time step), high-fidelity time step, max tracked fragments, packet hand-over distance, fragment escape distance, random start phase, seed |
| RL hyperparameter | training episodes, PPO steps per update, mini-batch, epochs, learning rate, discount factor, evaluation interval |
| Reward weight (scoring only) | progress, eccentricity, inclination, energy, mass, time, success, final miss, failure |
| Physical constant (fixed) | G, masses of Sun / Earth / Moon, AU, year, Earth radius, Moon distance, solar constant, speed of light |
| Viewer setting | simulation speed (1 day to 500 years of simulated time per real second) |

**Validation (Python is authoritative):** values outside the allowed range are rejected, never
clipped or reinterpreted. Cross-checks: min speed < max speed ≤ 3e7 m/s (10 % of c); packet mass ≤
asteroid mass; PPO batch ≤ PPO steps; target 1.1–5 AU; 2–1,000 decisions per episode. "Reset to
defaults" restores the defaults of the current preset. Units never change.

Presets:

- **`hypothetical`** (default): 3e22 kg asteroid (0.4 lunar masses), 1e18 kg packets, ≤ 5,000 km/s
  (1.7 % of c), 3e35 J energy budget. These values are deliberately chosen so the task is *just*
  feasible. They are not realistic engineering.
- **`realistic`**: a Ceres-mass asteroid, a 30 km/s launcher, and all sunlight intercepted by Earth
  for 10,000 years as energy. The gate reports **INFEASIBLE**: the mass-limited Δv is only 3.7e-4 of
  what is needed. The program says so instead of pretending.

## Physics model (simplified and documented)

- **Feasibility gate (analytic, runs before training).** It computes:
  - the Δv for a slow spiral (11.66 km/s for 1→2.7 AU) and for a Hohmann transfer;
  - the change in orbital energy and angular momentum;
  - the rocket-equation reaction mass, with launch from Earth's surface including escape speed;
  - the minimum launch energy and average power;
  - the thrust-rate limit and the combined mass + time + energy check.
- **Fast model (used for training) – the Moon IS simulated.**
  - Hierarchical two-body model. The Earth–Moon barycentre (Earth + Moon + docked asteroid) follows
    an exact Kepler orbit around the Sun, **and the Moon follows an exact Kepler orbit around
    Earth(+asteroid) under their mutual gravity** (G·(M_Earth + m_asteroid + M_Moon)). Earth wobbles
    about 4,700 km around the barycentre because of the Moon's pull.
  - Each ejection is an impulsive, exactly momentum-conserving recoil **applied to Earth** (where the
    launcher is). The Moon is not pushed directly; it follows through gravity, so the Earth–Moon
    relative velocity changes by exactly Earth's recoil (check 14). The packet's final speed accounts
    for climbing out of Earth's, the asteroid's and the Moon's gravity.
  - The Moon's distance and boundedness are part of the agent's observation, and losing the Moon
    ends an episode as a failure.
  - **Approximation:** the Sun's tidal pull on the Moon is neglected here (it is in the N-body
    model). The Moon's distance agrees with the N-body Moon within ~10,000 km (2.7 %) over a year,
    but its orbital phase drifts by about 22° per year (check 14). Packets hitting the Moon and
    fragment trajectories after escape exist only in the high-fidelity model.
  - Launch energy is ½·m·u²/efficiency and is drawn from the finite budget. The finite asteroid
    mass decreases.
  - Packets slower than escape speed fall back and give no net recoil.
  - Packets are tracked only in a ledger. A Moon-boundedness check runs each decision.
  - Speed: about 1.4 ms per 100-year decision (training: 1,000 episodes in about 4 minutes).
- **High-fidelity model (used for validation and the viewer).**
  - Full N-body Sun, Earth (+ asteroid), Moon and explicit fragments. Fragments feel the Sun,
    Earth and Moon and pull back on them. **Fragment–fragment gravity is neglected** (the
    fragments are tiny compared with Earth and the Moon and spread out quickly).
  - Integrator: 4th-order Yoshida, dt = 6 h.
  - Packets are handed over at 2e6 km from Earth with their exact escape-corrected velocity.
  - Up to 48 representative fragments exist at once. When the slots are full, the closest
    fragments merge inelastically (momentum conserved, energy loss recorded in the ledger).
  - Fragments that hit Earth, the Moon or the Sun merge with that body. Unbound fragments
    beyond 60 AU are recorded as *escaped* with their mass, momentum and energy. Nothing silently
    disappears.
  - After the mission, the orbit is integrated for 200 years with no thrust to check that it is
    stable.
- **The agent never sets Earth's velocity.** Its action per 100-year interval is: throttle (packet
  mass), two thrust angles in Earth's radial/along-track/normal frame, and launch speed. The physics
  turns these into packets, and all energy comes from the finite budget.
- **Recorded states for the viewer.** Each ejection event produces one record at the ejection
  instant: time, position, velocity just before and just after the event, packet mass, launch
  speed, relative speed, packet heliocentric velocity, fate (leaves the Solar System / bound to the
  Sun / fell back / hit the Moon), remaining asteroid mass and energy, recoil Δv, energy drawn and
  launch direction. Both models also record the Moon 8 times per event.
- **Asteroid.** The asteroid is docked to Earth (the launcher throws its material away piece by
  piece). Its gravity is included in both models: its mass adds to Earth's pull on the Sun, the Moon
  and every launched packet, and that pull weakens as it is used up (check 2). Because it is attached,
  its pull on Earth itself is an internal force with no net effect. Its own orbit and structure are
  not modelled.
- **Habitability:** insolation relative to today (S/S₀ = 1/a² ≈ 0.14 at 2.7 AU) and a black-body
  equilibrium temperature. This is a rough indicator, not a climate model.

## Validation (tested, `--check` / pytest, all PASS)

| check | result |
|---|---|
| Kepler closure / N-body vs Kepler (1 y) | 1.4e-14 / 261 m |
| Gravity of Sun, Earth, Moon, asteroid, fragment | analytic values reproduced |
| Equal and opposite forces | Σ m·a = 7.5e-17 (relative) |
| Ejection momentum / angular momentum | 2e-16 / 1.8e-16 (round-off) |
| Ejection energy vs ½μu² − G·dm·M/R | 2.6e-8 relative; budget never exceeded |
| Mass accounting incl. escaped fragments | 1.4e-16; reservoir cannot go negative |
| Reversed thrust direction | Δa +0.0225 vs −0.0217 AU |
| Fall-back / escape / Earth impact events | all handled and recorded in the ledger |
| Integrator convergence (12/6/3 h) | error ratios 13.2 and 16.0 (4th order → 16) |
| 100-year drift | \|dE/E\| 9e-12, \|dL/L\| 1e-15 |
| Fast vs high-fidelity (heuristic, 1,500 y) | a agrees to 1.5e-8 |
| Reproducibility | identical for a fixed seed |
| 13 Viewer continuity | consecutive records joined by a pure Kepler arc to 1.1e-11 AU (no hidden jumps); the viewer's Hermite curve is within 3.3e-5 AU (≈ 5,000 km) of the exact arc; recoil column = \|v_after − v_before\| |
| 14 Moon in the fast (training) model | 1 y without thrust: Earth–Moon distance 363,297–405,504 km vs N-body 361,807–399,330 km (max diff 10,422 km); phase drift 22°/yr (Sun's tide neglected); barycentre diff 99 km; Earth wobble 4,874 km; a launch changes the Earth–Moon relative velocity by exactly Earth's recoil (6e-12) |

## The viewer (what you see, and what is approximate)

Python computes all physics. The browser only draws recorded states; it never moves Earth on its own.

- **Controls:** ▶ Play / ❚❚ Pause / ⟲ Reset (re-runs the same mission from year 0 with the same seed,
  so it is identical), a logarithmic speed slider from **1 day to 500 years of simulated time per
  real second**, and "camera follows Earth" for a close-up. The HUD always shows simulated time and
  wall-clock time separately. The physics runs ahead of the display by at most 3 decision intervals
  or 2.5 s of display time (dim grey dots = computed but not yet shown).
- **Earth moves continuously.** Between two records (1/8 year apart) Earth is drawn on a cubic
  Hermite curve in polar coordinates through the recorded positions. The curve uses the velocity
  *after* the event at its start and *before* the next event at its end, so every recoil appears as a
  real kink, not smoothed away. Accuracy is checked by physics check 13. If two records are more
  than 2.5 event spacings apart (a data gap), Earth is **held** at the last record and a notice is
  shown. It never interpolates across a gap.
- **Labels:** Sun, Earth, Moon (with its current distance in km) and asteroid (with the % left) are
  labelled on screen, so each body can be found at any zoom level.
- **Ejections:** the asteroid is drawn as a brown rock sitting on Earth's marker, on the side the
  packets leave from (size ∝ remaining mass^1/3, labelled "asteroid (docked) · X % left"). It shrinks
  with every packet. Each
  packet is drawn along its own path, computed in Python with two-body solar gravity from its real
  release state, for up to 1 year or 60 AU. Orange = leaves the Solar System, yellow = stays bound to
  the Sun, red ✕ = fell back to Earth. At high speeds paths stay visible ≥ 0.6 s of real time, and
  only 1 of every k packets is drawn (an on-screen note says so). All packets remain in the physics
  ledger.
- **Recoil:** a green arrow on Earth shows the direction of v_after − v_before of the latest
  packet. A label gives the recoil per packet (≈ 0.4 m/s) and the total Δv so far.
- **Paths:** solid line = past path (dark → light = earlier → later); dashed = the coasting
  (osculating) orbit Earth would follow if thrust stopped now; green circle = target.
- **Moon:** simulated and drawn in **both** models, from its own recorded stream (distance ×100),
  with a thin line to Earth. A thin grey ellipse shows its current simulated orbit around Earth
  (osculating orbit from the recorded Earth–Moon state, same ×100 scale). The HUD shows the
  Earth–Moon distance and whether the Moon is still bound. Before a mission starts, the view shows
  the simulated year-0 state (Earth, Moon, full asteroid). The live training view shows the Moon
  and the asteroid at the end of the latest training episode.
- **Event log** (simulated time): at ≤ 1 simulated year per second, one line per packet ("Year 125.44:
  ejected 1.00e18 kg at 2,480 km/s → leaves the Solar System; Earth Δv 0.408 m/s; asteroid left …").
  Above that, one summary line per decision. Special events are always listed: first packet, packets
  falling back, Moon hits, coasting start/stop, resources used up, Moon lost, mission end, and the
  stability check.
- **Results & best model tab:** file name, path, size and date of `best_model.zip`; the selection rule
  and the deterministic tests during training; comparison with the no-ejection and heuristic
  baselines; final orbit vs each target criterion (pass/fail); a decision-by-decision table of the
  actual actions (timing, packet mass and count, speed, direction, Δv, Δa, e). Each decision is
  classified as helped / mixed / hurt / coast, with a reason built from the recorded numbers. The tab
  also lists training-episode outcomes, failed tests, and resource and constraint counts (packets
  that fell back or hit the Moon). It states explicitly that the policy is not proven optimal and
  that reward is only a training score.
- **Approximations flagged in the viewer:** not to scale (markers enlarged, Moon distance ×100);
  packet paths are shifted 0.047 AU so they start at the asteroid marker; packet paths use Sun-only
  gravity after release; in the fast model the Sun's tide on the Moon is neglected.

## Tested results (1,000 episodes, `example_results/`)

See `example_results/report.md`. These are the numbers from that run; on another PC the exact values
can differ slightly. "Best model" means **the best model found in the tested runs**; it is not a
proven global optimum.

Training: 1,000 episodes, 79,105 decisions, **254 s wall-clock** on 2 CPU cores, fast model (with the
Moon simulated).
`best_model.zip` = episode 750, chosen by deterministic evaluation on seeds 1001–1002.
Ranking order: success, then orbit error, then energy.

Comparison: physical metrics, high-fidelity N-body run on seed 2001, then 200 years without thrust.

| policy | fast model success (3 seeds) | high-fidelity result | final a [AU] | e | sim. years | asteroid used | energy drawn [J] | insolation | T_eq | stable 200 y |
|---|---|---|---|---|---|---|---|---|---|---|
| No ejection | 0 % | deadline | 1.0000 | 0.0167 | 10,000 | 0 % | 0 | 1.00× | 255 K | – |
| Hand-designed heuristic | 100 % | success | 2.7000 | 0.0080 | 3,600 | 95.0 % | 1.75e35 | 0.137× | 155 K | yes |
| **Best PPO model** | 100 % | **success** | 2.6675 | 0.0079 | 5,800 | 74.1 % | 2.29e35 | 0.141× | 156 K | yes |

- **Target reached** by the best PPO model, confirmed in the N-body check: |a − 2.7| = 0.033 AU, which
  is within the 0.05 tolerance, and e = 0.008. The Moon stayed bound (Earth–Moon distance at arrival
  373,919 km), and the orbit stayed within tolerance for 200 years without thrust (largest deviation
  of a from target 0.033 AU). All 58 of its decisions raised the orbit while keeping e within
  tolerance ("helped"). Of the 1,000 *exploring* training episodes, 914 ended with resources exhausted
  and 86 at the deadline. Only deterministic tests select the model.
- High-fidelity conservation (PPO run): momentum residual 2.0e-13, angular momentum 3.3e-14,
  mass 0, energy 1.7e-6 (see limitations).
- `example_results/` comes from a complete re-run (training + comparison) after the Moon was added
  to the training model. The earlier results (trained without the Moon) are superseded.
- What PPO learned:
  - almost pure along-track (prograde) recoil, cosine 0.99;
  - mass-weighted launch speed ≈ 3.2e6 m/s (minimum-energy speed 2.36e6 m/s; heuristic 2.48e6 m/s);
  - larger packets later in the transfer, where each interval gives more Δa.
- Compared with the heuristic, PPO uses 22 % less asteroid mass but more energy, and is slower. The
  heuristic is still the better strategy here. Training reward is shown separately in the viewer.
- Re-running with the same seed reproduced these numbers exactly in this test.

## Known limitations (honest list)

- The hypothetical preset needs about 0.2–0.4 lunar masses launched at about 1 % of c and about
  5e23 W of average launch power. That is physically extreme; the realistic preset is infeasible.
- Training uses the fast model: the Moon is simulated as an exact Earth–Moon two-body orbit, but
  without the Sun's tidal pull on it (phase drift ~22°/yr), and there are no fragment trajectories
  after escape. The policy is then checked in the full N-body model, on 1 seed only.
- The high-fidelity model uses at most 48 representative fragments, merged when the slots are
  full. Fragment fly-outs close to Earth are under-resolved at dt = 6 h. This gives an energy
  residual of about 1e-6 of |E| over a full mission. Momentum, angular momentum and mass stay at
  round-off level.
- The model is non-relativistic (fine below about 10 % of c). The Sun has a constant mass, there
  are no other planets, and there is no tidal or thermal physics.
- Fixed 100-year decision intervals. Within an interval the thrust direction is held constant in
  the rotating frame.
- The training episodes shown during learning are *stochastic* exploration and usually fail. The
  deterministic evaluations (yellow points) are what selects `best_model.zip`.
- With these settings PPO reaches the target but is **slower and less energy-efficient than the
  hand-designed heuristic**. It is a working baseline, not a superior strategy. More episodes or
  reward tuning (higher `w_time`, `w_energy`) are the obvious next step.
- The Three.js view is not to scale: the Moon distance is drawn ×100 and body sizes are enlarged.
- Packet paths in the view use two-body solar gravity after release. Earth's and the Moon's pull on
  a packet is included only in the analytic escape at launch (patched conic). Paths are drawn for
  ≤ 1 year / 60 AU; after that the packets still exist in the physics ledger.
- In the fast model a packet's gravity is included only up to its escape (its final speed and the
  recoil account for climbing out of Earth's, the asteroid's and the Moon's gravity); its pull after
  escape (negligible at ~2,500 km/s) is only in the N-body model. The Moon's pull on an escaping
  packet is credited to Earth rather than the Moon (relative effect ~1e-9).
- Very long sessions: the browser keeps all Earth records of a run (about 80,000 for 10,000 years),
  but only the most recent Moon and packet data.
