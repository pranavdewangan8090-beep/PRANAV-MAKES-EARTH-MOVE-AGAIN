#!/usr/bin/env python3
"""
OCTO-EARTH prototype: an RL agent ejects asteroid reaction mass to move Earth from ~1 AU
toward a stable, near-circular orbit at 2.7 AU within a mission time limit (default 10,000 y).

ALL Python logic lives in this file:
  physics constants and configuration, analytic feasibility gate, initial conditions,
  FAST training model (exact Kepler propagation of the Earth-Moon centre around the Sun AND of the
  Moon around Earth, impulsive momentum-conserving ejections that push Earth),
  HIGH-FIDELITY model (N-body Sun/Earth/Moon + explicit fragments, Yoshida-4 integrator,
  impact / escape / merge events with energy, momentum and angular-momentum ledgers),
  Gymnasium environment, reward, baseline policies, PPO training (Stable-Baselines3),
  best-model selection, evaluation and comparison, explanation and report generation,
  physics self-checks, and a small HTTP API that serves the Three.js viewer.

Run:   python octoearth_rl.py               train + compare + viewer (browser opens)
       python octoearth_rl.py --smoke       very short pipeline test
       python octoearth_rl.py --check       physics checks only
       python octoearth_rl.py --headless    no server/browser; prints the report
       python octoearth_rl.py --preset realistic    (see PRESETS)
On Windows simply double-click RUN_OCTOEARTH.bat.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import threading
import time
import traceback
import webbrowser
from dataclasses import asdict, dataclass, fields, replace
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse


def _dependency_guard():
    import importlib.util
    need = ["numpy", "numba", "gymnasium", "stable_baselines3", "torch"]
    missing = [m for m in need if importlib.util.find_spec(m) is None]
    if missing:
        print("\n[OCTO-EARTH] Missing Python packages: " + ", ".join(missing) +
              "\nDouble-click RUN_OCTOEARTH.bat (it installs them into .venv), or run:\n"
              "    python -m pip install -r requirements.txt\n")
        sys.exit(2)


if __name__ == "__main__":
    _dependency_guard()

import numpy as np  # noqa: E402
from numba import njit  # noqa: E402
import gymnasium as gym  # noqa: E402
from gymnasium import spaces  # noqa: E402

ROOT = Path(__file__).resolve().parent
FRONTEND = ROOT / "frontend"
OUTPUTS = ROOT / "outputs"
APP_NAME = "OCTO-EARTH prototype"

# =============================================================================================
# 1. CONSTANTS (SI)
# =============================================================================================
G = 6.67430e-11
AU = 1.495978707e11
YEAR = 365.25 * 86400.0                 # Julian year [s]
GM_SUN = 1.32712440041e20
GM_EARTH = 3.986004418e14
GM_MOON = 4.9028e12
M_SUN = GM_SUN / G
M_EARTH = GM_EARTH / G
M_MOON = GM_MOON / G
R_SUN = 6.957e8
R_EARTH = 6.371e6
R_MOON = 1.7374e6
MOON_A = 3.844e8
L_SUN = 3.828e26                        # W
S_1AU = 1361.0                          # W/m^2 solar constant
C_LIGHT = 299792458.0
EARTH_INTERCEPT_W = S_1AU * math.pi * R_EARTH ** 2   # sunlight intercepted by Earth [W]

# Ledger layout (float64 array). Every discrete event (ejection, impact, escape, merge) is
# measured as (state after - state before) so that conservation residuals isolate the
# numerical integration error.
LG_MASS_EJECTED, LG_N_PACKETS, LG_MASS_FALLBACK, LG_LAUNCH_KE, LG_E_DRAWN = 0, 1, 2, 3, 4
LG_DE_EJECT, LG_DE_EJECT_THEORY = 5, 6
LG_ESC_MASS, LG_ESC_PX, LG_ESC_PY, LG_ESC_PZ, LG_ESC_E = 7, 8, 9, 10, 11
LG_EARTH_IMPACT_M, LG_MOON_IMPACT_M, LG_SUN_IMPACT_M, LG_DE_IMPACT = 12, 13, 14, 15
LG_N_MERGES, LG_DE_MERGE = 16, 17
LG_PKT_BOUND_SUN_M, LG_PKT_UNBOUND_SUN_M = 18, 19
LG_PKT_PX, LG_PKT_PY, LG_PKT_PZ = 20, 21, 22
LG_MAX_IMPULSE, LG_N_LAUNCH_MOON_HITS, LG_N_FALLBACK = 23, 24, 25
LG_IMPULSE_TOTAL = 26                    # sum of momentum delivered to the Earth system [N s]
LG_EV_DE = 29                            # sum of all event energy changes
LG_EV_DP = 30                            # 30..32 sum of all event momentum changes
LG_EV_DL = 33                            # 33..35 sum of all event angular-momentum changes
LEDGER_N = 40

# Viewer sample = the physics state at ONE ejection instant (one row per ejection event).
# Between two consecutive rows the motion is purely ballistic (no event), so the viewer can
# interpolate with a cubic Hermite curve using v_post of the earlier row and v_pre of the later row.
S_T = 0              # simulated time [yr]
S_R = 1              # 1..3  Earth (fast model: Earth-Moon barycentre) heliocentric position [AU]
S_VPRE = 4           # 4..6  heliocentric velocity just BEFORE the ejection [km/s]
S_VPOST = 7          # 7..9  heliocentric velocity just AFTER the ejection (recoil applied) [km/s]
S_DM = 10            # packet mass [kg] (0 = no ejection at this event)
S_U = 11             # launch speed relative to Earth [m/s]
S_VREL = 12          # packet speed relative to Earth after climbing out of Earth's gravity [m/s]
S_VP = 13            # 13..15 packet heliocentric velocity at hand-over [km/s]
S_FATE = 16          # 0 none, 1 leaves the Solar System, 2 stays bound to the Sun, 3 fell back, 4 hit the Moon
S_MAST = 17          # asteroid mass remaining after the event [kg]
S_EREM = 18          # propulsion energy remaining after the event [J]
S_NFRAG = 19         # active explicit N-body fragments (high-fidelity only)
S_DV = 20            # recoil speed change of Earth from this packet [m/s]
S_EDRAW = 21         # energy drawn for this packet [J]
S_EHAT = 22          # 22..24 packet direction (unit vector, inertial frame)
N_SAMPLE_COLS = 25
MOON_SUB = 8         # high-fidelity: Moon state recorded 8x per ejection event (t, rel pos m, rel vel m/s)
# times after release at which packet positions are computed for the viewer [days]
PACKET_OFFSETS_DAYS = np.array([0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 4, 8, 16, 24, 32, 48, 64, 80, 96,
                                128, 160, 192, 224, 256, 288, 320, 365.25])
FATE_NAMES = {0: "no ejection", 1: "leaves the Solar System", 2: "stays bound to the Sun",
              3: "fell back to Earth (too slow)", 4: "hit the Moon"}


# =============================================================================================
# 2. CONFIGURATION (all user-tweakable parameters)
# =============================================================================================
@dataclass
class Config:
    preset: str = "hypothetical"
    # --- resource and launcher -------------------------------------------------------------
    asteroid_mass_kg: float = 3.0e22          # reaction-mass reservoir docked at Earth
    max_packet_mass_kg: float = 1.0e18        # per ejection event
    max_ejection_speed_ms: float = 5.0e6      # launch speed relative to Earth (non-relativistic)
    min_ejection_speed_ms: float = 2.0e4      # below Earth escape speed (11.2 km/s) packets fall back
    energy_budget_j: float = 3.0e35           # total launch energy available
    launch_efficiency: float = 0.5            # kinetic energy delivered / energy drawn
    ejections_per_year: float = 8.0           # ejection events per simulated year
    # --- mission ----------------------------------------------------------------------------
    target_au: float = 2.7
    mission_years: float = 10000.0
    decision_interval_years: float = 100.0    # one RL decision per interval
    tol_a_au: float = 0.05
    tol_e: float = 0.05
    tol_inc_deg: float = 5.0
    validate_years: float = 200.0             # ballistic check after the mission (high-fidelity)
    randomize_phase: bool = True
    # --- training ---------------------------------------------------------------------------
    train_episodes: int = 1000
    ppo_n_steps: int = 1000
    ppo_batch_size: int = 200
    ppo_epochs: int = 10
    ppo_learning_rate: float = 3e-4
    ppo_gamma: float = 0.995
    eval_every_episodes: int = 50
    seed: int = 1
    # --- reward weights (reward is NOT a physical metric) ------------------------------------
    w_progress: float = 100.0
    w_eccentricity: float = 1.0
    w_inclination: float = 1.0
    w_energy: float = 10.0
    w_mass: float = 5.0
    w_time: float = 0.02
    w_success: float = 100.0
    w_terminal_error: float = 50.0
    w_failure: float = 100.0
    # --- high-fidelity model -----------------------------------------------------------------
    hifi_dt_hours: float = 6.0
    max_fragments: int = 48
    release_radius_m: float = 2.0e9           # patched-conic hand-over radius (> Earth Hill radius)
    fragment_escape_au: float = 60.0

    def to_dict(self):
        return asdict(self)

    @property
    def max_steps(self) -> int:
        return int(math.ceil(self.mission_years / self.decision_interval_years - 1e-9))



PRESETS = {
    # Physically HYPOTHETICAL: chosen so that the transfer is just feasible.  Requires ~0.4
    # lunar masses ejected at ~1% of light speed and ~5e23 W average launch power.
    "hypothetical": dict(preset="hypothetical", asteroid_mass_kg=3.0e22, max_packet_mass_kg=1.0e18,
                         max_ejection_speed_ms=5.0e6, energy_budget_j=3.0e35, launch_efficiency=0.5),
    # Generous "realistic-ish": a Ceres-mass asteroid, a 30 km/s electromagnetic launcher, and
    # ALL sunlight intercepted by Earth for 10,000 years as the energy budget.
    "realistic": dict(preset="realistic", asteroid_mass_kg=9.4e20, max_packet_mass_kg=1.0e14,
                      max_ejection_speed_ms=3.0e4, energy_budget_j=EARTH_INTERCEPT_W * 1e4 * YEAR,
                      launch_efficiency=0.5),
}


# ---------------------------------------------------------------------------------------------
# 2b. PARAMETER / OUTPUT DOCUMENTATION (single source for the viewer's guide and tooltips)
# ---------------------------------------------------------------------------------------------
CAT_ENG = "Engineering assumption"
CAT_MIS = "Mission requirement"
CAT_SIM = "Simulation setting"
CAT_RL = "RL hyperparameter"
CAT_RW = "Reward weight (scoring only)"
CAT_CONST = "Physical constant"
CAT_VIEW = "Viewer setting"


def _p(key, label, unit, cat, lo, hi, meaning, why, up, down, constraint=""):
    return dict(key=key, label=label, unit=unit, category=cat, min=lo, max=hi, meaning=meaning, why=why,
                increase=up, decrease=down, constraint=constraint)


PARAM_DOCS = [
    _p("asteroid_mass_kg", "Initial asteroid mass", "kg", CAT_ENG, 1e15, 1e23,
       "Mass of the asteroid that is docked to Earth and used as throw-away reaction mass. "
       "For scale: the Moon is 7.3e22 kg, Ceres 9.4e20 kg.",
       "Momentum is conserved: Earth can only gain speed by throwing mass the other way. The total "
       "mass available limits the total speed change (rocket equation).",
       "More speed change is possible, at the same launch speed.",
       "Less mass to throw; below about 1.4e22 kg the default target cannot be reached even at maximum speed.",
       "Cannot exceed 1e23 kg in this model (the asteroid is treated as a small attachment to Earth)."),
    _p("max_packet_mass_kg", "Max packet mass", "kg", CAT_ENG, 1e3, 1e20,
       "Largest piece of asteroid that one ejection event can throw. The agent chooses a throttle between 0 and 100 % of this.",
       "Together with the ejection frequency it sets the maximum thrust (mass per second x speed).",
       "Faster transfer possible; each push is bigger (coarser control).",
       "Gentler, finer control, but the transfer may not finish within the mission time.",
       "Must not exceed the asteroid mass."),
    _p("max_ejection_speed_ms", "Max ejection speed (exhaust velocity)", "m/s", CAT_ENG, 1.2e4, 3.0e7,
       "Fastest launch speed of a packet relative to Earth's surface. This plays the role of a rocket's exhaust velocity.",
       "Momentum per packet = mass x speed, so faster packets push harder per kg. But energy = 1/2 mass x speed^2, so they "
       "cost much more energy per kg.",
       "Less asteroid mass needed, much more energy needed per kg.",
       "More mass needed. Below Earth's escape speed (11.2 km/s) packets fall back and give no net push.",
       "Capped at 3e7 m/s (10 % of light speed) because the model is non-relativistic."),
    _p("min_ejection_speed_ms", "Min ejection speed", "m/s", CAT_ENG, 1e3, 3.0e7,
       "Slowest launch speed the agent may choose.",
       "Sets the lower end of the agent's speed range (action a3 maps linearly between min and max).",
       "The agent cannot use slow, energy-cheap launches.",
       "Allows slow launches; below 11.2 km/s packets fall back to Earth (wasting energy, no push).",
       "Must be lower than the max ejection speed."),
    _p("energy_budget_j", "Propulsion energy budget", "J", CAT_ENG, 1e20, 1e37,
       "Total energy available to the launcher for the whole mission. For scale: Earth receives about 5.4e24 J of sunlight per year.",
       "Each packet needs 1/2 x mass x speed^2 / efficiency. When the budget is used up, no more packets can be launched. "
       "No free energy is ever created.",
       "More or faster packets possible.",
       "The mission may stop with asteroid mass left over ('resources exhausted').",
       "Never exceeded: packets are shrunk automatically so the budget cannot go negative."),
    _p("launch_efficiency", "Launch efficiency", "0-1", CAT_ENG, 0.01, 1.0,
       "Fraction of the energy drawn from the budget that ends up as packet kinetic energy (the rest is lost as heat).",
       "Energy drawn = kinetic energy / efficiency.",
       "More packets per joule.", "Wastes more energy per packet.", "Between 0.01 and 1 (100 %)."),
    _p("ejections_per_year", "Ejection frequency", "1/yr", CAT_ENG, 1.0, 52.0,
       "How many packets are launched per simulated year (evenly spaced). Each one is a separate recoil event.",
       "Sets the maximum mass flow (thrust) and the number of recorded ejection events.",
       "More thrust available and more, smaller recoil steps (smoother); slower simulation.",
       "Less thrust; the transfer may need longer than the mission allows.", "1 to 52 per year."),
    _p("target_au", "Target orbit radius", "AU", CAT_MIS, 1.1, 5.0,
       "Semi-major axis of the circular orbit Earth should reach (1 AU = today's Sun-Earth distance; Mars is at 1.52 AU, the asteroid belt 2.2-3.3 AU).",
       "Defines success and the orbital energy that must be added.",
       "Much more energy and mass needed; much less sunlight (insolation falls as 1/a^2).",
       "Easier transfer.", "1.1 to 5 AU."),
    _p("mission_years", "Mission duration", "yr", CAT_MIS, 100.0, 20000.0,
       "Simulated time allowed before the episode ends ('deadline').",
       "Limits how long thrust can act; together with max thrust it decides if the target is reachable in time.",
       "More time for slow, energy-efficient transfers; longer episodes (slower training).",
       "Forces higher thrust; may become infeasible.", "100 to 20,000 years; must be at least 2 decision intervals."),
    _p("decision_interval_years", "Decision interval", "yr", CAT_SIM, 1.0, 1000.0,
       "How often the agent chooses a new action. During one interval all packets use the same mass, speed and direction.",
       "It is the RL time step: an episode has mission / interval decisions.",
       "Fewer, coarser decisions (faster training, less precise control).",
       "Finer control, longer episodes, slower training.", "At most 1,000 decisions per episode."),
    _p("tol_a_au", "Tolerance on target radius", "AU", CAT_MIS, 0.001, 0.5,
       "How close the final semi-major axis must be to the target.",
       "Part of the success test.", "Easier to count as success.", "Stricter.", ""),
    _p("tol_e", "Tolerance on eccentricity", "-", CAT_MIS, 0.001, 0.3,
       "Largest allowed eccentricity (0 = perfect circle). Earth today has e = 0.0167.",
       "Part of the success test; 'stable circular orbit' means small e.", "Allows more elliptical orbits.",
       "Requires a rounder orbit.", "Must be above 0.0167 to be reachable easily."),
    _p("tol_inc_deg", "Tolerance on inclination", "deg", CAT_MIS, 0.1, 30.0,
       "Largest allowed tilt of the final orbit relative to today's orbital plane.",
       "Part of the success test.", "Allows out-of-plane drift.", "Stricter.", ""),
    _p("validate_years", "Stability check duration", "yr", CAT_MIS, 10.0, 2000.0,
       "After the mission, the high-fidelity model runs this long with NO ejections to check that the orbit stays within tolerance and the Moon stays bound.",
       "Confirms 'stable orbit' instead of a lucky snapshot.", "Stronger evidence, slower comparison.",
       "Weaker evidence.", ""),
    _p("randomize_phase", "Randomize starting position", "on/off", CAT_SIM, None, None,
       "If on, each training episode starts at a random point of Earth's orbit (and Moon phase).",
       "Prevents the agent from memorising one exact sequence.", "(on) more robust policies.",
       "(off) every episode is identical.", ""),
    _p("train_episodes", "Training episodes", "-", CAT_RL, 1, 100000,
       "Number of complete missions the agent practises on (fast model).",
       "More practice usually gives a better policy, with no guarantee.", "Longer training (about 0.15 s per episode).",
       "Quicker, usually worse policy.", ""),
    _p("ppo_n_steps", "PPO steps per update", "-", CAT_RL, 64, 20000,
       "Number of decisions collected before each neural-network update.",
       "Larger batches give more stable but less frequent updates.", "Smoother, slower learning.",
       "Faster, noisier learning.", "Must be at least the batch size."),
    _p("ppo_batch_size", "PPO mini-batch size", "-", CAT_RL, 16, 20000,
       "Decisions per gradient step inside an update.", "Affects the noise of each gradient step.",
       "Smoother gradients.", "Noisier gradients.", "At most the PPO steps per update."),
    _p("ppo_epochs", "PPO epochs per update", "-", CAT_RL, 1, 50,
       "How many passes over the collected data per update.", "More passes extract more from the data but can overfit.",
       "Faster learning per sample; risk of instability.", "Slower, more stable.", ""),
    _p("ppo_learning_rate", "Learning rate", "-", CAT_RL, 1e-6, 1e-2,
       "Step size of neural-network weight updates.", "Controls how fast the policy changes.",
       "Faster but possibly unstable learning.", "Slower, steadier learning.", ""),
    _p("ppo_gamma", "Discount factor (gamma)", "-", CAT_RL, 0.8, 0.9999,
       "How much the agent values future reward compared with immediate reward (per decision).",
       "A transfer takes ~40-100 decisions, so gamma must be close to 1 for the agent to care about arriving.",
       "More far-sighted.", "More short-sighted (may never learn to finish).", ""),
    _p("eval_every_episodes", "Evaluate every", "episodes", CAT_RL, 1, 10000,
       "How often the current policy is tested deterministically (no exploration) to pick the best model.",
       "Only evaluated policies can become best_model.zip.", "Fewer checkpoints considered.",
       "More checkpoints considered, slower training.", ""),
    _p("seed", "Random seed", "-", CAT_SIM, 0, 2 ** 31 - 1,
       "Starting value of the random-number generator.", "Same seed gives an identical run (reproducible).",
       "(no direction) different random run.", "(no direction) different random run.", ""),
    _p("w_progress", "Reward: progress weight", "points", CAT_RW, 0.0, 1000.0,
       "Points given for closing the gap in orbital energy toward the target (100 = full gap closed).",
       "Main learning signal. Reward is a scoring rule for training, NOT a physical result.",
       "Agent rushes outward more.", "Weaker signal; slower learning.", ""),
    _p("w_eccentricity", "Reward: eccentricity penalty", "points", CAT_RW, 0.0, 100.0,
       "Penalty per 100 years while e is above the tolerance.", "Encourages a round orbit.",
       "Stronger push for circular orbits.", "Agent may tolerate elliptical orbits.", ""),
    _p("w_inclination", "Reward: inclination penalty", "points", CAT_RW, 0.0, 100.0,
       "Penalty per 100 years while the orbit is tilted beyond tolerance.", "Discourages out-of-plane thrust.",
       "Stronger.", "Weaker.", ""),
    _p("w_energy", "Reward: energy-use penalty", "points", CAT_RW, 0.0, 1000.0,
       "Penalty for spending propulsion energy (10 = whole budget costs 10 points).", "Encourages efficiency.",
       "Agent saves energy (may arrive slower).", "Agent spends energy freely.", ""),
    _p("w_mass", "Reward: mass-use penalty", "points", CAT_RW, 0.0, 1000.0,
       "Penalty for using asteroid mass (5 = whole asteroid costs 5 points).", "Encourages saving mass.",
       "Agent saves mass (uses faster, energy-hungry launches).", "Agent uses mass freely.", ""),
    _p("w_time", "Reward: time penalty", "points/decision", CAT_RW, 0.0, 10.0,
       "Small penalty per decision.", "Encourages arriving sooner.", "Faster arrival preferred.",
       "No hurry.", ""),
    _p("w_success", "Reward: success bonus", "points", CAT_RW, 0.0, 10000.0,
       "Bonus when the orbit enters the target window (plus extra for leftover resources).",
       "Makes success clearly better than near-misses.", "Stronger focus on finishing.", "Weaker.", ""),
    _p("w_terminal_error", "Reward: final-miss penalty", "points", CAT_RW, 0.0, 1000.0,
       "Penalty at the deadline or when resources run out, proportional to the remaining orbit error.",
       "Distinguishes close misses from bad misses.", "Harsher.", "Milder.", ""),
    _p("w_failure", "Reward: failure penalty", "points", CAT_RW, 0.0, 10000.0,
       "Penalty for catastrophic outcomes (Earth unbound, too close to the Sun, Moon lost).",
       "Teaches the agent to avoid disasters.", "Harsher.", "Milder.", ""),
    _p("hifi_dt_hours", "High-fidelity time step", "h", CAT_SIM, 0.5, 24.0,
       "Integration step of the N-body (Sun, Earth, Moon, fragments) validation model.",
       "Smaller steps are more accurate but slower. The Moon's 27-day orbit needs about 6 h or less.",
       "Faster, less accurate (Moon orbit errors grow quickly above 12 h).",
       "More accurate, slower.", "0.5 to 24 h; the check '10 integrator convergence' shows the error at 12/6/3 h."),
    _p("max_fragments", "Max tracked fragments", "-", CAT_SIM, 4, 200,
       "How many ejected packets the N-body model integrates individually at the same time. When full, the two closest fragments merge (momentum kept, energy loss recorded).",
       "Representative-fragment approximation of thousands of packets.", "More detail, slower.",
       "Faster, coarser.", ""),
    _p("release_radius_m", "Packet hand-over distance", "m", CAT_SIM, 1.0e9, 1.0e10,
       "Distance from Earth at which a launched packet becomes an explicit N-body fragment (its climb out of Earth's gravity is computed analytically before that).",
       "Patched-conic approximation; it must be outside Earth's gravitational sphere of influence (Hill radius about 1.5e9 m).",
       "Less near-Earth detail.", "Below the Hill radius the approximation is invalid.", "1e9 to 1e10 m."),
    _p("fragment_escape_au", "Fragment escape distance", "AU", CAT_SIM, 10.0, 1000.0,
       "Fragments that are unbound from the Sun and farther than this leave the simulated region; their mass, momentum and energy are recorded in the ledger (never silently deleted).",
       "Keeps the N-body model small.", "Tracks fragments longer.", "Removes them sooner.", ""),
]
PARAM_BY_KEY = {d["key"]: d for d in PARAM_DOCS}
PARAM_SPEC = [(d["key"], d["label"], d["unit"]) for d in PARAM_DOCS]

CONSTANT_DOCS = [
    dict(name="Gravitational constant G", value=G, unit="m^3 kg^-1 s^-2", meaning="Strength of gravity in Newton's law F = G m1 m2 / r^2."),
    dict(name="Sun mass", value=M_SUN, unit="kg", meaning="From GM_sun = 1.32712440041e20 m^3/s^2."),
    dict(name="Earth mass", value=M_EARTH, unit="kg", meaning="From GM_earth = 3.986004418e14 m^3/s^2."),
    dict(name="Moon mass", value=M_MOON, unit="kg", meaning="From GM_moon = 4.9028e12 m^3/s^2."),
    dict(name="Astronomical unit (AU)", value=AU, unit="m", meaning="Mean Sun-Earth distance; the unit of orbit size."),
    dict(name="Year", value=YEAR, unit="s", meaning="Julian year (365.25 days); all 'simulated years' use it."),
    dict(name="Earth radius", value=R_EARTH, unit="m", meaning="Launch radius; escape speed from it is ~11.2 km/s."),
    dict(name="Moon distance", value=MOON_A, unit="m", meaning="Mean Earth-Moon distance (initial Moon orbit)."),
    dict(name="Solar constant at 1 AU", value=S_1AU, unit="W/m^2", meaning="Sunlight intensity today; used for the insolation estimate."),
    dict(name="Speed of light", value=C_LIGHT, unit="m/s", meaning="Only used to warn that the model is non-relativistic."),
]

SPEED_DOC = _p("sim_speed", "Simulation speed (time multiplier)", "simulated time per real second", CAT_VIEW, None, None,
               "How much simulated time the 3-D view advances per second of real (wall-clock) time. It does not change the physics.",
               "10,000 years cannot be watched in real time.",
               "Faster overview: Earth circles the Sun many times per second, so watch the trail.",
               "Slower: individual packets and the Moon become visible (packets cross Earth's orbit in hours at ~1% of light speed).",
               "1 day/s to 500 yr/s. The physics runs ahead of the display by at most 3 decision intervals or 2.5 s of display "
               "time (max 1,500 years); the dim grey dots show that computed-ahead part.")


def _o(key, label, unit, kind, measures, how, better, success):
    return dict(key=key, label=label, unit=unit, kind=kind, measures=measures, how=how, better=better, success=success)


K_SIM, K_CALC, K_EST, K_PRED, K_SCORE = ("simulated state", "calculated from simulated state", "estimate (simplified)",
                                         "prediction", "training score")
OUTPUT_DOCS = [
    _o("a", "Semi-major axis a", "AU", K_CALC, "Size of Earth's current orbit (half the longest diameter of the ellipse).",
       "From Earth's simulated position r and velocity v: a = 1 / (2/|r| - |v|^2/mu), mu = G(M_sun + M_earth + M_moon).",
       "Closer to the target (2.7 AU)", "Success needs |a - target| <= tolerance."),
    _o("e", "Eccentricity e", "-", K_CALC, "How elongated the orbit is: 0 = circle, 0.0167 = Earth today, 1 = escape.",
       "Length of the eccentricity vector computed from r and v.", "Lower", "Success needs e <= tolerance (round, stable orbit)."),
    _o("inc", "Inclination", "deg", K_CALC, "Tilt of the orbit plane relative to today's orbit plane.",
       "Angle between the orbital angular-momentum vector and the original orbit normal.", "Lower",
       "Success needs inclination <= tolerance."),
    _o("r", "Distance from Sun", "AU", K_SIM, "Earth's current distance from the Sun.", "|r| from the physics state.",
       "-", "Oscillates during each orbit; a is the better progress measure."),
    _o("speed", "Orbital speed", "km/s", K_SIM, "Earth's speed around the Sun (29.8 km/s today).",
       "|v| from the physics state.", "- (it DEcreases as the orbit grows: 18.1 km/s at 2.7 AU)",
       "Shows the paradox of orbit raising: pushing forward makes the orbit bigger and the speed smaller."),
    _o("E_orb", "Orbital energy", "J", K_CALC, "Kinetic + gravitational energy of the Earth-Moon system in its orbit around the Sun.",
       "E = M (|v|^2/2 - mu/|r|) = -mu M / (2a). Negative = bound to the Sun.", "Closer to zero (higher) until the target value",
       "Raising the orbit from 1 to 2.7 AU requires about +1.7e33 J of orbital energy."),
    _o("dv", "Velocity change delivered (delta-v)", "m/s", K_CALC,
       "Total speed change the recoil has given the Earth-Moon system so far.",
       "Sum over packets of recoil momentum / system mass (= impulse / M).",
       "Enough to reach the target (about 11.7 km/s for a slow spiral), but no more",
       "Compare with the required 11.7 km/s from the feasibility gate."),
    _o("thrust", "Average thrust", "N", K_CALC, "Average push on Earth during the last decision interval.",
       "Impulse delivered in the interval / interval duration.", "Depends: more thrust = faster transfer, more resources",
       "Tiny compared with Earth's weight in the Sun's gravity (3.5e22 N), so the orbit changes slowly."),
    _o("impulse", "Momentum delivered (impulse)", "N s", K_SIM,
       "Total momentum transferred to the Earth-Moon system by the recoil of all packets.",
       "Sum of reduced-mass x relative packet speed for every packet (exact momentum conservation).",
       "Enough for the target", "Momentum is conserved: packets carry exactly the opposite momentum."),
    _o("m_ast", "Remaining asteroid mass", "kg (%)", K_SIM, "Reaction mass still available.",
       "Initial mass minus all ejected packets (finite reservoir; cannot go below zero).",
       "Higher at arrival = more efficient", "Zero before arrival means the mission ran out of mass."),
    _o("E_used", "Energy consumed", "J (%)", K_SIM, "Propulsion energy drawn from the budget so far.",
       "Sum over packets of (1/2 reduced mass x launch speed^2) / efficiency.", "Lower for the same result",
       "At 100 % no more packets can be launched."),
    _o("packets", "Packets ejected", "-", K_SIM, "Number of ejection events so far.", "Counted by the physics ledger.",
       "-", "Each one produces a small recoil."),
    _o("recoil", "Recoil per packet", "m/s", K_SIM, "Speed change of Earth from the latest single packet.",
       "|v after - v before| at the ejection instant.", "-", "Tiny (well below 1 m/s) but repeated tens of thousands of times."),
    _o("moon", "Moon bound", "yes/no", K_CALC, "Whether the Moon still orbits Earth.",
       "Both models: negative Earth-Moon two-body energy and distance < 1/2 Hill radius (the Moon is simulated in both).",
       "yes", "Losing the Moon ends the episode as a failure."),
    _o("insolation", "Insolation", "x today", K_EST, "Sunlight per square metre at the final orbit relative to today.",
       "1 / a^2 (orbit-averaged, ignoring e).", "Not a goal of the optimisation; ~0.14 at 2.7 AU",
       "Habitability indicator only; no climate model."),
    _o("t_eq", "Equilibrium temperature", "K", K_EST, "Black-body temperature of Earth with today's albedo at the final orbit.",
       "255 K x a^-1/2 (no greenhouse effect).", "-", "Rough indicator only; real temperature depends on atmosphere."),
    _o("reward", "Reward (per decision)", "points", K_SCORE, "Score the agent receives after each decision.",
       "Weighted sum of progress, penalties and bonuses (see reward weights).", "Higher",
       "A scoring rule for training only. It is NOT a physical quantity; physical results are shown separately."),
    _o("return", "Episode return", "points", K_SCORE, "Sum of all rewards in one episode.", "Sum of per-decision rewards.",
       "Higher", "Good returns usually mean success, but always check a, e and the validation."),
    _o("progress", "Training progress", "episodes", K_SIM, "Completed training episodes out of the target.",
       "Counted after each finished episode.", "-", "More episodes do not guarantee a better policy."),
    _o("success_rate", "Success rate", "%", K_CALC, "Fraction of test episodes that reached the target window.",
       "Deterministic evaluation episodes (no exploration noise).", "Higher", "Used first when choosing the best model."),
    _o("cons_E", "Energy conservation error", "fraction of |E|", K_CALC,
       "High-fidelity only: total energy change NOT explained by recorded events (launches, impacts, escapes, merges).",
       "(E_now - E_start - sum of event energy changes) / |E_start|.", "Closer to 0",
       "Measures numerical error. Near 1e-6 here mainly because near-Earth fragment fly-outs are under-resolved."),
    _o("cons_P", "Momentum conservation error", "fraction", K_CALC, "Unexplained change of total momentum.",
       "|P_now - P_start - sum of event changes| / (M_earth x 30 km/s).", "Closer to 0", "Round-off level (~1e-13) means momentum is conserved."),
    _o("cons_L", "Angular-momentum conservation error", "fraction", K_CALC, "Unexplained change of total angular momentum.",
       "|L_now - L_start - sum of event changes| / |L_start|.", "Closer to 0", "Round-off level means it is conserved."),
    _o("cons_M", "Mass accounting error", "fraction", K_CALC, "Mass not explained by escaped fragments.",
       "(M_now + M_escaped - M_start) / M_start.", "0", "Zero means no mass appears or disappears."),
    _o("shortfall", "Shortfall", "AU", K_CALC, "Target radius minus final semi-major axis.", "target - a.",
       "0 (within tolerance)", "Positive = did not get far enough; negative = overshoot."),
]
OUTPUT_BY_KEY = {d["key"]: d for d in OUTPUT_DOCS}

GLOSSARY = [
    ("AU (astronomical unit)", "The average Sun-Earth distance, 149.6 million km. 2.7 AU is in the asteroid belt."),
    ("Orbit / ellipse", "The closed path of a planet around the Sun. It is described by its size (a), shape (e) and tilt (inclination)."),
    ("Semi-major axis (a)", "Half of the longest diameter of the orbit ellipse; the 'size' of the orbit. It fixes the orbital period and energy."),
    ("Eccentricity (e)", "How stretched the ellipse is: 0 is a circle; values near 1 are very elongated."),
    ("Inclination", "Tilt of the orbit relative to a reference plane (here: today's orbit)."),
    ("Delta-v", "A change of velocity. Mission planners measure how hard a manoeuvre is in delta-v."),
    ("Impulse / momentum", "Momentum = mass x velocity. Throwing a packet gives Earth the equal and opposite momentum (Newton's third law)."),
    ("Recoil", "The backward push on the thrower when mass is thrown (like a gun's kick)."),
    ("Reaction mass", "Mass thrown away to push the remaining body forward; here, pieces of the asteroid."),
    ("Exhaust / ejection velocity", "Speed of the thrown mass relative to the thrower. Higher speed = more push per kg but more energy per kg."),
    ("Rocket equation", "delta-v = v_exhaust x ln(M_start / M_end): the limit on speed change from finite reaction mass."),
    ("Escape speed", "Minimum launch speed to leave a body's gravity (11.2 km/s for Earth). Slower packets fall back."),
    ("Prograde / radial / normal (RTN frame)", "Directions relative to Earth's orbit: prograde = along the motion, radial = away from the Sun, normal = perpendicular to the orbit plane."),
    ("Kepler orbit (two-body)", "Exact motion of one body around another under gravity alone; used by the fast model between ejections."),
    ("N-body simulation", "Numerically integrating the mutual gravity of several bodies; used by the high-fidelity model."),
    ("Yoshida integrator", "A 4th-order time-stepping method that conserves energy well over long times."),
    ("Patched conic", "Treat a packet's climb out of Earth's gravity analytically, then follow it in the Sun's gravity."),
    ("Hill sphere", "Region where Earth's gravity dominates over the Sun's (~1.5 million km)."),
    ("Osculating (coasting) orbit", "The orbit Earth would follow from its current position and velocity if all thrust stopped now."),
    ("Insolation", "Sunlight power per square metre; falls as 1/distance^2."),
    ("Reinforcement learning (RL)", "A trial-and-error method: an agent tries actions, receives rewards, and adjusts a policy to get more reward."),
    ("Policy", "The agent's decision rule: observation in, action out (here a small neural network)."),
    ("PPO", "Proximal Policy Optimization, a standard RL algorithm that changes the policy in small, safe steps."),
    ("Episode", "One complete simulated mission from the start to success, deadline or failure."),
    ("Reward / return", "Reward = score per decision; return = total score per episode. A designed scoring rule, not a physical measurement."),
    ("Discount factor", "How much future rewards count compared with immediate ones."),
    ("Deterministic evaluation", "Testing the policy without random exploration, to measure what it has really learned."),
    ("Baseline", "A simple reference strategy (no ejection, hand-designed) to judge whether RL actually helps."),
    ("Conservation error", "How much a quantity that physics says must stay constant (energy, momentum...) drifts because of numerical approximations."),
]

REASON_DOCS = {
    "success": "Orbit entered the target window (a, e and inclination within tolerance).",
    "deadline": "Mission time ran out before the target was reached.",
    "resources_exhausted": "All asteroid mass or all propulsion energy was used before reaching the target.",
    "escaped_or_unbound": "Earth was pushed onto an escape orbit (constraint violation, catastrophic).",
    "too_close_to_sun": "Earth came closer than 0.2 AU to the Sun (catastrophic).",
    "moon_lost": "The Moon was no longer bound to Earth (catastrophic).",
}

CHECK_DOCS = {
    "1": "Gravity-only motion follows Kepler's exact solution, and the N-body integrator agrees with it.",
    "2": "Each gravitational pull (Sun, Earth, Moon, asteroid, fragment) has the textbook magnitude G m / r^2.",
    "3": "Every force has an equal and opposite partner (Newton's third law), so internal forces cancel.",
    "4": "Throwing a packet does not create momentum or angular momentum: Earth's gain equals the packet's loss.",
    "5": "The energy put into each launch matches 1/2 x mass x speed^2 (minus the climb out of gravity), and the budget can never go negative.",
    "7": "Mass is never created or lost: Earth + asteroid + fragments + escaped mass stays constant; the asteroid cannot go below zero.",
    "8": "Reversing the throw direction reverses the effect (forward push raises the orbit, backward push lowers it).",
    "9": "Slow packets fall back, fast ones escape, and packets aimed at Earth merge, all recorded in the ledger.",
    "10": "Halving the time step reduces the error 16x, as expected for a 4th-order method (the numerics converge).",
    "11": "The same seed gives exactly the same result.",
    "12": "The fast training model and the high-fidelity N-body model agree on the orbit.",
    "13": "The 3-D view's interpolation between recorded physics states is accurate and hides no jumps.",
    "14": "The training model really simulates the Moon: its distance from Earth agrees with the N-body Moon, its pull makes Earth wobble around the Earth-Moon centre, and a launch pushes only Earth.",
}


ASSUMPTIONS = [
    ("Scale is hypothetical", "The default asteroid (3e22 kg, 40 % of the Moon) and energy budget (3e35 J, about 25 years "
     "of the Sun's total output) are far beyond any real technology. The 'realistic' preset shows the honest answer: infeasible."),
    ("Asteroid docked to Earth", "The asteroid is attached to Earth (same position and velocity); the launcher throws its "
     "material away piece by piece. Its gravity IS included: its mass adds to Earth's pull on the Sun, the Moon and every "
     "launched packet, and that pull weakens as the asteroid is used up. Because it is attached, its pull on Earth itself "
     "is an internal force (no net push). Its own orbit and structure are not modelled. The 3-D view draws it as a brown rock "
     "sitting on Earth's marker on the side the packets are thrown from (not to scale)."),
    ("Instant launches", "Each packet is launched instantly from Earth's surface; launcher hardware, heat and atmosphere are not modelled."),
    ("Non-relativistic", "Newtonian mechanics; launch speeds up to 1.7 % of light speed give errors of order 0.01 %."),
    ("Fast model (training)", "Hierarchical two-body model: the Earth-Moon centre of mass follows an exact Kepler orbit "
     "around the Sun, and the Moon follows an exact Kepler orbit around Earth(+asteroid) under their mutual gravity. Each "
     "packet pushes Earth only; the Moon follows through gravity. The packet's escape from Earth's, the asteroid's and the "
     "Moon's gravity is included in its final speed. APPROXIMATION: the Sun's tidal pull on the Moon is neglected, so the "
     "Moon's orbital phase drifts by ~20 degrees per year relative to the N-body Moon (its distance agrees within ~3 %, "
     "check 14); packets hitting the Moon and fragment trajectories after escape are only in the high-fidelity model."),
    ("High-fidelity model", "Sun, Earth, Moon and up to max_fragments representative fragments with full mutual gravity, "
     "except fragment-fragment gravity, which is neglected (fragments are tiny). Other planets are not included."),
    ("Packet paths in the 3-D view", "Computed with exact two-body (Sun-only) gravity from each packet's real release state, "
     "drawn for at most 1 year or until 60 AU. Earth's and the Moon's pull on the packet after release is approximated by "
     "the analytic escape (patched conic)."),
    ("Interpolation in the 3-D view", "Between recorded physics states (8 per year) Earth's position is drawn on a cubic "
     "curve through the recorded positions and velocities (error < 1e-4 AU, check 13). At each ejection the curve uses the "
     "velocity before and after the event separately, so the recoil is never smoothed away."),
    ("Moon distance exaggerated in the view", "The Moon is drawn 100x farther from Earth than it really is, otherwise it "
     "would sit inside Earth's marker. Its motion is the simulated one in both models. Its simulated orbit around Earth is "
     "drawn as a thin grey ellipse (same 100x scale)."),
    ("Habitability", "Insolation and equilibrium temperature are rough indicators only; no climate, tides or day length."),
]


def validate_params(cfg_dict: dict) -> list:
    """Return a list of human-readable problems (empty = valid). Values are never silently changed."""
    errs = []
    for d in PARAM_DOCS:
        k, v = d["key"], cfg_dict.get(d["key"])
        if d["min"] is None or v is None or isinstance(v, bool):
            continue
        if not (d["min"] <= v <= d["max"]):
            errs.append(f"{d['label']} = {v:g} {d['unit']} is outside the allowed range {d['min']:g} .. {d['max']:g}")
    c = cfg_dict
    if c["min_ejection_speed_ms"] >= c["max_ejection_speed_ms"]:
        errs.append("Min ejection speed must be lower than max ejection speed")
    if c["max_packet_mass_kg"] > c["asteroid_mass_kg"]:
        errs.append("Max packet mass cannot exceed the asteroid mass")
    if c["ppo_batch_size"] > c["ppo_n_steps"]:
        errs.append("PPO mini-batch size cannot exceed PPO steps per update")
    steps = c["mission_years"] / c["decision_interval_years"]
    if steps < 2 or steps > 1000:
        errs.append(f"Mission / decision interval = {steps:g} decisions; must be between 2 and 1,000")
    return errs


def make_config(_preset: str = "hypothetical", **overrides) -> Config:
    """Preset values first, then overrides (strings are converted to the field type).  If an
    override changes one of the preset's physical parameters, the preset label becomes 'custom'."""
    name = _preset if _preset in PRESETS else "hypothetical"
    base = replace(Config(), **PRESETS[name])
    valid = {f.name for f in fields(Config)}
    clean = {}
    for k, v in overrides.items():
        if k in valid and k != "preset" and v is not None and v != "":
            cur = getattr(base, k)
            if isinstance(cur, bool):
                clean[k] = v if isinstance(v, bool) else str(v).lower() in ("1", "true", "yes")
            else:
                clean[k] = type(cur)(float(v)) if isinstance(cur, int) else type(cur)(v)
    unknown = [k for k in overrides if k not in valid]
    if unknown:
        raise ValueError(f"unknown parameter(s): {', '.join(unknown)}")
    cfg = replace(base, **clean)
    if any(k in PRESETS[name] and getattr(cfg, k) != PRESETS[name][k] for k in clean):
        cfg = replace(cfg, preset="custom")
    elif _preset == "custom" or overrides.get("preset") == "custom":
        cfg = replace(cfg, preset="custom") if any(getattr(cfg, k) != v for k, v in PRESETS[name].items() if k != "preset") else cfg
    return cfg


def checked_config(_preset: str = "hypothetical", **overrides) -> Config:
    """make_config + range / consistency validation.  Invalid values raise ValueError; they are
    never clipped or reinterpreted."""
    cfg = make_config(_preset, **overrides)
    errs = validate_params(cfg.to_dict())
    if errs:
        raise ValueError("; ".join(errs))
    return cfg


def smoke_config(preset="hypothetical") -> Config:
    return make_config(preset, train_episodes=30, mission_years=4000.0, decision_interval_years=100.0,
                       ppo_n_steps=200, ppo_batch_size=50, eval_every_episodes=10, validate_years=50.0)


# =============================================================================================
# 3. ANALYTIC FEASIBILITY GATE
# =============================================================================================
def launch_vinf(u: float, m_launcher: float) -> float:
    """Asymptotic relative speed of a packet launched at speed u from Earth's surface."""
    v2 = u * u - 2.0 * G * m_launcher / R_EARTH
    return math.sqrt(v2) if v2 > 0 else 0.0


def feasibility(cfg: Config) -> dict:
    m_sys = M_EARTH + cfg.asteroid_mass_kg + M_MOON          # Earth + reservoir + Moon (moved together)
    m_launch = M_EARTH + cfg.asteroid_mass_kg
    mu = GM_SUN + G * m_sys
    a0, at = 1.0 * AU, cfg.target_au * AU
    v0, vt = math.sqrt(mu / a0), math.sqrt(mu / at)
    dv_spiral = abs(v0 - vt)                                  # slow-spiral delta-v (circular->circular)
    a_tr = 0.5 * (a0 + at)                                    # Hohmann (two impulses) reference
    dv_hoh = abs(math.sqrt(mu * (2 / a0 - 1 / a_tr)) - v0) + abs(vt - math.sqrt(mu * (2 / at - 1 / a_tr)))
    dE_orb = G * M_SUN * m_sys * (1 / (2 * a0) - 1 / (2 * at))
    dL_orb = m_sys * (math.sqrt(mu * at) - math.sqrt(mu * a0))
    T = cfg.mission_years * YEAR
    vinf_max = launch_vinf(cfg.max_ejection_speed_ms, m_launch)
    vesc = math.sqrt(2 * G * m_launch / R_EARTH)
    # mass-limited capacity at maximum speed
    ratio = m_sys / (m_sys - cfg.asteroid_mass_kg)
    dv_cap_mass = vinf_max * math.log(ratio)
    mass_needed_umax = m_sys * (1 - math.exp(-dv_spiral / vinf_max)) if vinf_max > 0 else float("inf")
    # minimum-energy strategy: use ALL reaction mass at the lowest sufficient speed
    vinf_req = dv_spiral / math.log(ratio)
    u_req = math.sqrt(vinf_req ** 2 + vesc ** 2)
    e_kin_min = 0.5 * cfg.asteroid_mass_kg * u_req ** 2
    e_draw_min = e_kin_min / cfg.launch_efficiency
    # thrust-rate limit (packet size x frequency at max speed)
    dv_rate_max = cfg.ejections_per_year * cfg.max_packet_mass_kg * vinf_max / m_sys     # m/s per year
    years_min = dv_spiral / dv_rate_max if dv_rate_max > 0 else float("inf")
    # combined: slowest launch speed that satisfies BOTH the mass limit and the time limit,
    # and the energy that this strategy needs
    vinf_time = dv_spiral * m_sys / (cfg.mission_years * cfg.ejections_per_year * cfg.max_packet_mass_kg)
    vinf_comb = max(vinf_req, vinf_time)
    u_comb = math.sqrt(vinf_comb ** 2 + vesc ** 2)
    m_comb = m_sys * (1 - math.exp(-dv_spiral / vinf_comb))
    e_draw_comb = 0.5 * m_comb * u_comb ** 2 / cfg.launch_efficiency
    checks = {
        "mass": dict(ok=u_req <= cfg.max_ejection_speed_ms,
                     detail=f"needs launch speed {u_req:.3e} m/s with ALL {cfg.asteroid_mass_kg:.2e} kg; "
                            f"max allowed {cfg.max_ejection_speed_ms:.3e} m/s "
                            f"(mass-limited delta-v capacity {dv_cap_mass:.4g} m/s vs required {dv_spiral:.4g} m/s)"),
        "energy": dict(ok=e_draw_min <= cfg.energy_budget_j,
                       detail=f"minimum energy drawn {e_draw_min:.3e} J vs budget {cfg.energy_budget_j:.3e} J"),
        "rate": dict(ok=years_min <= cfg.mission_years,
                     detail=f"at max packet rate and speed the transfer needs >= {years_min:,.0f} years "
                            f"(mission {cfg.mission_years:,.0f} y)"),
        "combined": dict(ok=u_comb <= cfg.max_ejection_speed_ms and e_draw_comb <= cfg.energy_budget_j
                         and m_comb <= cfg.asteroid_mass_kg,
                         detail=f"slowest speed meeting mass AND time limits {u_comb:.3e} m/s -> needs "
                                f"{m_comb:.3e} kg and {e_draw_comb:.3e} J"),
    }
    feasible = all(c["ok"] for c in checks.values())
    shortfall = dv_cap_mass / dv_spiral
    lines = [
        f"Required slow-spiral delta-v 1 -> {cfg.target_au} AU: {dv_spiral:,.0f} m/s "
        f"(Hohmann two-impulse: {dv_hoh:,.0f} m/s)",
        f"Orbital energy change of Earth+Moon: {dE_orb:.3e} J; angular momentum change {dL_orb:.3e} kg m^2/s",
        f"Average power just for the orbital energy change over {cfg.mission_years:,.0f} y: "
        f"{dE_orb / T:.3e} W  (= {dE_orb / T / EARTH_INTERCEPT_W:.2e} x all sunlight intercepted by Earth)",
        f"Minimum launch energy with this reservoir: {e_draw_min:.3e} J -> average power {e_draw_min / T:.3e} W "
        f"(= {e_draw_min / T / L_SUN:.2e} x total solar luminosity)",
        f"Reaction mass needed at max speed: {mass_needed_umax:.3e} kg "
        f"({mass_needed_umax / M_MOON:.3g} lunar masses); available {cfg.asteroid_mass_kg:.3e} kg",
        f"Max launch speed = {cfg.max_ejection_speed_ms / C_LIGHT * 100:.3g}% of c "
        f"(non-relativistic model; error grows above ~10% c)",
    ]
    for k, c in checks.items():
        lines.append(f"[{'OK ' if c['ok'] else 'FAIL'}] {k}: {c['detail']}")
    verdict = ("FEASIBLE within this model (physically hypothetical scale)" if feasible
               else "INFEASIBLE with these parameters (failed: " +
               ", ".join(k for k, c in checks.items() if not c["ok"]) +
               (f"; mass-limited delta-v reaches {shortfall:.2e} of the requirement)" if shortfall < 1 else ")"))
    plain = [
        f"To move Earth from 1 AU to {cfg.target_au} AU slowly, its orbital speed must be changed by about "
        f"{dv_spiral / 1000:,.1f} km/s in total (it ends up slower, but on a much bigger orbit).",
        ("Mass: OK - throwing the whole asteroid at " if checks["mass"]["ok"] else
         "Mass: NOT ENOUGH - even throwing the whole asteroid at ") +
        f"{min(u_req, cfg.max_ejection_speed_ms) / 1000:,.0f} km/s " +
        ("gives enough push." if checks["mass"]["ok"] else
         f"the maximum speed gives only {shortfall:.2%} of the required push."),
        ("Energy: OK - " if checks["energy"]["ok"] else "Energy: NOT ENOUGH - ") +
        f"the cheapest way needs {e_draw_min:.2e} J; the budget is {cfg.energy_budget_j:.2e} J.",
        ("Time: OK - " if checks["rate"]["ok"] else "Time: TOO SHORT - ") +
        f"at full thrust the transfer takes at least {years_min:,.0f} years; the mission allows {cfg.mission_years:,.0f}.",
        ("All limits together: OK." if checks["combined"]["ok"] else
         "All limits together: NOT met (meeting the time limit needs faster, more energy-hungry launches)."),
        "This is an analytic estimate for a slow spiral; the simulation then shows what actually happens.",
    ]
    return dict(feasible=feasible, verdict=verdict, lines=lines, plain=plain, dv_required=dv_spiral, dv_hohmann=dv_hoh,
                dE_orbit=dE_orb, dL_orbit=dL_orb, dv_capacity_mass=dv_cap_mass, u_required=u_req,
                energy_min_draw=e_draw_min, mass_needed_at_umax=mass_needed_umax, years_min=years_min,
                u_combined=u_comb, energy_combined=e_draw_comb, mass_combined=m_comb,
                avg_power_min=e_draw_min / T, checks=checks, vesc=vesc, vinf_max=vinf_max)


def habitability(a_au: float, e: float) -> dict:
    """Simplified energy balance (NOT a climate model): orbit-averaged insolation and the
    blackbody equilibrium temperature for Bond albedo 0.3 plus today's ~33 K greenhouse offset."""
    if not (a_au > 0) or not math.isfinite(a_au) or e >= 1:
        return dict(insolation=float("nan"), t_eq=float("nan"), t_surface_naive=float("nan"))
    s = S_1AU / (a_au ** 2 * math.sqrt(1 - e * e))
    t_eq = (s * (1 - 0.3) / (4 * 5.670374419e-8)) ** 0.25
    return dict(insolation=s, insolation_rel=s / S_1AU, t_eq=t_eq, t_surface_naive=t_eq + 33.0)


# =============================================================================================
# 4. INITIAL CONDITIONS (J2000 mean elements, ecliptic frame)
# =============================================================================================
def elements_to_state(a, e, inc, raan, argp, M, mu):
    E = M
    for _ in range(50):
        E -= (E - e * math.sin(E) - M) / (1 - e * math.cos(E))
    cE, sE = math.cos(E), math.sin(E)
    r = a * (1 - e * cE)
    rp = np.array([a * (cE - e), a * math.sqrt(1 - e * e) * sE, 0.0])
    vp = math.sqrt(mu * a) / r * np.array([-sE, math.sqrt(1 - e * e) * cE, 0.0])
    cO, sO, ci, si, cw, sw = math.cos(raan), math.sin(raan), math.cos(inc), math.sin(inc), math.cos(argp), math.sin(argp)
    Rm = np.array([[cO * cw - sO * sw * ci, -cO * sw - sO * cw * ci, sO * si],
                   [sO * cw + cO * sw * ci, -sO * sw + cO * cw * ci, -cO * si],
                   [sw * si, cw * si, ci]])
    return Rm @ rp, Rm @ vp


def initial_conditions(cfg: Config, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    dL = rng.uniform(0, 2 * math.pi) if cfg.randomize_phase else 0.0
    moon_phase = rng.uniform(0, 2 * math.pi) if cfg.randomize_phase else 0.0
    m_e = M_EARTH + cfg.asteroid_mass_kg
    mu_h = GM_SUN + G * (m_e + M_MOON)
    varpi, L0 = math.radians(102.93768193), math.radians(100.46457166)
    r_emb, v_emb = elements_to_state(1.00000261 * AU, 0.01671123, 0.0, 0.0, varpi, (L0 - varpi + dL) % (2 * math.pi), mu_h)
    r_rel, v_rel = elements_to_state(MOON_A, 0.0549, math.radians(5.145), math.radians(125.08),
                                     math.radians(318.15), moon_phase, G * (m_e + M_MOON))
    return dict(r_emb=r_emb, v_emb=v_emb, r_rel=r_rel, v_rel=v_rel, m_earth=m_e, seed=seed,
                phase_offset=dL, moon_phase=moon_phase)


# =============================================================================================
# 5. NUMBA KERNELS
# =============================================================================================
@njit(cache=True, nogil=True)
def _stumpC(z):
    if z > 1e-8:
        s = math.sqrt(z)
        return (1.0 - math.cos(s)) / z
    if z < -1e-8:
        s = math.sqrt(-z)
        return (math.cosh(s) - 1.0) / (-z)
    return 0.5 - z / 24.0


@njit(cache=True, nogil=True)
def _stumpS(z):
    if z > 1e-8:
        s = math.sqrt(z)
        return (s - math.sin(s)) / (s * s * s)
    if z < -1e-8:
        s = math.sqrt(-z)
        return (math.sinh(s) - s) / (s * s * s)
    return 1.0 / 6.0 - z / 120.0


@njit(cache=True, nogil=True)
def kepler_step(r, v, mu, dt):
    """Exact two-body propagation (universal variables), in place."""
    r0 = math.sqrt(r[0] * r[0] + r[1] * r[1] + r[2] * r[2])
    v0sq = v[0] * v[0] + v[1] * v[1] + v[2] * v[2]
    rv = r[0] * v[0] + r[1] * v[1] + r[2] * v[2]
    smu = math.sqrt(mu)
    alpha = 2.0 / r0 - v0sq / mu
    x = smu * abs(alpha) * dt if alpha > 1e-30 else smu * dt / r0
    for _ in range(80):
        z = alpha * x * x
        C = _stumpC(z)
        S = _stumpS(z)
        F = rv / smu * x * x * C + (1.0 - alpha * r0) * x * x * x * S + r0 * x - smu * dt
        dF = rv / smu * x * (1.0 - z * S) + (1.0 - alpha * r0) * x * x * C + r0
        dx = F / dF
        x -= dx
        if abs(dx) <= 1e-14 * abs(x) + 1e-12:
            break
    z = alpha * x * x
    C = _stumpC(z)
    S = _stumpS(z)
    f = 1.0 - x * x / r0 * C
    g = dt - x * x * x / smu * S
    nr0 = f * r[0] + g * v[0]
    nr1 = f * r[1] + g * v[1]
    nr2 = f * r[2] + g * v[2]
    rn = math.sqrt(nr0 * nr0 + nr1 * nr1 + nr2 * nr2)
    fd = smu / (rn * r0) * x * (z * S - 1.0)
    gd = 1.0 - x * x / rn * C
    v0, v1, v2 = v[0], v[1], v[2]
    v[0] = fd * r[0] + gd * v0
    v[1] = fd * r[1] + gd * v1
    v[2] = fd * r[2] + gd * v2
    r[0], r[1], r[2] = nr0, nr1, nr2


@njit(cache=True, nogil=True)
def orbit_elements(r, v, mu):
    """Returns (a, e, inc, specific_energy). a = inf for unbound orbits."""
    hx = r[1] * v[2] - r[2] * v[1]
    hy = r[2] * v[0] - r[0] * v[2]
    hz = r[0] * v[1] - r[1] * v[0]
    hm = math.sqrt(hx * hx + hy * hy + hz * hz)
    rm = math.sqrt(r[0] * r[0] + r[1] * r[1] + r[2] * r[2])
    vsq = v[0] * v[0] + v[1] * v[1] + v[2] * v[2]
    eps = 0.5 * vsq - mu / rm
    ex = (v[1] * hz - v[2] * hy) / mu - r[0] / rm
    ey = (v[2] * hx - v[0] * hz) / mu - r[1] / rm
    ez = (v[0] * hy - v[1] * hx) / mu - r[2] / rm
    e = math.sqrt(ex * ex + ey * ey + ez * ez)
    inc = math.acos(max(-1.0, min(1.0, hz / hm))) if hm > 0 else 0.0
    a = -mu / (2.0 * eps) if eps < 0 else np.inf
    return a, e, inc, eps


@njit(cache=True, nogil=True)
def rtn_to_inertial(r, v, d_rtn):
    rm = math.sqrt(r[0] * r[0] + r[1] * r[1] + r[2] * r[2])
    R = r / rm
    h = np.cross(r, v)
    N = h / math.sqrt(h[0] * h[0] + h[1] * h[1] + h[2] * h[2])
    T = np.cross(N, R)
    return d_rtn[0] * R + d_rtn[1] * T + d_rtn[2] * N


@njit(cache=True, nogil=True)
def eject_core(M, m_ast, E_rem, dm_req, u, eta, r_launch, phi_extra):
    """Shared launch physics.  Returns (dm, v_inf, E_draw, E_kin, fell_back).
    phi_extra = additional gravitational potential depth [J/kg] the packet must climb out of
    (fast model: the Moon, G M_moon / d_EM; high-fidelity: 0 because the Moon is explicit).
    Two-body Earth(+reservoir) / packet momentum exchange with reduced mass mu = dm(M-dm)/M:
    kinetic energy given by the launcher in the centre-of-mass frame E_kin = mu u^2 / 2,
    energy drawn from the budget E_kin / eta.  Packets slower than escape speed fall back."""
    dm = min(dm_req, m_ast)
    if dm <= 0.0 or E_rem <= 0.0:
        return 0.0, 0.0, 0.0, 0.0, 0
    mu_red = dm * (M - dm) / M
    E_kin = 0.5 * mu_red * u * u
    E_draw = E_kin / eta
    if E_draw > E_rem:
        dm = dm * E_rem / E_draw
        mu_red = dm * (M - dm) / M
        E_kin = 0.5 * mu_red * u * u
        E_draw = min(E_kin / eta, E_rem)
    if dm <= 0.0:
        return 0.0, 0.0, 0.0, 0.0, 0
    vesc2 = 2.0 * G * M / r_launch + 2.0 * phi_extra
    if u * u <= vesc2:
        return dm, 0.0, E_draw, E_kin, 1
    return dm, math.sqrt(u * u - vesc2), E_draw, E_kin, 0


# ---------------------------------------------------------------- FAST model kernel ----------
@njit(cache=True, nogil=True)
def _fast_eject(st, ledger, k, samples, dm_req, u, d_rtn, eta, r_launch):
    """One packet launch in the FAST model.  The packet leaves EARTH (not the barycentre):
    Earth(+docked asteroid) recoils, the Moon keeps its velocity, and the new Earth-Moon
    barycentre and Earth-Moon relative orbit follow from momentum conservation."""
    M = st[6]
    mm = M_MOON
    r = st[0:3]
    v = st[3:6]
    rrel = st[11:14]
    vrel = st[14:17]
    d_em = math.sqrt(rrel[0] ** 2 + rrel[1] ** 2 + rrel[2] ** 2)
    phi_moon = G * mm / d_em
    dm, vinf, Ed, Ek, fb = eject_core(M, st[7], st[8], dm_req, u, eta, r_launch, phi_moon)
    if dm <= 0.0:
        return
    st[8] -= Ed
    ledger[LG_E_DRAWN] += Ed
    ledger[LG_LAUNCH_KE] += Ek
    samples[k, S_DM] = dm
    samples[k, S_U] = u
    samples[k, S_EDRAW] = Ed
    if fb == 1:
        ledger[LG_N_FALLBACK] += 1.0
        ledger[LG_MASS_FALLBACK] += dm
        samples[k, S_FATE] = 3
        return
    fe = mm / (M + mm)
    fm = M / (M + mm)
    rE = r - fe * rrel
    vE = v - fe * vrel
    rM = r + fm * rrel
    vM = v + fm * vrel
    ehat = -rtn_to_inertial(r, v, d_rtn)
    vp = vE + ((M - dm) / M) * vinf * ehat           # packet velocity after escaping Earth+Moon
    vE2 = vE - (dm / M) * vinf * ehat                # Earth(+asteroid) recoil
    M2 = M - dm
    v_old = v.copy()
    for c in range(3):
        st[0 + c] = (M2 * rE[c] + mm * rM[c]) / (M2 + mm)
        st[3 + c] = (M2 * vE2[c] + mm * vM[c]) / (M2 + mm)
        st[14 + c] = vM[c] - vE2[c]
    st[6] = M2
    st[7] -= dm
    mu_red = dm * (M - dm) / M
    ledger[LG_MASS_EJECTED] += dm
    ledger[LG_N_PACKETS] += 1.0
    for c in range(3):
        ledger[LG_PKT_PX + c] += dm * vp[c]
    imp = mu_red * vinf
    ledger[LG_IMPULSE_TOTAL] += imp
    if dm * u > ledger[LG_MAX_IMPULSE]:
        ledger[LG_MAX_IMPULSE] = dm * u
    ledger[LG_DE_EJECT_THEORY] += Ek - G * dm * (M - dm) / r_launch - dm * phi_moon
    rm = math.sqrt(rE[0] * rE[0] + rE[1] * rE[1] + rE[2] * rE[2])
    if 0.5 * (vp[0] ** 2 + vp[1] ** 2 + vp[2] ** 2) - GM_SUN / rm < 0:
        ledger[LG_PKT_BOUND_SUN_M] += dm
        samples[k, S_FATE] = 2
    else:
        ledger[LG_PKT_UNBOUND_SUN_M] += dm
        samples[k, S_FATE] = 1
    samples[k, S_VREL] = vinf
    dv2 = 0.0
    for c in range(3):
        dv2 += (st[3 + c] - v_old[c]) ** 2
    samples[k, S_DV] = math.sqrt(dv2)
    for c in range(3):
        samples[k, S_VP + c] = vp[c] / 1000.0
        samples[k, S_EHAT + c] = ehat[c]


@njit(cache=True, nogil=True)
def fast_interval(st, ledger, n_events, dt_event, dm_req, u, d_rtn, eta, r_launch, samples, moon):
    """FAST model (hierarchical two-body): the Earth-Moon barycentre (EMB) moves on an exact
    Kepler orbit about the Sun, and the Moon moves on an exact Kepler orbit about Earth(+docked
    asteroid) under their mutual gravity G (M_earth + m_asteroid + M_moon).  Between packets both
    motions are exact; each packet launch is an instantaneous momentum-conserving event applied to
    EARTH (see _fast_eject).  Neglected here (kept in the high-fidelity model): the Sun's tidal
    perturbation of the Moon's orbit and the packets' gravity after they have escaped.
    st = [r_emb(3), v_emb(3), m_earth+asteroid, m_ast, E_rem, t, unused, r_rel(3), v_rel(3)]
    (r_rel, v_rel = Moon minus Earth).  If moon has rows, MOON_SUB Moon states per event are
    recorded for the viewer (same format as the high-fidelity model)."""
    r = st[0:3]
    v = st[3:6]
    rrel = st[11:14]
    vrel = st[14:17]
    rec = moon.shape[0] > 0
    half = 0.5 * dt_event
    hs = MOON_SUB // 2
    sub = dt_event / MOON_SUB
    status = 0
    n_done = 0
    rb = np.zeros(3)
    vb = np.zeros(3)
    for k in range(n_events):
        for c in range(10, N_SAMPLE_COLS):
            samples[k, c] = 0.0
        for h in range(2):
            if h == 1:
                vpre = v.copy()
                if dm_req > 0.0:
                    _fast_eject(st, ledger, k, samples, dm_req, u, d_rtn, eta, r_launch)
                # record the state AT the ejection instant (position continuous; velocity jumps)
                samples[k, S_T] = (st[9] + half) / YEAR
                for c in range(3):
                    samples[k, S_R + c] = r[c] / AU
                    samples[k, S_VPRE + c] = vpre[c] / 1000.0
                    samples[k, S_VPOST + c] = v[c] / 1000.0
                samples[k, S_MAST] = st[7]
                samples[k, S_EREM] = st[8]
            mu_m = G * (st[6] + M_MOON)
            if rec:      # viewer samples: exact Kepler from the half-interval start (state untouched)
                for j in range(hs):
                    for c in range(3):
                        rb[c] = rrel[c]
                        vb[c] = vrel[c]
                    kepler_step(rb, vb, mu_m, (j + 1) * sub)
                    row = k * MOON_SUB + h * hs + j
                    moon[row, 0] = (st[9] + h * half + (j + 1) * sub) / YEAR
                    for c in range(3):
                        moon[row, 1 + c] = rb[c]
                        moon[row, 4 + c] = vb[c]
            kepler_step(r, v, GM_SUN + G * (st[6] + M_MOON), half)
            kepler_step(rrel, vrel, mu_m, half)
        st[9] += dt_event
        n_done = k + 1
        mu = GM_SUN + G * (st[6] + M_MOON)
        rm = math.sqrt(r[0] * r[0] + r[1] * r[1] + r[2] * r[2])
        vsq = v[0] * v[0] + v[1] * v[1] + v[2] * v[2]
        if rm < 0.2 * AU:
            status = 2
            break
        if rm > 30.0 * AU or 0.5 * vsq - mu / rm >= 0.0:
            status = 1
            break
    return n_done, status


# ---------------------------------------------------------------- HIGH-FIDELITY kernels ------
# bodies: 0 Sun, 1 Earth (+ docked reservoir), 2 Moon, 3.. fragments.
# Pairs: every major body with every body; fragment-fragment gravity is neglected (documented).
_W1 = 1.0 / (2.0 - 2.0 ** (1.0 / 3.0))
_W0 = -(2.0 ** (1.0 / 3.0)) * _W1
YC1 = _W1 / 2.0
YC2 = (_W0 + _W1) / 2.0
YD1 = _W1
YD2 = _W0


@njit(cache=True, nogil=True)
def nb_accel(pos, mass, nb, acc):
    for i in range(nb):
        acc[i, 0] = 0.0
        acc[i, 1] = 0.0
        acc[i, 2] = 0.0
    for i in range(3):
        for j in range(i + 1, nb):
            dx = pos[j, 0] - pos[i, 0]
            dy = pos[j, 1] - pos[i, 1]
            dz = pos[j, 2] - pos[i, 2]
            r2 = dx * dx + dy * dy + dz * dz
            inv3 = 1.0 / (r2 * math.sqrt(r2))
            fi = G * mass[j] * inv3
            fj = G * mass[i] * inv3
            acc[i, 0] += fi * dx
            acc[i, 1] += fi * dy
            acc[i, 2] += fi * dz
            acc[j, 0] -= fj * dx
            acc[j, 1] -= fj * dy
            acc[j, 2] -= fj * dz


@njit(cache=True, nogil=True)
def yoshida_step(pos, vel, mass, nb, h, acc):
    for i in range(nb):
        for c in range(3):
            pos[i, c] += YC1 * h * vel[i, c]
    nb_accel(pos, mass, nb, acc)
    for i in range(nb):
        for c in range(3):
            vel[i, c] += YD1 * h * acc[i, c]
            pos[i, c] += YC2 * h * vel[i, c]
    nb_accel(pos, mass, nb, acc)
    for i in range(nb):
        for c in range(3):
            vel[i, c] += YD2 * h * acc[i, c]
            pos[i, c] += YC2 * h * vel[i, c]
    nb_accel(pos, mass, nb, acc)
    for i in range(nb):
        for c in range(3):
            vel[i, c] += YD1 * h * acc[i, c]
            pos[i, c] += YC1 * h * vel[i, c]


@njit(cache=True, nogil=True)
def nb_totals(pos, vel, mass, nb, out):
    """out = [E, Px, Py, Pz, Lx, Ly, Lz, M] with the same pair set as nb_accel."""
    for k in range(8):
        out[k] = 0.0
    for i in range(nb):
        m = mass[i]
        out[0] += 0.5 * m * (vel[i, 0] ** 2 + vel[i, 1] ** 2 + vel[i, 2] ** 2)
        for c in range(3):
            out[1 + c] += m * vel[i, c]
        out[4] += m * (pos[i, 1] * vel[i, 2] - pos[i, 2] * vel[i, 1])
        out[5] += m * (pos[i, 2] * vel[i, 0] - pos[i, 0] * vel[i, 2])
        out[6] += m * (pos[i, 0] * vel[i, 1] - pos[i, 1] * vel[i, 0])
        out[7] += m
    for i in range(3):
        for j in range(i + 1, nb):
            d = math.sqrt((pos[j, 0] - pos[i, 0]) ** 2 + (pos[j, 1] - pos[i, 1]) ** 2 + (pos[j, 2] - pos[i, 2]) ** 2)
            out[0] -= G * mass[i] * mass[j] / d


@njit(cache=True, nogil=True)
def _record_event(before, after, ledger, slot_e):
    de = after[0] - before[0]
    ledger[LG_EV_DE] += de
    if slot_e >= 0:
        ledger[slot_e] += de
    for c in range(3):
        ledger[LG_EV_DP + c] += after[1 + c] - before[1 + c]
        ledger[LG_EV_DL + c] += after[4 + c] - before[4 + c]
    return de


@njit(cache=True, nogil=True)
def _remove(pos, vel, mass, nbarr, k):
    last = nbarr[0] - 1
    if k != last:
        for c in range(3):
            pos[k, c] = pos[last, c]
            vel[k, c] = vel[last, c]
        mass[k] = mass[last]
    mass[last] = 0.0
    nbarr[0] = last


@njit(cache=True, nogil=True)
def _merge_into(pos, vel, mass, nbarr, k, b, ledger, tb, ta):
    """Perfectly inelastic impact of fragment k on body b (mass and momentum conserved)."""
    nb_totals(pos, vel, mass, nbarr[0], tb)
    mf, mb = mass[k], mass[b]
    for c in range(3):
        vel[b, c] = (mb * vel[b, c] + mf * vel[k, c]) / (mb + mf)
    mass[b] = mb + mf
    if b == 1:
        ledger[LG_EARTH_IMPACT_M] += mf
    elif b == 2:
        ledger[LG_MOON_IMPACT_M] += mf
    else:
        ledger[LG_SUN_IMPACT_M] += mf
    _remove(pos, vel, mass, nbarr, k)
    nb_totals(pos, vel, mass, nbarr[0], ta)
    _record_event(tb, ta, ledger, LG_DE_IMPACT)


@njit(cache=True, nogil=True)
def _merge_closest_fragments(pos, vel, mass, nbarr, ledger, tb, ta):
    nb = nbarr[0]
    bi, bj, best = -1, -1, 1e300
    for i in range(3, nb):
        for j in range(i + 1, nb):
            d = (pos[i, 0] - pos[j, 0]) ** 2 + (pos[i, 1] - pos[j, 1]) ** 2 + (pos[i, 2] - pos[j, 2]) ** 2
            if d < best:
                best, bi, bj = d, i, j
    if bi < 0:
        return
    nb_totals(pos, vel, mass, nb, tb)
    mi, mj = mass[bi], mass[bj]
    for c in range(3):
        pos[bi, c] = (mi * pos[bi, c] + mj * pos[bj, c]) / (mi + mj)
        vel[bi, c] = (mi * vel[bi, c] + mj * vel[bj, c]) / (mi + mj)
    mass[bi] = mi + mj
    _remove(pos, vel, mass, nbarr, bj)
    nb_totals(pos, vel, mass, nbarr[0], ta)
    ledger[LG_N_MERGES] += 1.0
    _record_event(tb, ta, ledger, LG_DE_MERGE)


@njit(cache=True, nogil=True)
def hifi_advance(pos, vel, mass, nbarr, ledger, T, dt_max, prev, acc, esc_r, tb, ta):
    n = max(1, int(math.ceil(T / dt_max - 1e-9)))
    h = T / n
    radii = np.array([R_SUN, R_EARTH, R_MOON])
    for _ in range(n):
        nb = nbarr[0]
        if nb > 3:
            for i in range(nb):
                for c in range(3):
                    prev[i, c] = pos[i, c]
        yoshida_step(pos, vel, mass, nb, h, acc)
        if nb > 3:
            k = 3
            while k < nbarr[0]:
                hit = -1
                for b in range(3):
                    r00 = prev[k, 0] - prev[b, 0]
                    r01 = prev[k, 1] - prev[b, 1]
                    r02 = prev[k, 2] - prev[b, 2]
                    d0 = (pos[k, 0] - pos[b, 0]) - r00
                    d1 = (pos[k, 1] - pos[b, 1]) - r01
                    d2 = (pos[k, 2] - pos[b, 2]) - r02
                    dd = d0 * d0 + d1 * d1 + d2 * d2
                    ts = 0.0
                    if dd > 0:
                        ts = -(r00 * d0 + r01 * d1 + r02 * d2) / dd
                        ts = min(1.0, max(0.0, ts))
                    cx, cy, cz = r00 + ts * d0, r01 + ts * d1, r02 + ts * d2
                    if cx * cx + cy * cy + cz * cz < radii[b] * radii[b]:
                        hit = b
                        break
                if hit >= 0:
                    _merge_into(pos, vel, mass, nbarr, k, hit, ledger, tb, ta)
                else:
                    k += 1
    # escape bookkeeping: unbound fragments beyond esc_r leave the simulated subsystem
    k = 3
    while k < nbarr[0]:
        rx, ry, rz = pos[k, 0] - pos[0, 0], pos[k, 1] - pos[0, 1], pos[k, 2] - pos[0, 2]
        vx, vy, vz = vel[k, 0] - vel[0, 0], vel[k, 1] - vel[0, 1], vel[k, 2] - vel[0, 2]
        rr = math.sqrt(rx * rx + ry * ry + rz * rz)
        if rr > esc_r and 0.5 * (vx * vx + vy * vy + vz * vz) - GM_SUN / rr > 0:
            nb_totals(pos, vel, mass, nbarr[0], tb)
            ledger[LG_ESC_MASS] += mass[k]
            _remove(pos, vel, mass, nbarr, k)
            nb_totals(pos, vel, mass, nbarr[0], ta)
            de = _record_event(tb, ta, ledger, -1)
            ledger[LG_ESC_E] -= de
            for c in range(3):
                ledger[LG_ESC_PX + c] -= ta[1 + c] - tb[1 + c]
        else:
            k += 1


@njit(cache=True, nogil=True)
def eject_hifi(pos, vel, mass, nbarr, stv, ledger, dm_req, u, d_rtn, eta, r_launch, r_release, tb, ta, pk):
    """pk (output, len 12): dm, u, v_rel, vp(3) [barycentric m/s], fate, E_draw, ehat(3), unused."""
    nmax = mass.shape[0]
    M = mass[1]
    for c in range(pk.shape[0]):
        pk[c] = 0.0
    dm, vinf, Ed, Ek, fb = eject_core(M, stv[0], stv[1], dm_req, u, eta, r_launch, 0.0)
    if dm <= 0.0:
        return
    pk[0], pk[1], pk[7] = dm, u, Ed
    stv[1] -= Ed
    ledger[LG_E_DRAWN] += Ed
    ledger[LG_LAUNCH_KE] += Ek
    if fb == 1:
        ledger[LG_N_FALLBACK] += 1.0
        ledger[LG_MASS_FALLBACK] += dm
        pk[6] = 3
        return
    if nbarr[0] >= nmax:
        _merge_closest_fragments(pos, vel, mass, nbarr, ledger, tb, ta)
    me, mm = mass[1], mass[2]
    remb = (me * pos[1] + mm * pos[2]) / (me + mm) - pos[0]
    vemb = (me * vel[1] + mm * vel[2]) / (me + mm) - vel[0]
    ehat = -rtn_to_inertial(remb, vemb, d_rtn)
    nb_totals(pos, vel, mass, nbarr[0], tb)
    s = pos[2] - pos[1]
    p = s[0] * ehat[0] + s[1] * ehat[1] + s[2] * ehat[2]
    perp = s - p * ehat
    hit_moon = (p > r_launch) and (p < r_release) and \
        (math.sqrt(perp[0] ** 2 + perp[1] ** 2 + perp[2] ** 2) < R_MOON)
    rr = p if hit_moon else r_release
    vrel2 = u * u - 2.0 * G * M * (1.0 / r_launch - 1.0 / rr)
    vrel = math.sqrt(max(vrel2, 0.0))
    v1 = vel[1].copy()
    p1 = pos[1].copy()
    vp = v1 + ((M - dm) / M) * vrel * ehat
    for c in range(3):
        vel[1, c] = v1[c] - (dm / M) * vrel * ehat[c]
    mass[1] = M - dm
    stv[0] -= dm
    pk[2] = vrel
    for c in range(3):
        pk[3 + c] = vp[c]
        pk[8 + c] = ehat[c]
    if hit_moon:
        pk[6] = 4
    else:
        rh = p1 + r_release * ehat - pos[0]
        vh = vp - vel[0]
        if 0.5 * (vh[0] ** 2 + vh[1] ** 2 + vh[2] ** 2) - GM_SUN / math.sqrt(rh[0] ** 2 + rh[1] ** 2 + rh[2] ** 2) < 0:
            pk[6] = 2
        else:
            pk[6] = 1
    if hit_moon:
        for c in range(3):
            vel[2, c] = (mm * vel[2, c] + dm * vp[c]) / (mm + dm)
        mass[2] = mm + dm
        ledger[LG_N_LAUNCH_MOON_HITS] += 1.0
        ledger[LG_MOON_IMPACT_M] += dm
    else:
        k = nbarr[0]
        nbarr[0] = k + 1
        for c in range(3):
            pos[k, c] = p1[c] + ((M - dm) / M) * r_release * ehat[c]
            pos[1, c] = p1[c] - (dm / M) * r_release * ehat[c]
            vel[k, c] = vp[c]
        mass[k] = dm
    nb_totals(pos, vel, mass, nbarr[0], ta)
    _record_event(tb, ta, ledger, LG_DE_EJECT)
    mu_red = dm * (M - dm) / M
    ledger[LG_MASS_EJECTED] += dm
    ledger[LG_N_PACKETS] += 1.0
    ledger[LG_IMPULSE_TOTAL] += mu_red * vinf
    if dm * u > ledger[LG_MAX_IMPULSE]:
        ledger[LG_MAX_IMPULSE] = dm * u
    ledger[LG_DE_EJECT_THEORY] += Ek - G * dm * (M - dm) / r_launch
    for c in range(3):
        ledger[LG_PKT_PX + c] += dm * vp[c]


@njit(cache=True, nogil=True)
def hifi_interval(pos, vel, mass, nbarr, stv, ledger, n_events, dt_event, dt_max, dm_req, u, d_rtn,
                  eta, r_launch, r_release, esc_r, samples, moon, prev, acc):
    """stv = [m_ast, E_rem, t].  Each event: 4 sub-chunks, ejection, 4 sub-chunks; one viewer
    sample at the ejection instant and MOON_SUB Moon states per event.  Returns (n_samples, status)."""
    tb = np.zeros(8)
    ta = np.zeros(8)
    pk = np.zeros(12)
    status = 0
    n_done = 0
    sub = dt_event / MOON_SUB
    for k in range(n_events):
        for j in range(MOON_SUB):
            if j == MOON_SUB // 2:
                vpre = vel[1] - vel[0]
                if dm_req > 0.0:
                    eject_hifi(pos, vel, mass, nbarr, stv, ledger, dm_req, u, d_rtn, eta, r_launch, r_release,
                               tb, ta, pk)
                else:
                    for c in range(pk.shape[0]):
                        pk[c] = 0.0
                samples[k, S_T] = (stv[2] + 0.5 * dt_event) / YEAR
                for c in range(3):
                    samples[k, S_R + c] = (pos[1, c] - pos[0, c]) / AU
                    samples[k, S_VPRE + c] = vpre[c] / 1000.0
                    samples[k, S_VPOST + c] = (vel[1, c] - vel[0, c]) / 1000.0
                    samples[k, S_VP + c] = (pk[3 + c] - vel[0, c]) / 1000.0 if pk[0] > 0 and pk[6] != 3 else 0.0
                    samples[k, S_EHAT + c] = pk[8 + c]
                samples[k, S_DM] = pk[0]
                samples[k, S_U] = pk[1]
                samples[k, S_VREL] = pk[2]
                samples[k, S_FATE] = pk[6]
                samples[k, S_EDRAW] = pk[7]
                samples[k, S_MAST] = stv[0]
                samples[k, S_EREM] = stv[1]
                samples[k, S_NFRAG] = nbarr[0] - 3
                dvx = samples[k, S_VPOST] - samples[k, S_VPRE]
                dvy = samples[k, S_VPOST + 1] - samples[k, S_VPRE + 1]
                dvz = samples[k, S_VPOST + 2] - samples[k, S_VPRE + 2]
                samples[k, S_DV] = 1000.0 * math.sqrt(dvx * dvx + dvy * dvy + dvz * dvz)
            hifi_advance(pos, vel, mass, nbarr, ledger, sub, dt_max, prev, acc, esc_r, tb, ta)
            row = k * MOON_SUB + j
            moon[row, 0] = (stv[2] + (j + 1) * sub) / YEAR
            for c in range(3):
                moon[row, 1 + c] = pos[2, c] - pos[1, c]
                moon[row, 4 + c] = vel[2, c] - vel[1, c]
        stv[2] += dt_event
        n_done = k + 1
        me, mm = mass[1], mass[2]
        rx = (me * pos[1, 0] + mm * pos[2, 0]) / (me + mm) - pos[0, 0]
        ry = (me * pos[1, 1] + mm * pos[2, 1]) / (me + mm) - pos[0, 1]
        rz = (me * pos[1, 2] + mm * pos[2, 2]) / (me + mm) - pos[0, 2]
        rm = math.sqrt(rx * rx + ry * ry + rz * rz)
        if rm < 0.2 * AU:
            status = 2
            break
        if rm > 30.0 * AU:
            status = 1
            break
    return n_done, status


@njit(cache=True, nogil=True)
def packet_arcs(r0, v0, offsets_s, esc_r, out, nvalid):
    """Viewer helper (Python-side physics): propagate each released packet on its exact two-body
    orbit about the Sun (universal-variable Kepler solution) and store its position at the given
    times after release.  Stops once the packet is beyond esc_r (it has left the tracked region)."""
    for i in range(r0.shape[0]):
        r = r0[i].copy()
        v = v0[i].copy()
        tprev = 0.0
        nvalid[i] = 0
        for j in range(offsets_s.shape[0]):
            if offsets_s[j] > tprev:
                kepler_step(r, v, GM_SUN, offsets_s[j] - tprev)
                tprev = offsets_s[j]
            for c in range(3):
                out[i, j, c] = r[c]
            nvalid[i] = j + 1
            if math.sqrt(r[0] * r[0] + r[1] * r[1] + r[2] * r[2]) > esc_r:
                break


# =============================================================================================
# 6. ENGINES (Python wrappers)
# =============================================================================================
class FastEngine:
    """FAST training model (hierarchical two-body, see fast_interval): EMB on an exact Kepler
    orbit about the Sun + the Moon on an exact Kepler orbit about Earth(+docked asteroid) under
    their mutual gravity.  Ejections are impulsive, act on Earth and conserve momentum exactly.
    Packets are tracked as a ledger (mass, momentum, launch energy, bound/unbound to the Sun);
    their escape from Earth's AND the Moon's gravity is included in their final speed.
    Not modelled here (but in the high-fidelity model): the Sun's tide on the Moon's orbit,
    individual fragment trajectories after escape, packets hitting the Moon, fragment re-impacts."""
    fidelity = "fast"

    def __init__(self, cfg: Config, seed: int, record_moon: bool = True):
        self.cfg = cfg
        self.record_moon = record_moon
        ic = initial_conditions(cfg, seed)
        self.st = np.zeros(17)
        self.st[0:3], self.st[3:6] = ic["r_emb"], ic["v_emb"]
        self.st[6] = ic["m_earth"]
        self.st[7] = cfg.asteroid_mass_kg
        self.st[8] = cfg.energy_budget_j
        self.st[11:14], self.st[14:17] = ic["r_rel"], ic["v_rel"]
        self.ledger = np.zeros(LEDGER_N)
        self.last_moon = np.zeros((0, 7))

    t_years = property(lambda s: s.st[9] / YEAR)
    m_ast = property(lambda s: s.st[7])
    E_rem = property(lambda s: s.st[8])
    m_launcher = property(lambda s: s.st[6])

    def mu(self):
        return GM_SUN + G * (self.st[6] + M_MOON)

    def emb(self):
        return self.st[0:3].copy(), self.st[3:6].copy()

    def run_interval(self, years, dm, u, d_rtn):
        n = max(1, int(round(self.cfg.ejections_per_year * years)))
        samples = np.zeros((n, N_SAMPLE_COLS))
        moon = np.zeros((n * MOON_SUB if self.record_moon else 0, 7))
        k, status = fast_interval(self.st, self.ledger, n, years * YEAR / n, float(dm), float(u),
                                  np.asarray(d_rtn, dtype=np.float64), self.cfg.launch_efficiency,
                                  R_EARTH, samples, moon)
        self.last_moon = moon[:k * MOON_SUB]
        return samples[:k], int(status)

    def earth_state(self):
        """Viewer state: the fast model's rows are the Earth-Moon BARYCENTRE (4,700 km from Earth's
        centre, invisible at AU scale); third item = Moon relative to Earth (position, velocity)."""
        return self.st[0:3].copy(), self.st[3:6].copy(), (self.st[11:14].copy(), self.st[14:17].copy())

    def earth_moon(self):
        """Heliocentric Earth and Moon positions [m] (from barycentre + relative orbit)."""
        M = self.st[6]
        fe, fm = M_MOON / (M + M_MOON), M / (M + M_MOON)
        r, rr = self.st[0:3], self.st[11:14]
        return r - fe * rr, r + fm * rr

    def moon_info(self):
        rel, vrel = self.st[11:14], self.st[14:17]
        d = float(np.linalg.norm(rel))
        eps = 0.5 * float(vrel @ vrel) - G * (self.st[6] + M_MOON) / d
        hill = float(np.linalg.norm(self.st[0:3])) * ((self.st[6] + M_MOON) / (3 * M_SUN)) ** (1 / 3)
        return dict(modelled=True, bound=bool(eps < 0 and d < 0.5 * hill), dist_m=d, hill_m=hill)

    def conservation(self):
        return dict(model="fast", note="impulses conserve momentum exactly by construction; "
                                        "orbital motion is exact two-body Kepler (Sun-EMB and Earth-Moon)")


class HifiEngine:
    """HIGH-FIDELITY model.  N-body Sun, Earth(+docked reservoir), Moon and explicit fragments
    integrated with 4th-order Yoshida (fixed dt, default 6 h); ejection, impact, escape and
    merge events; full energy / momentum / angular-momentum bookkeeping."""
    fidelity = "hifi"

    def __init__(self, cfg: Config, seed: int):
        self.cfg = cfg
        ic = initial_conditions(cfg, seed)
        nmax = 3 + cfg.max_fragments
        self.pos = np.zeros((nmax, 3))
        self.vel = np.zeros((nmax, 3))
        self.mass = np.zeros(nmax)
        me = ic["m_earth"]
        f_e, f_m = M_MOON / (me + M_MOON), me / (me + M_MOON)
        self.mass[:3] = [M_SUN, me, M_MOON]
        self.pos[1] = ic["r_emb"] - f_e * ic["r_rel"]
        self.vel[1] = ic["v_emb"] - f_e * ic["v_rel"]
        self.pos[2] = ic["r_emb"] + f_m * ic["r_rel"]
        self.vel[2] = ic["v_emb"] + f_m * ic["v_rel"]
        mt = self.mass[:3].sum()
        self.pos[:3] -= (self.mass[:3, None] * self.pos[:3]).sum(0) / mt
        self.vel[:3] -= (self.mass[:3, None] * self.vel[:3]).sum(0) / mt
        self.nbarr = np.array([3], dtype=np.int64)
        self.stv = np.array([cfg.asteroid_mass_kg, cfg.energy_budget_j, 0.0])
        self.ledger = np.zeros(LEDGER_N)
        self.prev = np.zeros((nmax, 3))
        self.acc = np.zeros((nmax, 3))
        self.tot0 = self.totals()
        self.last_moon = np.zeros((0, 7))

    t_years = property(lambda s: s.stv[2] / YEAR)
    m_ast = property(lambda s: s.stv[0])
    E_rem = property(lambda s: s.stv[1])
    m_launcher = property(lambda s: s.mass[1])

    def totals(self):
        out = np.zeros(8)
        nb_totals(self.pos, self.vel, self.mass, int(self.nbarr[0]), out)
        return out

    def mu(self):
        return G * (self.mass[0] + self.mass[1] + self.mass[2])

    def emb(self):
        me, mm = self.mass[1], self.mass[2]
        r = (me * self.pos[1] + mm * self.pos[2]) / (me + mm) - self.pos[0]
        v = (me * self.vel[1] + mm * self.vel[2]) / (me + mm) - self.vel[0]
        return r, v

    def run_interval(self, years, dm, u, d_rtn):
        cfg = self.cfg
        n = max(1, int(round(cfg.ejections_per_year * years)))
        samples = np.zeros((n, N_SAMPLE_COLS))
        moon = np.zeros((n * MOON_SUB, 7))
        k, status = hifi_interval(self.pos, self.vel, self.mass, self.nbarr, self.stv, self.ledger, n,
                                  years * YEAR / n, cfg.hifi_dt_hours * 3600.0, float(dm), float(u),
                                  np.asarray(d_rtn, dtype=np.float64), cfg.launch_efficiency, R_EARTH,
                                  cfg.release_radius_m, cfg.fragment_escape_au * AU, samples, moon,
                                  self.prev, self.acc)
        self.last_moon = moon[:k * MOON_SUB]
        return samples[:k], int(status)

    def earth_state(self):
        return self.pos[1] - self.pos[0], self.vel[1] - self.vel[0], (self.pos[2] - self.pos[1], self.vel[2] - self.vel[1])

    def moon_info(self):
        rel = self.pos[2] - self.pos[1]
        vrel = self.vel[2] - self.vel[1]
        d = float(np.linalg.norm(rel))
        eps = 0.5 * float(vrel @ vrel) - G * (self.mass[1] + self.mass[2]) / d
        r_emb, _ = self.emb()
        hill = float(np.linalg.norm(r_emb)) * ((self.mass[1] + self.mass[2]) / (3 * self.mass[0])) ** (1 / 3)
        return dict(modelled=True, bound=bool(eps < 0 and d < 0.5 * hill), dist_m=d, hill_m=hill)

    def conservation(self):
        now = self.totals()
        L = self.ledger
        e_ref = abs(self.tot0[0])
        p_ref = M_EARTH * 3e4
        l_ref = float(np.linalg.norm(self.tot0[4:7]))
        resid_e = (now[0] - self.tot0[0] - L[LG_EV_DE]) / e_ref
        resid_p = float(np.linalg.norm(now[1:4] - self.tot0[1:4] - L[LG_EV_DP:LG_EV_DP + 3])) / p_ref
        resid_l = float(np.linalg.norm(now[4:7] - self.tot0[4:7] - L[LG_EV_DL:LG_EV_DL + 3])) / l_ref
        mass_resid = (now[7] + L[LG_ESC_MASS] - self.tot0[7]) / self.tot0[7]
        return dict(model="hifi", energy_residual_rel=float(resid_e), momentum_residual_rel=resid_p,
                    angmom_residual_rel=resid_l, mass_residual_rel=float(mass_resid),
                    energy_injected_by_ejections_J=float(L[LG_DE_EJECT]),
                    energy_theory_ejections_J=float(L[LG_DE_EJECT_THEORY]),
                    energy_removed_by_escapes_J=float(L[LG_ESC_E]), impact_heat_J=float(-L[LG_DE_IMPACT]),
                    merge_energy_J=float(L[LG_DE_MERGE]), merges=int(L[LG_N_MERGES]),
                    active_fragments=int(self.nbarr[0] - 3))


def make_engine(cfg: Config, fidelity: str, seed: int, record_moon: bool = True):
    return HifiEngine(cfg, seed) if fidelity == "hifi" else FastEngine(cfg, seed, record_moon)


# =============================================================================================
# 7. GYMNASIUM ENVIRONMENT
# =============================================================================================
OBS_DIM = 21
REWARD_KEYS = ["progress", "eccentricity", "inclination", "energy_use", "mass_use", "time",
               "success", "terminal_error", "failure"]


class OctoEarthEnv(gym.Env):
    """Action (Box[-1,1]^4):
        a0 throttle -> packet mass = throttle * max_packet (throttle < 0.02 -> no ejection)
        a1 in-plane thrust angle theta = a1*pi measured from prograde (+T) toward +R
        a2 out-of-plane thrust angle phi = a2*pi/2
        a3 launch speed u = u_min + (a3+1)/2 (u_max - u_min)
    The thrust direction is held fixed in the rotating RTN frame of the Earth-Moon barycentre
    during one decision interval; packets are ejected opposite to it at the configured
    frequency.  The agent never sets Earth's velocity: only packets are launched."""
    metadata = {"render_modes": ["ansi"]}

    def __init__(self, cfg: Config | None = None, fidelity: str = "fast", record: bool = True,
                 record_moon: bool = True):
        super().__init__()
        self.cfg = cfg or Config()
        self.fidelity = fidelity
        self.record = record
        self.record_moon = record_moon
        self.observation_space = spaces.Box(-10.0, 10.0, shape=(OBS_DIM,), dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float32)
        self.engine = None
        self.last_trail = None
        self.last_decisions = []

    # ------------------------------------------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is None:
            seed = int(self.np_random.integers(0, 2 ** 31 - 1))
        self.seed_used = int(seed)
        cfg = self.cfg
        self.engine = make_engine(cfg, self.fidelity, self.seed_used, self.record_moon)
        self.steps = 0
        self.prev_action = np.zeros(4, dtype=np.float32)
        self.m_ast0 = cfg.asteroid_mass_kg
        self.E0 = cfg.energy_budget_j
        self.mu_t = self.engine.mu()
        self.eps_t = -self.mu_t / (2 * cfg.target_au * AU)
        r, v = self.engine.emb()
        a0, e0, _, eps0 = orbit_elements(r, v, self.engine.mu())
        self.eps0 = eps0
        self.a0_au, self.e0 = a0 / AU, e0
        self.phi = self._potential()
        self.cum_reward = 0.0
        self.components = {k: 0.0 for k in REWARD_KEYS}
        self.decisions = []
        self.trail = []
        self.done_reason = None
        self.success = False
        self.validated = None
        self.dv_cum = 0.0
        return self._obs(), {}

    # ------------------------------------------------------------------------------------------
    def elements(self):
        r, v = self.engine.emb()
        a, e, inc, eps = orbit_elements(r, v, self.engine.mu())
        return dict(a_au=a / AU if math.isfinite(a) else float("inf"), e=e, inc_deg=math.degrees(inc),
                    r_au=float(np.linalg.norm(r)) / AU, speed=float(np.linalg.norm(v)), eps=eps)

    def _potential(self):
        el = self.elements()
        return -min(3.0, abs(el["eps"] - self.eps_t) / abs(self.eps0 - self.eps_t))

    def dv_needed(self, el=None):
        el = el or self.elements()
        a = el["a_au"] * AU
        vc = math.sqrt(self.engine.mu() / a) if math.isfinite(a) and a > 0 else 0.0
        return abs(vc - math.sqrt(self.engine.mu() / (self.cfg.target_au * AU)))

    def _obs(self):
        cfg = self.cfg
        r, v = self.engine.emb()
        el = self.elements()
        moon = self.engine.moon_info()
        M = self.engine.m_launcher + M_MOON
        m_rem = self.engine.m_ast
        vinf_max = launch_vinf(cfg.max_ejection_speed_ms, self.engine.m_launcher)
        need = self.dv_needed(el) + 1.0
        cap_m = vinf_max * math.log(M / max(M - m_rem, 1.0)) + 1e-9
        cap_e = math.sqrt(2 * cfg.launch_efficiency * max(self.engine.E_rem, 0) * m_rem) / M + 1e-9
        a_rel = el["a_au"] / cfg.target_au if math.isfinite(el["a_au"]) else 5.0
        o = np.concatenate([
            r / (cfg.target_au * AU), v / 3.0e4,
            [min(a_rel, 5.0), min(el["e"] / cfg.tol_e, 10.0), min(el["inc_deg"] / cfg.tol_inc_deg, 10.0),
             max(-5.0, min(5.0, (el["eps"] - self.eps_t) / abs(self.eps0 - self.eps_t))),
             moon["dist_m"] / MOON_A, 1.0 if moon["bound"] else 0.0,
             m_rem / self.m_ast0, max(self.engine.E_rem, 0) / self.E0,
             max(0.0, 1.0 - self.engine.t_years / cfg.mission_years),
             max(-3.0, min(3.0, math.log10(cap_m / need))) / 3.0,
             max(-3.0, min(3.0, math.log10(cap_e / need))) / 3.0],
            self.prev_action])
        return np.clip(np.nan_to_num(o, nan=0.0, posinf=10.0, neginf=-10.0), -10, 10).astype(np.float32)

    # ------------------------------------------------------------------------------------------
    def decode_action(self, action):
        cfg = self.cfg
        a = np.nan_to_num(np.asarray(action, dtype=np.float64).reshape(4), nan=0.0)
        a = np.clip(a, -1.0, 1.0)
        throttle = 0.5 * (a[0] + 1.0)
        if throttle < 0.02:
            throttle = 0.0
        theta, phi = a[1] * math.pi, a[2] * math.pi / 2
        u = cfg.min_ejection_speed_ms + 0.5 * (a[3] + 1.0) * (cfg.max_ejection_speed_ms - cfg.min_ejection_speed_ms)
        d = np.array([math.sin(theta) * math.cos(phi), math.cos(theta) * math.cos(phi), math.sin(phi)])
        return dict(throttle=throttle, dm=throttle * cfg.max_packet_mass_kg, u=u, d_rtn=d,
                    theta_deg=math.degrees(theta), phi_deg=math.degrees(phi), clipped=a)

    def in_target(self, el=None):
        el = el or self.elements()
        c = self.cfg
        return (math.isfinite(el["a_au"]) and abs(el["a_au"] - c.target_au) <= c.tol_a_au and
                el["e"] <= c.tol_e and el["inc_deg"] <= c.tol_inc_deg)

    def step(self, action):
        cfg = self.cfg
        dec = self.decode_action(action)
        eng = self.engine
        years = min(cfg.decision_interval_years, cfg.mission_years - eng.t_years)
        led0 = eng.ledger.copy()
        m0, E0 = eng.m_ast, eng.E_rem
        samples, status = eng.run_interval(years, dec["dm"], dec["u"], dec["d_rtn"])
        self.steps += 1
        dm_used = m0 - eng.m_ast
        e_used = E0 - eng.E_rem
        el = self.elements()
        moon = eng.moon_info()
        phi_new = self._potential()
        frac = years / 100.0
        rc = dict(progress=cfg.w_progress * (phi_new - self.phi),
                  eccentricity=-cfg.w_eccentricity * min(5.0, max(0.0, el["e"] - cfg.tol_e) / cfg.tol_e) * frac,
                  inclination=-cfg.w_inclination * min(5.0, max(0.0, el["inc_deg"] - cfg.tol_inc_deg) / cfg.tol_inc_deg) * frac,
                  energy_use=-cfg.w_energy * e_used / self.E0,
                  mass_use=-cfg.w_mass * dm_used / self.m_ast0,
                  time=-cfg.w_time, success=0.0, terminal_error=0.0, failure=0.0)
        self.phi = phi_new
        terminated = False
        reason = None
        if status == 1:
            terminated, reason = True, "escaped_or_unbound"
        elif status == 2:
            terminated, reason = True, "too_close_to_sun"
        elif not moon["bound"]:
            terminated, reason = True, "moon_lost"
        elif self.in_target(el):
            terminated, reason = True, "success"
        elif eng.t_years >= cfg.mission_years - 1e-6:
            terminated, reason = True, "deadline"
        elif eng.m_ast <= 1e-9 * self.m_ast0 or eng.E_rem <= 1e-9 * self.E0:
            terminated, reason = True, "resources_exhausted"
        if terminated:
            if reason == "success":
                left = 0.5 * (eng.m_ast / self.m_ast0 + max(eng.E_rem, 0) / self.E0)
                rc["success"] = cfg.w_success * (1.0 + 0.5 * left)
                self.success = True
            elif reason in ("deadline", "resources_exhausted"):
                rc["terminal_error"] = -cfg.w_terminal_error * self.error_norm(el)
            else:
                rc["failure"] = -cfg.w_failure
        reward = float(sum(rc.values()))
        self.cum_reward += reward
        for k, v in rc.items():
            self.components[k] += v
        self.prev_action = dec["clipped"].astype(np.float32)
        led1 = eng.ledger
        impulse = led1[LG_IMPULSE_TOTAL] - led0[LG_IMPULSE_TOTAL]
        M_emb = eng.m_launcher + M_MOON
        dv_int = impulse / M_emb                 # speed change delivered to the Earth-Moon system
        self.dv_cum += dv_int
        rec = dict(step=self.steps, t_years=eng.t_years, a_au=el["a_au"], e=el["e"], inc_deg=el["inc_deg"],
                   r_au=el["r_au"], throttle=dec["throttle"], dm_kg=dm_used, u_ms=dec["u"],
                   theta_deg=dec["theta_deg"], phi_deg=dec["phi_deg"], d_rtn=dec["d_rtn"].tolist(),
                   impulse_Ns=impulse, energy_J=e_used, m_ast_frac=eng.m_ast / self.m_ast0,
                   energy_frac=max(eng.E_rem, 0) / self.E0, reward=reward, cum_reward=self.cum_reward,
                   moon_dist_m=moon["dist_m"], moon_bound=moon["bound"],
                   fallback=int(led1[LG_N_FALLBACK] - led0[LG_N_FALLBACK]),
                   n_packets=int(led1[LG_N_PACKETS] - led0[LG_N_PACKETS]),
                   moon_hits=int(led1[LG_N_LAUNCH_MOON_HITS] - led0[LG_N_LAUNCH_MOON_HITS]),
                   dv_ms=dv_int, dv_cum_ms=self.dv_cum, thrust_N=impulse / max(years * YEAR, 1.0),
                   impulse_cum_Ns=float(led1[LG_IMPULSE_TOTAL]), speed_kms=el["speed"] / 1000.0,
                   E_orb_J=el["eps"] * M_emb, E_used_cum_J=self.E0 - eng.E_rem,
                   m_ast_kg=eng.m_ast, E_rem_J=max(eng.E_rem, 0.0), years=years, reason=reason)
        if eng.fidelity == "hifi":
            rec["conservation"] = eng.conservation()
        self.decisions.append(rec)
        self.last_samples = samples
        if self.record and len(samples):
            # every 3rd sample (3/8 year apart) so successive points sit at different orbital phases
            self.trail.extend(samples[::3, 1:4].round(4).tolist())
        if terminated:
            self.done_reason = reason
            self.last_trail = self.trail
            self.last_decisions = self.decisions
        info = dict(reason=reason, metrics=self.metrics(el), components=rc, decision=rec)
        if terminated:
            info["episode_summary"] = self.summary()
        return self._obs(), reward, terminated, False, info

    def error_norm(self, el=None):
        el = el or self.elements()
        a = el["a_au"] if math.isfinite(el["a_au"]) else 10.0
        return min(3.0, abs(a - self.cfg.target_au) / (self.cfg.target_au - 1.0)) + 0.1 * min(10.0, el["e"] / self.cfg.tol_e)

    def metrics(self, el=None):
        el = el or self.elements()
        eng = self.engine
        L = eng.ledger
        hab = habitability(el["a_au"], el["e"])
        return dict(t_years=eng.t_years, a_au=el["a_au"], e=el["e"], inc_deg=el["inc_deg"], r_au=el["r_au"],
                    shortfall_au=self.cfg.target_au - el["a_au"] if math.isfinite(el["a_au"]) else float("nan"),
                    mass_used_kg=self.m_ast0 - eng.m_ast, mass_used_frac=1 - eng.m_ast / self.m_ast0,
                    energy_used_J=self.E0 - eng.E_rem, energy_used_frac=1 - max(eng.E_rem, 0) / self.E0,
                    impulse_Ns=float(L[LG_IMPULSE_TOTAL]), n_packets=int(L[LG_N_PACKETS]),
                    fallback_packets=int(L[LG_N_FALLBACK]), max_packet_impulse_Ns=float(L[LG_MAX_IMPULSE]),
                    insolation_rel=hab.get("insolation_rel", float("nan")), t_eq_K=hab["t_eq"],
                    in_target=self.in_target(el))

    def summary(self):
        el = self.elements()
        s = dict(fidelity=self.fidelity, seed=self.seed_used, steps=self.steps, reason=self.done_reason,
                 success=self.success, total_reward=self.cum_reward, components=dict(self.components),
                 **self.metrics(el))
        L = self.engine.ledger
        s["ledger"] = dict(mass_ejected=float(L[LG_MASS_EJECTED]), packets=int(L[LG_N_PACKETS]),
                           launch_ke=float(L[LG_LAUNCH_KE]), energy_drawn=float(L[LG_E_DRAWN]),
                           escaped_mass=float(L[LG_ESC_MASS]), earth_impact_mass=float(L[LG_EARTH_IMPACT_M]),
                           moon_impact_mass=float(L[LG_MOON_IMPACT_M]), sun_impact_mass=float(L[LG_SUN_IMPACT_M]),
                           packets_bound_to_sun_mass=float(L[LG_PKT_BOUND_SUN_M]),
                           packets_unbound_from_sun_mass=float(L[LG_PKT_UNBOUND_SUN_M]),
                           fallback_mass=float(L[LG_MASS_FALLBACK]), merges=int(L[LG_N_MERGES]),
                           n_fallback=int(L[LG_N_FALLBACK]), n_launch_moon_hits=int(L[LG_N_LAUNCH_MOON_HITS]))
        s["a0_au"], s["e0"] = self.a0_au, self.e0
        if self.fidelity == "hifi":
            s["conservation"] = self.engine.conservation()
            s["moon"] = self.engine.moon_info()
        return s

    def render(self):
        el = self.elements()
        return (f"t={self.engine.t_years:8.1f} y  a={el['a_au']:.4f} AU  e={el['e']:.4f}  "
                f"ast={self.engine.m_ast / self.m_ast0:6.1%}  E={max(self.engine.E_rem, 0) / self.E0:6.1%}")


gym.register(id="OctoEarth-v0", entry_point=lambda **kw: OctoEarthEnv(**kw))


# =============================================================================================
# 8. BASELINE POLICIES
# =============================================================================================
def policy_none(env, obs):
    """No ejection at all (baseline)."""
    return np.array([-1.0, 0.0, 0.0, -1.0], dtype=np.float32)


def heuristic_speed(cfg: Config) -> float:
    """Hand-designed choice of launch speed: the lowest speed that (a) lets the whole reservoir
    supply the required delta-v (energy-optimal) and (b) still finishes within 85% of the
    mission at the maximum packet rate."""
    fz = feasibility(cfg)
    m_sys = M_EARTH + cfg.asteroid_mass_kg + M_MOON
    vinf_mass = fz["dv_required"] / math.log(m_sys / (m_sys - 0.95 * cfg.asteroid_mass_kg))
    vinf_time = fz["dv_required"] * m_sys / (0.85 * cfg.mission_years * cfg.ejections_per_year * cfg.max_packet_mass_kg)
    vinf = max(vinf_mass, vinf_time)
    u = math.sqrt(vinf ** 2 + fz["vesc"] ** 2)
    return min(max(u, cfg.min_ejection_speed_ms), cfg.max_ejection_speed_ms)


def policy_heuristic(env, obs):
    """Hand-designed strategy: prograde (tangential) thrust, speed from heuristic_speed(),
    throttle proportional to the remaining slow-spiral delta-v so it lands on the target."""
    cfg = env.cfg
    if not hasattr(env, "_heur_u"):
        env._heur_u = heuristic_speed(cfg)
    u = env._heur_u
    el = env.elements()
    vinf = launch_vinf(u, env.engine.m_launcher)
    M = env.engine.m_launcher + M_MOON
    n_ev = cfg.ejections_per_year * cfg.decision_interval_years
    dv_max = n_ev * cfg.max_packet_mass_kg * vinf / M
    need = env.dv_needed(el)
    a = el["a_au"]
    theta = 0.0 if a < cfg.target_au else 1.0               # retro-thrust if overshooting
    throttle = min(1.0, need / max(dv_max, 1e-9))
    if abs(a - cfg.target_au) < 0.25 * cfg.tol_a_au:
        throttle = 0.0
    a3 = 2 * (u - cfg.min_ejection_speed_ms) / (cfg.max_ejection_speed_ms - cfg.min_ejection_speed_ms) - 1
    return np.array([2 * throttle - 1, theta, 0.0, a3], dtype=np.float32)


def model_policy(model):
    def pol(env, obs):
        a, _ = model.predict(obs, deterministic=True)
        return a
    return pol


def run_episode(cfg: Config, policy, seed: int, fidelity: str = "fast", validate: bool = False,
                on_step=None, stop_event=None, on_reset=None) -> dict:
    env = OctoEarthEnv(cfg, fidelity, record=True)
    obs, _ = env.reset(seed=seed)
    if on_reset:
        on_reset(env)
    done = False
    t0 = time.time()
    while not done:
        if stop_event is not None and stop_event.is_set():
            break
        obs, r, term, trunc, info = env.step(policy(env, obs))
        done = term or trunc
        if on_step:
            on_step(env, info)
    summ = env.summary()
    summ["decisions"] = env.decisions
    summ["wall_s"] = time.time() - t0
    if validate and fidelity == "hifi" and summ["reason"] in ("success", "deadline", "resources_exhausted"):
        summ["validation"] = validate_orbit(env)
    return summ


def validate_orbit(env: OctoEarthEnv) -> dict:
    """Ballistic (zero ejection) propagation after the mission in the HIGH-FIDELITY model; the
    orbit must stay within tolerances and the Moon must stay bound."""
    cfg = env.cfg
    chunks = 10
    worst_da, worst_e, moon_ok = 0.0, 0.0, True
    for _ in range(chunks):
        env.engine.run_interval(cfg.validate_years / chunks, 0.0, cfg.min_ejection_speed_ms, [0.0, 1.0, 0.0])
        el = env.elements()
        da = abs(el["a_au"] - cfg.target_au) if math.isfinite(el["a_au"]) else float("inf")
        worst_da, worst_e = max(worst_da, da), max(worst_e, el["e"])
        moon_ok = moon_ok and env.engine.moon_info()["bound"]
    ok = worst_da <= cfg.tol_a_au and worst_e <= cfg.tol_e and moon_ok
    return dict(years=cfg.validate_years, max_abs_da_au=worst_da, max_e=worst_e, moon_bound=moon_ok, passed=bool(ok))


# =============================================================================================
# 9. PHYSICS SELF-CHECKS (run before training; also used by tests/test_octoearth.py)
# =============================================================================================
def _fresh_hifi(cfg, seed=3):
    return HifiEngine(cfg, seed)


def run_physics_checks(full: bool = False, log=print) -> list:
    cfg = make_config("hypothetical", randomize_phase=False)
    out = []

    def check(name, fn):
        t0 = time.time()
        try:
            ok, detail = fn()
        except Exception as exc:  # pragma: no cover
            ok, detail = False, f"exception: {exc!r}"
        out.append(dict(name=name, passed=bool(ok), detail=detail, seconds=round(time.time() - t0, 2)))
        log(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")

    def kepler():
        ic = initial_conditions(cfg, 1)
        mu = GM_SUN + G * (ic["m_earth"] + M_MOON)
        r, v = ic["r_emb"].copy(), ic["v_emb"].copy()
        a, e, _, _ = orbit_elements(r, v, mu)
        P = 2 * math.pi * math.sqrt(a ** 3 / mu)
        r1, v1 = r.copy(), v.copy()
        for _ in range(37):
            kepler_step(r1, v1, mu, P / 37)
        closure = np.linalg.norm(r1 - r) / np.linalg.norm(r)
        # N-body integrator vs Kepler for an isolated Sun-Earth pair (Moon massless and far away)
        pos = np.zeros((4, 3)); vel = np.zeros((4, 3)); mass = np.array([M_SUN, M_EARTH, 0.0, 0.0])
        pos[1], vel[1] = r, v
        pos[2] = [60 * AU, 0, 0]
        nbarr = np.array([3], dtype=np.int64)
        led = np.zeros(LEDGER_N)
        hifi_advance(pos, vel, mass, nbarr, led, YEAR, 6 * 3600.0, np.zeros((4, 3)), np.zeros((4, 3)),
                     1e30, np.zeros(8), np.zeros(8))
        rk, vk = r.copy(), v.copy()
        kepler_step(rk, vk, G * (M_SUN + M_EARTH), YEAR)
        err = np.linalg.norm((pos[1] - pos[0]) - rk)
        return closure < 1e-9 and err < 1e4, f"Kepler 1-orbit closure {closure:.1e}; N-body vs Kepler after 1 y: {err:.1f} m"

    def gravity():
        h = _fresh_hifi(cfg)
        acc = np.zeros_like(h.pos)
        def src(i, j):
            d = h.pos[j] - h.pos[i]
            return G * h.mass[j] * d / np.linalg.norm(d) ** 3
        sun_e = np.linalg.norm(src(1, 0)); moon_e = np.linalg.norm(src(1, 2)); earth_m = np.linalg.norm(src(2, 1))
        nb_accel(h.pos, h.mass, 3, acc)
        tot_ok = np.allclose(acc[1], src(1, 0) + src(1, 2), rtol=1e-12)
        h0 = _fresh_hifi(replace(cfg, asteroid_mass_kg=0.0))
        ratio = np.linalg.norm(src(2, 1)) / (G * h0.mass[1] / np.linalg.norm(h.pos[2] - h.pos[1]) ** 2)
        ast_ok = abs(ratio - (M_EARTH + cfg.asteroid_mass_kg) / M_EARTH) < 1e-9
        # fragment gravity on Earth
        h.pos[3] = h.pos[1] + np.array([1e9, 0, 0]); h.mass[3] = 1e20
        acc2 = np.zeros_like(h.pos)
        nb_accel(h.pos, h.mass, 4, acc2)
        frag = np.linalg.norm(acc2[1] - acc[1])
        frag_ok = abs(frag - G * 1e20 / 1e18) / (G * 1e20 / 1e18) < 1e-6
        ok = 5.6e-3 < sun_e < 6.2e-3 and 2.8e-5 < moon_e < 3.8e-5 and 2.4e-3 < earth_m < 3.2e-3 and tot_ok and ast_ok and frag_ok
        return ok, (f"Sun->Earth {sun_e:.3e}, Moon->Earth {moon_e:.3e}, Earth(+asteroid)->Moon {earth_m:.3e} m/s^2; "
                    f"asteroid mass scales Earth's pull by {ratio:.6f}; fragment(1e20 kg @1e6 km)->Earth {frag:.3e} m/s^2")

    def newton3():
        h = _fresh_hifi(cfg)
        h.pos[3] = h.pos[1] + [3e9, 1e9, 0]; h.mass[3] = 1e19
        h.pos[4] = h.pos[0] + [0, 2 * AU, 0]; h.mass[4] = 1e18
        acc = np.zeros_like(h.pos)
        nb_accel(h.pos, h.mass, 5, acc)
        f = (h.mass[:5, None] * acc[:5])
        rel = np.linalg.norm(f.sum(0)) / np.abs(f).sum()
        return rel < 1e-13, f"|sum m a| / sum|m a| = {rel:.1e}"

    def eject_momentum():
        h = _fresh_hifi(cfg)
        tb, ta = h.totals(), None
        eject_hifi(h.pos, h.vel, h.mass, h.nbarr, h.stv, h.ledger, 1e18, 3e6, np.array([0.0, 1.0, 0.0]),
                   cfg.launch_efficiency, R_EARTH, cfg.release_radius_m, np.zeros(8), np.zeros(8), np.zeros(12))
        ta = h.totals()
        dp = np.linalg.norm(ta[1:4] - tb[1:4]) / (M_EARTH * 3e4)       # vs Earth's orbital momentum
        dl = np.linalg.norm(ta[4:7] - tb[4:7]) / np.linalg.norm(tb[4:7])
        f = FastEngine(cfg, 3)
        memb = f.st[6] + M_MOON
        p0 = memb * f.st[3:6]
        f.run_interval(1e-3 / YEAR, 1e18, 3e6, [0, 1, 0])           # one packet, 1 ms of orbital motion
        p1 = (f.st[6] + M_MOON) * f.st[3:6] + f.ledger[LG_PKT_PX:LG_PKT_PX + 3]
        dpf = np.linalg.norm(p1 - p0) / (1e18 * 3e6)
        return dp < 1e-14 and dl < 1e-12 and dpf < 1e-4, (f"high-fidelity ejection: |dP|/(M_E v_E) = {dp:.1e} (round-off), |dL|/|L| = {dl:.1e}; "
                                                           f"fast model |dP|/(dm u) = {dpf:.1e} (incl. 1 ms of solar gravity)")

    def energy_accounting():
        h = _fresh_hifi(cfg)
        eject_hifi(h.pos, h.vel, h.mass, h.nbarr, h.stv, h.ledger, 1e18, 3e6, np.array([0.0, 1.0, 0.0]),
                   cfg.launch_efficiency, R_EARTH, cfg.release_radius_m, np.zeros(8), np.zeros(8), np.zeros(12))
        L = h.ledger
        rel = abs(L[LG_DE_EJECT] - L[LG_DE_EJECT_THEORY]) / L[LG_LAUNCH_KE]
        draw_ok = abs(L[LG_E_DRAWN] - L[LG_LAUNCH_KE] / cfg.launch_efficiency) <= 1e-12 * L[LG_E_DRAWN]
        budget_ok = abs(cfg.energy_budget_j - h.stv[1] - L[LG_E_DRAWN]) <= 1e-9 * cfg.energy_budget_j
        hs = _fresh_hifi(replace(cfg, energy_budget_j=1e30))
        eject_hifi(hs.pos, hs.vel, hs.mass, hs.nbarr, hs.stv, hs.ledger, 1e18, 3e6, np.array([0.0, 1.0, 0.0]),
                   cfg.launch_efficiency, R_EARTH, cfg.release_radius_m, np.zeros(8), np.zeros(8), np.zeros(12))
        clip_ok = hs.stv[1] >= 0 and hs.ledger[LG_MASS_EJECTED] < 1e18
        return rel < 1e-6 and draw_ok and budget_ok and clip_ok, (
            f"measured system energy change vs 1/2 mu u^2 - G dm M/R: rel diff {rel:.1e}; drawn = KE/eta; "
            f"budget never exceeded (tiny budget -> packet reduced to {hs.ledger[LG_MASS_EJECTED]:.2e} kg)")

    def mass_accounting():
        c2 = replace(cfg, max_fragments=3)
        h = _fresh_hifi(c2)
        m0 = h.totals()[7]
        h.run_interval(3.0, 1e18, 3e4, [0.3, 0.9, 0.0])           # slow packets stay bound -> merges
        h.run_interval(2.0, 1e18, 3e6, [0, 1, 0])                  # fast packets escape
        c = h.conservation()
        small = HifiEngine(replace(cfg, asteroid_mass_kg=5e18), 3)
        small.run_interval(5.0, 1e18, 3e6, [0, 1, 0])
        ok = abs(c["mass_residual_rel"]) < 1e-14 and small.m_ast >= 0 and small.ledger[LG_MASS_EJECTED] <= 5e18 * (1 + 1e-12)
        return ok, (f"total mass + escaped mass - initial = {c['mass_residual_rel']:.1e} (rel); merges={c['merges']}, "
                    f"escaped {h.ledger[LG_ESC_MASS]:.2e} kg; reservoir cannot go negative "
                    f"(5e18 kg reservoir -> ejected {small.ledger[LG_MASS_EJECTED]:.2e} kg)")

    def reverse():
        res = {}
        for name, d in (("pro", [0, 1, 0]), ("retro", [0, -1, 0])):
            f = FastEngine(cfg, 3)
            a0 = orbit_elements(*f.emb(), f.mu())[0]
            f.run_interval(200, 5e17, 2.5e6, d)
            res[name] = orbit_elements(*f.emb(), f.mu())[0] - a0
        sym = abs(res["pro"] + res["retro"]) / abs(res["pro"])
        return res["pro"] > 0 > res["retro"] and sym < 0.2, (
            f"prograde da = {res['pro'] / AU:+.4f} AU, retrograde da = {res['retro'] / AU:+.4f} AU (asymmetry {sym:.2f})")

    def fragments():
        f = FastEngine(cfg, 3)
        v0 = f.st[3:6].copy()
        f.run_interval(1e-6, 1e18, 5e3, [0, 1, 0])               # 5 km/s < escape speed
        fb_ok = f.ledger[LG_N_FALLBACK] == 1 and f.m_ast == cfg.asteroid_mass_kg and f.ledger[LG_E_DRAWN] > 0
        h = _fresh_hifi(cfg)
        h.run_interval(0.125, 1e18, 3e6, [0, 1, 0])
        h.run_interval(1.0, 0.0, 3e6, [0, 1, 0])
        esc_ok = abs(h.ledger[LG_ESC_MASS] - 1e18) < 1e6
        g = _fresh_hifi(cfg)
        g.pos[3] = g.pos[1] + [2e8, 0, 0]; g.vel[3] = g.vel[1] + [-2e4, 0, 0]; g.mass[3] = 1e18; g.nbarr[0] = 4
        p0 = g.totals()[1:4].copy()
        g.run_interval(0.02, 0.0, 3e6, [0, 1, 0])
        p1 = g.totals()[1:4]
        imp_ok = g.ledger[LG_EARTH_IMPACT_M] == 1e18 and g.nbarr[0] == 3 and np.linalg.norm(p1 - p0) / (1e18 * 2e4) < 1e-6
        return fb_ok and esc_ok and imp_ok, (
            f"sub-escape packet falls back (no net recoil, energy lost): {fb_ok}; fast packet leaves the system "
            f"and is ledgered as escaped: {esc_ok}; fragment aimed at Earth merges with momentum conserved: {imp_ok}")

    def reproducible():
        a = run_episode(replace(cfg, randomize_phase=True, mission_years=1000.0), policy_heuristic, 7)
        b = run_episode(replace(cfg, randomize_phase=True, mission_years=1000.0), policy_heuristic, 7)
        return a["a_au"] == b["a_au"] and a["total_reward"] == b["total_reward"], f"identical results for seed 7 (a={a['a_au']:.6f} AU)"

    def convergence():
        errs = []
        def run(dt_h):
            h = HifiEngine(replace(cfg, hifi_dt_hours=dt_h), 3)
            h.run_interval(10.0, 0.0, 3e6, [0, 1, 0])
            return h.pos[1] - h.pos[0]
        ref = run(0.75)
        for dt in (12.0, 6.0, 3.0):
            errs.append(float(np.linalg.norm(run(dt) - ref)))
        r1, r2 = errs[0] / errs[1], errs[1] / errs[2]
        return errs[2] < errs[1] < errs[0] and r2 > 6, (f"Earth position error after 10 y at dt 12/6/3 h: "
                                                          f"{errs[0]:.0f}/{errs[1]:.0f}/{errs[2]:.0f} m (ratios {r1:.1f}, {r2:.1f}; 4th order -> 16)")

    def energy_drift():
        h = HifiEngine(cfg, 3)
        h.run_interval(100.0, 0.0, 3e6, [0, 1, 0])
        c = h.conservation()
        return abs(c["energy_residual_rel"]) < 1e-10 and c["angmom_residual_rel"] < 1e-12, (
            f"100 y ballistic: |dE/E| = {abs(c['energy_residual_rel']):.1e}, |dL|/|L| = {c['angmom_residual_rel']:.1e}")

    def fast_vs_hifi():
        c2 = replace(cfg, mission_years=1500.0)
        a = run_episode(c2, policy_heuristic, 11, "fast")
        b = run_episode(c2, policy_heuristic, 11, "hifi")
        rel = abs(a["a_au"] - b["a_au"]) / b["a_au"]
        de = abs(a["e"] - b["e"])
        return rel < 1e-3 and de < 5e-3, f"heuristic, 1,500 y: a fast {a['a_au']:.5f} vs high-fidelity {b['a_au']:.5f} AU (rel {rel:.1e}), |de| {de:.1e}"

    def polar_hermite(r0, v0, r1, v1, h, s_):
        """Same interpolation as frontend/app.js: cubic Hermite in (radius, angle, z) using the
        recorded velocities as end-point derivatives."""
        def pol(r, v):
            rho2 = r[0] ** 2 + r[1] ** 2
            rho = math.sqrt(rho2)
            return rho, math.atan2(r[1], r[0]), r[2], (r[0] * v[0] + r[1] * v[1]) / rho, \
                (r[0] * v[1] - r[1] * v[0]) / rho2, v[2]
        a0, a1 = pol(r0, v0), pol(r1, v1)
        th1 = a1[1]
        while th1 - a0[1] < -math.pi:
            th1 += 2 * math.pi
        while th1 - a0[1] > math.pi:
            th1 -= 2 * math.pi
        h00, h10, h01, h11 = 2 * s_ ** 3 - 3 * s_ ** 2 + 1, s_ ** 3 - 2 * s_ ** 2 + s_, -2 * s_ ** 3 + 3 * s_ ** 2, s_ ** 3 - s_ ** 2
        rho = h00 * a0[0] + h10 * h * a0[3] + h01 * a1[0] + h11 * h * a1[3]
        th = h00 * a0[1] + h10 * h * a0[4] + h01 * th1 + h11 * h * a1[4]
        z = h00 * a0[2] + h10 * h * a0[5] + h01 * a1[2] + h11 * h * a1[5]
        return np.array([rho * math.cos(th), rho * math.sin(th), z])

    def visual_continuity():
        """The viewer interpolates between recorded states.  Verify that (a) consecutive fast-model
        rows are connected by a pure ballistic Kepler arc (no hidden jump), (b) the cubic Hermite
        curve the viewer draws stays within 0.001 AU of the exact arc, (c) the recoil recorded in
        each row equals |v_post - v_pre|, and (d) packet paths follow two-body solar gravity."""
        f = FastEngine(cfg, 3)
        smp, _ = f.run_interval(5.0, 1e18, 3e6, [0.0, 1.0, 0.0])
        mu = f.mu()
        worst_join, worst_herm, worst_dv = 0.0, 0.0, 0.0
        for i in range(len(smp) - 1):
            a, b = smp[i], smp[i + 1]
            h = (b[S_T] - a[S_T]) * YEAR
            r = a[S_R:S_R + 3] * AU
            v = a[S_VPOST:S_VPOST + 3] * 1000.0
            r_end, v_end = r.copy(), v.copy()
            kepler_step(r_end, v_end, mu, h)
            worst_join = max(worst_join, float(np.linalg.norm(r_end - b[S_R:S_R + 3] * AU)) / AU)
            r_mid, v_mid = r.copy(), v.copy()
            kepler_step(r_mid, v_mid, mu, 0.5 * h)
            herm = polar_hermite(r, v, b[S_R:S_R + 3] * AU, b[S_VPRE:S_VPRE + 3] * 1000.0, h, 0.5)
            worst_herm = max(worst_herm, float(np.linalg.norm(herm - r_mid)) / AU)
            dv = np.linalg.norm(a[S_VPOST:S_VPOST + 3] - a[S_VPRE:S_VPRE + 3]) * 1000.0
            worst_dv = max(worst_dv, abs(dv - a[S_DV]) / a[S_DV])
        # packet arc: energy of the two-body path must be constant; 1-day position ~ straight line
        r0 = smp[0, S_R:S_R + 3][None, :] * AU
        v0 = smp[0, S_VP:S_VP + 3][None, :] * 1000.0
        offs = PACKET_OFFSETS_DAYS * 86400.0
        out = np.zeros((1, len(offs), 3))
        nv = np.zeros(1, dtype=np.int64)
        packet_arcs(r0, v0, offs, cfg.fragment_escape_au * AU, out, nv)
        j = int(np.searchsorted(PACKET_OFFSETS_DAYS, 1.0))
        straight = r0[0] + v0[0] * 86400.0
        dev = float(np.linalg.norm(out[0, j] - straight)) / float(np.linalg.norm(out[0, j] - r0[0]))
        ok = worst_join < 1e-9 and worst_herm < 1e-4 and worst_dv < 1e-6 and 0 < dev < 1e-3 and nv[0] > 3
        return ok, (f"ballistic join between rows {worst_join:.1e} AU (no hidden jumps); viewer Hermite curve vs exact "
                    f"arc max {worst_herm:.1e} AU; recoil |v_post - v_pre| matches recorded value to {worst_dv:.1e}; "
                    f"packet path bends {dev:.1e} (relative) from a straight line after 1 day under solar gravity")

    def fast_moon():
        """The training model simulates the Moon as a gravitating body (Earth-Moon two-body orbit
        about their barycentre).  Compare with the N-body Moon for 1 year without thrust, check the
        Earth's wobble produced by the Moon's pull, and check that a packet launch changes the
        Earth-Moon relative velocity by exactly Earth's recoil (the Moon is not pushed directly)."""
        f, h = FastEngine(cfg, 3), _fresh_hifi(cfg)
        f.run_interval(1.0, 0.0, 3e6, [0, 1, 0])
        h.run_interval(1.0, 0.0, 3e6, [0, 1, 0])
        df = np.linalg.norm(f.last_moon[:, 1:4], axis=1)
        dh = np.linalg.norm(h.last_moon[:, 1:4], axis=1)
        dmax = float(np.abs(df - dh).max())
        a1, a2 = f.last_moon[-1, 1:4], h.last_moon[-1, 1:4]
        phase = math.degrees(math.acos(max(-1.0, min(1.0, float(a1 @ a2) / (np.linalg.norm(a1) * np.linalg.norm(a2))))))
        emb = float(np.linalg.norm(f.emb()[0] - h.emb()[0]))
        rE, _ = f.earth_moon()
        wobble = float(np.linalg.norm(rE - f.st[0:3]))
        g, g0 = FastEngine(cfg, 3), FastEngine(cfg, 3)           # same interval with and without a packet
        g.run_interval(1e-9, 1e18, 3e6, [0, 1, 0])
        g0.run_interval(1e-9, 0.0, 3e6, [0, 1, 0])
        dvr = float(np.linalg.norm(g.st[14:17] - g0.st[14:17]))
        m_l = M_EARTH + cfg.asteroid_mass_kg
        vinf = math.sqrt(3e6 ** 2 - 2 * G * m_l / R_EARTH - 2 * G * M_MOON / np.linalg.norm(g0.st[11:14]))
        expect = 1e18 / m_l * vinf
        rel_dv = abs(dvr - expect) / expect
        ok = (f.moon_info()["bound"] and h.moon_info()["bound"] and dmax < 0.05 * MOON_A and emb < 1e6
              and 4.0e6 < wobble < 5.2e6 and rel_dv < 1e-6)
        return ok, (f"1 y without thrust: Earth-Moon distance fast {df.min() / 1e3:,.0f}-{df.max() / 1e3:,.0f} km vs N-body "
                    f"{dh.min() / 1e3:,.0f}-{dh.max() / 1e3:,.0f} km (max diff {dmax / 1e3:,.0f} km); Moon's orbital phase "
                    f"drifts {phase:.0f} deg/yr (Sun's tide on the Moon not in the fast model); barycentre diff "
                    f"{emb / 1e3:.0f} km; Earth wobble from the Moon's pull {wobble / 1e3:,.0f} km; recoil changes the "
                    f"Earth-Moon relative velocity by Earth's recoil only (rel. error {rel_dv:.1e}); Moon bound: yes")

    log("[CHECK] physics checks" + (" (full)" if full else " (quick)"))
    check("1 two-body motion vs Kepler", kepler)
    check("2 gravity of Sun / Earth / Moon / asteroid / fragments", gravity)
    check("3 equal and opposite forces", newton3)
    check("4+6 momentum & angular momentum in ejection", eject_momentum)
    check("5 energy accounting of propulsion", energy_accounting)
    check("7 finite asteroid mass & fragment mass accounting", mass_accounting)
    check("8 reversed ejection direction", reverse)
    check("9 fragment fall-back / escape / Earth impact", fragments)
    check("11 reproducibility (fixed seed)", reproducible)
    check("13 viewer continuity (interpolation of recorded states)", visual_continuity)
    check("14 Moon in the fast (training) model", fast_moon)
    if full:
        check("10 integrator convergence", convergence)
        check("10b energy / angular-momentum drift", energy_drift)
        check("12 fast model vs high-fidelity N-body", fast_vs_hifi)
    return out


# =============================================================================================
# 10. APPLICATION STATE, TRAINING, COMPARISON, EXPLANATION
# =============================================================================================
# Row layout sent to the browser (see frontend/app.js, constant R_*):
# t[yr], x,y,z[AU], v_pre(3)[km/s], v_post(3)[km/s], dm[kg], u[m/s], v_rel[m/s], fate, m_ast[kg],
# E_rem[J], n_frag, recoil dv[m/s], E_draw[J], ehat(3)
def _g(x, n=5):
    return float(f"{float(x):.{n}g}")


def _viewer_rows(samples):
    out = []
    for q in samples:
        out.append([round(float(q[S_T]), 6)] + [round(float(x), 7) for x in q[S_R:S_R + 3]] +
                   [round(float(x), 5) for x in q[S_VPRE:S_VPRE + 3]] + [round(float(x), 5) for x in q[S_VPOST:S_VPOST + 3]] +
                   [_g(q[S_DM], 4), int(q[S_U]), int(q[S_VREL]), int(q[S_FATE]), _g(q[S_MAST], 6), _g(q[S_EREM], 6),
                    int(q[S_NFRAG]), _g(q[S_DV], 4), _g(q[S_EDRAW], 4)] + [round(float(x), 4) for x in q[S_EHAT:S_EHAT + 3]])
    return out


def _initial_row(eng):
    r, v, _ = eng.earth_state()
    return [0.0] + [round(float(x) / AU, 7) for x in r] + [round(float(x) / 1000, 5) for x in v] * 2 + \
        [0.0, 0, 0, 0, _g(eng.m_ast, 6), _g(eng.E_rem, 6), 0, 0.0, 0.0, 0.0, 0.0, 0.0]


def _moon_rows(moon):
    return [[round(float(m[0]), 6)] + [int(round(float(x) / 1000.0)) for x in m[1:4]] + [round(float(x), 1) for x in m[4:7]]
            for m in moon]


def _packet_records(samples, cfg, fidelity):
    """One record per ejected packet: [t, fate, dm, u, n_points, x0, y0, z0, x1, ...] with the
    packet's positions [AU] at PACKET_OFFSETS_DAYS after release, computed here with exact
    two-body solar gravity from the packet's release state (patched conic)."""
    idx = [i for i in range(len(samples)) if samples[i, S_DM] > 0]
    if not idx:
        return []
    sm = samples[idx]
    flying = (sm[:, S_FATE] == 1) | (sm[:, S_FATE] == 2)
    r0 = sm[:, S_R:S_R + 3] * AU
    if fidelity == "hifi":
        r0 = r0 + cfg.release_radius_m * sm[:, S_EHAT:S_EHAT + 3]
    v0 = sm[:, S_VP:S_VP + 3] * 1000.0
    offs = PACKET_OFFSETS_DAYS * 86400.0
    out = np.zeros((len(idx), len(offs), 3))
    nv = np.zeros(len(idx), dtype=np.int64)
    fl = np.where(flying)[0]
    if len(fl):
        o2 = np.zeros((len(fl), len(offs), 3))
        n2 = np.zeros(len(fl), dtype=np.int64)
        packet_arcs(np.ascontiguousarray(r0[fl]), np.ascontiguousarray(v0[fl]), offs, cfg.fragment_escape_au * AU, o2, n2)
        out[fl], nv[fl] = o2, n2
    recs = []
    for j, q in enumerate(sm):
        if flying[j]:
            pts = [round(float(x) / AU, 5) for x in out[j, :nv[j]].ravel()]
            n = int(nv[j])
        else:
            pts = [round(float(x), 5) for x in q[S_R:S_R + 3]]
            n = 1
        recs.append([round(float(q[S_T]), 6), int(q[S_FATE]), _g(q[S_DM], 4), int(q[S_U]), n] + pts)
    return recs


def _policy_label(name):
    return {"none": "No ejection", "heuristic": "Hand-designed heuristic", "ppo": "Best PPO model"}.get(name, name)


class App:
    def __init__(self, cfg: Config, out_dir: Path):
        from collections import deque
        self.lock = threading.RLock()
        self.cfg = cfg
        self.out = Path(out_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        self.feas = feasibility(cfg)
        self.checks = []
        self.logbuf = deque(maxlen=400)
        self.train = dict(state="idle", episode=0, target=cfg.train_episodes, timesteps=0, ppo_iter=0,
                          losses={}, best=None, error=None, wall_s=0.0, eps_per_s=0.0)
        self.history = []
        self.evals = []
        self.comparison = None
        self.compare_state = "idle"
        self.viewer = dict(run_id=0, mode="idle", label="", fidelity="", rows=[], rbase=0, moon=[], mbase=0,
                           packets=[], pbase=0, trail=[], decisions=[], done=True, produced_years=0.0, final=None)
        self.disp_years = 0.0
        self.stop_train = threading.Event()
        self.stop_play = threading.Event()
        self.shutdown = threading.Event()
        self.train_thread = None
        self.play_thread = None
        self.compare_thread = None
        self.auto_compare = True
        self._csv_path = self.out / "training_log.csv"
        self._load_previous()

    def _load_previous(self):
        """Show results of an earlier run (from outputs/) until a new training run replaces them."""
        comp = _read_json(self.out / "comparison.json")
        if comp:
            self.comparison = comp
            self.compare_state = "done (previous run)"
        best = _read_json(self.out / "best_model_info.json")
        if best and (self.out / "best_model.zip").exists():
            self.train["best"] = best
        try:
            with open(self._csv_path) as f:
                for r in csv.DictReader(f):
                    fl = lambda k: float(r[k]) if r[k] not in ("", "None") else None
                    self.history.append(dict(ep=int(r["episode"]), reward=fl("reward"), success=r["success"] == "1",
                                             a=fl("a_au"), e=fl("e"), mass=fl("mass_used_frac"),
                                             energy=fl("energy_used_frac"), years=fl("years"), reason=r["reason"]))
        except (OSError, KeyError, ValueError):
            pass
        ev = _read_json(self.out / "evals.json")
        if isinstance(ev, list):
            self.evals = ev

    # ------------------------------------------------------------------------------------------
    def log(self, msg):
        line = time.strftime("%H:%M:%S ") + msg
        print(line, flush=True)
        with self.lock:
            self.logbuf.append(line)

    def set_config(self, updates: dict):
        with self.lock:
            if self.train["state"] == "running":
                raise RuntimeError("stop training before changing parameters")
            if updates.pop("reset_defaults", False):
                base = self.cfg.preset if self.cfg.preset in PRESETS else "hypothetical"
                cfg = make_config(base)
                self.cfg, self.feas = cfg, feasibility(cfg)
                self.train["target"] = cfg.train_episodes
                self.log(f"[CONFIG] all parameters reset to the '{base}' defaults")
                return cfg
            preset = updates.pop("preset", None)
            if preset in PRESETS:             # switching preset resets the physical parameters
                keep = {k: v for k, v in self.cfg.to_dict().items() if k not in PRESETS[preset]}
                cfg = checked_config(preset, **{**keep, **updates})
            else:
                base = self.cfg.preset if self.cfg.preset in PRESETS else "hypothetical"
                cur = {k: v for k, v in self.cfg.to_dict().items() if k not in PRESETS[base]}
                phys = {k: getattr(self.cfg, k) for k in PRESETS[base] if k != "preset"}
                cfg = checked_config(base, **{**cur, **phys, **updates})
            self.cfg = cfg
            self.feas = feasibility(cfg)
            self.train["target"] = cfg.train_episodes
        self.log(f"[CONFIG] preset={cfg.preset}  feasibility: {self.feas['verdict']}")
        return cfg

    # ---------------------------------------------------------------- training ----------------
    def start_training(self):
        with self.lock:
            if self.train_thread and self.train_thread.is_alive():
                return False
            self.stop_train.clear()
            self.history, self.evals = [], []
            self.train.update(state="running", episode=0, timesteps=0, ppo_iter=0, losses={}, best=None,
                              error=None, target=self.cfg.train_episodes)
        self.train_thread = threading.Thread(target=self._train_worker, daemon=True, name="ppo")
        self.train_thread.start()
        return True

    def _eval_policy_fast(self, model, cfg, seeds=(1001, 1002)):
        res = [run_episode(cfg, model_policy(model), s, "fast") for s in seeds]
        sr = float(np.mean([r["success"] for r in res]))
        err = float(np.mean([OctoEarthEnv.error_norm_static(cfg, r) for r in res]))
        ef = float(np.mean([r["energy_used_frac"] for r in res]))
        yrs = float(np.mean([r["t_years"] for r in res]))
        return dict(success_rate=sr, error=err, energy_frac=ef, years=yrs, reward=float(np.mean([r["total_reward"] for r in res])),
                    a_au=float(np.mean([r["a_au"] if math.isfinite(r["a_au"]) else 99 for r in res])),
                    e=float(np.mean([r["e"] for r in res])), score=(sr, -round(err, 4), -ef))

    def _train_worker(self):
        try:
            import torch
            from stable_baselines3 import PPO
            from stable_baselines3.common.callbacks import BaseCallback
            from stable_baselines3.common.vec_env import DummyVecEnv
            torch.set_num_threads(1)
            cfg = self.cfg
            app = self
            env = OctoEarthEnv(cfg, "fast", record=True, record_moon=False)   # same physics, no viewer Moon samples
            vec = DummyVecEnv([lambda: env])
            model = PPO("MlpPolicy", vec, n_steps=cfg.ppo_n_steps, batch_size=cfg.ppo_batch_size,
                        n_epochs=cfg.ppo_epochs, learning_rate=cfg.ppo_learning_rate, gamma=cfg.ppo_gamma,
                        gae_lambda=0.95, ent_coef=0.0, seed=cfg.seed, device="cpu", verbose=0,
                        policy_kwargs=dict(net_arch=[64, 64], log_std_init=-0.5))
            hp = dict(algorithm="PPO (Stable-Baselines3)", net_arch=[64, 64], n_steps=cfg.ppo_n_steps,
                      batch_size=cfg.ppo_batch_size, n_epochs=cfg.ppo_epochs, learning_rate=cfg.ppo_learning_rate,
                      gamma=cfg.ppo_gamma, gae_lambda=0.95, log_std_init=-0.5, seed=cfg.seed)
            t0 = time.time()
            best = dict(score=None)
            last_view = [0.0]
            with open(self._csv_path, "w", newline="") as fcsv:
                wr = csv.writer(fcsv)
                wr.writerow(["episode", "timesteps", "reward", "success", "reason", "years", "a_au", "e",
                             "mass_used_frac", "energy_used_frac"])

                class CB(BaseCallback):
                    def _on_rollout_start(self):
                        vals = dict(self.model.logger.name_to_value)
                        if "train/value_loss" in vals:
                            app.train["ppo_iter"] += 1
                            app.train["losses"] = {k.split("/")[1]: float(v) for k, v in vals.items() if k.startswith("train/")}
                            L = app.train["losses"]
                            app.log(f"[PPO update {app.train['ppo_iter']}] policy_loss={L.get('policy_gradient_loss', float('nan')):.4g} "
                                    f"value_loss={L.get('value_loss', float('nan')):.4g} entropy={L.get('entropy_loss', float('nan')):.4g} "
                                    f"approx_kl={L.get('approx_kl', float('nan')):.3g} explained_var={L.get('explained_variance', float('nan')):.3g}")

                    def _on_step(self):
                        for info, done in zip(self.locals["infos"], self.locals["dones"]):
                            if done and "episode_summary" in info:
                                app._on_episode(info["episode_summary"], self.num_timesteps, wr, t0, env, last_view)
                                ep = app.train["episode"]
                                if ep % cfg.eval_every_episodes == 0 or ep == cfg.train_episodes:
                                    ev = app._eval_policy_fast(self.model, cfg)
                                    ev.update(episode=ep, timesteps=self.num_timesteps)
                                    app.evals.append(ev)
                                    (app.out / "evals.json").write_text(json.dumps(_clean(app.evals)))
                                    tag = ""
                                    if best["score"] is None or ev["score"] > best["score"]:
                                        best["score"] = ev["score"]
                                        self.model.save(str(app.out / "best_model.zip"))
                                        info_b = dict(episode=ep, timesteps=self.num_timesteps, eval=ev, hyperparameters=hp,
                                                      selection="lexicographic: success rate, then lowest terminal orbit "
                                                                "error, then lowest energy use (deterministic, seeds 1001-1002)")
                                        app.train["best"] = info_b
                                        (app.out / "best_model_info.json").write_text(json.dumps(info_b, indent=2, default=str))
                                        tag = "  << NEW BEST -> best_model.zip"
                                    app.log(f"[EVAL ep {ep}] deterministic: success={ev['success_rate']:.0%} a={ev['a_au']:.4f} AU "
                                            f"e={ev['e']:.4f} energy={ev['energy_frac']:.1%} years={ev['years']:.0f} "
                                            f"reward={ev['reward']:.1f}{tag}")
                        return not (app.stop_train.is_set() or app.train["episode"] >= cfg.train_episodes)

                total = int(cfg.train_episodes * cfg.max_steps * 1.05) + cfg.ppo_n_steps
                app.log(f"[TRAIN] PPO started: target {cfg.train_episodes} episodes (<= {cfg.max_steps} decisions each, "
                        f"{cfg.decision_interval_years:g} y per decision) on the FAST model")
                model.learn(total_timesteps=total, callback=CB())
            model.save(str(self.out / "last_model.zip"))
            if not (self.out / "best_model.zip").exists():
                model.save(str(self.out / "best_model.zip"))
            self.train.update(state="stopped" if self.stop_train.is_set() else "finished", wall_s=time.time() - t0)
            self.log(f"[TRAIN] {self.train['state']} after {self.train['episode']} episodes, "
                     f"{self.train['timesteps']} timesteps, {time.time() - t0:.0f} s. best_model.zip saved.")
            if self.auto_compare and not self.shutdown.is_set():
                self.start_comparison()
        except Exception as exc:
            self.train.update(state="error", error=f"{type(exc).__name__}: {exc}")
            self.log(f"[ERROR] training failed: {exc}\n{traceback.format_exc()}")

    def _on_episode(self, s, timesteps, wr, t0, env, last_view):
        with self.lock:
            self.train["episode"] += 1
            self.train["timesteps"] = int(timesteps)
            ep = self.train["episode"]
            el = time.time() - t0
            self.train["wall_s"] = el
            self.train["eps_per_s"] = ep / max(el, 1e-9)
            rec = dict(ep=ep, reward=round(s["total_reward"], 3), success=bool(s["success"]),
                       a=round(s["a_au"], 5) if math.isfinite(s["a_au"]) else None, e=round(s["e"], 5),
                       mass=round(s["mass_used_frac"], 4), energy=round(s["energy_used_frac"], 4),
                       years=round(s["t_years"], 1), reason=s["reason"])
            self.history.append(rec)
        wr.writerow([ep, timesteps, rec["reward"], int(rec["success"]), rec["reason"], rec["years"], rec["a"], rec["e"],
                     rec["mass"], rec["energy"]])
        if ep % 10 == 0 or ep == 1:
            rw = [h["reward"] for h in self.history[-50:]]
            sr = np.mean([h["success"] for h in self.history[-50:]])
            self.log(f"[EPISODE {ep}/{self.cfg.train_episodes}] reward={rec['reward']:.2f} avg50={np.mean(rw):.2f} "
                     f"success50={sr:.0%} final a={rec['a']} AU e={rec['e']:.4f} asteroid used={rec['mass']:.1%} "
                     f"energy used={rec['energy']:.1%} sim={rec['years']:.0f} y ({rec['reason']})")
        now = time.time()
        if now - last_view[0] > 1.5 and env.last_trail:
            last_view[0] = now
            with self.lock:
                if self.viewer["mode"] in ("idle", "training"):
                    nv = self._new_viewer("training", f"training episode {ep} (fast model, exploring)", "fast")
                    nv.update(rows=[[0.0, *p] for p in env.last_trail],
                              decisions=[{k: v for k, v in d.items() if k != "conservation"} for d in env.last_decisions],
                              done=True, produced_years=env.engine.t_years,
                              final=dict(s, moon_rel_km=[round(float(x) / 1000.0) for x in env.engine.st[11:14]],
                                         m_ast_left=float(env.engine.m_ast)))
                    self.viewer = nv

    # ---------------------------------------------------------------- comparison --------------
    def start_comparison(self):
        if self.compare_thread and self.compare_thread.is_alive():
            return False
        self.compare_state = "running"
        self.compare_thread = threading.Thread(target=self._compare_worker, daemon=True, name="compare")
        self.compare_thread.start()
        return True

    def load_ppo(self):
        p = self.out / "best_model.zip"
        if not p.exists():
            return None
        from stable_baselines3 import PPO
        return PPO.load(str(p), device="cpu")

    def _compare_worker(self):
        try:
            cfg = self.cfg
            self.log("[COMPARE] evaluating policies on the FAST model (3 seeds) and the HIGH-FIDELITY N-body model "
                     f"(1 seed + {cfg.validate_years:g} y ballistic validation) ...")
            pols = [("none", policy_none), ("heuristic", policy_heuristic)]
            model = self.load_ppo()
            if model is not None:
                pols.append(("ppo", model_policy(model)))
            rows, traces = [], {}
            for name, pol in pols:
                fast = [run_episode(cfg, pol, s, "fast") for s in (2001, 2002, 2003)]
                hifi = run_episode(cfg, pol, 2001, "hifi", validate=True)
                row = dict(policy=name, label=_policy_label(name),
                           fast=dict(success_rate=float(np.mean([f["success"] for f in fast])),
                                     reward=float(np.mean([f["total_reward"] for f in fast])),
                                     a_au=[f["a_au"] for f in fast], e=[f["e"] for f in fast],
                                     years=[f["t_years"] for f in fast], reasons=[f["reason"] for f in fast],
                                     mass_used_frac=float(np.mean([f["mass_used_frac"] for f in fast])),
                                     energy_used_frac=float(np.mean([f["energy_used_frac"] for f in fast]))),
                           hifi={k: hifi[k] for k in ("success", "reason", "total_reward", "a_au", "e", "inc_deg", "t_years",
                                                      "mass_used_kg", "mass_used_frac", "energy_used_J", "energy_used_frac",
                                                      "impulse_Ns", "n_packets", "insolation_rel", "t_eq_K", "shortfall_au",
                                                      "max_packet_impulse_Ns", "ledger", "conservation", "moon")},
                           validation=hifi.get("validation"),
                           model_agreement_da_au=abs(fast[0]["a_au"] - hifi["a_au"]) if math.isfinite(hifi["a_au"]) else None)
                row["mission_success"] = bool(hifi["success"] and (hifi.get("validation") or {}).get("passed", False))
                rows.append(row)
                traces[name] = dict(fast=[dict(t=d["t_years"], a=d["a_au"], e=d["e"], m=d["m_ast_frac"], E=d["energy_frac"],
                                               u=d["u_ms"], th=d["theta_deg"], ph=d["phi_deg"], thr=d["throttle"],
                                               dm=d["dm_kg"], imp=d["impulse_Ns"], d_rtn=d["d_rtn"])
                                          for d in fast[0]["decisions"]],
                                    hifi=[_trace_rec(d) for d in hifi["decisions"]],
                                    hifi_a0=hifi.get("a0_au", 1.0), hifi_e0=hifi.get("e0", 0.0167))
                self.log(f"[COMPARE] {row['label']:24s} fast: success {row['fast']['success_rate']:.0%}, a={np.mean(row['fast']['a_au']):.4f} AU | "
                         f"high-fidelity: {hifi['reason']}, a={hifi['a_au']:.4f} AU e={hifi['e']:.4f}, "
                         f"validated={row['mission_success']}, asteroid used {hifi['mass_used_frac']:.1%}, "
                         f"energy {hifi['energy_used_J']:.3e} J")
            expl = explain(rows, traces, cfg, self.feas, self.train.get("best"))
            best_info = self.train.get("best") or _read_json(self.out / "best_model_info.json")
            panel = best_panel(rows, traces, cfg, best_info, self.out / "best_model.zip", self.history, self.evals)
            comp = dict(rows=rows, traces=traces, explanation=expl, config=cfg.to_dict(), feasibility=self.feas,
                        best_model=best_info, best_panel=panel,
                        created=time.strftime("%Y-%m-%d %H:%M:%S"))
            with self.lock:
                self.comparison = comp
                self.compare_state = "done"
            (self.out / "comparison.json").write_text(json.dumps(_clean(comp), indent=1))
            (self.out / "report.md").write_text(make_report(self, comp))
            self.log(f"[COMPARE] done -> {self.out / 'comparison.json'} and report.md")
            for line in expl["summary"]:
                self.log("  " + line)
        except Exception as exc:
            self.compare_state = "error"
            self.log(f"[ERROR] comparison failed: {exc}\n{traceback.format_exc()}")

    # ---------------------------------------------------------------- playback ----------------
    def start_playback(self, policy: str, fidelity: str):
        self.stop_play.set()
        if self.play_thread and self.play_thread.is_alive():
            self.play_thread.join(5)
        self.stop_play.clear()
        if policy == "ppo":
            model = self.load_ppo()
            if model is None:
                raise RuntimeError("no best_model.zip yet - train first")
            pol = model_policy(model)
        else:
            pol = policy_none if policy == "none" else policy_heuristic
        with self.lock:
            self.disp_years = 0.0
            self.viewer = self._new_viewer("playback", f"{_policy_label(policy)} - "
                                           f"{'high-fidelity N-body' if fidelity == 'hifi' else 'fast model'}", fidelity)
            self.viewer["policy"] = policy
            run_id = self.viewer["run_id"]
        self.play_thread = threading.Thread(target=self._play_worker, args=(pol, fidelity, run_id), daemon=True)
        self.play_thread.start()

    def _new_viewer(self, mode, label, fidelity):
        return dict(run_id=self.viewer["run_id"] + 1, mode=mode, label=label, fidelity=fidelity, rows=[], rbase=0,
                    moon=[], mbase=0, packets=[], pbase=0, trail=[], decisions=[], done=False, produced_years=0.0,
                    final=None)

    @staticmethod
    def _append_window(v, key, base_key, items, limit):
        v[key].extend(items)
        if len(v[key]) > limit:            # keep memory bounded; the browser already has the old part
            drop = len(v[key]) - limit // 2
            del v[key][:drop]
            v[base_key] += drop

    def _play_worker(self, pol, fidelity, run_id):
        try:
            cfg = self.cfg
            first = [True]

            def on_step(env, info):
                rec = {k: v for k, v in info["decision"].items() if k != "conservation"}
                if "conservation" in info["decision"]:
                    rec["conservation"] = {k: v for k, v in info["decision"]["conservation"].items()
                                           if k.endswith("_rel") or k in ("active_fragments", "merges")}
                rows = _viewer_rows(env.last_samples)
                moon = _moon_rows(env.engine.last_moon)
                pk = _packet_records(env.last_samples, cfg, fidelity)
                with self.lock:
                    v = self.viewer
                    if v["run_id"] != run_id:
                        return
                    if first[0]:
                        rows.insert(0, env._first_row)
                        if env._first_moon is not None:
                            moon.insert(0, env._first_moon)
                        first[0] = False
                    self._append_window(v, "rows", "rbase", rows, 24000)
                    self._append_window(v, "moon", "mbase", moon, 64000)
                    self._append_window(v, "packets", "pbase", pk, 24000)
                    v["trail"].extend([[r[0]] + r[1:4] for r in rows[::8]])
                    v["decisions"].append(rec)
                    v["produced_years"] = env.engine.t_years
                while (not self.stop_play.is_set() and self.viewer["run_id"] == run_id and
                       env.engine.t_years - self.disp_years > max(3 * cfg.decision_interval_years,
                                                                  getattr(self, "ahead_years", 0.0))):
                    time.sleep(0.05)

            def first_state(env):
                with self.lock:
                    if self.viewer["run_id"] == run_id:
                        self.viewer["meta"] = dict(mu=float(env.engine.mu()), m_ast0=float(env.m_ast0), E0=float(env.E0),
                                                   ejections_per_year=cfg.ejections_per_year,
                                                   decision_interval_years=cfg.decision_interval_years,
                                                   target_au=cfg.target_au, tol_a_au=cfg.tol_a_au, tol_e=cfg.tol_e,
                                                   mission_years=cfg.mission_years, release_radius_m=cfg.release_radius_m,
                                                   offsets_days=PACKET_OFFSETS_DAYS.tolist(), seed=2001)
                env._first_row = _initial_row(env.engine)
                _, _, mo = env.engine.earth_state()
                env._first_moon = None if mo is None else \
                    [0.0] + [int(round(float(x) / 1000.0)) for x in mo[0]] + [round(float(x), 1) for x in mo[1]]

            s = run_episode(cfg, pol, 2001, fidelity, validate=(fidelity == "hifi"), on_step=on_step,
                            stop_event=self.stop_play, on_reset=first_state)
            s.pop("decisions", None)
            with self.lock:
                if self.viewer["run_id"] == run_id:
                    self.viewer["done"] = True
                    self.viewer["final"] = s
        except Exception as exc:
            self.log(f"[ERROR] playback failed: {exc}\n{traceback.format_exc()}")

    # ---------------------------------------------------------------- API views ---------------
    def state_view(self, run, since, msince, psince, dsince, disp, pstride=1, ahead=0.0):
        with self.lock:
            if disp is not None:
                self.disp_years = disp
                self.ahead_years = min(1500.0, max(0.0, float(ahead)))
            v = self.viewer
            reset = run != v["run_id"]
            if reset:
                since = msince = psince = dsince = 0

            def chunk(key, base_key, start, n):
                st = max(start, v[base_key])
                return st, v[key][st - v[base_key]: st - v[base_key] + n]
            rs, rows = chunk("rows", "rbase", since, 3000)
            ms, moon = chunk("moon", "mbase", msince, 16000)
            pstride = max(1, int(pstride))
            ps, pk = chunk("packets", "pbase", psince, 3000 * pstride)
            pnext = ps + len(pk)
            if pstride > 1:                   # high display speed: send every k-th packet only
                pk = pk[::pstride]
            return dict(train=dict(self.train, best=None), best=self.train.get("best"), compare_state=self.compare_state,
                        init=self._init_view(),
                        viewer=dict(run_id=v["run_id"], mode=v["mode"], label=v["label"], fidelity=v["fidelity"],
                                    policy=v.get("policy"), done=v["done"], final=v["final"],
                                    produced_years=v["produced_years"], reset=reset,
                                    rstart=rs, rows=rows, mstart=ms, moon=moon, pstart=ps, pnext=pnext, pstride=pstride,
                                    packets=pk, meta=v.get("meta"),
                                    trail=v["trail"] if (reset or since < v["rbase"]) else None,
                                    dstart=dsince, decisions=v["decisions"][dsince:]),
                        log=list(self.logbuf)[-40:], feasible=self.feas["feasible"], verdict=self.feas["verdict"],
                        preset=self.cfg.preset, checks_ok=all(c["passed"] for c in self.checks) if self.checks else None)

    def _init_view(self):
        """Simulated start state (year 0, playback seed 2001) shown while nothing is running."""
        key = json.dumps(self.cfg.to_dict(), sort_keys=True, default=str)
        if getattr(self, "_init_key", None) != key:
            ic = initial_conditions(self.cfg, 2001)
            me = ic["m_earth"]
            r_e = ic["r_emb"] - M_MOON / (me + M_MOON) * ic["r_rel"]
            vh = ic["v_emb"] / np.linalg.norm(ic["v_emb"])
            self._init_cache = dict(earth=[round(float(x) / AU, 7) for x in r_e],
                                    moon_rel_km=[round(float(x) / 1000.0, 1) for x in ic["r_rel"]],
                                    moon_v_kms=[round(float(x) / 1000.0, 6) for x in ic["v_rel"]],
                                    ast_dir=[round(float(-x), 5) for x in vh], m_ast0=float(self.cfg.asteroid_mass_kg))
            self._init_key = key
        return self._init_cache

    def results_view(self):
        with self.lock:
            base = self.cfg.preset if self.cfg.preset in PRESETS else "hypothetical"
            return dict(config=self.cfg.to_dict(), defaults=make_config(base).to_dict(), param_spec=PARAM_SPEC,
                        param_docs=PARAM_DOCS, output_docs=OUTPUT_DOCS, glossary=GLOSSARY, reason_docs=REASON_DOCS,
                        check_docs=CHECK_DOCS, constants=CONSTANT_DOCS, speed_doc=SPEED_DOC, assumptions=ASSUMPTIONS,
                        presets=list(PRESETS), feasibility=self.feas, checks=self.checks, history=self.history[-3000:],
                        evals=self.evals, comparison=self.comparison, train=dict(self.train),
                        training_outcomes=failure_counts(self.history), out_dir=str(self.out.resolve()),
                        packet_offsets_days=PACKET_OFFSETS_DAYS.tolist())


def _read_json(p):
    try:
        return json.loads(Path(p).read_text())
    except Exception:
        return None


def _error_norm_static(cfg, s):
    a = s["a_au"] if math.isfinite(s["a_au"]) else 10.0
    return min(3.0, abs(a - cfg.target_au) / (cfg.target_au - 1.0)) + 0.1 * min(10.0, s["e"] / cfg.tol_e)


OctoEarthEnv.error_norm_static = staticmethod(_error_norm_static)


def _clean(o):
    if isinstance(o, float):
        return o if math.isfinite(o) else None
    if isinstance(o, (np.floating,)):
        f = float(o)
        return f if math.isfinite(f) else None
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return _clean(o.tolist())
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    return o


def _trace_rec(d):
    return dict(t=d["t_years"], a=d["a_au"], e=d["e"], inc=d["inc_deg"], thr=d["throttle"], dm=d["dm_kg"], u=d["u_ms"],
                th=d["theta_deg"], ph=d["phi_deg"], d_rtn=d["d_rtn"], imp=d["impulse_Ns"], E=d["energy_J"],
                dv=d.get("dv_ms"), npk=d.get("n_packets"), fb=d.get("fallback"), mh=d.get("moon_hits"),
                m=d["m_ast_frac"], Ef=d["energy_frac"], rw=d["reward"], moon=d["moon_bound"])


def failure_counts(history):
    """Outcome counts over all training episodes (stochastic exploration, fast model)."""
    c = {}
    for h in history:
        c[h["reason"]] = c.get(h["reason"], 0) + 1
    return dict(sorted(c.items(), key=lambda kv: -kv[1]))


def best_panel(rows, traces, cfg, best, path, history, evals):
    """Everything the 'Best model' panel shows, computed only from recorded simulation data."""
    by = {r["policy"]: r for r in rows}
    out = dict(available="ppo" in by)
    p = Path(path)
    out["file"] = dict(name=p.name, path=str(p.resolve()), exists=p.exists(),
                       size_kb=round(p.stat().st_size / 1024, 1) if p.exists() else None,
                       modified=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(p.stat().st_mtime)) if p.exists() else None)
    out["selection"] = best
    out["training_outcomes"] = failure_counts(history)
    out["training_episodes"] = len(history)
    out["failed_evals"] = [dict(episode=e.get("episode"), success_rate=e.get("success_rate"), a_au=e.get("a_au"),
                                reason=",".join(e.get("reasons", [])) if isinstance(e.get("reasons"), list) else e.get("reason"))
                           for e in evals if e.get("success_rate", 0) < 1]
    out["disclaimer"] = [
        "This is the best policy found in this training run, judged on a few deterministic test episodes. It is not "
        "proven to be globally optimal; other strategies (e.g. the heuristic) can be better on some measures.",
        "Reward is only the scoring rule used to train the agent. Mission success is judged separately from physics: "
        "final a, e and inclination, a ballistic stability check, and the Moon staying bound."]
    if "ppo" not in by:
        return out
    r = by["ppo"]
    h = r["hifi"]
    tr = traces["ppo"]["hifi"]
    # ----- criteria
    v = r.get("validation") or {}
    out["criteria"] = [
        dict(name="Semi-major axis a", value=h["a_au"], unit="AU", required=f"{cfg.target_au} +/- {cfg.tol_a_au}",
             passed=bool(abs(h["a_au"] - cfg.target_au) <= cfg.tol_a_au)),
        dict(name="Eccentricity e", value=h["e"], unit="", required=f"<= {cfg.tol_e}", passed=bool(h["e"] <= cfg.tol_e)),
        dict(name="Inclination", value=h["inc_deg"], unit="deg", required=f"<= {cfg.tol_inc_deg}",
             passed=bool(h["inc_deg"] <= cfg.tol_inc_deg)),
        dict(name="Arrival time", value=h["t_years"], unit="yr", required=f"<= {cfg.mission_years:g}",
             passed=bool(h["success"])),
        dict(name="Moon still bound", value=bool(h["moon"]["bound"]), unit="", required="yes", passed=bool(h["moon"]["bound"])),
        dict(name=f"Stays in the target window for {cfg.validate_years:g} yr without thrust (largest deviation of a from target)",
             value=v.get("max_abs_da_au"), unit="AU", required=f"<= {cfg.tol_a_au} AU, e <= {cfg.tol_e}, Moon bound",
             passed=bool(v.get("passed", False))),
    ]
    # ----- per-decision analysis (high-fidelity run)
    a_prev, e_prev = traces["ppo"].get("hifi_a0", 1.0), traces["ppo"].get("hifi_e0", 0.0167)
    acts, total_da = [], 0.0
    for d in tr:
        da, de = d["a"] - a_prev, d["e"] - e_prev
        if d["dm"] <= 0:
            cls, why = "coast", "no packets launched in this interval"
        elif da > 0 and (d["e"] <= cfg.tol_e or de <= 0):
            cls = "helped"
            why = (f"recoil {d['d_rtn'][1]:+.2f} along the motion (prograde cosine) raised the orbit by {da:+.4f} AU"
                   + (f" and kept e at {d['e']:.4f}" if d["e"] <= cfg.tol_e else f" while lowering e by {-de:.4f}"))
        elif da > 0:
            cls = "mixed"
            why = f"raised the orbit by {da:+.4f} AU but e grew to {d['e']:.4f} (above tolerance {cfg.tol_e})"
        else:
            cls = "hurt"
            why = f"orbit shrank by {da:+.4f} AU (prograde cosine {d['d_rtn'][1]:+.2f})"
        if d["a"] > cfg.target_au + cfg.tol_a_au and da > 0:
            cls, why = "hurt", f"overshot the target: a = {d['a']:.4f} AU"
        total_da += max(da, 0.0)
        acts.append(dict(t=d["t"], dt=cfg.decision_interval_years, thr=d["thr"], dm=d["dm"], u=d["u"], th=d["th"], ph=d["ph"],
                         d_rtn=d["d_rtn"], npk=d["npk"], dv=d["dv"], E=d["E"], da=da, de=de, a=d["a"], e=d["e"],
                         fb=d["fb"], cls=cls, why=why, rw=d["rw"]))
        a_prev, e_prev = d["a"], d["e"]
    for x in acts:
        x["share"] = x["da"] / total_da if total_da > 0 and x["da"] > 0 else 0.0
        x["da_per_1e19kg"] = x["da"] / (x["dm"] / 1e19) if x["dm"] > 0 else None
        x["da_per_1e33J"] = x["da"] / (x["E"] / 1e33) if x["E"] > 0 else None
    out["actions"] = acts
    cnt = {}
    for x in acts:
        cnt[x["cls"]] = cnt.get(x["cls"], 0) + 1
    out["action_classes"] = cnt
    act = [x for x in acts if x["dm"] > 0]
    out["top_helpful"] = sorted(act, key=lambda x: -x["da"])[:5]
    out["least_helpful"] = sorted(act, key=lambda x: x["da"])[:3]
    # ----- resources / constraints
    L = h["ledger"]
    out["constraints"] = [
        dict(name="Asteroid mass used", value=h["mass_used_frac"], unit="fraction",
             note="ran out" if h["reason"] == "resources_exhausted" and h["mass_used_frac"] > 0.999 else "within limit"),
        dict(name="Energy budget used", value=h["energy_used_frac"], unit="fraction",
             note="ran out" if h["reason"] == "resources_exhausted" and h["energy_used_frac"] > 0.999 else "within limit"),
        dict(name="Packets that fell back to Earth (wasted)", value=int(L.get("n_fallback", 0)), unit="packets",
             note="launch slower than escape speed"),
        dict(name="Packets that hit the Moon", value=int(L.get("n_launch_moon_hits", 0)), unit="packets", note=""),
        dict(name="Outcome of high-fidelity run", value=h["reason"], unit="", note=REASON_DOCS.get(h["reason"], "")),
    ]
    out["comparison"] = [dict(policy=x["label"], success=x["mission_success"], a=x["hifi"]["a_au"], e=x["hifi"]["e"],
                              years=x["hifi"]["t_years"], mass=x["hifi"]["mass_used_frac"], energy_J=x["hifi"]["energy_used_J"],
                              reward=x["hifi"]["total_reward"]) for x in rows]
    return out


def explain(rows, traces, cfg, feas, best):
    """Plain-language explanation of what the best PPO model learned, from its own decisions."""
    by = {r["policy"]: r for r in rows}
    out = dict(summary=[], ppo=None)
    if "ppo" not in by:
        out["summary"].append("No trained PPO model available yet - only baselines were compared.")
        return out

    def stats(tr):
        imp = np.array([d["imp"] for d in tr]); D = np.array([d["d_rtn"] for d in tr]); dm = np.array([d["dm"] for d in tr])
        u = np.array([d["u"] for d in tr]); t = np.array([d["t"] for d in tr])
        tot = imp.sum()
        if tot <= 0:
            return dict(active=False)
        act = dm > 0
        return dict(active=True, prograde=float((imp * D[:, 1]).sum() / tot), radial=float((imp * np.abs(D[:, 0])).sum() / tot),
                    normal=float((imp * np.abs(D[:, 2])).sum() / tot), retro=float(imp[D[:, 1] < 0].sum() / tot),
                    u_mean=float((dm * u).sum() / dm.sum()), first=float(t[act][0]) if act.any() else None,
                    last=float(t[act][-1]) if act.any() else None, steps=len(tr), active_steps=int(act.sum()))

    sp, sh = stats(traces["ppo"]["fast"]), stats(traces["heuristic"]["fast"])
    p, h, n = by["ppo"], by["heuristic"], by["none"]
    hp = p["hifi"]
    lines = []
    if p["mission_success"]:
        lines.append(f"RESULT: the best PPO model reached the target in the high-fidelity N-body check: a = {hp['a_au']:.4f} AU, "
                     f"e = {hp['e']:.4f} after {hp['t_years']:,.0f} simulated years, and the orbit stayed within tolerance "
                     f"for {cfg.validate_years:g} ballistic years with the Moon bound.")
    else:
        sf = hp["shortfall_au"]
        lines.append(f"RESULT: the best PPO model did NOT complete the mission in the high-fidelity check "
                     f"({hp['reason']}); final a = {hp['a_au']:.4f} AU (short by {sf:+.4f} AU), e = {hp['e']:.4f}.")
    if sp.get("active"):
        lines.append(f"Direction: the impulse-weighted recoil direction on Earth has along-track (prograde) cosine "
                     f"{sp['prograde']:.2f}, |radial| {sp['radial']:.2f}, |out-of-plane| {sp['normal']:.2f}; "
                     f"{sp['retro']:.1%} of the impulse was retrograde. Along-track thrust raises orbital energy fastest "
                     f"(Gauss: da/dt = 2 a^2 v f_T / mu, with f_T the along-track acceleration); radial/normal components mostly waste reaction mass.")
        lines.append(f"Launch speed: mass-weighted mean {sp['u_mean']:.3e} m/s vs {feas['u_required']:.3e} m/s, the "
                     f"minimum-energy speed that uses the whole asteroid (heuristic uses {sh.get('u_mean', float('nan')):.3e} m/s). "
                     f"Momentum per joule falls as 2/u, so a lower speed saves energy but costs more mass.")
        tr = traces["ppo"]["fast"]
        a_prev = [1.0] + [d["a"] for d in tr[:-1]]
        eff = sorted(((d["a"] - a0, d) for d, a0 in zip(tr, a_prev) if d["dm"] > 0), key=lambda x: -x[0])
        if eff:
            fmt = lambda x: (f"t={x[1]['t']:.0f} y: da={x[0]:+.4f} AU (prograde cos {x[1]['d_rtn'][1]:.2f}, "
                             f"u={x[1]['u']:.2e} m/s, throttle {x[1]['thr']:.2f})")
            lines.append("Most effective decisions (largest da per 100-y interval, fast model): " +
                         "; ".join(fmt(x) for x in eff[:3]) + ". Least effective: " + fmt(eff[-1]) +
                         ". At similar throttle, gains per interval grow with a, because at constant thrust acceleration da/dt ~ a^1.5.")
        stopped = sp["active_steps"] < sp["steps"]
        succ = p["fast"]["success_rate"] > 0
        lines.append(f"Timing: ejected in {sp['active_steps']} of {sp['steps']} decision intervals "
                     f"(first {sp['first']:.0f} y, last {sp['last']:.0f} y)" +
                     ("; it coasts in some intervals." if stopped else ".") +
                     (" It keeps thrusting until the orbit enters the target window, where the episode ends."
                      if succ else " It never learned to coast; throttle control near the target is a weakness."
                      if not stopped else ""))
    lines.append(f"Resources (high-fidelity run): PPO used {hp['mass_used_frac']:.1%} of the asteroid and "
                 f"{hp['energy_used_frac']:.1%} of the energy budget; heuristic {h['hifi']['mass_used_frac']:.1%} / "
                 f"{h['hifi']['energy_used_frac']:.1%}; no-ejection baseline stays at a = {n['hifi']['a_au']:.4f} AU.")
    if h["mission_success"] and p["mission_success"]:
        better = "faster" if hp["t_years"] < h["hifi"]["t_years"] else "slower"
        lines.append(f"Versus the hand-designed heuristic: PPO arrives {better} ({hp['t_years']:,.0f} vs {h['hifi']['t_years']:,.0f} y) "
                     f"and uses {'less' if hp['energy_used_J'] < h['hifi']['energy_used_J'] else 'more'} energy.")
    elif h["mission_success"] and not p["mission_success"]:
        lines.append("The hand-designed heuristic succeeds where PPO does not: more training or reward tuning is needed.")
    lines.append("This is the best model found in the tested training run, not a proven global optimum.")
    out["summary"] = lines
    out["ppo"] = sp
    out["heuristic"] = sh
    return out


def make_report(app: "App", comp: dict) -> str:
    cfg = app.cfg
    L = ["# OCTO-EARTH - results report", "", f"Generated {comp['created']}  |  preset **{cfg.preset}**", "",
         "## Feasibility gate (analytic)", "", f"**{app.feas['verdict']}**", ""]
    L += [f"- {x}" for x in app.feas["lines"]]
    L += ["", "## Physics checks", ""]
    L += [f"- {'PASS' if c['passed'] else 'FAIL'} - {c['name']}: {c['detail']}" for c in app.checks] or ["- not run"]
    tr = app.train
    if tr["state"] == "idle" and app.history:
        L += ["", "## Training", "", f"Loaded from a previous run in this folder: {len(app.history)} episodes "
              f"(training_log.csv); this session did not train (comparison re-run only)."]
    else:
        L += ["", "## Training", "", f"State {tr['state']}, {tr['episode']} episodes, {tr['timesteps']} timesteps, "
              f"{tr.get('wall_s', 0):.0f} s wall time (fast model)."]
    b = comp.get("best_model")
    if b:
        L += [f"Best model: episode {b['episode']}, selection = {b['selection']}.",
              f"Hyperparameters: {json.dumps(b['hyperparameters'])}", f"Deterministic eval at selection: {json.dumps(_clean(b['eval']))}"]
    L += ["", "## Comparison (physical metrics; reward shown separately)", "",
          "| policy | fast success (3 seeds) | hi-fi result | hi-fi a [AU] | e | years | asteroid used | energy used [J] | validated | reward (hi-fi) |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for r in comp["rows"]:
        h = r["hifi"]
        L.append(f"| {r['label']} | {r['fast']['success_rate']:.0%} | {h['reason']} | {h['a_au']:.4f} | {h['e']:.4f} | "
                 f"{h['t_years']:,.0f} | {h['mass_used_frac']:.1%} | {h['energy_used_J']:.3e} | {r['mission_success']} | {h['total_reward']:.1f} |")
    bp = comp.get("best_panel") or {}
    if bp.get("criteria"):
        L += ["", "## Best model: final orbit vs target (high-fidelity run)", "", "| criterion | value | required | result |", "|---|---|---|---|"]
        for k in bp["criteria"]:
            val = f"{k['value']:.5g}" if isinstance(k["value"], float) else str(k["value"])
            L.append(f"| {k['name']} | {val} {k['unit']} | {k['required']} | {'PASS' if k['passed'] else 'FAIL'} |")
        L += ["", "## Best model: decisions (high-fidelity run)", "",
              "Classes: " + ", ".join(f"{k} {v}" for k, v in bp["action_classes"].items()), "",
              "| years | throttle | packets | packet kg | speed km/s | prograde cos | dv m/s | da AU | e | effect |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        for a in bp["actions"]:
            L.append(f"| {a['t'] - a['dt']:.0f}-{a['t']:.0f} | {a['thr']:.2f} | {a['npk']} | {a['dm']:.3g} | {a['u'] / 1000:.0f} | "
                     f"{a['d_rtn'][1]:.2f} | {a['dv'] or 0:.1f} | {a['da']:+.4f} | {a['e']:.4f} | {a['cls']} |")
    if bp.get("training_outcomes"):
        L += ["", "## Training episode outcomes (stochastic exploration)", ""]
        L += [f"- {k}: {v} ({REASON_DOCS.get(k, '')})" for k, v in bp["training_outcomes"].items()]
    L += [""] + [f"> {d}" for d in bp.get("disclaimer", [])]
    L += ["", "## Why the best model behaves as it does", ""] + [f"- {x}" for x in comp["explanation"]["summary"]]
    L += ["", "## Conservation (high-fidelity runs)", ""]
    for r in comp["rows"]:
        c = r["hifi"]["conservation"]
        L.append(f"- {r['label']}: energy residual {c['energy_residual_rel']:.1e} (rel. to |E0|), momentum {c['momentum_residual_rel']:.1e}, "
                 f"angular momentum {c['angmom_residual_rel']:.1e}, mass {c['mass_residual_rel']:.1e}; fragment merges {c['merges']}")
    L += ["", "## Limitations", "", "See README.md - simplified prototype, not a validated engineering or climate model."]
    return "\n".join(L) + "\n"


# =============================================================================================
# 11. HTTP API (stdlib) - Python is the only physics source; the browser only renders
# =============================================================================================
import re as _re  # noqa: E402

_NONFINITE = _re.compile(r"\bNaN\b|-?\bInfinity\b")


def _dumps(obj) -> bytes:
    return _NONFINITE.sub("null", json.dumps(obj, default=lambda o: _clean(o) if not isinstance(o, Path) else str(o),
                                             separators=(",", ":"))).encode()


def make_handler(app: App):
    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *a, **kw):
            super().__init__(*a, directory=str(FRONTEND), **kw)

        def log_message(self, *a):
            pass

        def end_headers(self):
            self.send_header("Cache-Control", "no-store")
            super().end_headers()

        def _send(self, obj, code=200):
            body = _dumps(_clean(obj) if not isinstance(obj, (bytes,)) else obj)
            try:
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                pass                                   # browser tab closed / reloaded

        def do_GET(self):
            u = urlparse(self.path)
            if not u.path.startswith("/api/"):
                return super().do_GET()
            q = {k: v[0] for k, v in parse_qs(u.query).items()}
            try:
                if u.path == "/api/state":
                    disp = float(q["disp"]) if "disp" in q else None
                    return self._send(app.state_view(int(q.get("run", -1)), int(q.get("since", 0)),
                                                     int(q.get("msince", 0)), int(q.get("psince", 0)),
                                                     int(q.get("dsince", 0)), disp, int(q.get("pstride", 1)),
                                                     float(q.get("ahead", 0) or 0)))
                if u.path == "/api/results":
                    return self._send(app.results_view())
                if u.path == "/api/health":
                    return self._send(dict(ok=True, app=APP_NAME))
                return self._send(dict(error="not found"), 404)
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                return None
            except Exception as exc:
                app.log(f"[HTTP] GET {u.path} failed: {exc}")
                return self._send(dict(error=str(exc)), 500)

        def do_POST(self):
            u = urlparse(self.path)
            n = int(self.headers.get("Content-Length", 0) or 0)
            try:
                body = json.loads(self.rfile.read(n) or b"{}") if n else {}
                if u.path == "/api/config":
                    app.set_config(dict(body))
                    return self._send(dict(ok=True, config=app.cfg.to_dict(), feasibility=app.feas))
                if u.path == "/api/train":
                    if body.get("action") == "stop":
                        app.stop_train.set()
                        return self._send(dict(ok=True))
                    return self._send(dict(ok=app.start_training()))
                if u.path == "/api/compare":
                    return self._send(dict(ok=app.start_comparison()))
                if u.path == "/api/play":
                    app.start_playback(body.get("policy", "heuristic"), body.get("fidelity", "fast"))
                    return self._send(dict(ok=True))
                if u.path == "/api/stop_play":
                    app.stop_play.set()
                    return self._send(dict(ok=True))
                if u.path == "/api/shutdown":
                    app.shutdown.set()
                    return self._send(dict(ok=True))
                return self._send(dict(error="not found"), 404)
            except Exception as exc:
                return self._send(dict(ok=False, error=str(exc)), 400)

    return Handler


def find_port(host, start):
    import socket
    for p in range(start, start + 30):
        with socket.socket() as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                s.bind((host, p))
                return p
            except OSError:
                continue
    raise RuntimeError("no free port")


# =============================================================================================
# 12. MAIN
# =============================================================================================
def parse_args(argv=None):
    p = argparse.ArgumentParser(description=APP_NAME)
    p.add_argument("--preset", default="hypothetical", choices=list(PRESETS))
    p.add_argument("--smoke", action="store_true", help="short pipeline test (30 episodes, 2,000-year missions)")
    p.add_argument("--check", action="store_true", help="run all physics checks and exit")
    p.add_argument("--headless", action="store_true", help="no server/browser: checks, train, compare, report")
    p.add_argument("--no-train", action="store_true", help="start the viewer without training")
    p.add_argument("--no-browser", action="store_true")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("PORT", "8800"))
    )
    p.add_argument("--episodes", type=int, default=None)
    p.add_argument("--out", default=str(OUTPUTS))
    p.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="override any Config field")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.check:
        res = run_physics_checks(full=True)
        return 0 if all(r["passed"] for r in res) else 1
    over = dict(kv.split("=", 1) for kv in args.set)
    if args.episodes:
        over["train_episodes"] = args.episodes
    cfg = smoke_config(args.preset) if args.smoke else make_config(args.preset)
    try:
        cfg = checked_config(cfg.preset, **{**cfg.to_dict(), **over})
    except ValueError as exc:
        print(f"[ERROR] invalid parameter: {exc}")
        return 2
    app = App(cfg, Path(args.out))
    print("=" * 100)
    print(f" {APP_NAME} | preset={cfg.preset} | mission {cfg.mission_years:,.0f} y | target {cfg.target_au} AU | "
          f"{cfg.train_episodes} training episodes")
    print("=" * 100)
    app.log("[INIT] compiling physics kernels (first run can take ~30-60 s; cached afterwards) ...")
    t0 = time.time()
    app.checks = run_physics_checks(full=False, log=app.log)
    app.log(f"[INIT] physics ready ({time.time() - t0:.1f} s)")
    checks_ok = all(c["passed"] for c in app.checks)
    (app.out / "physics_checks.json").write_text(json.dumps(app.checks, indent=1))
    (app.out / "config.json").write_text(json.dumps(cfg.to_dict(), indent=1))
    (app.out / "feasibility.json").write_text(json.dumps(_clean(app.feas), indent=1))
    app.log(f"[FEASIBILITY] {app.feas['verdict']}")
    for line in app.feas["lines"]:
        app.log("    " + line)
    if not checks_ok:
        app.log("[ERROR] physics checks failed - training is disabled until the physics is fixed.")
    if args.headless:
        if not checks_ok:
            return 1
        app.start_training()
        try:
            while app.train_thread.is_alive() or (app.compare_thread and app.compare_thread.is_alive()) \
                    or app.compare_state == "running":
                time.sleep(0.5)
        except KeyboardInterrupt:
            app.stop_train.set()
            app.train_thread.join(60)
        return 0 if app.train["state"] in ("finished", "stopped") else 1
    port = args.port
    class _Server(ThreadingHTTPServer):
        daemon_threads = True

        def handle_error(self, request, client_address):     # no tracebacks for closed browser tabs
            exc = sys.exc_info()[1]
            if not isinstance(exc, (BrokenPipeError, ConnectionResetError, ConnectionAbortedError)):
                super().handle_error(request, client_address)

    server = _Server((args.host, port), make_handler(app))
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True, name="http").start()
    url = f"http://{'127.0.0.1' if args.host in ('0.0.0.0', '') else args.host}:{port}/"
    app.log(f"[SERVER] viewer at {url}   (close this window or press Ctrl+C to stop)")
    if not args.no_browser:
        try:
            webbrowser.open(url)
        except Exception:
            pass
    import signal

    def _on_signal(signum, frame):            # Ctrl+C, closing the console (SIGBREAK), SIGTERM
        app.shutdown.set()
    for name in ("SIGINT", "SIGTERM", "SIGBREAK"):
        if hasattr(signal, name):
            try:
                signal.signal(getattr(signal, name), _on_signal)
            except (ValueError, OSError):
                pass
    if checks_ok and not args.no_train:
        app.start_training()
    try:
        while not app.shutdown.is_set():
            time.sleep(0.3)
    except KeyboardInterrupt:
        pass
    app.log("[SHUTDOWN] stopping ... (saving the model if training)")
    app.stop_train.set()
    app.stop_play.set()
    if app.train_thread and app.train_thread.is_alive():
        app.train_thread.join(120)
    server.shutdown()
    app.log("[SHUTDOWN] done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
