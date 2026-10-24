"""pytest suite for the OCTO-EARTH prototype:  python -m pytest -q tests"""
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import octoearth_rl as R  # noqa: E402


@pytest.fixture(scope="module")
def checks():
    return R.run_physics_checks(full=True, log=lambda *a: None)


def test_all_physics_checks_pass(checks):
    failed = [(c["name"], c["detail"]) for c in checks if not c["passed"]]
    assert not failed, failed
    assert len(checks) >= 12


def test_feasibility_gate_is_honest():
    assert R.feasibility(R.make_config("hypothetical"))["feasible"]
    real = R.feasibility(R.make_config("realistic"))
    assert not real["feasible"] and "INFEASIBLE" in real["verdict"]


def test_gym_env_api():
    from gymnasium.utils.env_checker import check_env
    env = R.OctoEarthEnv(R.make_config(), "fast")
    check_env(env, skip_render_check=True)


def test_no_free_energy_and_mass_limits():
    cfg = R.make_config(energy_budget_j=1e33, asteroid_mass_kg=1e20)
    env = R.OctoEarthEnv(cfg, "fast")
    env.reset(seed=3)
    for _ in range(20):
        _, _, term, trunc, info = env.step(np.ones(4, dtype=np.float32))
        if term or trunc:
            break
    eng = env.engine
    assert eng.E_rem >= -1e-6 * cfg.energy_budget_j
    assert eng.m_ast >= 0.0
    # Earth only moved by the energy it was allowed to spend: a stays close to 1 AU
    assert info["decision"]["a_au"] < 1.01


def test_no_ejection_baseline_stays_put():
    s = R.run_episode(R.make_config(), R.policy_none, 1001, "fast")
    assert abs(s["a_au"] - 1.0) < 1e-3 and not s["success"]


def test_heuristic_reaches_target_fast_model():
    s = R.run_episode(R.make_config(), R.policy_heuristic, 1001, "fast")
    assert s["success"], s["reason"]


def test_json_is_browser_safe():
    out = R._dumps(R._clean({"a": float("nan"), "b": [float("inf"), 1.0], "c": np.float64(2.0)}))
    d = json.loads(out)
    assert d["a"] is None and d["b"][0] is None and d["c"] == 2.0


def test_config_overrides_and_presets():
    c = R.make_config("hypothetical", asteroid_mass_kg="1e21", train_episodes="50")
    assert c.asteroid_mass_kg == 1e21 and c.train_episodes == 50 and c.preset == "custom"
    assert R.make_config("realistic").preset == "realistic"


# ----------------------------------------------------------------------------- viewer data
def _one_decision(fid):
    env = R.OctoEarthEnv(R.make_config(randomize_phase=False), fid)
    env.reset(seed=5)
    env.step(R.policy_heuristic(env, None))
    return env


@pytest.mark.parametrize("fid", ["fast", "hifi"])
def test_recorded_states_are_continuous(fid):
    """One row per ejection event at the ejection instant; evenly spaced; recoil column equals
    |v_post - v_pre|; Earth moves no farther between rows than its orbital speed allows."""
    env = _one_decision(fid)
    s = env.last_samples
    cfg = env.cfg
    assert s.shape[1] == 25 and len(s) == int(round(cfg.ejections_per_year * cfg.decision_interval_years))
    dt = np.diff(s[:, R.S_T])
    assert np.allclose(dt, 1.0 / cfg.ejections_per_year, rtol=1e-9)
    dv = np.linalg.norm(s[:, R.S_VPOST:R.S_VPOST + 3] - s[:, R.S_VPRE:R.S_VPRE + 3], axis=1) * 1000.0
    assert np.allclose(dv, s[:, R.S_DV], rtol=1e-6)
    step = np.linalg.norm(np.diff(s[:, R.S_R:R.S_R + 3], axis=0), axis=1) * R.AU
    vmax = np.linalg.norm(s[:, R.S_VPRE:R.S_VPRE + 3], axis=1).max() * 1000.0
    assert step.max() <= vmax * dt.max() * R.YEAR * 1.001          # no jumps
    assert np.all(np.diff(s[:, R.S_MAST]) <= 0)                       # asteroid only loses mass
    if fid == "hifi":
        assert len(env.engine.last_moon) == 8 * len(s)
        assert np.all(np.diff(env.engine.last_moon[:, 0]) > 0)


def test_packet_arcs_match_kepler():
    r0 = np.array([[R.AU, 0.0, 0.0]])
    v0 = np.array([[0.0, 40e3, 1e3]])                                  # bound, inclined
    offs = R.PACKET_OFFSETS_DAYS * 86400.0
    out = np.zeros((1, len(offs), 3))
    nv = np.zeros(1, dtype=np.int64)
    R.packet_arcs(r0, v0, offs, 60 * R.AU, out, nv)
    assert nv[0] == len(offs)
    for j in (5, 12, len(offs) - 1):
        r, v = r0[0].copy(), v0[0].copy()
        R.kepler_step(r, v, R.GM_SUN, offs[j])
        assert np.linalg.norm(out[0, j] - r) < 1e-6 * R.AU


def test_viewer_payload_is_json_safe():
    for fid in ("fast", "hifi"):
        env = _one_decision(fid)
        rows = R._viewer_rows(env.last_samples)
        pk = R._packet_records(env.last_samples, env.cfg, fid)
        moon = R._moon_rows(env.engine.last_moon)
        assert len(rows[0]) == 22 and len(pk) == len(rows)
        assert all(p[4] * 3 + 5 == len(p) for p in pk)
        txt = json.dumps(dict(rows=rows, pk=pk, moon=moon), allow_nan=False)
        assert len(txt) > 1000


def test_parameter_validation_rejects_bad_values():
    assert R.validate_params(R.make_config("hypothetical").to_dict()) == []
    assert R.validate_params(R.make_config("realistic").to_dict()) == []
    bad = [dict(min_ejection_speed_ms=6e6), dict(max_ejection_speed_ms=1e8), dict(target_au=0.5),
           dict(ppo_batch_size=5000), dict(decision_interval_years=5.0), dict(max_packet_mass_kg=1e23, asteroid_mass_kg=1e22),
           dict(nonexistent_parameter=1)]
    for b in bad:
        with pytest.raises(ValueError):
            R.checked_config("hypothetical", **b)


def test_every_parameter_and_output_is_documented():
    fields = set(R.make_config().to_dict()) - {"preset"}
    assert fields == set(R.PARAM_BY_KEY)
    for d in R.PARAM_DOCS:
        for k in ("label", "unit", "category", "meaning", "why", "increase", "decrease"):
            assert d[k], (d["key"], k)
    for k in ("a", "e", "inc", "E_orb", "dv", "thrust", "impulse", "m_ast", "E_used", "reward", "return",
              "progress", "cons_E", "cons_P", "cons_L", "cons_M"):
        assert k in R.OUTPUT_BY_KEY
    assert set(R.REASON_DOCS) >= {"success", "deadline", "resources_exhausted"}


def test_fast_model_simulates_moon():
    """The training (fast) model simulates the Moon: recorded stream, real distance, bound,
    Earth's recoil changes the Earth-Moon relative velocity, and the training env gets the same
    physics whether or not Moon samples are recorded for the viewer."""
    cfg = R.make_config("hypothetical", randomize_phase=False)
    f = R.FastEngine(cfg, 3)
    smp, status = f.run_interval(2.0, 1e18, 2.5e6, [0, 1, 0])
    assert status == 0 and len(f.last_moon) == R.MOON_SUB * len(smp)
    assert np.all(np.diff(f.last_moon[:, 0]) > 0)
    d = np.linalg.norm(f.last_moon[:, 1:4], axis=1)
    assert 3.4e8 < d.min() and d.max() < 4.2e8           # real Earth-Moon distances, varying
    assert d.max() - d.min() > 1e7
    info = f.moon_info()
    assert info["modelled"] and info["bound"]
    g = R.FastEngine(cfg, 3, record_moon=False)
    g.run_interval(2.0, 1e18, 2.5e6, [0, 1, 0])
    assert np.array_equal(f.st, g.st)                     # recording does not alter the physics
    chk = {c["name"]: c for c in R.run_physics_checks(False, log=lambda m: None)}
    assert chk["14 Moon in the fast (training) model"]["passed"]
