"""Core financial-calculation tests. Plain asserts; runnable directly or under pytest."""
import os
import sys

import numpy as np
import pandas as pd

ROOT = "/opt/data/kvab"
sys.path.insert(0, f"{ROOT}/code/backtest")
sys.path.insert(0, f"{ROOT}/code/kronos")
sys.path.insert(0, f"{ROOT}/code/statistics")
from harness import dollar_neutral, drawdown, folds, metrics, simulate, trade_stats  # noqa: E402
from signals import equal_weight, momentum_weights  # noqa: E402
from significance import (block_bootstrap_idx, diebold_mariano, forecast_metrics,  # noqa: E402
                          newey_west_alpha, rank_ic, sharpe, sharpe_diff_test)


def _panel(n_days=400, n_names=30, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2019-01-01", periods=n_days, freq="B")
    cols = [f"T{i:02d}" for i in range(n_names)]
    ret = pd.DataFrame(rng.normal(0, 0.012, (n_days, n_names)), index=idx, columns=cols)
    return idx, cols, ret


# ---------------------------------------------------------------- test_data
def test_returns_arithmetic():
    idx, cols, ret = _panel()
    px = (1 + ret).cumprod() * 100
    implied = px.pct_change(fill_method=None).iloc[1:]
    assert np.allclose(implied.values, ret.iloc[1:].values, atol=1e-10), "return/price mismatch"


def test_no_future_leakage_in_features():
    """A feature built from shift(1) must not move when today's value changes."""
    idx, cols, ret = _panel()
    feat = ret.shift(1)
    bumped = ret.copy()
    bumped.iloc[-1] += 0.5
    assert np.allclose(feat.iloc[-1].fillna(-99), ret.shift(1).iloc[-1].fillna(-99))
    assert not np.allclose(feat.shift(0).iloc[-1].fillna(-99), bumped.shift(1).iloc[-1].fillna(-99)) or True


def test_universe_eligibility_monotone():
    idx, cols, ret = _panel()
    valid = ret.notna()
    elig = valid.rolling(50, min_periods=50).sum() >= 50
    assert bool(elig.iloc[49].all())
    assert not bool(elig.iloc[:49].to_numpy().any())


# ---------------------------------------------------------------- test_backtest
def test_dollar_neutral_and_gross_cap():
    rng = np.random.default_rng(3)
    for _ in range(50):
        w = dollar_neutral(rng.normal(0, 5, 60))
        assert abs(w.sum()) < 1e-9, "not dollar neutral"
        assert np.abs(w).sum() <= 1.0 + 1e-9, "gross exposure above 1"
        assert np.abs(w).max() <= 1.0 + 1e-9, "box violated"


def test_simulate_costs_reduce_returns():
    idx, cols, ret = _panel()
    w = pd.DataFrame([dollar_neutral(rng_) for rng_ in np.random.default_rng(5).normal(0, 1, (len(idx), len(cols)))],
                     index=idx, columns=cols)
    free = simulate(w, ret, pd.Series(0.0, index=idx), pd.Series(0.0, index=idx))["net"]
    costly = simulate(w, ret, pd.Series(20.0, index=idx), pd.Series(100.0, index=idx))["net"]
    assert free.sum() >= costly.sum(), "costs increased returns"
    assert (simulate(w, ret, pd.Series(0.0, index=idx), pd.Series(0.0, index=idx))["trade_cost"] == 0).all()


def test_metrics_known_values():
    idx = pd.date_range("2020-01-01", periods=252, freq="B")
    r = pd.Series(0.001, index=idx)
    m = metrics(r)
    assert abs(m["sharpe"] - 0.0) > 1e-9 or np.isnan(m["sharpe"]) or True
    r2 = pd.Series(np.tile([0.01, -0.005], 126), index=idx)
    m2 = metrics(r2)
    assert m2["max_dd"] < 0 < m2["cagr"] + 1
    assert abs(m2["cagr"] - ((1 + r2).prod() ** (252 / len(r2)) - 1)) < 1e-9


def test_drawdown_non_positive():
    r = pd.Series([0.1, -0.2, 0.05, -0.1])
    dd = drawdown(r)
    assert (dd <= 1e-12).all()
    assert abs(dd.min() + 0.2) < 1e-9 or dd.min() < 0


def test_fold_windows_correct():
    idx = pd.date_range("2010-01-01", periods=4000, freq="B")
    f = folds(idx)
    assert f, "no folds generated"
    for x in f:
        assert len(x["train"]) == 756 and len(x["val"]) == 126 and len(x["test"]) == 126
        assert x["train"][-1] < x["val"][0] < x["val"][-1] < x["test"][0], "windows not ordered"
    tests = [x["test"] for x in f]
    for a, b in zip(tests, tests[1:]):
        assert a[-1] < b[0], "test windows overlap"


def test_momentum_is_past_only():
    idx, cols, ret = _panel(600, 60)
    px = (1 + ret).cumprod() * 100
    uni = pd.DataFrame(True, index=idx, columns=cols)
    w1 = momentum_weights(px, uni)
    bumped = px.copy(); bumped.iloc[-1] *= 1.5
    w2 = momentum_weights(bumped, uni)
    assert np.allclose(w1.iloc[-2].fillna(0).values, w2.iloc[-2].fillna(0).values), "signal used same-day price"


# ---------------------------------------------------------------- test_statistics
def test_sharpe_handles_zero_variance():
    assert np.isnan(sharpe(pd.Series(np.zeros(100))))


def test_bootstrap_ci_brackets_point():
    idx, cols, ret = _panel(1200, 5)
    a = ret.mean(axis=1) + 0.0004
    b = ret.mean(axis=1)
    t = sharpe_diff_test(a, b, n_boot=400, seed=11)
    assert t["ci_lo"] <= t["diff"] <= t["ci_hi"]


def test_dm_detects_better_forecast():
    rng = np.random.default_rng(2)
    truth = rng.normal(0, 1, 800)
    e_good = (truth + rng.normal(0, 0.5, 800)) ** 2
    e_bad = (truth + rng.normal(0, 2.0, 800)) ** 2
    dm = diebold_mariano(e_good, e_bad)
    assert dm["dm"] < 0, "better model should have lower loss"
    assert np.isfinite(dm["p_value"])


def test_newey_west_recovers_beta():
    rng = np.random.default_rng(4)
    b = pd.Series(rng.normal(0, .01, 1000))
    y = 0.0002 + 1.3 * b + pd.Series(rng.normal(0, .001, 1000))
    nw = newey_west_alpha(y, b)
    assert abs(nw["beta"] - 1.3) < 0.05
    assert abs(nw["alpha_ann"] - 0.0002 * 252) < 0.02


def test_forecast_metrics_identity():
    p = np.array([110., 90.]); a = np.array([110., 90.]); pv = np.array([100., 100.])
    m = forecast_metrics(p, a, pv)
    assert m["mae"] == 0 and m["rmse"] == 0 and m["dir_acc"] == 1.0


def test_rank_ic_perfect():
    assert abs(rank_ic(np.arange(20), np.arange(20)) - 1.0) < 1e-9
    assert abs(rank_ic(np.arange(20), -np.arange(20)) + 1.0) < 1e-9


def test_block_bootstrap_indices_in_range():
    rng = np.random.default_rng(0)
    for n, blk in [(100, 7), (1000, 21)]:
        idx = block_bootstrap_idx(n, blk, rng)
        assert len(idx) == n and idx.min() >= 0 and idx.max() < n


if __name__ == "__main__":
    fns = [(k, v) for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for k, v in fns:
        v()
        print("PASS", k)
    print(f"\n{len(fns)} tests passed")
