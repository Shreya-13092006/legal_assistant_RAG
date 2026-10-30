"""Paired statistics (SDD 7.8): exact McNemar, Wilcoxon signed-rank, paired bootstrap CIs, Holm,
Cohen's kappa (plain and weighted). Estimation first; p-values are secondary."""
from __future__ import annotations

import math
import random
from collections import Counter


def mcnemar_exact(a: list[bool], b: list[bool]) -> dict:
    """Two-sided exact McNemar test on paired binary outcomes."""
    assert len(a) == len(b)
    n01 = sum(1 for x, y in zip(a, b) if not x and y)
    n10 = sum(1 for x, y in zip(a, b) if x and not y)
    n = n01 + n10
    if n == 0:
        return {"b": n10, "c": n01, "p": 1.0}
    k = min(n01, n10)
    p = sum(math.comb(n, i) for i in range(0, k + 1)) / 2 ** n
    return {"b": n10, "c": n01, "p": min(1.0, 2 * p)}


def wilcoxon(a: list[float], b: list[float]) -> dict:
    d = [y - x for x, y in zip(a, b) if x is not None and y is not None]
    nz = [x for x in d if x != 0]
    if not nz:
        return {"statistic": 0.0, "p": 1.0, "rank_biserial": 0.0, "n": len(d)}
    try:
        from scipy.stats import wilcoxon as _w
        res = _w(nz, zero_method="wilcox", alternative="two-sided")
        stat, p = float(res.statistic), float(res.pvalue)
    except ImportError:  # normal approximation
        stat, p = float("nan"), float("nan")
    ranks = _ranks([abs(x) for x in nz])
    wpos = sum(r for r, x in zip(ranks, nz) if x > 0)
    wneg = sum(r for r, x in zip(ranks, nz) if x < 0)
    rb = (wpos - wneg) / (wpos + wneg)
    return {"statistic": stat, "p": p, "rank_biserial": rb, "n": len(d)}


def _ranks(xs: list[float]) -> list[float]:
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return ranks


def bootstrap_ci(values: list[float], n: int = 10_000, seed: int = 13, alpha: float = 0.05) -> tuple[float, float, float]:
    vals = [float(v) for v in values if v is not None]
    if not vals:
        return (float("nan"),) * 3
    rng = random.Random(seed)
    m = len(vals)
    stats = sorted(sum(vals[rng.randrange(m)] for _ in range(m)) / m for _ in range(n))
    return sum(vals) / m, stats[int(alpha / 2 * n)], stats[int((1 - alpha / 2) * n) - 1]


def paired_bootstrap_diff(a: list[float], b: list[float], n: int = 10_000, seed: int = 13) -> tuple[float, float, float]:
    """CI on mean(b - a) resampling items (cluster = item)."""
    pairs = [(float(x), float(y)) for x, y in zip(a, b) if x is not None and y is not None]
    return bootstrap_ci([y - x for x, y in pairs], n, seed)


def holm(pvals: dict[str, float], alpha: float = 0.05) -> dict[str, dict]:
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    out, running = {}, 0.0
    for i, (k, p) in enumerate(items):
        adj = min(1.0, max(running, (m - i) * p))
        running = adj
        out[k] = {"p": p, "p_holm": adj, "reject": adj < alpha}
    return out


def cohen_kappa(a: list, b: list) -> float:
    n = len(a)
    if n == 0:
        return float("nan")
    po = sum(1 for x, y in zip(a, b) if x == y) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum(ca[k] * cb.get(k, 0) for k in ca) / n ** 2
    return 1.0 if pe == 1 else (po - pe) / (1 - pe)


def weighted_kappa(a: list[int], b: list[int], categories=(0, 1, 2), quadratic: bool = False) -> float:
    k = len(categories)
    idx = {c: i for i, c in enumerate(categories)}
    n = len(a)
    if n == 0:
        return float("nan")
    w = [[((i - j) ** 2 if quadratic else abs(i - j)) / ((k - 1) ** (2 if quadratic else 1)) for j in range(k)]
         for i in range(k)]
    obs = [[0.0] * k for _ in range(k)]
    for x, y in zip(a, b):
        obs[idx[x]][idx[y]] += 1 / n
    ra = [sum(r) for r in obs]
    rb = [sum(obs[i][j] for i in range(k)) for j in range(k)]
    num = sum(w[i][j] * obs[i][j] for i in range(k) for j in range(k))
    den = sum(w[i][j] * ra[i] * rb[j] for i in range(k) for j in range(k))
    return 1 - num / den if den else 1.0
