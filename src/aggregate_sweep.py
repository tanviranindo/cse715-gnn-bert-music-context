"""Aggregate the Task 3 seed sweep into mean +/- std and a paired comparison.

Both arms are run on the same seeds, so the per-seed difference is paired and a
paired t-test is the right test: it removes seed-to-seed variance that affects
both arms equally.
"""

import json
import math
import pathlib
import sys

ROOT = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "results/_sweep")
SEEDS = [42, 1, 2, 3, 4]
METRICS = ["test_macro_f1", "test_micro_f1", "test_auc_pr"]


def load(arm, seed):
    p = ROOT / ("%s_s%d" % (arm, seed)) / "metrics_task3.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())["runs"]["crossattn"]


def mean_sd(xs):
    n = len(xs)
    m = sum(xs) / n
    if n < 2:
        return m, 0.0
    var = sum((x - m) ** 2 for x in xs) / (n - 1)
    return m, math.sqrt(var)


def t_cdf_upper(t, df):
    """P(T > t) for Student's t, via the incomplete beta function."""
    x = df / (df + t * t)
    return 0.5 * betainc(df / 2.0, 0.5, x) if t > 0 else 1.0 - 0.5 * betainc(df / 2.0, 0.5, x)


def betainc(a, b, x):
    """Regularised incomplete beta I_x(a,b), continued fraction (NR 6.4)."""
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    lbeta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    front = math.exp(lbeta + a * math.log(x) + b * math.log(1 - x))
    if x < (a + 1) / (a + b + 2):
        return front * _cf(a, b, x) / a
    return 1.0 - math.exp(lbeta + b * math.log(1 - x) + a * math.log(x)) * _cf(b, a, 1 - x) / b


def _cf(a, b, x, itmax=200, eps=3e-16):
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    if abs(d) < 1e-30:
        d = 1e-30
    d = 1.0 / d
    h = d
    for m in range(1, itmax + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        c = 1.0 + aa / c
        if abs(d) < 1e-30: d = 1e-30
        if abs(c) < 1e-30: c = 1e-30
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        c = 1.0 + aa / c
        if abs(d) < 1e-30: d = 1e-30
        if abs(c) < 1e-30: c = 1e-30
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return h


rows = {a: {m: [] for m in METRICS} for a in ("shared", "gnnlr")}
have = []
for seed in SEEDS:
    s, g = load("shared", seed), load("gnnlr", seed)
    if not s or not g:
        print("  (missing seed %d: shared=%s gnnlr=%s)" % (seed, bool(s), bool(g)))
        continue
    have.append(seed)
    for m in METRICS:
        rows["shared"][m].append(s[m])
        rows["gnnlr"][m].append(g[m])

print("seeds with both arms: %s (n=%d)\n" % (have, len(have)))
print("%-14s %-22s %-22s %s" % ("metric", "shared lr (2e-5)", "separate GNN lr 1e-3", "paired diff"))
print("-" * 88)
for m in METRICS:
    a, b = rows["shared"][m], rows["gnnlr"][m]
    if not a:
        continue
    ma, sa = mean_sd(a)
    mb, sb = mean_sd(b)
    diffs = [y - x for x, y in zip(a, b)]
    md, sd = mean_sd(diffs)
    n = len(diffs)
    if n > 1 and sd > 0:
        t = md / (sd / math.sqrt(n))
        p = 2 * t_cdf_upper(abs(t), n - 1)
        sig = "p=%.4f" % p
    else:
        sig = "n/a"
    print("%-14s %.4f +/- %.4f      %.4f +/- %.4f      %+.4f +/- %.4f  %s"
          % (m.replace("test_", ""), ma, sa, mb, sb, md, sd, sig))

print("\nper-seed macro-F1:")
for i, seed in enumerate(have):
    print("  seed %-3d shared=%.4f  gnnlr=%.4f  diff=%+.4f"
          % (seed, rows["shared"]["test_macro_f1"][i],
             rows["gnnlr"]["test_macro_f1"][i],
             rows["gnnlr"]["test_macro_f1"][i] - rows["shared"]["test_macro_f1"][i]))
