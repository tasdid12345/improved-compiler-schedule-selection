"""Compare a standard random-forest schedule-cost surrogate (baseline) with a
class-balanced, CV-tuned LightGBM model with native categorical features (improved).
Data: seeded synthetic loop-nest / schedule simulator (numpy only, no download)."""
import json, os, sys, time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import lightgbm as lgb
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import OneHotEncoder

T0 = time.time()
SEED = 42
SEEDS = [42, 43, 44]                       # model seeds (data split fixed at SEED)
N_PROG, N_CAND = 600, 40                   # loop-nest programs, sampled schedules per program
ORDERS = ["ijk", "ikj", "jik", "jki", "kij", "kji"]
UNROLLS = [1, 2, 4, 8]
TILES = ["small", "medium", "large"]
CAT_COLS = ["stride_loop", "order", "vectorize", "parallelize", "unroll", "tile"]
NUM_COLS = ["ext_i", "ext_j", "ext_k", "flops_log2", "footprint_log2", "intensity", "reduction"]


def generate(rng):
    """Synthetic loop nests x sampled schedules; runtime = analytic cost * lognormal noise."""
    grid = [(o, v, p, u, t) for o in ORDERS for v in (0, 1) for p in (0, 1)
            for u in UNROLLS for t in TILES]           # 288 possible schedules
    rows = []
    for pid in range(N_PROG):
        ext = rng.integers(4, 13, size=3)              # log2 extents of loops i, j, k
        flops_log2 = ext.sum() + rng.uniform(0, 2)     # log2 flop count
        foot_log2 = rng.uniform(10, 26)                # log2 working-set size
        prog = dict(ext_i=int(ext[0]), ext_j=int(ext[1]), ext_k=int(ext[2]),
                    flops_log2=flops_log2, footprint_log2=foot_log2,
                    intensity=flops_log2 - foot_log2, reduction=int(rng.integers(0, 2)),
                    stride_loop="ijk"[rng.integers(0, 3)])  # memory stride-1 loop
        ideal_tile = 0 if foot_log2 > 22 else (2 if foot_log2 < 16 else 1)
        for c in rng.choice(len(grid), size=N_CAND, replace=False):
            o, v, p, u, t = grid[c]
            inner = o[-1]                              # innermost loop of the order
            match = inner == prog["stride_loop"]
            order_pen = 1.0 if match else 1.6          # cache-unfriendly order
            tile_pen = 1 + 0.35 * abs(TILES.index(t) - ideal_tile)
            vec_s = 2.5 if (v and match and prog["ext_" + inner] >= 6) else (0.95 if v else 1.0)
            par_s = 3.0 if (p and flops_log2 > 16) else (0.8 if p else 1.0)
            ideal_u = 4 if prog["reduction"] else 2
            unroll_pen = 1 + 0.1 * abs(np.log2(u) - np.log2(ideal_u))
            t_comp = 2.0 ** flops_log2 * 1e-9 / (vec_s * par_s)
            t_mem = 2.0 ** foot_log2 * 1e-9 * order_pen * tile_pen
            runtime = (t_comp + t_mem) * unroll_pen * np.exp(rng.normal(0, 0.05))
            rows.append(dict(pid=pid, order=o, vectorize=v, parallelize=p, unroll=u,
                             tile=t, runtime=runtime, **prog))
    return pd.DataFrame(rows)


def baseline_matrix(df, enc=None):
    """One-hot encoded schedule choices + numeric program features (random forest)."""
    if enc is None:
        enc = OneHotEncoder(handle_unknown="ignore").fit(df[CAT_COLS].astype(str))
    cat = enc.transform(df[CAT_COLS].astype(str)).toarray()
    return np.hstack([cat, df[NUM_COLS].to_numpy(float)]), enc


def improved_frame(df, cats):
    """Native pandas categoricals for LightGBM; category levels fixed on training data."""
    X = df[CAT_COLS + NUM_COLS].copy()
    for c in CAT_COLS:
        X[c] = pd.Categorical(X[c].astype(str), categories=cats[c])
    return X


def standardised_target(df):
    """Log-runtime z-scored within each program; weights give each program equal total weight."""
    lt = np.log(df["runtime"])
    g = lt.groupby(df["pid"])
    z = (lt - g.transform("mean")) / (g.transform("std") + 1e-12)
    w = 1.0 / df.groupby("pid")["pid"].transform("size")
    return z.to_numpy(), (w / w.mean()).to_numpy()


def evaluate(df, pred):
    """Mean regret (chosen/best - 1), top-1 accuracy, mean per-program Spearman rho."""
    d = df[["pid", "runtime"]].assign(pred=np.asarray(pred))
    regrets, top1, rhos = [], [], []
    for _, g in d.groupby("pid"):
        best = g["runtime"].min()
        k = int(g["pred"].to_numpy().argmin())
        regrets.append(g["runtime"].iloc[k] / best - 1.0)
        top1.append(float(g["runtime"].iloc[k] == best))
        rho = spearmanr(g["pred"], g["runtime"]).correlation
        if np.isfinite(rho):
            rhos.append(rho)
    return float(np.mean(regrets)), float(np.mean(top1)), float(np.mean(rhos))


def tune_lgbm(train, cats):
    """Choose LightGBM hyper-parameters by 3-fold GroupKFold CV on TRAINING programs only."""
    X = improved_frame(train, cats)
    y, w = standardised_target(train)
    grid = [dict(num_leaves=nl, learning_rate=lr, n_estimators=ne, min_child_samples=mc)
            for nl in (15, 31) for lr in (0.05, 0.1) for ne in (200, 400) for mc in (20, 50)]
    best = None
    for hp in grid:
        oof = np.zeros(len(train))
        for tr, va in GroupKFold(n_splits=3).split(X, y, groups=train["pid"]):
            m = lgb.LGBMRegressor(**hp, random_state=SEED, verbose=-1)
            oof[va] = m.fit(X.iloc[tr], y[tr], sample_weight=w[tr]).predict(X.iloc[va])
        r = evaluate(train, oof)[0]               # out-of-fold regret on training programs
        print(f"  cv regret {r:.4f} {hp}")
        if best is None or r < best[0]:
            best = (r, hp)
    print(f"selected {best[1]} (cv regret {best[0]:.4f})")
    return best[1]


def main():
    rng = np.random.default_rng(SEED)
    df = generate(rng)
    print(f"generated {len(df)} rows from {N_PROG} programs [{time.time() - T0:.0f}s]")
    # program-level 70/30 split, done once
    perm = rng.permutation(N_PROG)
    is_tr = df["pid"].isin(set(perm[: int(0.7 * N_PROG)]))
    train = df[is_tr].reset_index(drop=True)
    test = df[~is_tr].reset_index(drop=True)
    print(f"train rows {len(train)}, test rows {len(test)}")

    cats = {c: sorted(train[c].astype(str).unique()) for c in CAT_COLS}
    hp = tune_lgbm(train, cats)

    Xb_tr, enc = baseline_matrix(train)
    Xb_te, _ = baseline_matrix(test, enc)
    Xi_tr, Xi_te = improved_frame(train, cats), improved_frame(test, cats)
    y_z, w = standardised_target(train)

    res = {"baseline": [], "improved": []}
    for s in SEEDS:
        rf = RandomForestRegressor(n_estimators=300, min_samples_leaf=2, n_jobs=-1, random_state=s)
        res["baseline"].append(evaluate(test, rf.fit(Xb_tr, train["runtime"]).predict(Xb_te)))
        gb = lgb.LGBMRegressor(**hp, random_state=s, verbose=-1)
        res["improved"].append(evaluate(test, gb.fit(Xi_tr, y_z, sample_weight=w).predict(Xi_te)))
        print(f"seed {s}: baseline regret {res['baseline'][-1][0]:.4f}, "
              f"improved regret {res['improved'][-1][0]:.4f} [{time.time() - T0:.0f}s]")

    m = {}
    for k in ("baseline", "improved"):
        a = np.array(res[k])
        m[f"{k}_score"] = float(a[:, 0].mean())
        m[f"{k}_score_std"] = float(a[:, 0].std())
        m[f"{k}_top1_accuracy"] = float(a[:, 1].mean())
        m[f"{k}_spearman"] = float(a[:, 2].mean())
    m["improvement"] = m["baseline_score"] - m["improved_score"]   # positive = improved is better
    if not all(np.isfinite(v) for v in m.values()):
        sys.exit("non-finite metric; aborting")

    os.makedirs("figures", exist_ok=True)
    plt.figure(figsize=(5, 4))
    plt.bar(["baseline (RF)", "improved (LGBM)"], [m["baseline_score"], m["improved_score"]],
            yerr=[m["baseline_score_std"], m["improved_score_std"]], capsize=6, color=["gray", "steelblue"])
    plt.ylabel("mean regret (lower is better)")
    plt.title("Held-out programs, mean over 3 seeds")
    plt.tight_layout()
    plt.savefig("figures/regret_comparison.png", dpi=120)

    out = {
        "question": "Can a class-balanced, CV-tuned gradient-boosting cost model with categorical "
                    "schedule features choose faster compiler schedules on held-out programs than "
                    "a standard random-forest cost surrogate?",
        "method": "Seeded synthetic loop-nest simulator; program-level 70/30 split; each test program's "
                  "candidate schedules ranked by predicted runtime; mean regret as main metric.",
        "dataset": {
            "name": "Synthetic loop-nest schedule simulator (seeded numpy generator)",
            "source_url": "https://numpy.org",
            "n_rows": int(len(df)), "n_train": int(len(train)), "n_test": int(len(test)),
            "same_as_paper": False, "baseline_is_papers_method": False,
        },
        "metric_name": "mean regret (chosen runtime / best candidate runtime - 1)",
        "higher_is_better": False,
        "baseline": "RandomForestRegressor (300 trees, min_samples_leaf=2) on one-hot schedule "
                    "features plus numeric program features, predicting raw runtime.",
        "improved": "LightGBM regressor with native categorical features, hyper-parameters chosen by "
                    "3-fold GroupKFold CV on training programs, per-program z-scored log-runtime target "
                    "and program-balanced sample weights.",
        "metrics": m,
        "notes": "Synthetic data, not the paper's dataset; the paper's own method is not reproduced "
                 "(only a generic random-forest baseline is used). The baseline uses fixed hyper-parameters "
                 "while the improved model is CV-tuned, so gains may partly reflect tuning. Regret and top-1 "
                 "use noisy observed runtimes on a modest number of test programs. Seeds vary model "
                 "randomness only; the data split and generator are fixed at seed 42.",
    }
    with open("results.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"done in {time.time() - T0:.0f}s; wrote results.json")


if __name__ == "__main__":
    main()
