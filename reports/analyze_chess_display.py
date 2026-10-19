"""Reproduce the observational board-display analysis; no API calls.
Run from the repository root: .venv/bin/python reports/analyze_chess_display.py
Dependencies: pandas, numpy, scipy, matplotlib (Agg).
"""
import collections
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import fisher_exact, norm

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports"
frames = []
for path in sorted((ROOT / "exp/chess_comparison/direct_qa").glob("*/puzzles.csv")):
    frame = pd.read_csv(path, keep_default_na=False)
    frame["complete"] = frame.complete.astype(str).str.lower().eq("true")
    frame["correct"] = frame.correct.astype(str).str.lower().eq("true")
    frame["display"] = pd.to_numeric(frame.display_board_unicode_calls).gt(0)
    frame["display_calls"] = pd.to_numeric(frame.display_board_unicode_calls)
    received, first_display = [], []
    for trace_text in frame.tool_trace:
        trace = json.loads(trace_text or "[]")
        received.append(any("board" in t.get("result", {}) for t in trace))
        turn, first = -1, None
        for event in trace:
            if "stop_reason" in event:
                turn += 1
            if (
                "result" in event
                and event["tool_call"]["function"]["name"] == "display_board_unicode"
                and first is None
            ):
                first = turn
        first_display.append(-1 if first is None else first)
    frame["received_board"] = received
    frame["first_display_response_index"] = first_display
    frames.append(frame)
all_rows = pd.concat(frames, ignore_index=True)
data = all_rows[all_rows.complete].copy()
assert len(all_rows) == 1050 and len(data) == 1020
for row in pd.read_csv(ROOT / "exp/chess_comparison/results.csv").itertuples():
    group = data[data.model.eq(row.model)]
    assert len(group) == row.completed and np.isclose(group.correct.mean(), row.all)


def estimate(frame, controls, exposure="display", linear=False):
    """Common exposure coefficient; puzzle-clustered sandwich covariance.
    Logistic risk differences standardize over this frame's covariates.
    OLS with puzzle and model dummies is a two-way fixed-effect sensitivity.
    """
    y = frame.correct.to_numpy(float)
    parts = [np.ones((len(frame), 1)), frame[exposure].to_numpy(float)[:, None]]
    for control in controls:
        if control == "rating_poly":
            rating = (frame.rating.to_numpy(float) - 1700) / 1000
            parts += [rating[:, None], (rating**2)[:, None]]
        else:
            parts.append(
                pd.get_dummies(frame[control], drop_first=True).to_numpy(float)
            )
    x = np.column_stack(parts)
    if linear:
        beta = np.linalg.lstsq(x, y, rcond=None)[0]
        mu = x @ beta
        hessian = x.T @ x
    else:

        def objective(beta):
            z = x @ beta
            return np.logaddexp(0, z).sum() - y @ z, x.T @ (expit(z) - y)

        fit = minimize(
            objective,
            np.zeros(x.shape[1]),
            jac=True,
            method="BFGS",
            options={"gtol": 1e-6, "maxiter": 1000},
        )
        assert np.linalg.norm(fit.jac) < 1e-5, fit.message
        beta = fit.x
        mu = expit(x @ beta)
        hessian = x.T @ ((mu * (1 - mu))[:, None] * x)
    scores = x * (y - mu)[:, None]
    ids = frame.puzzleId.to_numpy()
    sums = np.array([scores[ids == puzzle].sum(0) for puzzle in np.unique(ids)])
    bread = np.linalg.pinv(hessian)
    n, k = x.shape
    g = len(sums)
    covariance = bread @ (sums.T @ sums) @ bread * (g / (g - 1)) * ((n - 1) / (n - k))
    se = np.sqrt(covariance[1, 1])
    coefficient = beta[1]
    result = {
        "n": n,
        "coefficient": float(coefficient),
        "se": float(se),
        "p": float(2 * norm.sf(abs(coefficient / se))),
    }
    if linear:
        result.update(
            risk_difference=float(coefficient),
            risk_difference_ci=[
                float(coefficient - 1.96 * se),
                float(coefficient + 1.96 * se),
            ],
        )
    else:
        x1, x0 = x.copy(), x.copy()
        x1[:, 1], x0[:, 1] = 1, 0
        p1, p0 = expit(x1 @ beta), expit(x0 @ beta)
        rd = (p1 - p0).mean()
        gradient = ((p1 * (1 - p1))[:, None] * x1 - (p0 * (1 - p0))[:, None] * x0).mean(
            0
        )
        rd_se = np.sqrt(gradient @ covariance @ gradient)
        result.update(
            odds_ratio=float(np.exp(coefficient)),
            odds_ratio_ci=np.exp(
                [coefficient - 1.96 * se, coefficient + 1.96 * se]
            ).tolist(),
            risk_difference=float(rd),
            risk_difference_ci=[float(rd - 1.96 * rd_se), float(rd + 1.96 * rd_se)],
        )
    return result


fits = {
    "unadjusted": estimate(data, []),
    "model": estimate(data, ["model"]),
    "model_bucket": estimate(data, ["model", "rating_bucket"]),
    "model_rating_quadratic": estimate(data, ["model", "rating_poly"]),
    "model_puzzle_fixed_effects_linear": estimate(
        data, ["model", "puzzleId"], linear=True
    ),
    "received_board_model_bucket": estimate(
        data, ["model", "rating_bucket"], exposure="received_board"
    ),
    "exclude_oss20b_model_bucket": estimate(
        data[data.model.ne("openai/gpt-oss-20b")], ["model", "rating_bucket"]
    ),
    "haiku_luna_only_model_bucket": estimate(
        data[data.model.isin(["claude-haiku-5-5", "gpt-6-luna"])],
        ["model", "rating_bucket"],
    ),
}


def wilson(k, n):
    p, z = k / n, 1.96
    center = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return center - half, center + half


def comparison(group):
    shown, hidden = group[group.display], group[~group.display]
    a, b = int(shown.correct.sum()), len(shown)
    c, d = int(hidden.correct.sum()), len(hidden)
    result = {
        "shown_correct": a,
        "shown_n": b,
        "not_shown_correct": c,
        "not_shown_n": d,
    }
    if b and d:
        p1, p0 = a / b, c / d
        l1, u1 = wilson(a, b)
        l0, u0 = wilson(c, d)
        diff = p1 - p0
        result.update(
            difference=diff,
            ci_low=diff - np.sqrt((p1 - l1) ** 2 + (u0 - p0) ** 2),
            ci_high=diff + np.sqrt((u1 - p1) ** 2 + (p0 - l0) ** 2),
            fisher_p=float(fisher_exact([[a, b - a], [c, d - c]]).pvalue),
        )
    return result


by_model = [
    {"model": name, **comparison(group)} for name, group in data.groupby("model")
]
by_bucket = [
    {"model": name, "bucket": bucket, **comparison(group)}
    for (name, bucket), group in data.groupby(["model", "rating_bucket"])
]
pd.DataFrame(by_model).to_csv(OUT / "chess_display_by_model.csv", index=False)
pd.DataFrame(by_bucket).to_csv(OUT / "chess_display_by_model_bucket.csv", index=False)
data[
    [
        "model",
        "puzzleId",
        "rating",
        "rating_bucket",
        "correct",
        "display",
        "display_calls",
        "received_board",
        "first_display_response_index",
    ]
].to_csv(OUT / "chess_display_observations.csv", index=False)
(OUT / "chess_display_statistics.json").write_text(json.dumps(fits, indent=2) + "\n")

order = [
    "gpt-6.1-sol",
    "claude-haiku-5-5",
    "claude-sonnet-5-5",
    "gpt-6-luna",
    "gpt-5-mini",
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
]
lookup = {r["model"]: r for r in by_model}
fig, ax = plt.subplots(figsize=(10, 5.5), layout="constrained")
for i, model in enumerate(order):
    row = lookup[model]
    color = "#206b9b" if min(row["shown_n"], row["not_shown_n"]) >= 10 else "#90969c"
    point = row["difference"] * 100
    ax.errorbar(
        point,
        i,
        xerr=[[point - row["ci_low"] * 100], [row["ci_high"] * 100 - point]],
        fmt="o",
        color=color,
        capsize=3,
        markersize=6,
    )
ax.axvline(0, color="#444444", linewidth=1, linestyle="--")
ax.set_yticks(
    range(len(order)),
    [f"{m}  ({lookup[m]['shown_n']} / {lookup[m]['not_shown_n']})" for m in order],
)
ax.invert_yaxis()
ax.set_xlim(-100, 100)
ax.set_xlabel(
    "Accuracy with display minus accuracy without display (percentage points)"
)
ax.set_title(
    "Board display: within-model associations", loc="left", weight="bold", pad=34
)
ax.text(
    0,
    1.02,
    "Labels show display / no-display sample sizes. Bars: 95% Newcombe intervals.",
    transform=ax.transAxes,
    fontsize=9,
    color="#555555",
)
ax.grid(axis="x", alpha=0.18)
ax.spines[["top", "right", "left"]].set_visible(False)
fig.savefig(OUT / "chess_display_association.png", dpi=180)
plt.close(fig)
print(json.dumps(fits, indent=2))
print("Model comparisons:", json.dumps(by_model, indent=2))
