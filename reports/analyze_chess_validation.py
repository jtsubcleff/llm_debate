"""Analyze saved chess validation use without issuing any model requests.
Run: OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=/tmp/chess-validation-mpl .venv/bin/python reports/analyze_chess_validation.py
"""
import collections
import hashlib
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
results = collections.Counter()
for path in sorted((ROOT / "exp/chess_comparison/direct_qa").glob("*/puzzles.csv")):
    frame = pd.read_csv(path, keep_default_na=False)
    frame["complete"] = frame.complete.astype(str).str.lower().eq("true")
    frame["correct"] = frame.correct.astype(str).str.lower().eq("true")
    for col in [
        "validate_move_sequence_calls",
        "display_board_unicode_calls",
        "validate_move_sequence_illegal_calls",
    ]:
        frame[col] = pd.to_numeric(frame[col])
    frame["validation"] = frame.validate_move_sequence_calls.gt(0)
    frame["display"] = frame.display_board_unicode_calls.gt(0)
    extra = []
    for row in frame.itertuples():
        trace = json.loads(row.tool_trace or "[]")
        events = [t for t in trace if "result" in t]
        validations = [
            t
            for t in events
            if t["tool_call"]["function"]["name"] == "validate_move_sequence"
        ]
        boards = [
            t
            for t in events
            if t["tool_call"]["function"]["name"] == "display_board_unicode"
        ]
        assert len(validations) == row.validate_move_sequence_calls
        assert len(boards) == row.display_board_unicode_calls
        responses = [t for t in trace if "stop_reason" in t]
        tool_rounds = sum(bool(t.get("tool_calls")) for t in responses)
        first = None
        turn = -1
        for t in trace:
            if "stop_reason" in t:
                turn += 1
            if (
                "result" in t
                and t["tool_call"]["function"]["name"] == "validate_move_sequence"
            ):
                if first is None:
                    first = turn
                r = t["result"]
                results["calls"] += 1
                results["legal"] += r.get("legal") is True
                results["illegal"] += r.get("legal") is False
                results["error"] += "error" in r
        key = hashlib.sha256(row.puzzleId.encode()).hexdigest()
        caches = sorted(
            (path.parent / "cache").glob(key + "_*.json"),
            key=lambda p: int(p.stem.rsplit("_", 1)[1]),
        )
        extra.append(
            dict(
                truncated=bool(
                    responses and responses[-1]["stop_reason"] == "MAX_TOKENS"
                ),
                total_tool_calls=len(events),
                tool_rounds=tool_rounds,
                api_responses=len(responses),
                first_validation_response_index=-1 if first is None else first,
                legal_result=any(t["result"].get("legal") is True for t in validations),
                source=str(path),
                cache=str(caches[-1]) if caches else "",
            )
        )
    frame = pd.concat([frame.reset_index(drop=True), pd.DataFrame(extra)], axis=1)
    frames.append(frame)
all_rows = pd.concat(frames, ignore_index=True)
data = all_rows[all_rows.complete].copy()
assert len(data) == 1020 and len(all_rows) == 1050
for row in pd.read_csv(ROOT / "exp/chess_comparison/results.csv").itertuples():
    group = data[data.model.eq(row.model)]
    assert len(group) == row.completed and np.isclose(group.correct.mean(), row.all)
# Sol uses validation on every attempt and supplies no within-model contrast.
overlap_models = [m for m, g in data.groupby("model") if g.validation.nunique() == 2]
contrast = data[data.model.isin(overlap_models)].copy()


def estimate(frame, controls, exposure="validation", linear=False):
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


def wilson(k, n):
    p, z = k / n, 1.96
    center = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return center - half, center + half


def comparison(group):
    shown, hidden = group[group.validation], group[~group.validation]
    a, b = int(shown.correct.sum()), len(shown)
    c, d = int(hidden.correct.sum()), len(hidden)
    result = {
        "validated_correct": a,
        "validated_n": b,
        "unvalidated_correct": c,
        "unvalidated_n": d,
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


fits = {
    "normal_finishes_overlap_diagnostic": estimate(
        contrast[~contrast.truncated], ["model", "rating_bucket"]
    ),
    "haiku_normal_finishes_diagnostic": estimate(
        data[data.model.eq("claude-haiku-5-5") & ~data.truncated], ["rating_bucket"]
    ),
    "haiku_bucket": estimate(
        data[data.model.eq("claude-haiku-5-5")], ["rating_bucket"]
    ),
    "haiku_rating": estimate(data[data.model.eq("claude-haiku-5-5")], ["rating_poly"]),
    "unadjusted_all": estimate(data, []),
    "model_all": estimate(data, ["model"]),
    "model_bucket_all": estimate(data, ["model", "rating_bucket"]),
    "model_bucket_overlap": estimate(contrast, ["model", "rating_bucket"]),
    "model_rating_overlap": estimate(contrast, ["model", "rating_poly"]),
    "model_puzzle_fixed_effects_overlap": estimate(
        contrast, ["model", "puzzleId"], linear=True
    ),
    "model_bucket_display_overlap": estimate(
        contrast, ["model", "rating_bucket", "display"]
    ),
    "successful_legal_result_overlap": estimate(
        contrast, ["model", "rating_bucket"], exposure="legal_result"
    ),
    "exclude_oss20b_overlap": estimate(
        contrast[contrast.model.ne("openai/gpt-oss-20b")], ["model", "rating_bucket"]
    ),
}
by_model = [{"model": m, **comparison(g)} for m, g in data.groupby("model")]
by_bucket = [
    {"model": m, "bucket": b, **comparison(g)}
    for (m, b), g in data.groupby(["model", "rating_bucket"])
]
maxima = []
for m, g in data.groupby("model"):
    max_total = int(g.total_tool_calls.max())
    maxima.append(
        dict(
            model=m,
            max_total_tool_calls=max_total,
            max_validation_calls=int(g.validate_move_sequence_calls.max()),
            max_display_calls=int(g.display_board_unicode_calls.max()),
            max_tool_rounds=int(g.tool_rounds.max()),
            max_api_responses=int(g.api_responses.max()),
            max_total_puzzles=",".join(g[g.total_tool_calls.eq(max_total)].puzzleId),
        )
    )
max_rows = data[data.total_tool_calls.eq(data.total_tool_calls.max())]
cols = [
    "model",
    "puzzleId",
    "rating",
    "rating_bucket",
    "correct",
    "validation",
    "validate_move_sequence_calls",
    "display_board_unicode_calls",
    "total_tool_calls",
    "tool_rounds",
    "api_responses",
    "first_validation_response_index",
    "legal_result",
    "truncated",
    "source",
    "cache",
]
data[cols].to_csv(OUT / "chess_validation_observations.csv", index=False)
pd.DataFrame(by_model).to_csv(OUT / "chess_validation_by_model.csv", index=False)
pd.DataFrame(by_bucket).to_csv(
    OUT / "chess_validation_by_model_bucket.csv", index=False
)
pd.DataFrame(maxima).to_csv(OUT / "chess_tool_call_maxima.csv", index=False)
(OUT / "chess_validation_statistics.json").write_text(json.dumps(fits, indent=2) + "\n")
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
for i, m in enumerate(order):
    row = lookup[m]
    if not row["unvalidated_n"]:
        ax.text(
            0,
            i,
            "No no-validation attempts",
            ha="center",
            va="center",
            color="#666666",
            fontsize=9,
        )
        continue
    color = (
        "#206b9b" if min(row["validated_n"], row["unvalidated_n"]) >= 10 else "#90969c"
    )
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
ax.axvline(0, color="#444444", linestyle="--", linewidth=1, zorder=0)
ax.set_yticks(
    range(len(order)),
    [
        f"{m}  ({lookup[m]['validated_n']} / {lookup[m]['unvalidated_n']})"
        for m in order
    ],
)
ax.set_ylim(len(order) - 0.6, -0.6)
ax.set_xlim(-100, 100)
ax.set_xlabel(
    "Accuracy with validation minus accuracy without validation (percentage points)"
)
ax.set_title(
    "Move validation: within-model associations", loc="left", weight="bold", pad=34
)
ax.text(
    0,
    1.02,
    "Labels show validation / no-validation sample sizes. Bars: 95% Newcombe intervals.",
    transform=ax.transAxes,
    fontsize=9,
    color="#555555",
)
ax.grid(axis="x", alpha=0.18)
ax.spines[["top", "right", "left"]].set_visible(False)
fig.savefig(OUT / "chess_validation_association.png", dpi=180)
plt.close(fig)
print("FITS", json.dumps(fits, indent=2))
print("MODELS", json.dumps(by_model, indent=2))
print("MAXIMA", json.dumps(maxima, indent=2))
print("RESULTS", results)
print(
    "FIRST VALIDATION",
    data[data.validation]
    .first_validation_response_index.value_counts()
    .sort_index()
    .to_dict(),
)
print(
    "POOLED",
    data.groupby("validation").correct.agg(["sum", "count", "mean"]).to_string(),
)
print("MAX ROWS", max_rows[cols].to_dict("records"))
print(
    "POOLED ACROSS MODELS MAX",
    data.groupby("puzzleId").total_tool_calls.sum().nlargest(3).to_dict(),
)
data.groupby("validate_move_sequence_calls").correct.agg(
    ["sum", "count", "mean"]
).to_csv(OUT / "chess_validation_by_call_count.csv")
print(
    "DOSE",
    data.groupby("validate_move_sequence_calls")
    .correct.agg(["sum", "count", "mean"])
    .to_string(),
)
