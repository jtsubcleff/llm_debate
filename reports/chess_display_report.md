**Is board display positively associated with chess-puzzle performance?**

These results do not provide convincing evidence of a positive overall association. The pooled association is negative, but most of that difference disappears after accounting for model and puzzle difficulty. The remaining estimate is uncertain enough to allow either benefit or harm. This analysis also cannot establish that displaying the board causes a performance change.

**Data and definitions**

The source is `exp/chess_comparison`, the same run used in the failure report. There are 1,050 assigned model–puzzle pairs and 1,020 completed attempts. The 30 incomplete GPT-OSS-20B attempts are excluded; their eventual tool use and outcomes are unknown. Completed attempts include token-limit exits, scored incorrect under the experiment's existing rule. The outcome is saved strict-SAN correctness, not an engine assessment of every alternative move.

“Display” means at least one invocation of `display_board_unicode` in that attempt, rather than the total number of invocations. There are 543 display calls across 478 completed attempts. Although 23 individual calls return errors, retries usually recover: 476 of those attempts receive at least one actual board, and only two display attempts (both Mini failures) receive none. A sensitivity analysis uses actual board receipt instead of invocation.

Exposure and correctness were checked against the saved per-puzzle CSVs and tool traces, and all seven model accuracy aggregates match the original results. No model calls or experiment changes were made.

**Observed accuracy**

| Model | With display | Without display | Difference, percentage points |
|---|---:|---:|---:|
| GPT-6.1-Sol | 7/9 = 77.8% | 101/141 = 71.6% | +6.1 |
| Claude Haiku 5.5 | 15/37 = 40.5% | 58/113 = 51.3% | −10.8 |
| Claude Sonnet 5.5 | 1/5 = 20.0% | 60/145 = 41.4% | −21.4 |
| GPT-6-Luna | 32/127 = 25.2% | 6/23 = 26.1% | −0.9 |
| GPT-5-Mini | 29/149 = 19.5% | 0/1 = 0.0% | +19.5 |
| GPT-OSS-120B | 48/145 = 33.1% | 2/5 = 40.0% | −6.9 |
| GPT-OSS-20B | 0/6 = 0.0% | 10/114 = 8.8% | −8.8 |
| **All completed attempts** | **132/478 = 27.6%** | **237/542 = 43.7%** | **−16.1** |

The pooled gap is not a fair within-model comparison. Sol, the strongest model in this run, displays a board on only 9/150 attempts. Mini displays on 149/150, and GPT-OSS-120B on 145/150. These differences in model composition confound the overall comparison. Conversely, GPT-OSS-20B is weak and rarely displays, so the composition story is not simply “weaker models always display more.”

Mini has only one no-display observation; Sonnet, Sol, and both GPT-OSS models have only 5–9 observations in one of the two groups. Haiku and Luna are the only models with at least ten observations on both sides. All seven individual model differences have 95% intervals spanning zero; none of the within-model two-sided Fisher exact tests is significant (smallest p = 0.264). This does not establish equivalence, because several comparisons are extremely imprecise.

![Within-model accuracy differences and uncertainty](/Users/jacksonsubcleff/llm_debate/reports/chess_display_association.png)

Bars are approximate 95% Newcombe intervals for independent binomial proportion differences within each model. Gray denotes fewer than ten observations in at least one group; blue denotes at least ten in each group. The intervals are not adjusted for simultaneous comparisons.

**Adjusted association**

The primary adjusted analysis fits logistic regression with a board-display indicator, categorical model effects, and categorical difficulty-bucket effects. Uncertainty uses a sandwich covariance clustered by puzzle ID, because different models solve the same puzzles. The model assumes a common display association across models and buckets; sparse overlap limits how well the data can check this assumption.

| Adjustment | Odds ratio for display | Approximate 95% CI | Two-sided p |
|---|---:|---:|---:|
| None | 0.49 | 0.38–0.63 | <0.001 |
| Model | 0.75 | 0.47–1.22 | 0.254 |
| Model + difficulty bucket | **0.85** | **0.51–1.44** | **0.556** |
| Model + numeric rating and rating squared | 0.82 | 0.48–1.41 | 0.482 |

An odds ratio above one would indicate a positive adjusted association. The main estimate is below one, but its interval includes one and substantial associations in either direction. Standardizing the model-plus-bucket fit over the observed covariates gives an estimated accuracy difference of **−2.8 percentage points**, with an approximate 95% interval of **−12.3 to +6.7 points**. This is a model-based adjusted association, not an experimentally measured treatment effect; standardization extrapolates especially for models that almost always or almost never display.

A second analysis controls for each exact puzzle's identity, not just its rating, using a linear probability model with model and puzzle fixed effects and puzzle-clustered uncertainty. It estimates **−3.5 points**, interval **−15.3 to +8.4**, p = 0.567. This similarly provides no clear positive or negative association after adjustment. It controls shared puzzle characteristics but cannot remove model-specific reasons for choosing the tool.

**Sensitivity checks and a subgroup worth following up**

- Requiring receipt of a successful board, rather than merely invoking the tool: adjusted odds ratio **0.87**, interval **0.52–1.45**, p = 0.598. The conclusion is unchanged.
- Excluding GPT-OSS-20B, whose run is incomplete and dominated by truncation: **0.89**, interval **0.52–1.54**, p = 0.682.
- Restricting to Haiku and Luna, the two models with at least ten display and ten no-display attempts: **0.80**, interval **0.42–1.54**, p = 0.505. This check changes the target population to those two models.
- Sol on **High** puzzles is the most interesting positive subgroup: **5/6 = 83.3%** with display versus **15/44 = 34.1%** without it, a **+49.2-point** raw difference. Two-sided Fisher p = 0.0317 before adjustment for searching subgroups. There are 17 model–bucket strata with both exposure groups; a Bonferroni correction over those exploratory tests gives p ≈ 0.54. Six displayed puzzles are too few to establish a reproducible benefit. On Sol's Medium puzzles, the only displayed attempt is incorrect; its two displayed Low puzzles are correct.

**What can and cannot be inferred**

The data do not support a broad claim that board display is positively correlated with performance after accounting for obvious confounders. They also do not establish that it hurts. Tool use was self-selected, model identity is strongly related to tool use, and the groups are often small or unbalanced.

In 472 of the 478 display attempts, the first display request occurs in the first model response; only six first invoke it later. Thus a story in which models usually fail several tool rounds and only then request a board does not fit these traces. However, choosing display in the first response can still reflect position-specific uncertainty or private reasoning, so selection bias remains. The fitted association is not isolated from other tool-use habits or unobserved model–puzzle interactions.

Dropping token-limit exits would condition on a downstream outcome that might itself be affected by tool use. The primary analysis therefore retains them as failures, consistent with the saved scores. Likewise, incomplete attempts are not silently counted as no-display failures.

A direct test of benefit would randomize otherwise identical attempts to receive a board or not, within each model and puzzle, keeping reasoning effort, token limits, prompts, temperature, and other tools consistent. Sol's High subgroup is a useful hypothesis for that experiment, alongside models with currently limited overlap. The present results alone do not justify enabling or removing the tool on performance grounds.

**Artifacts and reproduction**

- [Analysis script](/Users/jacksonsubcleff/llm_debate/reports/analyze_chess_display.py)
- [All 1,020 completed observations](/Users/jacksonsubcleff/llm_debate/reports/chess_display_observations.csv)
- [Per-model counts, intervals, and Fisher tests](/Users/jacksonsubcleff/llm_debate/reports/chess_display_by_model.csv)
- [Per-model, per-bucket comparisons](/Users/jacksonsubcleff/llm_debate/reports/chess_display_by_model_bucket.csv)
- [Adjusted estimates and sensitivity results](/Users/jacksonsubcleff/llm_debate/reports/chess_display_statistics.json)
- [Original aggregate](/Users/jacksonsubcleff/llm_debate/exp/chess_comparison/results.csv)

Run from the repository root:

```sh
OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=/tmp/chess-display-mpl .venv/bin/python reports/analyze_chess_display.py
```

The script rebuilds the observation tables, statistical estimates, and figure. Confidence intervals and p-values are approximate; no population effect or causal benefit is established by these observational comparisons.
