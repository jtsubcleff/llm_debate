**Move validation, performance, and maximum tool calls**

Validation has a more positive observed association with correctness than board display, but the adjusted overall estimate remains inconclusive. Much of the raw association is tied to whether an attempt reaches an answer before exhausting its token budget. The largest recorded individual attempt uses **11 tool calls**: **9 validations and 2 board displays**, by GPT-5-Mini on puzzle **00IpT**.

**Scope and measurement**

The source is the same `exp/chess_comparison` run: seven models, 150 assigned puzzles each, and 1,020 completed attempts. Thirty incomplete GPT-OSS-20B rows are excluded from performance comparisons. Token-limit exits count as completed but incorrect, matching the original scorer.

“Validation used” means one or more invocations of `validate_move_sequence`; it does not require the tool to accept the sequence. An illegal result can be useful feedback. The 1,131 recorded validation calls comprise **820 legal results, 306 illegal results, and 5 argument/tool errors**. The script verifies call counters against actual tool-result events and checks correctness aggregates against the saved original results.

All estimates concern the saved strict-SAN outcome, not an engine's evaluation of alternative moves. These are self-selected tool uses, not randomized treatments. No model requests or source experiment changes were made.

**Accuracy with and without validation**

| Model | With validation | Without validation | Difference, percentage points |
|---|---:|---:|---:|
| GPT-6.1-Sol | 108/150 = 72.0% | No observations | Not estimable |
| Claude Haiku 5.5 | 45/76 = 59.2% | 28/74 = 37.8% | +21.4 |
| Claude Sonnet 5.5 | 4/7 = 57.1% | 57/143 = 39.9% | +17.3 |
| GPT-6-Luna | 36/141 = 25.5% | 2/9 = 22.2% | +3.3 |
| GPT-5-Mini | 25/139 = 18.0% | 4/11 = 36.4% | −18.4 |
| GPT-OSS-120B | 12/37 = 32.4% | 38/113 = 33.6% | −1.2 |
| GPT-OSS-20B | 1/4 = 25.0% | 9/116 = 7.8% | +17.2 |
| **All completed attempts** | **231/554 = 41.7%** | **138/466 = 29.6%** | **+12.1** |

The raw pooled association is positive. Model composition matters: Sol is strong and validates every puzzle, so it boosts the pooled validation group without supplying any validation/no-validation contrast of its own. Sonnet, Luna, and GPT-OSS-20B have fewer than ten observations in one group. Mini has only eleven no-validation observations.

![Within-model validation associations](/Users/jacksonsubcleff/llm_debate/reports/chess_validation_association.png)

Bars are approximate 95% Newcombe intervals. Gray indicates fewer than ten observations in one group; blue indicates at least ten on both sides. Sol has no comparison interval. The intervals are not corrected for simultaneous comparisons.

**Adjusted comparison**

The primary adjusted fit uses the **six models with both validation and no-validation observations**, totaling **870 attempts**. Sol is excluded from that fit because there is no within-model contrast. The model is logistic regression with a common validation coefficient, categorical model effects, and categorical difficulty-bucket effects. Standard errors are clustered by puzzle ID because models share the same puzzles. Risk differences standardize the fitted probabilities over the comparison sample; they depend on model assumptions and extrapolate where overlap is sparse.

| Analysis | Estimated association | Approximate 95% interval | Two-sided p |
|---|---:|---:|---:|
| Six models, model + bucket adjustment | Odds ratio **1.38** | **0.88–2.16** | **0.157** |
| Same fit, standardized accuracy difference | **+5.8 pp** | **−2.2 to +13.7 pp** | — |
| Numeric rating and rating squared instead of bucket | Odds ratio 1.38 | 0.88–2.18 | 0.161 |
| Model + exact-puzzle fixed effects, linear probability model | +5.5 pp | −4.7 to +15.7 pp | 0.287 |
| Also adjusting for board-display invocation | Odds ratio 1.54 | 0.97–2.45 | 0.070 |
| Excluding GPT-OSS-20B as well | Odds ratio 1.34 | 0.85–2.12 | 0.203 |

Including Sol in the logistic model-plus-bucket fit gives a similar odds ratio, **1.37** (0.86–2.16). The primary result is positive in direction, but uncertainty includes no association and modest negative association. These results do not establish a general benefit or harm.

Board display is another self-selected behavior and can occur before or after validation; controlling it is a sensitivity check, not a complete causal adjustment. Puzzle fixed effects address shared position difficulty but cannot remove model-specific reasons for invoking a tool.

**Haiku is the strongest positive pattern, with a completion caveat**

Haiku has a reasonably balanced comparison, and its validation association is positive in all three buckets:

| Bucket | With validation | Without validation |
|---|---:|---:|
| Low | 22/28 = 78.6% | 12/22 = 54.5% |
| Medium | 14/26 = 53.8% | 8/24 = 33.3% |
| High | 9/22 = 40.9% | 8/28 = 28.6% |

Haiku's raw Fisher exact p = **0.0096**. Adjusting for bucket gives odds ratio **2.30**, interval **1.15–4.59**, p = **0.018**; numeric-rating adjustment gives a similar result. These are exploratory model-specific tests. A Bonferroni correction over the six raw model comparisons gives p ≈ **0.058** for Haiku, so the apparent subgroup evidence should not be treated as a confirmed discovery.

The completion pattern substantially changes the interpretation:

| Group | Token-limit exits / attempts |
|---|---:|
| All models, with validation | **12/554 = 2.2%** |
| All models, without validation | **161/466 = 34.5%** |
| Haiku, with validation | **3/76 = 3.9%** |
| Haiku, without validation | **23/74 = 31.1%** |

Among Haiku attempts that finish normally, accuracy is **45/73 = 61.6%** with validation versus **28/51 = 54.9%** without, a much smaller **+6.7-point** raw difference. Its bucket-adjusted normal-finish-only estimate is also uncertain: odds ratio **1.25**, interval **0.58–2.70**. Across the six comparison models, normal finishes alone give adjusted odds ratio **0.96**, interval **0.59–1.54**.

These normal-finish comparisons are diagnostics, not replacements for the primary analysis: excluding truncations conditions on an outcome that tool use could itself affect. They show that the observed positive association is closely connected to completion. Validation might help models progress or reach a fresh API-call budget, or an attempt that reaches a tool call might already be on a more productive path. Saved observational traces do not distinguish those explanations. A positive association should not automatically be interpreted as improved chess calculation conditional on finishing.

**Timing and useful versus unproductive checking**

Validation first occurs in the initial model response on **249/554** validation attempts, the second response on **287**, and the third on **18**. A second-response invocation frequently follows a board-display call, so later validation does not by itself establish that the model was stuck. The traces do not reveal a randomized choice or the model's complete private reasoning.

An accepted sequence is not an evaluation: it says the moves are legal, not that they are best or forced. A separate exploratory association for receiving at least one legal validation result gives odds ratio **1.63**, interval **1.07–2.51** after model/bucket adjustment in the six-model sample. That changes the exposure to successful candidate generation, which itself reflects chess ability; it is not evidence that a legal tool result causes improvement.

More validation is not monotonically associated with better raw results: one-call attempts score **146/298 = 49.0%**, two-call attempts **48/124 = 38.7%**, and all four nine-call attempts fail. These pooled dose comparisons mix models and difficulty, so they should not be interpreted as evidence that additional calls cause harm.

**Maximum recorded calls on a single model–puzzle attempt**

The overall maximum is **GPT-5-Mini on 00IpT**, rating 2130 (Medium):

- **11 total tool invocations:** 9 `validate_move_sequence` and 2 `display_board_unicode`.
- **8 tool rounds**, followed by a final answer, for **9 model API responses** in total.
- The attempt is scored **incorrect**.

[Source puzzle CSV with the full tool trace](/Users/jacksonsubcleff/llm_debate/exp/chess_comparison/direct_qa/gpt-5-mini_f6a6fd96/puzzles.csv). [Final cached response](/Users/jacksonsubcleff/llm_debate/exp/chess_comparison/direct_qa/gpt-5-mini_f6a6fd96/cache/8efcfe0c3399663693d4ca2a97cfd47721c1c87d881b49886837e83399e1ec2f_8.json).

| Model | Maximum total calls | Maximum validation calls | Maximum board calls |
|---|---:|---:|---:|
| GPT-6.1-Sol | 6 | 6 | 1 |
| Claude Haiku 5.5 | 9 | 8 | 2 |
| Claude Sonnet 5.5 | 4 | 3 | 1 |
| GPT-6-Luna | 6 | 5 | 2 |
| GPT-5-Mini | **11** | **9** | **5** |
| GPT-OSS-120B | 4 | 3 | 3 |
| GPT-OSS-20B | 1 | 1 | 1 |

The three maxima in a row need not occur on the same puzzle; do not add them. The maximum validation count, **9**, occurs on four Mini puzzles: `00IpT`, `03q1Y`, `01LO7`, and `00DKr`. The maximum board count, **5**, occurs on Mini's `00UB0`.

The configured cap is **10 tool rounds**, not 10 individual invocations. Multiple calls can be issued in a round, explaining the 11-call attempt. The largest observed number of rounds is **10** and the largest number of API responses is **11**, consistent with ten tool-producing responses plus a final response with tools disabled. Thirteen attempts reach ten rounds, all Mini; two are correct. [Tool-loop implementation](/Users/jacksonsubcleff/llm_debate/core/chess/protocols/direct_qa.py).

If “individual puzzle” instead means adding calls across all seven models, the largest recorded combined count is **25**, also on `00IpT`. The main maximum of 11 refers to one model's attempt, which is the relevant unit for the per-puzzle tool budget. Counts include recorded executed tool calls returning errors; they exclude model API calls themselves. Incomplete rows contain no executed tool traces; their two saved partial responses request board display only and do not change any observed maximum.

**Artifacts and reproduction**

- [Analysis script](/Users/jacksonsubcleff/llm_debate/reports/analyze_chess_validation.py)
- [Completed observations, call counts, timing, truncation, and source paths](/Users/jacksonsubcleff/llm_debate/reports/chess_validation_observations.csv)
- [Per-model comparisons](/Users/jacksonsubcleff/llm_debate/reports/chess_validation_by_model.csv)
- [Per-model, per-bucket comparisons](/Users/jacksonsubcleff/llm_debate/reports/chess_validation_by_model_bucket.csv)
- [Accuracy by validation-call count](/Users/jacksonsubcleff/llm_debate/reports/chess_validation_by_call_count.csv)
- [Adjusted statistics and diagnostic fits](/Users/jacksonsubcleff/llm_debate/reports/chess_validation_statistics.json)
- [Maximum counts by model](/Users/jacksonsubcleff/llm_debate/reports/chess_tool_call_maxima.csv)

Run from the repository root:

```sh
OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=/tmp/chess-validation-mpl .venv/bin/python reports/analyze_chess_validation.py
```

The script regenerates tables, statistics, and the figure. The statistical estimates assume a common validation association in the pooled adjusted fits; confidence intervals and p-values are approximate. Source experiment artifacts remain unchanged.
