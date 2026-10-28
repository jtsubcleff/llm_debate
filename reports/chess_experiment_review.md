# Chess experiment design review

**Subsequent word-limit policy:** public initial solves now use a 300-word prompt
instruction and a 350-word maximum across all chess formats. Later debate
arguments use 150/200 words; later consultant responses use 300/350 words.
Questioner responses also use the same 300/350-word limits. Opening truncation reserves the final allowed word
for the original move token; scoring and full audit responses remain unchanged.
Limits and per-turn truncation metrics are saved, and new manifests pin the policy.
See `core/chess/word_limits.py` and the README for the current behavior.

The original turn ordering and locked-move scoring matched the intended protocols,
but the surrounding execution path did not yet provide a controlled four-condition
comparison. The changes below add that path while preserving standalone commands,
existing SAN scoring, native tool adapters, caches, and error recovery.

## Findings and fixes

| Finding | Why it mattered | Change |
| --- | --- | --- |
| Modes and proposer/questioner pairs generated independent openings with different prompts. | Judge accuracy could change because the underlying proposals changed, rather than because auditing helped. | Added `protocol=research_suite`: one neutral opening per proposer/puzzle, shared across all selected conditions and judges. |
| Single- and multi-round conditions regenerated overlapping work. | Added cost and stochastic variation obscured the round-count comparison. | Single consultancy is the opening; single debate is the exact first exchange of the longer debate. A strong debate is also shared across different final judges. |
| Consultancy/debate generation required a separate manual judging command. | Pairing the wrong questioner/judge or judging the wrong transcripts was easy. | Suite orchestrates generation, fresh final judging, and scoring. Explicit `[proposer, judge]` pairs enforce the same questioner/final-judge profile and avoid Cartesian pair expansion. Any subset of the four conditions can run end-to-end. |
| Later-role prompts inherited solver instructions by splitting a sentence in Direct QA prose. | Brittle coupling and conflicting instructions could turn the questioner back into a solver. | Explicit neutral context, optional tool descriptions, privacy policy, opening task, and role-specific continuations. The opening is identical in both modes. |
| Public tool-loop preambles remained in shared evidence; model IDs were included in headers. | Scratch commentary inflated context, and model names could bias the judge. | Share only the final public response of each agent turn. Keep role/round chronology, omit model identities, tool calls/results, native reasoning, and tagged thinking. Preserve audit data separately. |
| Completed truncations/refusals could yield a false correctness label even when a final judge was being asked to assess an unusable proposal. | A correct judge could be penalized for a generation failure. | Preserve legacy completion/scoring semantics, but add proposal/interaction/verdict validity fields. Suite counts invalid openings and avoids buying follow-ups or final judgements for them. Later truncated arguments remain flagged, with no hidden exclusion. |
| Judge summaries pooled proposers; overall accuracy could conceal a mostly-correct-proposal base rate. | An always-accepting weak judge could appear successful. | Keep the legacy overview, add `pair_results.csv`, and provide suite summaries by condition/proposer/judge with class counts, confusion counts, balanced accuracy, invalid counts, and coverage. |
| Missing results could silently alter denominators across conditions. | Comparisons on different surviving puzzles could be biased. | `evaluations.csv` contains the full planned grid, including absent results. Matched accuracy uses only the common judged puzzle IDs across selected conditions for each pair. |
| Shared prefixes/openings would duplicate reported logical costs. | Estimated spending could be overstated despite fewer actual calls. | Suite `costs.csv` counts each successful functional response cache exactly once, retains unknown-cost counts, and distinguishes actual recorded usage from per-condition logical costs. |
| Prompts were reread during turns, and manifests did not pin implementation changes. | Mid-run edits or resumed code changes could mix protocols. | Freeze prompts at stage/suite startup. Pin resolved profiles, settings, tool definitions, exact samples, source proposal hashes, transcript policy, relevant implementation hashes, and Python/library versions. Reject incompatible resumes. Normalize CSV optional values so legitimate resumes do not change hashes. |
| Documented command options were missing from the main Hydra config. | Standard `num_rounds=...` and transcript overrides could fail before execution. | Restored `num_rounds` and `transcripts_dir`, and added the suite-condition selector. |
| Duplicate YAML catalog mappings silently shadowed model roles and one provider profile. | A supposed strong/weak pairing could resolve unexpectedly. | Removed shadowed entries while preserving effective model choices and profile values. Added duplicate-key rejection, retaining YAML merge support. Use explicit model catalog keys for the experiment pairs. |

## Resulting experiment logic

1. Resolve explicit proposer/judge pairs and validate the whole suite before model
   requests. Sample once, balanced across the existing rating buckets.
2. Generate one proposal/argument per proposer/puzzle using the common opening
   prompt. Lock its selected SAN move. Refusals, truncated openings, and absent
   selections remain visible as invalid proposals rather than judge errors.
3. Branch into strong-proposer/strong-critic debate and strong-proposer/weak-judge
   consultancy. Every later agent receives the prior public arguments, original
   FEN, and locked move. There is no cross-examiner in debate.
4. Debate ends with the critic after N responses from each participant.
   Consultancy ends with the proposer after N proposer responses and N−1 questions.
   Derive single-round conditions from these unchanged prefixes. Agents are not
   told the future round count, so prefix reuse does not change their instructions.
5. Run fresh final-judge conversations on each condition, with the same judge model,
   prompt, tool access, token/tool settings, and original proposed move. The judge
   does not receive either the dataset solution or saved correctness label.
6. Checkpoint and score independently; save coverage, paired puzzle identities,
   proposal accuracy, judge accuracy, and class-conditioned metrics. Cached calls
   survive transient failures, and a failed stage remains explicit until resumed.

This review used offline tests only. To launch a complete experiment:

```bash
python -m core.chess.run protocol=research_suite \
  models='[[openrouter_gpt_oss_120b,openrouter_gpt_oss_20b]]' \
  num_rounds=3 limit=30 seed=42 exp_dir=./exp/chess_research
```

Use `suite_conditions='[single_consultancy]'`, `[multi_consultancy]`,
`[single_debate]`, or `[multi_debate]` to run just that condition end-to-end.
Use a new directory when changing the selected conditions or other pinned inputs.

## Interpretation limits retained explicitly

- **Truth definition:** scoring uses the prepared Lichess reference first move,
  canonical SAN rules, and the existing alternative-mate-in-one exception. It is
  not an engine proof of unique optimality; other equally good moves can still
  be labelled incorrect. Changing this requires a separate evaluation policy and
  reference labels, rather than an unannounced scoring change.
- **Round versus compute matching:** at N rounds, debate has 2N agent turns and
  consultancy has 2N−1. Additional auditing also adds tokens and tool opportunities.
  These conditions test the requested protocols, not an equal-compute intervention.
  Saved token/call/cost data support subsequent budget-controlled analysis.
- **Sampling versus generation randomness:** the seed fixes puzzle sampling.
  Cached responses reproduce a recorded run; regenerated API responses need not
  be identical. Mutable provider model IDs and routing can still change externally.
  Prefer dated snapshots where the provider offers them. No claim about a model's
  relative strength is inferred automatically from its name.
- **Coverage:** invalid openings are excluded from final-judge accuracy and counted
  separately. Invalid completed judge responses count as incorrect. Missing work is
  never counted as a successful judgement. Compare matched accuracy alongside
  coverage and class counts; balanced accuracy is undefined if one class is absent.
- **Cost:** unique cached responses account for recorded usage, including failed
  interactions' successful calls. Provider retries/errors without returned usage
  may incur additional costs that cannot be recovered from the current adapters.
- **Existing runs:** no saved experiment outputs were rewritten. New prompt and
  transcript policies require new experiment directories. Absolute source paths
  in the existing standalone judge manifests still tie those manifests to their
  current location; archive the full run directory and functional caches.

## Validation

Offline tests cover all four end-to-end conditions, multiple judge pairings,
identical openings, exact prefix reuse, fresh final judging, model-name removal,
private tool isolation, final-response-only sharing, planned/matched denominators,
class/confusion counts, invalid outputs, unique-cache cost accounting, transient
failure recovery, incompatible resume rejection, frozen prompt plumbing, and
duplicate-catalog-key handling. Existing Direct QA, judging, rescoring, cost,
provider-tool, redaction, and model-configuration tests are included in regression
validation. `python -m unittest discover -s tests` passed all **90 tests**.
Black/isort checks and `git diff --check` passed. Hydra configuration rendering
also succeeded for the full suite and a single-condition command, without
launching model requests. No live experiment or paid API request was made by
this review.
