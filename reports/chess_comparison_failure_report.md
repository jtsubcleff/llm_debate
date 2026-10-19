**Chess comparison: model failure analysis**

Analyzed `exp/chess_comparison` (singular), the closest match to the requested `exp/chess_comparisons`. This report does not combine the separate `chess_comparison2` or `chess_comparison2yestools` runs. Model names are the identifiers recorded in these artifacts.

**Scope and method**

All seven models were assigned the same 150 puzzles: 50 Low (<1300), 50 Medium (1300–2300), and 50 High (>2300), seed 42. Tools were enabled, with up to 10 tool rounds, an 8,192-token budget per API call, and configured temperature 0.2. Provider profiles differ; these are outcomes under the saved configuration, not a controlled comparison of equal reasoning effort. No new model calls or engine evaluations were made.

Counts below cover every saved puzzle row and its tool trace. I replayed final moves and the cited board claims with python-chess. The evaluator scores the final whitespace-separated token against canonical SAN, after removing matching Markdown wrappers; legal alternative mates in one are accepted. Truncation counts as a completed but incorrect attempt. Missing results are excluded from the saved aggregate.

“Different legal move” means the parsed move differs from the reference first move; it does not establish an engine evaluation or prove every alternative is losing. “Same move, notation error” identifies the reference piece movement after permissive SAN parsing and, where needed, removing move numbers/punctuation. This includes substantive check/mate and capture-notation mistakes, not just cosmetic formatting. No claim is made that all such responses show correct chess understanding.

Failure categories are exclusive: incomplete first, then truncation, then move/notation classification. A truncated response is counted under truncation regardless of any unfinished text.

**Census**

| Model | Correct / completed | Different legal move¹ | Same move, notation error | Illegal or unparseable final | Token limit | No completed result |
|---|---:|---:|---:|---:|---:|---:|
| claude-haiku-5-5 | 73/150 (48.7%) | 48 | 1 | 2 | 26 | 0 |
| claude-sonnet-5-5 | 61/150 (40.7%) | 62 | 2 | 11 | 14 | 0 |
| gpt-5-mini | 29/150 (19.3%) | 73 | 12 | 36 | 0 | 0 |
| gpt-6-luna | 38/150 (25.3%) | 97 | 12 | 3 | 0 | 0 |
| gpt-6.1-sol | 108/150 (72.0%) | 41 | 0 | 1 | 0 | 0 |
| openai/gpt-oss-120b | 50/150 (33.3%) | 53 | 5 | 9 | 33 | 0 |
| openai/gpt-oss-20b | 10/120 (8.3%) | 5 | 2 | 3 | 100 | 30 |

¹ Includes eight Mini answers with a different legal move plus extra formatting. Illegal/unparseable finals include prose instead of a move, ambiguous SAN, and genuinely illegal moves.

GPT-OSS-20B’s saved 8.3% is 10/120 completed attempts. Relative to all 150 assigned puzzles it has 10 recorded successes (6.7%); 30 rows have no completed result, empty errors and no CSV response trace. Two of these have a cached partial response requesting a tool; the other 28 have no cached response. These artifacts do not establish why completion is missing, so the rows are not assigned an API-failure or chess-failure cause.

**Model findings and examples**

**claude-haiku-5-5: legal tactical misjudgments, followed by truncation.** Of 77 completed failures, 48 select a different legal move (62.3%) and 26 hit the token limit (33.8%). Only two final moves are illegal and one uses the right move with wrong notation. Validation was used on 76/150 puzzles. The explanation examples show missed immediate tactics and mistaken attack/defense analysis.

- [00kT1](/Users/jacksonsubcleff/llm_debate/exp/chess_comparison/direct_qa/claude-haiku-5-5_26f6cf65/cache/f190aa013ee336bd5e331cac0039fc3a97c9ed8a8f43f82952f8d3bb6d047a50_1.json) (rating 924): Returns `Qxe3`, describing a free knight and rejecting `Qh7+` because “nothing defends h7.” The bishop on c2 actually protects h7 along c2–d3–e4–f5–g6–h7 once the queen leaves d3. `Qh7#` is immediate mate. Validating `Qxe3` established legality, not superiority.
- [00rB5](/Users/jacksonsubcleff/llm_debate/exp/chess_comparison/direct_qa/claude-haiku-5-5_26f6cf65/cache/257985690a557bcaa068183534853a77af779b1e8458c046b2b281453a4c01f3_0.json) (rating 1616): Returns `Rxe3` to win the queen, overlooking the e1-rook’s pin to Kh1 by the c1-rook. Moving it exposes check, so the final move is illegal. The reference starts `Qxc8+`.
- [00GuD](/Users/jacksonsubcleff/llm_debate/exp/chess_comparison/direct_qa/claude-haiku-5-5_26f6cf65/cache/466b88b75ec735d9572219eabfa76b6ec5c81c0a169f2957303f9e3cfcd5ab33_2.json) (rating 1152): Finds and validates the correct mating sequence `Rxh6+ gxh6 g7+ Kh7 g8=Q#`, but ends with `g8=Q#`. The task asks for the first move; the final promotion is illegal from the original FEN. This is an answer-selection failure despite solving the line.
- [019wq](/Users/jacksonsubcleff/llm_debate/exp/chess_comparison/direct_qa/claude-haiku-5-5_26f6cf65/cache/562499b086b9fb184572a98e0a9168d0f89c53b24ba083aa834234579c2b9710_0.json) (rating 2110): Ends `MAX_TOKENS` at 8,192 output tokens with no final answer; reference `Nxg6`.

**claude-sonnet-5-5: tactical misjudgments with very little tool verification.** Of 89 failures, 62 select a different legal move (69.7%), 14 truncate (15.7%), and 11 give illegal final moves (12.4%). Only 7/150 puzzles invoke the validator (9 calls total). Lack of checking is an observed association, not evidence that adding tool calls alone would fix the errors.

- [00GuD](/Users/jacksonsubcleff/llm_debate/exp/chess_comparison/direct_qa/claude-sonnet-5-5_c13e6668/cache/466b88b75ec735d9572219eabfa76b6ec5c81c0a169f2957303f9e3cfcd5ab33_0.json) (rating 1152): Returns `Qxe2` to take a rook, but its queen on g2 is blocked by its own pawn on f2. This is an illegal move caused by missed board occupancy; reference `Rxh6+`.
- [04XTY](/Users/jacksonsubcleff/llm_debate/exp/chess_comparison/direct_qa/claude-sonnet-5-5_c13e6668/cache/dfe6bc2e93c567ad107fbe82b0cb8bfc8db0d6c55ee6d27f82784e2bdd6fb2fa_0.json) (rating 2328): Returns `Rxg7`, overlooking White’s pawn on g5 blocking the rook’s path from g3. The final move is illegal; reference `Qg4+`. The response explicitly says it did not use the tools.
- [00Dib](/Users/jacksonsubcleff/llm_debate/exp/chess_comparison/direct_qa/claude-sonnet-5-5_c13e6668/cache/e8dd33ab68873081c5982548531c1894a6e32abcce951ae4d46c46dbb32f6d6b_0.json) (rating 1698): Returns `Rh6+` while White’s king on f6 is already in check from Bh4 along h4–g5–f6. Moving the rook does not answer that check. The reference `g5` blocks the diagonal.
- [008LD](/Users/jacksonsubcleff/llm_debate/exp/chess_comparison/direct_qa/claude-sonnet-5-5_c13e6668/cache/2337972d9aff1c2603223eed44dd155b8f9701afc121fe122dd96707fae71630_0.json) (rating 403): Returns `Rf1+` rather than `Rf1#` and invents a continuation with `Re1` as a blocking move. `Rf1` is already mate; there is no legal blocking reply.

**gpt-5-mini: wrong choices, failure to obey the final-answer contract, and failure to act on negative tool feedback.** Of 121 failures, 73 choose a different legal move (including 8 with formatting errors), 36 have illegal/unparseable finals, and 12 identify the reference move but use incorrect notation. Of the 36, 30 are invalid move syntax and 6 are illegal chess moves. All 150 attempts finish normally; truncation is not the issue. Mini makes 494 validation calls, including 173 rejected sequences, across 139 puzzles. Rejected exploratory lines are not inherently failures; the important issue is whether feedback changes the answer.

- [00Dib](/Users/jacksonsubcleff/llm_debate/exp/chess_comparison/direct_qa/gpt-5-mini_f6a6fd96/cache/e8dd33ab68873081c5982548531c1894a6e32abcce951ae4d46c46dbb32f6d6b_6.json) (rating 1698): The validator rejects `Rh6+`, its UCI equivalent `g6h6`, and the proposed pawn push `e7`; nevertheless the final answer is `Rh6+`. White is in check from Bh4, and the rook move leaves that check unresolved. This directly demonstrates ignored tool feedback.
- [00Rlv](/Users/jacksonsubcleff/llm_debate/exp/chess_comparison/direct_qa/gpt-5-mini_f6a6fd96/cache/3623079eaa64f56556607aed4229687d8af473323a6c1c968c887d51694e32b5_1.json) (rating 612): Opens with `15.Rhe1`, follows with an explanation, and ends with `king.`. The extracted answer is therefore prose, not SAN. It also misplaces White’s rook on c1 (the FEN has the king on c1 and rook on d1). Reference `Qxd8+`.
- [00FHX](/Users/jacksonsubcleff/llm_debate/exp/chess_comparison/direct_qa/gpt-5-mini_f6a6fd96/cache/11ba86c310b70727f288a66755dec025b774024a253b43a844d0b4325e798383_4.json) (rating 473): Returns `Rxc8+` instead of `Rxc8#` and says Black’s king must move, suggesting `...Kh7`. The move is mate: h7 is occupied by Black’s pawn, and no legal reply exists. This is a mate-recognition mistake as well as a notation failure.

**gpt-6-luna: legal but unproven tactical lines and inaccurate SAN/mate claims.** Of 112 failures, 97 choose a different legal move (86.6%), 12 have the reference piece movement with wrong notation (10.7%), and 3 have unparseable finals. The latter are two ambiguous SAN answers and one prose ending; none is classified as an outright illegal final movement. Validation is common (141/150 puzzles), but a legal sample continuation does not establish best play.

- [01EFz](/Users/jacksonsubcleff/llm_debate/exp/chess_comparison/direct_qa/gpt-6-luna_9a7e60cd/cache/714a82afb93a910bc5dcf84211a5ee05ad8077a445b9331928d380dd21e1f9ae_3.json) (rating 761): Chooses `Nxf7` and validates a multi-move attacking line, overlooking immediate `Qh7#`. This is an example of searching a plausible attack while missing a simpler decisive move.
- [00FAe](/Users/jacksonsubcleff/llm_debate/exp/chess_comparison/direct_qa/gpt-6-luna_9a7e60cd/cache/177f67db4a1dba2e6c61644a3b9822f2758339d78c63a0cb6b4a15878c9cabef_4.json) (rating 1621): Tests several illegal king escapes after `Qe1+`, then concludes `Qe1#`. White can block with `Rf1`, which the model never tests. The validator accepts the move text with `#` because it checks move legality, not the truth of that suffix; canonical SAN is `Qe1+`.
- [00rB5](/Users/jacksonsubcleff/llm_debate/exp/chess_comparison/direct_qa/gpt-6-luna_9a7e60cd/cache/257985690a557bcaa068183534853a77af779b1e8458c046b2b281453a4c01f3_5.json) (rating 1616): Correctly recognizes that `Rxe3` is pinned, then returns `Rxc1` even after that SAN is rejected. Both white rooks can capture c1; the trace eventually validates `b1c1`, whose canonical SAN is `Rbxc1`. The final answer remains ambiguous, and it also differs from reference `Qxc8+`.

**gpt-6.1-sol: remaining failures are chiefly choice/search quality, concentrated in harder puzzles.** Of 42 failures, 41 select a different legal move (97.6%) and one is illegal. There are no truncations or reference-move notation failures. All 150 puzzles use validation; 37 of the 42 failures contain an accepted legal line beginning with the final candidate. This measures legal-line coverage, not soundness against best defense. High puzzles contribute 30 failures, Medium 11, Low 1. It correctly solves all 22 reference mates in one.

- [04zCL](/Users/jacksonsubcleff/llm_debate/exp/chess_comparison/direct_qa/gpt-6.1-sol_dedfefd5/cache/a2f2308fdcaace8dff0b307cc3cadebc2e2797f931f55af633b7a63307ce97a1_1.json) (rating 2322): Chooses `Be4` rather than `Bg4` and supplies a validator-approved line ending in mate after `...Be4 Qf4 Bxd5 Bxd5 Qe2 Bf3 Qe1+ Rxe1 Rxe1#`. The line is legal and does end in mate, but it covers one chosen sequence of White replies rather than proving the first move best. This illustrates the dominant pattern of a legal, persuasive continuation paired with the wrong reference choice.
- [01we5](/Users/jacksonsubcleff/llm_debate/exp/chess_comparison/direct_qa/gpt-6.1-sol_dedfefd5/cache/9066c7ec6c1c1ef168192a8858ad132726f03cd3fe9d0a8d9c6fb7c25793dd5e_1.json) (rating 2414): Chooses `h2` rather than reference `a3` and describes a promotion race. The tool checked `h2 Rh1 Rg1 Rxh2 Rxh1 c6 Rc1`, but the explanation switches to `h2 Rh1 a3 c6 a2 c7 Rg1`. Its presented justification is not the line it validated.
- [04QLf](/Users/jacksonsubcleff/llm_debate/exp/chess_comparison/direct_qa/gpt-6.1-sol_dedfefd5/cache/a9a9e6ecc37684e8f35b67b2a979b302737896d41a0c7d46d7ea5955287041b7_1.json) (rating 2595): Returns `Qd3`, attempting to pin the queen to the king, while White is already in check from Nf3. That knight check must be answered; `Qd3` is illegal. The trace had tried the correct `Rxf3`, but the tool rejected its proposed follow-up `...Qxf3`; that rejection does not invalidate the first move. Reference `Rxf3` is legal.

**openai/gpt-oss-120b: wrong legal moves plus a substantial token-budget bottleneck.** Of 100 failures, 53 select a different legal move, 33 truncate, 9 give illegal moves, and 5 use the reference move with wrong notation. Every truncation ends at 8,192 output tokens with an empty final answer. The model makes 150 board-display calls but validates only 37/150 puzzles; displaying a board does not ensure the final move is checked. Saved public answers are often only a move, limiting diagnosis of internal reasoning.

- [01LO7](/Users/jacksonsubcleff/llm_debate/exp/chess_comparison/direct_qa/openai_gpt-oss-120b_298e8d39/cache/ec2d01d632674fbfcdb5d62ae8b3c82e2d778740c8855042e0c83820702d3778_1.json) (rating 1117): Returns `Qg7#`. The move is legal but only check: Black can play `Bxg7`. This is a false mate claim, and the candidate was not validated; reference `Nf6+`.
- [015pp](/Users/jacksonsubcleff/llm_debate/exp/chess_comparison/direct_qa/openai_gpt-oss-120b_298e8d39/cache/a41510643464e6fe820542d5509667babb32cd646076ba0fbda73f4cdb5214c1_2.json) (rating 1600): Returns `Kg3`, moving into attack from both the rook on a3 and king on e3. The tool validated `Rxf4`, not the final king move. Reference `Rh3+`.
- [019wq](/Users/jacksonsubcleff/llm_debate/exp/chess_comparison/direct_qa/openai_gpt-oss-120b_298e8d39/cache/562499b086b9fb184572a98e0a9168d0f89c53b24ba083aa834234579c2b9710_1.json) (rating 2110): Displays the board, then reaches `MAX_TOKENS` at 8,192 output tokens with no final answer; reference `Nxg6`.

**openai/gpt-oss-20b: inability to produce a final answer within the configured budget dominates.** Of 120 completed attempts, 100 truncate at 8,192 output tokens with empty final answers (83.3%). Those account for 90.9% of its 110 completed failures. Only 20 completed attempts finish normally: 10 correct, 5 different legal moves, 3 illegal finals, and 2 reference moves with incorrect notation. Another 30 assigned puzzles have no completed result. The 50% success rate among the 20 normal finishes is highly selected and should not be treated as an unbiased accuracy estimate.

- [02im6](/Users/jacksonsubcleff/llm_debate/exp/chess_comparison/direct_qa/openai_gpt-oss-20b_a741d960/cache/efef9b0f2b4eb2710cc80ebf7e9b140499cd77c17cfb24578099baf48add0c12_0.json) (rating 2356): Ends `MAX_TOKENS` at 8,192 output tokens without a final answer, representative of the dominant failure; reference `Bxf6`.
- [011ON](/Users/jacksonsubcleff/llm_debate/exp/chess_comparison/direct_qa/openai_gpt-oss-20b_a741d960/cache/3b4481c1d45f4fa204f20324fecf994f9efa5a949b845529ed1b443ab6dd8307_0.json) (rating 794): Returns `Rxb7` although Black is to move and has no rook. The only rook is White’s rook on b1. This is a concrete side-to-move / piece-ownership error; reference `Nc3+`.
- [04zCL](/Users/jacksonsubcleff/llm_debate/exp/chess_comparison/direct_qa/openai_gpt-oss-20b_a741d960/cache/a2f2308fdcaace8dff0b307cc3cadebc2e2797f931f55af633b7a63307ce97a1_0.json) (rating 2322): Returns `Nxb6` although Black has no knight in the starting position. This is another impossible piece choice; reference `Bg4`.

**Implications for evaluation**

- Preserve strict SAN accuracy, but report intended-move agreement separately. Mini and Luna each have 12 failures where the intended piece movement matches the reference; several still reveal real misunderstanding of check, mate, or captures. Their scores should not simply be “corrected” without distinguishing these cases.
- Treat token exhaustion and missing results as separate outcomes. GPT-OSS-20B primarily tests whether this configuration can finish within budget; a new budget/effort configuration requires a separate run. Haiku, Sonnet and GPT-OSS-120B are also affected by truncation.
- Distinguish move legality from tactical correctness. Tool acceptance does not evaluate a move, quantify an advantage, prove a line forced, or validate a `#` claim. The Luna false-mate example makes this limitation explicit.
- For Sol, the largest remaining opportunity in this sample is broader candidate and defensive-response search. For Mini, final-answer compliance and acting on rejected moves are additional concrete issues. These are diagnostic hypotheses grounded in the saved examples, not tested interventions.

**Traceability**

- [Original aggregate results](/Users/jacksonsubcleff/llm_debate/exp/chess_comparison/results.csv)
- [Run settings and puzzle manifest](/Users/jacksonsubcleff/llm_debate/exp/chess_comparison/chess_run.json)
- [Scoring contract](/Users/jacksonsubcleff/llm_debate/core/chess/protocols/direct_qa.py:26)
- [Full 1,050-row audit, including final answers, responses, source CSV paths and cache paths](/Users/jacksonsubcleff/llm_debate/reports/chess_comparison_failure_audit.csv)
Each linked example above opens the final cached response for that puzzle; tool events are available in the corresponding source CSV’s `tool_trace` column. No source experiment artifacts were modified.
