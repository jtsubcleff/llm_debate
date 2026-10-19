# Chess experiments

This repository contains experiments testing LLM chess puzzle solving abilities, in various evaluations protocols. Puzzles are sourced from [Lichess Database](https://database.lichess.org/#puzzles)

## Setup

Use Python 3.11 and install the repository's `requirements.txt`. API credentials
use the existing `SECRETS` setup described in [README.md](README.md). Model names,
roles, and API IDs resolve through [core/config/models.yaml](core/config/models.yaml).
Use modern Chat Completions, Responses, or Claude Messages profiles supporting
function tools. Historical Completions profiles are rejected.

The experimental dataset should already be present at `chess_data/lichess_puzzles.csv`. To load more puzzles that adhere to the format required by this repository, use `scripts/prepare_lichess_puzzles.py`

## Run Direct QA

Direct QA asks a model for the best first move from a FEN. It can reason and
call `validate_move_sequence` to check candidate lines with python-chess. No
engine evaluation or solution is exposed to the model.

From the repository root with the virtual environment activated:

```bash
# One model: 30 puzzles total, ten per bucket
python -m core.chess.run exp_dir=./exp/chess_direct_qa models='[judge]' limit=30

# Several models on the same 300 puzzles, 100 per bucket
python -m core.chess.run exp_dir=./exp/chess_comparison \
  models='[openai_small,anthropic_small,openrouter_gpt_oss_20b]' limit=300 seed=42

# Disable tools: one model call per puzzle, using a prompt without tool references
python -m core.chess.run exp_dir=./exp/chess_no_tools models='[judge]' \
  limit=30 use_tools=false

# Adjust concurrency and the per-puzzle tool/per-call token budgets
python -m core.chess.run exp_dir=./exp/chess_custom models='[judge]' \
  limit=30 num_threads=5 max_tokens=8192 max_tool_rounds=10 temperature=0.2
```

`limit` is the **total per model**, must be positive and divisible by three,
and fails before any API calls if a bucket has insufficient puzzles. Selection
is seeded random sampling without replacement; all models receive the same
sample. Default buckets are Low `<1300`, Medium `1300–2300` inclusive, and
High `>2300`.

The default input is `chess_data/lichess_puzzles.csv`; override `dataset_path`
for another prepared CSV. Required columns are `PuzzleId,FEN,Moves,Rating`.
**FEN must already be the solver's position**, and `Moves` must start with the
solution in UCI. The repository's preparation script has already played and
removed the opponent's setup move. Do not pass a raw Lichess database directly.
The loader validates positions and solution sequences.

## Tools and scoring

Example tool arguments: `{"moves": ["Nf3", "d5"]}` or
`{"moves": ["g1f3", "d7d5"]}`. SAN and UCI may be mixed. Every invocation
starts from the original puzzle FEN, independently of earlier calls and other
puzzles. Null moves are illegal. Results have `legal` (boolean), `fen` (new FEN
if fully legal, otherwise null), and `first_illegal_move` (null if legal).
Malformed arguments return an error to the model so it can correct them.

The model's final whitespace-separated token must exactly match canonical SAN
for the first solution move, including check/mate/promotion notation. Markdown,
extra punctuation, or a UCI final answer is incorrect. Truncated and refused
responses count as incorrect. The prompt is in
[core/config/prompts/chess_direct_qa.yaml](core/config/prompts/chess_direct_qa.yaml).

`use_tools` defaults to `true`. With `use_tools=false`, the runner uses
[the no-tools prompt](core/config/prompts/chess_direct_qa_no_tools.yaml), sends
no tool definitions or tool-choice parameters, and makes one API call per puzzle.
`max_tool_rounds` is ignored in this mode; `max_tokens` is the output budget for
the single call. Any unexpected structured tool request is recorded as an error
and is never executed. Use a separate `exp_dir` when switching modes.

With tools enabled, there are at most `max_tool_rounds` tool rounds, followed by one final API call
with tools disabled if needed. A round can contain several independent tool
calls. `max_tokens` applies to each API call (including provider reasoning tokens
where applicable), rather than the whole puzzle. Provider adapters retain
native tool/reasoning history and their existing rate limits, retries, and cost
handling. Actual input/output usage is summed over all calls for each puzzle.

## Results and resuming

Each model has `exp_dir/direct_qa/<safe-model-name>_<hash>/puzzles.csv` with:

- `puzzleId`, `rating`, `rating_bucket`, `FEN`, `moves`, and `model`;
- `model_response`, `final_move`, `expected_move`, `correct`, `complete`, and `error`;
- `use_tools`, `input_tokens`, `output_tokens`, `cost` (USD; blank if unknown), `api_calls`,
  and `tool_trace` (JSON containing responses, tool calls, and results).

Rows are checkpointed atomically after each completed or failed puzzle, and
successful API turns are cached under the model's `cache/` directory. Run the
same command to resume: completed puzzles are skipped, and cached turns are
reused without billing again. API failures stay incomplete and can be retried;
an incorrect answer is a completed evaluation.

`exp_dir/results.csv` updates after every puzzle. One row per model contains
success fractions in `Low`, `Medium`, `High`, and `all`, corresponding counts,
and planned/completed totals. Only completed evaluations enter the denominator;
partial-run buckets may be unbalanced until all planned puzzles finish.

`chess_run.json` pins the sample, prompt, tool schema, and generation settings;
per-model `model.json` pins model profiles. Changed settings require a new
`exp_dir`. Additional models can be added to the same experiment. Avoid running
two processes writing the same experiment directory concurrently.

## Visualize

Open [scripts/plot_chess.ipynb](scripts/plot_chess.ipynb), set `EXP_DIR`, and run
all cells. It uses the same pandas/matplotlib/seaborn stack as `plot_minimal.ipynb`
and saves PNGs to `<EXP_DIR>/figures/`:

1. Overall correctness by model.
2. Correctness by model and rating bucket.
3. Success versus puzzle rating, with one ordinary least squares line per model.

The notebook also includes an **Inspect a single puzzle request** section. Set
`MODEL` and `PUZZLE_ID` to view its board, rating, correct SAN move, full model
output, and readable tool-call sequences and validation results.

Regression fits per-puzzle binary correctness; points show observed success in
100-rating intervals. Linear predictions can fall outside 0–100%.

## Extending chess protocols

`core/load/chess.py` owns puzzle loading and balanced sampling;
`core/chess/tools.py` owns board validation; `core/chess/results.py` owns CSV
storage and summaries. Direct QA lives in `core/chess/protocols/direct_qa.py`.
Register future implementations in `core/chess/run.py:PROTOCOLS` and add their
prompt templates. The runner passes the selected protocol to result discovery.
