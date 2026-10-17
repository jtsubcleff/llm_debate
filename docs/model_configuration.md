# Model configuration and extensibility

`core/config/models.yaml` is the single source of model IDs, snapshot aliases,
provider capabilities, pricing, compatibility quirks, default output budgets,
and experiment rosters. No Python allowlist needs updating for a new version
that uses an existing provider endpoint.

## Choosing debaters and judges

Edit the `roles` section. A selection may be a catalog key, registered API ID,
snapshot alias, or another role. For example:

```yaml
roles:
  debater: openai_balanced
  correct_debater: debater
  incorrect_debater: anthropic_strong
  judge: judge_4o_mini
  preference: judge
  critic: debater
  critique_judge: judge
  cross_examiner: judge
  concession_judge: judge
```

The defaults pair strong debaters with a smaller, older judge. `correct_debater`
and `incorrect_debater` refer to the assigned answer, not a permanent player
identity; counterbalance models across both assignments in experiments.

For independent experiments, copy the complete catalog and select it **before
starting any process**:

```bash
export LLM_MODELS_CONFIG=/absolute/path/to/my-models.yaml
python -m core.debate exp_dir=./exp/my_run +experiment=debate_generic
python -m core.judge exp_dir=./exp/my_run +experiment=debate_generic
python -m core.scoring.accuracy exp_dir=./exp/my_run +experiment=debate_generic
```

The catalog is loaded once per process. Restart a running web backend after
editing it. The same role resolver is used by Hydra, standalone YAML configs,
tournament generation, and the web backend. The default result directory label
resolves the configured `judge` role. If overriding an agent's model directly
through Hydra, also set `judge_name` when judging/scoring, or edit the role instead.

Use a **fresh `exp_dir` when changing models or prompts**. Existing completed CSV
rows and round caches are intentionally resumable; they are not invalidated by a
catalog edit. Archive the selected catalog with your experiment and pin snapshots
where the provider offers them. Runtime logs include the selected model IDs.

## Included current/weak model choices

| Catalog key | API ID | Endpoint |
|---|---|---|
| `openai_strong` | `gpt-6-astra` | Responses |
| `openai_balanced` | `gpt-6.1-sol` | Responses |
| `openai_small` | `gpt-6-luna` | Responses |
| `anthropic_strong` | `claude-opus-5-5` | Messages |
| `anthropic_balanced` | `claude-sonnet-5-5` | Messages |
| `anthropic_small` | `claude-haiku-5-5` | Messages |
| `judge_4o_mini` | `gpt-4o-mini` | Chat Completions |
| `judge_41_mini` | `gpt-4.1-mini` | Chat Completions |
| `judge_5_mini` | `gpt-5-mini` | Responses |
| `judge_5_nano` | `gpt-5-nano` | Responses |

The dated weak-judge snapshots discussed in the chat are registered aliases and
are passed to the API unchanged when selected. The `groups` section controls
figure scripts and the tournament roster. The tournament runner generates player
configs using the generic debater template; all configured judges score debates,
and the first judge supplies the Swiss pairing score. Historical player files
remain available with `legacy_*` catalog references.

Provider availability is separate from adapter support. As checked on 2026-10-17,
GPT-4o mini and GPT-4.1 mini retire on 2027-01-14, and the GPT-5 mini/nano snapshots
retire on 2026-12-11. Legacy Claude/GPT IDs are preserved for old configurations,
not promised to work on a provider's live API.

## Adding a new version

Add a catalog entry and select it through a role:

```yaml
models:
  my_new_version:
    id: YOUR_PROVIDER_MODEL_ID
    profile: openai_reasoning
    # Optional: prices in USD per million input/output tokens.
    # price_per_million: [INPUT_PRICE, OUTPUT_PRICE]
    # Optional account-specific throttling (defaults: 60 RPM / 1,000,000 TPM).
    requests_per_minute: 30
    tokens_per_minute: 500000
roles:
  debater: my_new_version
```

Use `openai_chat`, `openai_reasoning`, `openai_base`, `anthropic_messages`, or
`anthropic_legacy` as the profile. These are **API compatibility profiles**, not
claims about intelligence. If a new model has different parameter support, create
another profile or override metadata on its entry. IDs are never inferred from
prefixes. Fine-tuned model IDs must also be explicitly registered with the right
profile, pricing, and context limits.

`request_defaults` supplies provider-specific request settings. OpenAI reasoning
profiles currently use low effort. Sampling controls from historical prompts are
omitted when the profile marks them unsupported; temperature changes therefore
do not vary those models' reasoning behavior. Review effective settings before
comparing models. A genuinely new provider/endpoint still requires an adapter.

Pricing estimates use API token usage. An omitted price produces `cost=None` and
increments `ModelAPI.unknown_cost_responses`; `running_cost` is only the known
subtotal. Candidate costs include discarded/invalid responses. Cache discounts,
cache writes, and other provider billing adjustments are not modeled. Legacy
Anthropic completions have unknown cost because they lack authoritative usage.

## Generic prompt templates and future tasks

- `core/config/prompts/debater.yaml`: task, rubric, context, question, assigned
  answer, opponent answer, transcript, and word limit; returns `<argument>` tags.
- `core/config/prompts/judge.yaml`: task, rubric, question, candidate answers,
  transcript, and word limit; returns `Answer: A` or `Answer: B`. It deliberately
  receives **no source context or correct-answer label**.
- `core/prompts.py`: provider-independent template loading and strict rendering.
- `core/config/experiment/debate_generic.yaml`: runnable QuALITY experiment using
  both templates. Existing research prompts remain selectable.

Select a template through `prompts.template` and customize constants with
`prompts.vars`, for example `TASK` and `RUBRIC`. Missing variables fail clearly.
Inserted task text is not recursively rendered. `PromptConfig` loads the same
standalone templates for CLI and web use.

These templates do not make the current dataset/scorer generic: `quality` agents,
the transcript schema, quote verification, two-answer conventions, and accuracy
scoring still implement QuALITY. A future task supplies its own context mapping,
answer/output contract, and scorer while reusing the catalog and provider adapters.

## Files requiring special treatment

Search for `SPECIAL TREATMENT` comments in runtime code. The remaining boundaries
are explicit:

| File | Why handling differs |
|---|---|
| `core/llm_api/openai_llm.py` | Chat/Responses/legacy endpoint formats, reasoning versus sampling parameters, output extraction, and account throttling. Responses returns one completion per request. |
| `core/llm_api/anthropic_llm.py` | Separate system prompt, content blocks, no token logprobs, no native multiple completions, and some models reject assistant prefill. |
| `core/llm_api/registry.py` | Local tokenizer estimates differ from authoritative provider token usage. |
| `core/llm_api/base_llm.py` | Provider finish reasons must map to a common result contract; unsupported reasons fail explicitly. |
| `core/agents/debater_quality.py` | Opt-in legacy output parsing quirks and configured prompt-token limits. |
| `core/agents/judge_quality.py` | Logprob support and opt-in legacy question parsing; explicit discrete-choice scoring when logprobs are disabled. |
| `core/tournament.py` | Optional model concurrency limits from the catalog. |

Token-logprob preference scoring and discrete-choice scoring are different
experimental signals. Preference/critique configs retain `use_logprobs: true`;
using a reasoning/Claude model there fails with a clear message. To opt into
coarser discrete scoring, set `use_logprobs: false`, use a sufficient `max_tokens`
budget (typically null for the profile default), and keep a prompt that requests
only A/B or Y/N. No silent substitution of scoring methods occurs.

Historical concession/preference prompts ending with an assistant prefill must be
rewritten as user-side formatting instructions before using Claude models that
reject prefill. The adapter fails before submitting these requests.

Historical model names also remain as compatibility labels in
`core/scoring/ratings.py`, notebook saved outputs, README examples, and legacy
player filenames. They do not select runtime model IDs.

## Validation

```bash
python -m unittest discover -s tests -v
python -m core.debate --cfg job --resolve +experiment=debate_generic exp_dir=./exp/check
python -m core.llm_api.registry --role judge
python -m core.llm_api.registry --group debaters
```

Tests use mocked OpenAI/Anthropic HTTP transports. They check request schemas,
reasoning parameter removal, text-block extraction, weak-judge snapshots, new
versions added only in YAML, role resolution, prompt isolation, and candidate
accounting. Tokenizers may download public encoding data on first use. These
checks do not establish account access or empirical model strength.

Sources checked for API compatibility and lifecycle:
[OpenAI models](https://developers.openai.com/api/docs/models),
[OpenAI reasoning compatibility](https://developers.openai.com/api/docs/guides/latest-model),
[OpenAI deprecations](https://developers.openai.com/api/docs/deprecations),
[Claude models](https://platform.claude.com/docs/en/models/overview),
[Claude Messages API](https://platform.claude.com/docs/en/api/messages/create).
