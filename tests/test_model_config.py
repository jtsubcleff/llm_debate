"""Offline contract checks for central configuration and provider compatibility."""
import asyncio
import json
import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

import httpx
import yaml
from anthropic import AsyncAnthropic
from hydra import compose, initialize_config_dir
from openai import AsyncOpenAI

from core.llm_api.anthropic_llm import AnthropicChatModel
from core.llm_api.base_llm import LLMResponse, PromptConfig, StopReason
from core.llm_api.llm import ModelAPI
from core.llm_api.openai_llm import OpenAIChatModel
from core.llm_api.registry import DEFAULT_CONFIG, catalog, model_id, resolve_model
from core.prompts import render


class RegistryTests(unittest.TestCase):
    def test_all_roles_groups_and_legacy_configs_resolve(self):
        # Resolve every central role and roster entry, then scan the checked-in
        # agent configs for references that would otherwise fail when a run starts.
        config = catalog()
        for role in config["roles"]:
            self.assertTrue(resolve_model(role)["id"])
        for group in config["groups"].values():
            for name in group:
                self.assertTrue(resolve_model(name)["provider"])
        for root in (
            "core/config/experiment",
            "scripts/tournament_players",
            "scripts/human_trial_example",
        ):
            for path in Path(root).rglob("*.yaml"):
                data = yaml.safe_load(path.read_text())
                if data and data.get("language_model"):
                    resolve_model(data["language_model"]["model"])
                if data and data.get("preference_model"):
                    resolve_model(data["preference_model"])

    def test_new_version_only_needs_yaml(self):
        # Load an alternate catalog containing a fictional new model to demonstrate
        # that an existing API profile needs no Python changes; unknown IDs must fail clearly.
        config = catalog()
        config["models"]["future"] = {
            "id": "test-future-version",
            "profile": "openai_reasoning",
        }
        config["roles"]["judge"] = "future"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "models.yaml"
            path.write_text(yaml.safe_dump(config))
            with patch.dict(os.environ, {"LLM_MODELS_CONFIG": str(path)}):
                self.assertEqual(model_id("judge"), "test-future-version")
                self.assertEqual(resolve_model("judge")["endpoint"], "responses")
        with self.assertRaisesRegex(ValueError, "Unknown model"):
            resolve_model("test-unknown")

    def test_snapshot_is_preserved(self):
        # Keep an explicitly selected dated model ID unchanged so a reproducible
        # experiment cannot silently switch to the provider's moving alias.
        self.assertEqual(model_id("gpt-4o-mini-2024-07-18"), "gpt-4o-mini-2024-07-18")

    def test_cycle_fails_clearly(self):
        # Introduce circular role references and require a descriptive error
        # rather than an infinite resolution loop.
        config = catalog()
        config["roles"].update(a="b", b="a")
        with patch("core.llm_api.registry.catalog", return_value=config):
            with self.assertRaisesRegex(ValueError, "Cyclic"):
                resolve_model("a")

    def test_template_contract(self):
        # Check that the generic judge template excludes source context, inserted
        # values are not recursively rendered, XML tags survive, and missing variables fail.
        judge = PromptConfig(template="judge")
        self.assertEqual(len(judge.messages), 2)
        self.assertNotIn("<CONTEXT>", "".join(m["content"] for m in judge.messages))
        self.assertEqual(
            render("<TASK><argument></argument>", {"TASK": "<UNTRUSTED>"}),
            "<UNTRUSTED><argument></argument>",
        )
        with self.assertRaisesRegex(ValueError, "Missing prompt variable"):
            render("<TASK>", {})

    def test_hydra_and_standalone_agents_use_same_roles(self):
        # Build agents from Hydra using the shared template loader and check their
        # resolved roles and output label. Only the debater should receive the source context.
        from core.create_agents import setup_debate, setup_judge
        from core.file_handler import Method
        from core.rollouts.utils import Answers, DebaterNames, TranscriptConfig

        config_dir = str(DEFAULT_CONFIG.parent.resolve())
        with initialize_config_dir(version_base=None, config_dir=config_dir):
            cfg = compose(
                config_name="config",
                overrides=[
                    "+experiment=debate_generic",
                    "exp_dir=/private/tmp/test-debate",
                ],
            )
        api = ModelAPI()
        with tempfile.TemporaryDirectory() as directory:
            rollout = setup_debate(cfg, Path(directory), api)
            judge = setup_judge(cfg, cfg.judge, api)
        self.assertEqual(
            rollout.correct_debater.config.language_model.model,
            model_id("correct_debater"),
        )
        self.assertEqual(judge.config.language_model.model, model_id("judge"))
        self.assertEqual(cfg.judge_name, model_id("judge"))
        transcript = TranscriptConfig(
            index=0,
            story="SECRET CONTEXT <TASK>",
            story_title="Test",
            question="Choose?",
            question_set_id="q",
            answers=Answers(correct="True", incorrect="False"),
            names=DebaterNames(correct="A", incorrect="B"),
            swap=False,
            rollout_type="quality_sim",
        )
        debater_messages = rollout.correct_debater.construct_messages(transcript)
        judge_messages = judge.construct_messages(transcript)
        self.assertIn("SECRET CONTEXT <TASK>", debater_messages[-1]["content"])
        self.assertNotIn(
            "SECRET CONTEXT", "".join(m["content"] for m in judge_messages)
        )
        self.assertIn("Answer: A", judge_messages[0]["content"])

    def test_existing_hydra_prompt_lists_and_optional_judges(self):
        # Construct agents for the existing experiment presets, including best-of-N
        # judges, to catch regressions converting nested Hydra prompt lists for validation.
        from core.create_agents import setup_debate, setup_judge

        for experiment in (
            "debate",
            "debate_seq",
            "debate_interactive",
            "debate_critique_story",
            "consultancy",
            "consultancy_critique_story",
            "blind",
            "oracle",
        ):
            with self.subTest(experiment=experiment):
                overrides = [
                    f"+experiment={experiment}",
                    "exp_dir=/private/tmp/test-debate",
                ]
                if experiment.startswith("consultancy"):
                    overrides.append("method_type=correct")
                with initialize_config_dir(
                    version_base=None, config_dir=str(DEFAULT_CONFIG.parent.resolve())
                ):
                    cfg = compose(config_name="config", overrides=overrides)
                if experiment in ("debate", "debate_seq"):
                    cfg.correct_debater.BoN = 2
                    cfg.incorrect_debater.BoN = 2
                with tempfile.TemporaryDirectory() as directory:
                    rollout = setup_debate(cfg, Path(directory), ModelAPI())
                    judge = setup_judge(cfg, cfg.judge, ModelAPI())
                self.assertTrue(judge.config.prompts.messages)
                self.assertTrue(rollout.correct_debater.config.prompts.messages)

    def test_independent_debater_roles_and_judge_capability_error(self):
        # Assign different models to the two sides without mutating their shared
        # input config, and reject logprob judging when the selected model lacks that capability.
        from core.agents.debater_base import DebaterConfig
        from core.agents.debater_quality import DebaterQuality
        from core.agents.judge_base import JudgeConfig
        from core.agents.judge_quality import JudgeQuality
        from core.rollouts.utils import RolloutConfig

        config = catalog()
        config["roles"]["correct_debater"] = "openai_small"
        config["roles"]["incorrect_debater"] = "anthropic_balanced"
        with patch("core.llm_api.registry.catalog", return_value=config):
            debater_config = DebaterConfig(
                language_model={"model": "debater"}, prompts={"template": "debater"}
            )
            correct = DebaterQuality("debate", debater_config, True, ModelAPI())
            incorrect = DebaterQuality("debate", debater_config, False, ModelAPI())
            self.assertEqual(
                correct.config.language_model.model, model_id("openai_small")
            )
            self.assertEqual(
                incorrect.config.language_model.model, model_id("anthropic_balanced")
            )
            self.assertEqual(debater_config.language_model.model, "debater")
        judge_config = JudgeConfig(
            language_model={"model": "anthropic_strong"},
            prompts={"template": "judge"},
            use_logprobs=True,
        )
        rollout_config = RolloutConfig(
            rollout_type="quality_sim", num_steps=1, name1="A", name2="B"
        )
        with self.assertRaisesRegex(ValueError, "does not support logprobs"):
            JudgeQuality("debate", judge_config, rollout_config, ModelAPI())


class AdapterTests(unittest.IsolatedAsyncioTestCase):
    async def test_debate_round_and_judgement_with_generic_prompts(self):
        # Run one simultaneous round and a final judgment with mocked model outputs.
        # Verify both arguments reach the transcript and refinement variables render safely.
        import pandas as pd

        from core.create_agents import setup_debate, setup_judge
        from core.rollouts.utils import Answers, DebaterNames, TranscriptConfig

        api = ModelAPI()
        adapter = AsyncMock(
            return_value=[
                LLMResponse(
                    model_id="test",
                    completion="<argument>The evidence supports my answer.</argument>",
                    stop_reason="stop",
                    cost=0.1,
                    duration=1,
                    api_duration=0.5,
                )
            ]
        )
        with initialize_config_dir(
            version_base=None, config_dir=str(DEFAULT_CONFIG.parent.resolve())
        ):
            cfg = compose(
                config_name="config",
                overrides=[
                    "+experiment=debate_generic",
                    "exp_dir=/private/tmp/test-debate",
                ],
            )
        row = pd.Series(
            {
                "story": "Some evidence.",
                "story_title": "Story",
                "question": "Which?",
                "question_set_id": "q",
                "correct answer": "First",
                "negative answer": "Second",
            }
        )
        with tempfile.TemporaryDirectory() as directory:
            rollout = setup_debate(cfg, Path(directory), api)
            rollout.config.num_steps = 1
            with patch.object(ModelAPI, "_adapter", return_value=adapter):
                result = await rollout.run(0, row)
            self.assertTrue(result["complete"])
            transcript = TranscriptConfig(**json.loads(result["transcript"]))
            self.assertEqual(len(transcript.rounds), 1)
            self.assertIn("evidence", transcript.rounds[0].correct)
            self.assertIn("evidence", transcript.rounds[0].incorrect)
            judge = setup_judge(cfg, cfg.judge, api)
            row["transcript"] = result["transcript"]
            with patch.object(
                ModelAPI, "call_single", new=AsyncMock(return_value="Answer: A")
            ):
                judgement = await judge.make_decision(0, row)
            self.assertTrue(judgement["complete"])
            self.assertEqual(judgement["judgement"], "Answer: A")
            refinement = rollout.correct_debater.config.prompts.messages1[0]["content"]
            text = rollout.correct_debater.fill_in_content(
                refinement, transcript, {"CRITIQUE": "Improve <TASK>"}
            )
            self.assertIn("Improve <TASK>", text)
            self.assertNotIn("<WORD_LIMIT>", text)

    async def test_every_openai_reasoning_model_request(self):
        # For each configured reasoning-model choice, inspect a mocked Responses
        # request for token-limit translation and removed sampling controls, then parse its answer.
        for selection in (
            "openai_strong",
            "openai_balanced",
            "openai_small",
            "judge_5_mini",
            "judge_5_nano",
        ):
            with self.subTest(selection=selection):
                requests = []

                def respond(request):
                    requests.append(json.loads(request.content))
                    self.assertEqual(request.url.path, "/v1/responses")
                    return httpx.Response(
                        200,
                        json={
                            "id": "resp_test",
                            "object": "response",
                            "created_at": 0,
                            "status": "completed",
                            "model": model_id(selection),
                            "output": [
                                {
                                    "id": "msg_test",
                                    "type": "message",
                                    "role": "assistant",
                                    "status": "completed",
                                    "content": [
                                        {
                                            "type": "output_text",
                                            "text": "Answer: A",
                                            "annotations": [],
                                        }
                                    ],
                                }
                            ],
                            "usage": {
                                "input_tokens": 10,
                                "output_tokens": 5,
                                "total_tokens": 15,
                            },
                        },
                    )

                with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
                    adapter = OpenAIChatModel(frac_rate_limit=0.9, organization=None)
                await adapter.client.close()
                async with AsyncOpenAI(
                    api_key="test-key",
                    http_client=httpx.AsyncClient(
                        transport=httpx.MockTransport(respond)
                    ),
                ) as client:
                    adapter.client = client
                    result = await adapter._make_api_call(
                        [{"role": "user", "content": "test"}],
                        model_id(selection),
                        time.time(),
                        temperature=0.4,
                        top_p=1,
                        max_tokens=8192,
                    )
                self.assertNotIn("temperature", requests[0])
                self.assertNotIn("top_p", requests[0])
                self.assertNotIn("max_tokens", requests[0])
                self.assertEqual(requests[0]["max_output_tokens"], 8192)
                self.assertEqual(result[0].completion, "Answer: A")
                self.assertIs(result[0].stop_reason, StopReason.STOP_SEQUENCE)

    async def test_every_claude_messages_model_request(self):
        # For each current Claude choice, inspect system/message separation and
        # sampling controls; extract only text blocks and preserve unknown pricing as None.
        for selection in ("anthropic_strong", "anthropic_balanced", "anthropic_small"):
            with self.subTest(selection=selection):
                requests = []

                def respond(request):
                    self.assertEqual(request.url.path, "/v1/messages")
                    requests.append(json.loads(request.content))
                    return httpx.Response(
                        200,
                        json={
                            "id": "msg_test",
                            "type": "message",
                            "role": "assistant",
                            "model": model_id(selection),
                            "content": [
                                {
                                    "type": "thinking",
                                    "thinking": "hidden",
                                    "signature": "signature",
                                },
                                {"type": "text", "text": "argument"},
                            ],
                            "stop_reason": "end_turn",
                            "stop_sequence": None,
                            "usage": {"input_tokens": 10, "output_tokens": 5},
                        },
                    )

                with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
                    adapter = AnthropicChatModel(num_threads=1)
                await adapter.client.close()
                async with AsyncAnthropic(
                    api_key="test-key",
                    http_client=httpx.AsyncClient(
                        transport=httpx.MockTransport(respond)
                    ),
                ) as client:
                    adapter.client = client
                    with tempfile.TemporaryDirectory() as directory, patch(
                        "core.llm_api.anthropic_llm.Path",
                        side_effect=lambda _: Path(directory),
                    ):
                        result = await adapter(
                            [model_id(selection)],
                            [
                                {"role": "system", "content": "rules"},
                                {"role": "user", "content": "test"},
                            ],
                            False,
                            1,
                            temperature=0.4,
                            top_p=1,
                        )
                self.assertEqual(requests[0]["system"], "rules")
                self.assertEqual(
                    requests[0]["messages"], [{"role": "user", "content": "test"}]
                )
                self.assertNotIn("temperature", requests[0])
                self.assertEqual(result[0].completion, "argument")
                self.assertIs(result[0].stop_reason, StopReason.STOP_SEQUENCE)
                if selection == "anthropic_small":
                    self.assertAlmostEqual(result[0].cost, 3.5e-6)

    async def test_unsupported_features_fail_before_request(self):
        # Reject unsupported assistant prefill and logprobs before submitting a
        # request; the logprob check must not even instantiate a provider adapter.
        with self.assertRaisesRegex(ValueError, "prefill"):
            AnthropicChatModel.messages_request(
                [{"role": "assistant", "content": "Answer:"}],
                resolve_model("anthropic_strong"),
                {},
            )
        api = ModelAPI()
        with self.assertRaisesRegex(ValueError, "logprobs"):
            await api("openai_strong", "test", 100, logprobs=5)
        self.assertEqual(api._adapters, {})

    async def test_candidate_fanout_and_invalid_cost_accounting(self):
        # Request two candidates through a single-completion API and discard one.
        # Both calls must count toward cost even though only the valid candidate is returned.
        api = ModelAPI()
        adapter = AsyncMock(
            side_effect=[
                [
                    LLMResponse(
                        model_id="test",
                        completion="invalid",
                        stop_reason="stop",
                        cost=0.2,
                        duration=2,
                        api_duration=1,
                    )
                ],
                [
                    LLMResponse(
                        model_id="test",
                        completion="valid",
                        stop_reason="stop",
                        cost=0.3,
                        duration=2,
                        api_duration=1,
                    )
                ],
            ]
        )
        with patch.object(ModelAPI, "_adapter", return_value=adapter):
            results = await api(
                "openai_strong",
                "test",
                100,
                num_candidates_per_completion=2,
                is_valid=lambda x: x == "valid",
            )
        self.assertEqual(adapter.await_count, 2)
        self.assertEqual(results[0].completion, "valid")
        self.assertAlmostEqual(api.running_cost, 0.5)


if __name__ == "__main__":
    unittest.main()
