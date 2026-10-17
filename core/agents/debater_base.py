import logging
from typing import Optional

from omegaconf import OmegaConf
from pydantic import BaseModel, validator

from core.file_handler import Method
from core.llm_api.base_llm import LanguageModelConfig, PromptConfig
from core.llm_api.llm import ModelAPI
from core.llm_api.registry import resolve_model, tokenizer
from core.rollouts.utils import Round, TranscriptConfig

LOGGER = logging.getLogger(__name__)


side_mapping = {
    True: "correct",
    False: "incorrect",
}


class DebaterConfig(BaseModel):
    language_model: LanguageModelConfig
    prompts: PromptConfig
    debater_type: Optional[str] = "standard"

    BoN: Optional[int] = 1
    cBoN: Optional[int] = 0
    bon_pm_config: Optional[str] = None
    critic_config: Optional[str] = None
    critique_pm_config: Optional[str] = None
    preference_model: Optional[str] = None

    @validator("preference_model")
    def resolve_preference(cls, value):
        return resolve_model(value)["id"] if value else value

    few_shot_num_samples: Optional[int] = 0
    transcript_quotes: Optional[str] = None
    permissions: Optional[dict] = None
    few_shot_base: Optional[str] = None


class DebaterBase:
    def __init__(
        self,
        method: Method,
        config: DebaterConfig,
        # None for cross-examiner
        correct: Optional[bool],
        api_handler: ModelAPI,
    ):
        if OmegaConf.is_config(config):
            config = OmegaConf.to_container(config, resolve=True)
        config = (
            config if isinstance(config, DebaterConfig) else DebaterConfig(**config)
        )
        selection = config.language_model.model
        if selection == "debater":
            selection = "correct_debater" if correct else "incorrect_debater"
        config = config.copy(deep=True)
        config.language_model.model = resolve_model(selection)["id"]
        self.method = method
        self.config = config
        self.correct = correct
        self.api_handler = api_handler
        self.partials = self.config.prompts.partials
        self.messages = self.config.prompts.messages
        self.constructed_messages = (
            []
        )  # note: don't use this for concurrent jobs, only used for live debates
        self.side = side_mapping[self.correct]

        if self.config.BoN > 1:
            LOGGER.info(f"Using BoN: {self.config.BoN} {self.side}")

        self.model_spec = resolve_model(self.config.language_model.model)
        self.tokenizer = tokenizer(self.config.language_model.model)

    def construct_messages(self, _: TranscriptConfig):
        raise NotImplementedError

    def answers_from_transcript(self, transcript: TranscriptConfig):
        answer_defending = (
            transcript.answers.correct if self.correct else transcript.answers.incorrect
        )
        answer_opposing = (
            transcript.answers.incorrect if self.correct else transcript.answers.correct
        )
        return answer_defending.strip(), answer_opposing.strip()

    def answer_letters_from_transcript(self, transcript: TranscriptConfig):
        if transcript.swap:
            answer_defending = "B" if self.correct else "A"
        else:
            answer_defending = "A" if self.correct else "B"

        answer_opposing = "B" if answer_defending == "A" else "A"
        return answer_defending.strip(), answer_opposing.strip()

    def names_from_transcript(self, transcript: TranscriptConfig):
        # We don't know our name without the transcript because it's different in a swap case
        # All we know is whether we're the correct or incorrect debater
        name = transcript.names.correct if self.correct else transcript.names.incorrect
        opponent_name = (
            transcript.names.incorrect if self.correct else transcript.names.correct
        )
        return name, opponent_name

    def args_from_round(self, round: Round):
        our_arg = round.correct if self.correct else round.incorrect
        opponent_arg = round.incorrect if self.correct else round.correct
        return our_arg, opponent_arg

    def our_args(self, transcript: TranscriptConfig):
        args = []
        for round in transcript.rounds:
            our_arg, _ = self.args_from_round(round)
            if our_arg is not None:
                args.append(our_arg)
        return args

    def opponent_args(self, transcript: TranscriptConfig):
        args = []
        for round in transcript.rounds:
            _, opponent_arg = self.args_from_round(round)
            if opponent_arg is not None:
                args.append(opponent_arg)
        return args
