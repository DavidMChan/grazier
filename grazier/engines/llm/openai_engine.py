
import os
from abc import abstractmethod
from typing import Any, List

import openai

from grazier.engines.llm import LLMEngine, register_engine
from grazier.utils.python import retry, singleton

# Setup openai api keys
openai.organization = os.getenv("OPENAI_API_ORG", None)
openai.api_key = os.getenv("OPENAI_API_KEY", None)


@singleton
class OpenAI:
    USAGE = 0.0


class OpenAICompletionLLMEngine(LLMEngine):

    @property
    @abstractmethod
    def cost_per_token(self) -> float:
        raise NotImplementedError()

    def __init__(self, model: str):
        self._model = model
        super().__init__(device='api')

    @retry(
        no_retry_on=(
            openai.error.AuthenticationError,
        )
    )
    def call(
        self, prompt: str, n_completions: int = 1, **kwargs: Any
    ) -> List[str]:
        cp = openai.Completion.create(
            model=self._model, prompt=prompt, n=n_completions, **kwargs
        )  # type: ignore
        OpenAI.USAGE += int(cp.usage.total_tokens) * self.cost_per_token
        return [i.text for i in cp.choices]  # type: ignore

@register_engine
@singleton
class GPT3Davinci3(OpenAICompletionLLMEngine):
    cost_per_token = 0.02 / 1000
    name = ("GPT-3 Davinci", "gpt3-davinci3")

    def __init__(self) -> None:
        super().__init__("text-davinci-003")

@register_engine
@singleton
class GPT3Davinci2(OpenAICompletionLLMEngine):
    cost_per_token = 0.03 / 1000
    name = ("GPT-3 Davinci", "gpt3-davinci2")

    def __init__(self) -> None:
        super().__init__("text-davinci-002")

@register_engine
@singleton
class GPT3Curie(OpenAICompletionLLMEngine):
    cost_per_token = 0.002 / 1000
    name = ("GPT-3 Curie", "gpt3-curie")

    def __init__(self) -> None:
        super().__init__("text-curie-001")

@register_engine
@singleton
class GPT3Babbage(OpenAICompletionLLMEngine):
    cost_per_token = 0.02 / 1000
    name = ("GPT-3 Babbage", "gpt3-babbage")

    def __init__(self) -> None:
        super().__init__("text-babbage-001")

@register_engine
@singleton
class GPT3Ada(OpenAICompletionLLMEngine):
    cost_per_token = 0.02 / 1000
    name = ("GPT-3 Ada", "gpt3-ada")

    def __init__(self) -> None:
        super().__init__("text-ada-001")
