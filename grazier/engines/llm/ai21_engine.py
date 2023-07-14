import os
from typing import Any, List

import ai21
from ai21.errors import (
    BadRequest,
    EmptyMandatoryListException,
    MissingApiKeyException,
    MissingInputException,
    NoSpecifiedRegionException,
    OnlyOneInputException,
    Unauthorized,
    UnprocessableEntity,
    UnsupportedDestinationException,
    UnsupportedInputException,
    WrongInputTypeException,
)

from grazier.engines.llm import LLMEngine, register_engine
from grazier.utils.python import retry, singleton

ai21.api_key = os.environ.get("AI21_API_KEY", None)


class AI21CompletionLLMEngine(LLMEngine):
    def __init__(self, model: str) -> None:
        self._model = model
        super().__init__(device="api")

    @retry(
        no_retry_on=(
            BadRequest,
            Unauthorized,
            UnprocessableEntity,
            MissingInputException,
            UnsupportedInputException,
            UnsupportedDestinationException,
            OnlyOneInputException,
            WrongInputTypeException,
            EmptyMandatoryListException,
            MissingApiKeyException,
            NoSpecifiedRegionException,
        )
    )
    def call(self, prompt: str, n_completions: int = 1, **kwargs: Any) -> List[str]:
        _default_params = {
            "numResults": n_completions,
            "maxTokens": 256,
            "temperature": 0.7,
            "stopSequences": ["##"],
        }

        response = ai21.Completion.execute(model=self._model, prompt=prompt, **(_default_params | kwargs))

        return [c["data"]["text"] for c in response["completions"]]

    @staticmethod
    def is_configured() -> bool:
        return ai21.api_key is not None


@register_engine
@singleton
class J2Light(AI21CompletionLLMEngine):
    name = ("J2 (Light)", "j2-light")

    def __init__(self) -> None:
        super().__init__("j2-light")


@register_engine
@singleton
class J2Mid(AI21CompletionLLMEngine):
    name = ("J2 (Mid)", "j2-mid")

    def __init__(self) -> None:
        super().__init__("j2-mid")


@register_engine
@singleton
class J2Ultra(AI21CompletionLLMEngine):
    name = ("J2 (Ultra)", "j2-ultra")

    def __init__(self) -> None:
        super().__init__("j2-ultra")
