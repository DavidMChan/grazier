import logging
from abc import abstractmethod
from typing import Any, List

import openai
from rich.prompt import Prompt

from grazier.engines.llm import LLMEngine, register_engine
from grazier.utils.python import retry, singleton
from grazier.utils.secrets import get_secret, set_secret


def _setup_api_keys():
    # Setup openai api keys
    openai.api_key = get_secret("OPENAI_API_KEY", None)
    openai.organization = get_secret("OPENAI_API_ORG", None)


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

        _setup_api_keys()

        super().__init__(device="api")

    @retry(no_retry_on=(openai.error.AuthenticationError,))
    def call(self, prompt: str, n_completions: int = 1, **kwargs: Any) -> List[str]:
        cp = openai.Completion.create(model=self._model, prompt=prompt, n=n_completions, **kwargs)  # type: ignore
        OpenAI.USAGE += int(cp.usage.total_tokens) * self.cost_per_token
        return [i.text for i in cp.choices]  # type: ignore

    @staticmethod
    def is_configured() -> bool:
        _setup_api_keys()
        return openai.api_key is not None

    @staticmethod
    def configure():
        if OpenAICompletionLLMEngine.is_configured():
            reconfigure = Prompt.ask("OpenAI API key is already configured. Reconfigure?", choices=["y", "n"])
            if reconfigure == "n":
                logging.info("OpenAI API key already configured.")
                return

        have_openai_key = Prompt.ask("Do you have an OpenAI API key?", choices=["y", "n"])
        if have_openai_key == "n":
            Prompt.ask(
                "Please follow the instructions here [u cyan]https://www.howtogeek.com/885918/how-to-get-an-openai-api-key/[/u cyan] to obtain an API key.\nWhen you have the key, press enter to continue",
            )

        openai_key = Prompt.ask("Please enter your OpenAI API key (It will be saved to your platform's secrets store)")
        set_secret("OPENAI_API_KEY", openai_key.strip())

        # Save the openai api key to the secrets store
        logging.info("OpenAI API successfully configured.")


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
