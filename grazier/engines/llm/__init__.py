import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar

from huggingface_hub import HfApi, ModelFilter

from grazier.utils.pytorch import select_device


class LLMEngine(ABC):

    @property
    @abstractmethod
    def name(self) -> Tuple[str, str]:
        """ Returns a tuple of (Pretty Name, CLI name) of the language model. """
        raise NotImplementedError()

    def __init__(self, device: Optional[str] = None) -> None:
        self.device = select_device(device)


    @property
    def prompt_prefix(self) -> str:
        return ""

    @property
    def prompt_suffix(self) -> str:
        return ""

    def __call__(
        self,
        prompt: str,
        n_completions: int = 1,
        **kwargs: Any
    ) -> List[str]:
        prompt = self.prompt_prefix + prompt + self.prompt_suffix
        return self.call(prompt, n_completions, **kwargs)


    @abstractmethod
    def call(
        self,
        prompt: str,
        n_completions: int = 1,
        **kwargs: Any
    ) -> List[str]:
        raise NotImplementedError()

    def __repr__(
        self,
    ) -> str:
        return f"{self.__class__.__name__}({self.name[0]})"


    @staticmethod
    def from_string(typestr: str, **kwargs: Any) -> "LLMEngine":

        if typestr in LM_ENGINES:
            return LM_ENGINES[typestr](**kwargs)  # type: ignore
        elif typestr in LM_ENGINES_CLI:
            return LM_ENGINES_CLI[typestr](**kwargs)  # type: ignore

        logging.info(f"Failed to find local LLM matching {typestr}. Fetching remote LLMs...")
        api = HfApi()
        if models := list(
            api.list_models(
                filter=ModelFilter(model_name=typestr, task='text-generation')
            )
        ):
            return HuggingFaceTextGenerationLMEngine.from_hub_model(typestr)(**kwargs)

        raise ValueError(f"Invalid language model type: {typestr}")


    @staticmethod
    def list_models() -> List[str]:
        return list(LM_ENGINES_CLI.keys())

LM_ENGINES: Dict[str, Type[LLMEngine]] = {}
LM_ENGINES_CLI: Dict[str, Type[LLMEngine]] = {}

T = TypeVar("T")
def register_engine(cls: T) -> T:

    LM_ENGINES[cls.name[0]] = cls # type: ignore
    LM_ENGINES_CLI[cls.name[1]] = cls # type: ignore
    return cls

def register_chat_engine(cls: T) -> T:
    LM_ENGINES[cls.name[0]] = cls # type: ignore
    if cls.name[1] not in LM_ENGINES_CLI:
        LM_ENGINES_CLI[cls.name[1]] = cls # type: ignore
    LM_ENGINES_CLI[f'{cls.name[1]}-chat'] = cls # type: ignore

    return cls

# Imports for engine modules
from grazier.engines.llm.ai21_engine import *  # noqa: F403, E402
from grazier.engines.llm.huggingface_engine import *  # noqa: F403, E402
from grazier.engines.llm.llama_engine import *  # noqa: F403, E402
from grazier.engines.llm.openai_engine import *  # noqa: F403, E402
from grazier.engines.llm.vertex_engine import *  # noqa: F403, E402
