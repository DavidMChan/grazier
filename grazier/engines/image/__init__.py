from abc import abstractmethod
from typing import Any, Dict, List, Optional, Type, TypeVar

from PIL import Image

from grazier.engines.default import Engine
from grazier.utils.pytorch import select_device


class ILMEngine(Engine):
    def __init__(self, device: Optional[str] = None) -> None:
        self.device = select_device(device)
        self.device_specified = device is not None

    @property
    def prompt_prefix(self) -> str:
        return ""

    @property
    def prompt_suffix(self) -> str:
        return ""

    def __call__(self, image: Image, prompt: Optional[str] = None, n_completions: int = 1, **kwargs: Any) -> List[str]:
        if prompt is not None:
            prompt = self.prompt_prefix + prompt + self.prompt_suffix
        return self.call(image, prompt, n_completions, **kwargs)

    @abstractmethod
    def call(self, image: Image, prompt: Optional[str] = None, n_completions: int = 1, **kwargs: Any) -> List[str]:
        raise NotImplementedError()

    def __repr__(
        self,
    ) -> str:
        return f"{self.__class__.__name__}({self.name[0]})"

    @staticmethod
    def from_string(typestr: str, **kwargs: Any) -> "ILMEngine":
        typestr = typestr.lower()

        if typestr in ILM_ENGINES:
            if not ILM_ENGINES[typestr].is_configured() and ILM_ENGINES[typestr].requires_configuration():
                raise ValueError(
                    f"Image/Language model type: {typestr} requires configuration. Please run `grazier configure {typestr}` first."
                )
            else:
                return ILM_ENGINES[typestr](**kwargs)  # type: ignore
        elif typestr in ILM_ENGINES_CLI:
            if not ILM_ENGINES_CLI[typestr].is_configured() and ILM_ENGINES_CLI[typestr].requires_configuration():
                raise ValueError(
                    f"Image/Language model type: {typestr} requires configuration. Please run `grazier configure {typestr}` first."
                )
            else:
                return ILM_ENGINES_CLI[typestr](**kwargs)  # type: ignore

        raise ValueError(f"Invalid Image/Language model type: {typestr}")

    @staticmethod
    def list_models() -> List[str]:
        return list(ILM_ENGINES_CLI.keys())


ILM_ENGINES: Dict[str, Type[ILMEngine]] = {}
ILM_ENGINES_CLI: Dict[str, Type[ILMEngine]] = {}

T = TypeVar("T")


def register_engine(cls: T) -> T:
    ILM_ENGINES[cls.name[0]] = cls  # type: ignore
    ILM_ENGINES_CLI[cls.name[1].lower()] = cls  # type: ignore
    return cls


# Imports for engine modules
from .blip_engine import *  # noqa: F403, E402
from .open_flamingo_engine import *  # noqa: F403, E402
