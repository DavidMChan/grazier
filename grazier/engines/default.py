from abc import ABC, abstractmethod
from typing import Tuple


class Engine(ABC):
    @property
    @abstractmethod
    def name(self) -> Tuple[str, str]:
        """Returns a tuple of (Pretty Name, CLI name) of the language model."""
        raise NotImplementedError()

    @staticmethod
    def configure(*args, **kwargs) -> None:
        """Configures the engine."""
        raise NotImplementedError("This engine does not support automated configuration.")

    @staticmethod
    def is_configured() -> bool:
        """Returns True if the engine is configured configuration, False otherwise."""
        return "Unknown"

    @staticmethod
    def requires_configuration() -> bool:
        """Returns True if the engine is requires configuration, False otherwise."""
        return True
