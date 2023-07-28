# isort: skip_file

from dotenv import load_dotenv
from typing import Union, List
import logging

from rich.logging import RichHandler

__version__ = "0.0.2"
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)],
)

# Ignore import sorting, since LLMEngine needs to be imported first
from grazier.engines.llm import LLMEngine  # noqa: E402
from grazier.engines.chat import Conversation, LLMChat, Speaker, ConversationTurn  # noqa: F401, E402


def get(name: str, chat: bool = False) -> Union[LLMChat, LLMEngine]:
    """Get an engine by name."""
    if chat:
        if name in LLMChat.list_models():
            return LLMChat.from_string(name)
        else:
            raise ValueError(f"Chat engine {name} not found")

    return LLMEngine.from_string(name)


def list_models(chat: bool = False) -> List[str]:
    """List available LLMs."""
    if chat:
        return LLMChat.list_models()
    return LLMEngine.list_models()
