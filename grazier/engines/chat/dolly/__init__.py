
import logging
from typing import Any, List, Optional

from transformers import AutoModelForCausalLM, TFAutoModelForCausalLM
from transformers.pipelines import PIPELINE_REGISTRY, pipeline

from grazier.engines.chat import Conversation, ConversationTurn, LLMChat, Speaker, register_engine
from grazier.utils.python import singleton

from .instruct_pipeline import InstructionTextGenerationPipeline

PIPELINE_REGISTRY.register_pipeline(
    task='dolly-instruct',
    pipeline_class=InstructionTextGenerationPipeline,
    pt_model = AutoModelForCausalLM,
    tf_model = TFAutoModelForCausalLM,
    type='text',
)

class DollyEngine(LLMChat):


    def __init__(self,
                model: str,
                device: Optional[str] = None,
    ) -> None:
        super().__init__(device=device)

        self._pipeline = pipeline(
            'dolly-instruct',
            model=model,
            tokenizer=model,
            framework="pt",
            device=self.device,
            trust_remote_code=True,
        )


    def call(
        self, conversation: Conversation, n_completions: int = 1, **kwargs: Any
    ) -> List[ConversationTurn]:

        # If the length of the conversation is not a single user turn, warn
        if len(conversation) != 1 or conversation[0].speaker != Speaker.USER:
            logging.warning("DollyEngine is designed to be used with a single user turn. This may not be the case.")

        input_string = [c for c in conversation.turns if c.speaker == Speaker.USER][-1].text

        outputs = self._pipeline(
            input_string,
            do_sample=n_completions > 1,
            num_return_sequences=n_completions,
            **kwargs,
        )

        return [ConversationTurn(text=o['generated_text'], speaker=Speaker.AI) for o in outputs] # type: ignore


@register_engine
@singleton
class DollyV23B(DollyEngine):
    name = ('Dolly v2 3B', 'dolly-v2-3b')
    def __init__(self, device: Optional[str] = None) -> None:
        super().__init__(
            model="databricks/dolly-v2-3b",
            device=device,
        )

@register_engine
@singleton
class DollyV27B(DollyEngine):
    name = ('Dolly v2 7B', 'dolly-v2-7b')
    def __init__(self, device: Optional[str] = None) -> None:
        super().__init__(
            model="databricks/dolly-v2-7b",
            device=device,
        )

@register_engine
@singleton
class DollyV212B(DollyEngine):
    name = ('Dolly v2 12B', 'dolly-v2-12b')
    def __init__(self, device: Optional[str] = None) -> None:
        super().__init__(
            model="databricks/dolly-v2-12b",
            device=device,
        )

@register_engine
@singleton
class DollyV16B(DollyEngine):
    name = ('Dolly v1 6B', 'dolly-v1-6b')
    def __init__(self, device: Optional[str] = None) -> None:
        super().__init__(
            model="databricks/dolly-v1-6b",
            device=device,
        )
