import logging
from typing import Any, List, Optional

from transformers import AutoModelForCausalLM, AutoTokenizer, TFAutoModelForCausalLM
from transformers.pipelines import PIPELINE_REGISTRY, pipeline

from grazier.engines.chat import Conversation, ConversationTurn, LLMChat, Speaker, register_engine
from grazier.utils.huggingface import check_huggingface_model_files_are_local
from grazier.utils.python import singleton

from .instruct_pipeline import InstructionTextGenerationPipeline

PIPELINE_REGISTRY.register_pipeline(
    task="dolly-instruct",
    pipeline_class=InstructionTextGenerationPipeline,
    pt_model=AutoModelForCausalLM,
    tf_model=TFAutoModelForCausalLM,
    type="text",
)


class DollyEngine(LLMChat):
    def __init__(
        self,
        model: str,
        device: Optional[str] = None,
    ) -> None:
        super().__init__(device=device)

        self._pipeline = pipeline(
            "dolly-instruct",
            model=model,
            tokenizer=AutoTokenizer.from_pretrained(model),
            framework="pt",
            device=self.device,
            trust_remote_code=True,
        )

    def call(self, conversation: Conversation, n_completions: int = 1, **kwargs: Any) -> List[ConversationTurn]:
        # If the length of the conversation is not a single user turn, warn
        system_turn_list = [c for c in conversation if c.speaker == Speaker.SYSTEM]
        if len(system_turn_list) > 1:
            logging.warning(
                "DollyEngine is designed to be used with a single user (and optionally, a single system) turn. This may not be the case."
            )
        if len(system_turn_list) > 0:
            system_turn = system_turn_list[-1]
        else:
            system_turn = None

        non_system_turns = [c for c in conversation if c.speaker != Speaker.SYSTEM]
        if len(non_system_turns) != 1 or non_system_turns[0].speaker != Speaker.USER:
            logging.warning(
                "DollyEngine is designed to be used with a single user (and optionally, a single system) turn. This may not be the case."
            )

        input_string = [c for c in non_system_turns if c.speaker == Speaker.USER][-1].text
        outputs = self._pipeline(
            input_string,
            do_sample=n_completions > 1,
            num_return_sequences=n_completions,
            intro_blurb=system_turn.text if system_turn is not None else None,
            **kwargs,
        )

        return [ConversationTurn(text=o["generated_text"], speaker=Speaker.AI) for o in outputs]  # type: ignore


@register_engine
@singleton
class DollyV23B(DollyEngine):
    name = ("Dolly v2 3B", "dolly-v2-3b")

    def __init__(self, device: Optional[str] = None) -> None:
        super().__init__(
            model="databricks/dolly-v2-3b",
            device=device,
        )

    @staticmethod
    def is_configured() -> bool:
        return check_huggingface_model_files_are_local("databricks/dolly-v2-3b")


@register_engine
@singleton
class DollyV27B(DollyEngine):
    name = ("Dolly v2 7B", "dolly-v2-7b")

    def __init__(self, device: Optional[str] = None) -> None:
        super().__init__(
            model="databricks/dolly-v2-7b",
            device=device,
        )

    @staticmethod
    def is_configured() -> bool:
        return check_huggingface_model_files_are_local("databricks/dolly-v2-7b")


@register_engine
@singleton
class DollyV212B(DollyEngine):
    name = ("Dolly v2 12B", "dolly-v2-12b")

    def __init__(self, device: Optional[str] = None) -> None:
        super().__init__(
            model="databricks/dolly-v2-12b",
            device=device,
        )

    @staticmethod
    def is_configured() -> bool:
        return check_huggingface_model_files_are_local("databricks/dolly-v2-12b")


@register_engine
@singleton
class DollyV16B(DollyEngine):
    name = ("Dolly v1 6B", "dolly-v1-6b")

    def __init__(self, device: Optional[str] = None) -> None:
        super().__init__(
            model="databricks/dolly-v1-6b",
            device=device,
        )

    @staticmethod
    def is_configured() -> bool:
        return check_huggingface_model_files_are_local("databricks/dolly-v1-6b")


@register_engine
@singleton
class MPT7BInstruct(DollyEngine):
    name = ("MPT 7B Instruct", "mpt-7b-instruct")

    def __init__(self, device: Optional[str] = None) -> None:
        super().__init__(
            model="mosaicml/mpt-7b-instruct",
            device=device,
        )

    @staticmethod
    def is_configured() -> bool:
        return check_huggingface_model_files_are_local("mosaicml/mpt-7b-instruct")


@register_engine
@singleton
class MPT30BInstruct(DollyEngine):
    name = ("MPT 30B Instruct", "mpt-30b-instruct")

    def __init__(self, device: Optional[str] = None) -> None:
        super().__init__(
            model="mosaicml/mpt-30b-instruct",
            device=device,
        )

    @staticmethod
    def is_configured() -> bool:
        return check_huggingface_model_files_are_local("mosaicml/mpt-30b-instruct")
