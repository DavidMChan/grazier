import copy
from typing import Any, List, Optional, Type

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TFAutoModelForCausalLM,
    is_tf_available,
    is_torch_available,
)
from transformers.pipelines import PIPELINE_REGISTRY, TextGenerationPipeline, pipeline

from grazier.engines.llm import LLMEngine, register_engine
from grazier.utils.python import singleton

# Some models are too new, and not directly supported by the text-generation pipeline, but do have support for
# AutoModelForCausalLM. We can register a new pipeline that doesn't check the model type, and use that instead.
_tgp = copy.copy(TextGenerationPipeline)
_tgp.check_model_type = lambda *args, **kwargs: None
PIPELINE_REGISTRY.register_pipeline(
    task="text-generation-no-check-models",
    pipeline_class=_tgp,
    tf_model=(TFAutoModelForCausalLM,) if is_tf_available() else (),
    pt_model=(AutoModelForCausalLM,) if is_torch_available() else (),
    default={"model": {"pt": ("t5-base", "686f1db"), "tf": ("t5-base", "686f1db")}},
    type="text",
)


class HuggingFaceTextGenerationLMEngine(LLMEngine):
    def __init__(self, model: str, device: Optional[str] = None, override_check_models: bool = False):
        super().__init__(device=device)

        self._generator = pipeline(
            "text-generation" if not override_check_models else "text-generation-no-check-models",
            model=model,
            tokenizer=AutoTokenizer.from_pretrained(model),
            framework="pt",
            device=self.device,
            trust_remote_code=True,
        )

    def call(self, prompt: str, n_completions: int = 1, **kwargs: Any) -> List[str]:
        # Handle the kwargs
        _params = {
            "max_new_tokens": kwargs.get("max_new_tokens", kwargs.pop("max_tokens", 256)),
            "min_new_tokens": kwargs.get("min_new_tokens", 10),
            "num_return_sequences": n_completions,
            "temperature": kwargs.get("temperature", None),
            "return_full_text": False,
        } | kwargs

        outputs = self._generator(
            prompt,
            do_sample=n_completions > 1,
            **_params,
        )

        outputs = [g["generated_text"].strip() for g in outputs]  # type: ignore

        return outputs

    @staticmethod
    def from_hub_model(modelstr: str) -> Type["HuggingFaceTextGenerationLMEngine"]:
        class _RemoteHFModel(HuggingFaceTextGenerationLMEngine):
            name = (modelstr, modelstr)

            def __init__(self, device: Optional[str] = None):
                super().__init__(modelstr, device=device)

        _cf_class = singleton(_RemoteHFModel)
        return _cf_class  # type: ignore


@register_engine
@singleton
class GPTJ6B(HuggingFaceTextGenerationLMEngine):
    name = ("GPT-J 6B", "gptj-6B")

    def __init__(self, device: Optional[str] = None) -> None:
        super().__init__("EleutherAI/gpt-j-6B", device=device)


@register_engine
@singleton
class GPT2(HuggingFaceTextGenerationLMEngine):
    name = ("GPT-2", "gpt2")

    def __init__(self, device: Optional[str] = None) -> None:
        super().__init__("gpt2", device=device)


@register_engine
@singleton
class GPT2Med(HuggingFaceTextGenerationLMEngine):
    name = ("GPT-2 Medium", "gpt2-med")

    def __init__(self, device: Optional[str] = None) -> None:
        super().__init__("gpt2-medium", device=device)


@register_engine
@singleton
class GPT2Lg(HuggingFaceTextGenerationLMEngine):
    name = ("GPT-2 Large", "gpt2-lg")

    def __init__(self, device: Optional[str] = None) -> None:
        super().__init__("gpt2-large", device=device)


@register_engine
@singleton
class GPT2XL(HuggingFaceTextGenerationLMEngine):
    name = ("GPT-2 XL", "gpt2-xl")

    def __init__(self, device: Optional[str] = None) -> None:
        super().__init__("gpt2-xl", device=device)


@register_engine
@singleton
class DistilGPT2(HuggingFaceTextGenerationLMEngine):
    name = ("DistilGPT-2", "distilgpt2")

    def __init__(self, device: Optional[str] = None) -> None:
        super().__init__("distilgpt2", device=device)


@register_engine
@singleton
class GPTNeo125M(HuggingFaceTextGenerationLMEngine):
    name = ("GPT-Neo 125M", "gptneo-125M")

    def __init__(self, device: Optional[str] = None) -> None:
        super().__init__("EleutherAI/gpt-neo-125M", device=device)


@register_engine
@singleton
class GPTNeo1B(HuggingFaceTextGenerationLMEngine):
    name = ("GPT-Neo 1.3B", "gptneo-1.3B")

    def __init__(self, device: Optional[str] = None) -> None:
        super().__init__("EleutherAI/gpt-neo-1.3B", device=device)


@register_engine
@singleton
class GPTNeo2B(HuggingFaceTextGenerationLMEngine):
    name = ("GPT-Neo 2.7B", "gptneo-2.7B")

    def __init__(self, device: Optional[str] = None) -> None:
        super().__init__("EleutherAI/gpt-neo-2.7B", device=device)


@register_engine
@singleton
class StableLMBase3B(HuggingFaceTextGenerationLMEngine):
    name = ("StableLM 3B", "stablelm-3b")

    def __init__(self, device: Optional[str] = "defer") -> None:
        super().__init__("stabilityai/stablelm-base-alpha-3b", device=device)


@register_engine
@singleton
class StableLMBase7B(HuggingFaceTextGenerationLMEngine):
    name = ("StableLM 7B", "stablelm-7b")

    def __init__(self, device: Optional[str] = "defer") -> None:
        super().__init__("stabilityai/stablelm-base-alpha-7b", device=device)


@register_engine
@singleton
class OPT125M(HuggingFaceTextGenerationLMEngine):
    name = ("OPT 125M", "opt-125M")

    def __init__(self, device: Optional[str] = None) -> None:
        super().__init__("facebook/opt-125m", device=device)


@register_engine
@singleton
class OPT350M(HuggingFaceTextGenerationLMEngine):
    name = ("OPT 350M", "opt-350M")

    def __init__(self, device: Optional[str] = None) -> None:
        super().__init__("facebook/opt-350m", device=device)


@register_engine
@singleton
class OPT1x3B(HuggingFaceTextGenerationLMEngine):
    name = ("OPT 1.3B", "opt-1.3b")

    def __init__(self, device: Optional[str] = None) -> None:
        super().__init__("facebook/opt-1.3b", device=device)


@register_engine
@singleton
class OPT2x7B(HuggingFaceTextGenerationLMEngine):
    name = ("OPT 2.7B", "opt-2.7b")

    def __init__(self, device: Optional[str] = None) -> None:
        super().__init__("facebook/opt-2.7b", device=device)


@register_engine
@singleton
class OPT6x7B(HuggingFaceTextGenerationLMEngine):
    name = ("OPT 6.7B", "opt-6.7b")

    def __init__(self, device: Optional[str] = None) -> None:
        super().__init__("facebook/opt-6.7b", device=device)


@register_engine
@singleton
class OPT13B(HuggingFaceTextGenerationLMEngine):
    name = ("OPT 13B", "opt-13b")

    def __init__(self, device: Optional[str] = None) -> None:
        super().__init__("facebook/opt-13b", device=device)


@register_engine
@singleton
class OPT30B(HuggingFaceTextGenerationLMEngine):
    name = ("OPT 30B", "opt-30b")

    def __init__(self, device: Optional[str] = None) -> None:
        super().__init__("facebook/opt-30b", device=device)


@register_engine
@singleton
class OPT66B(HuggingFaceTextGenerationLMEngine):
    name = ("OPT 66B", "opt-66b")

    def __init__(self, device: Optional[str] = None) -> None:
        super().__init__("facebook/opt-66b", device=device)


@register_engine
@singleton
class Falcon40B(HuggingFaceTextGenerationLMEngine):
    name = ("Falcon 40B", "falcon-40b")

    def __init__(self, device: Optional[str] = None) -> None:
        super().__init__("tiiuae/falcon-40b", device=device)


@register_engine
@singleton
class Falcon7B(HuggingFaceTextGenerationLMEngine):
    name = ("Falcon 7B", "falcon-7b")

    def __init__(self, device: Optional[str] = None) -> None:
        super().__init__("tiiuae/falcon-7b", device=device)


@register_engine
@singleton
class FalconRW7B(HuggingFaceTextGenerationLMEngine):
    name = ("Falcon RW 7B", "falcon-rw-7b")

    def __init__(self, device: Optional[str] = None) -> None:
        super().__init__("tiiuae/falcon-rw-7b", device=device, override_check_models=True)


@register_engine
@singleton
class FalconRW1B(HuggingFaceTextGenerationLMEngine):
    name = ("Falcon RW 1B", "falcon-rw-1b")

    def __init__(self, device: Optional[str] = None) -> None:
        super().__init__("tiiuae/falcon-rw-1b", device=device, override_check_models=True)
