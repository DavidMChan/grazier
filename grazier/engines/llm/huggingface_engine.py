import copy
from typing import Any, List, Optional, Type
import logging

import torch

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TFAutoModelForCausalLM,
    is_tf_available,
    is_torch_available,
    BitsAndBytesConfig,
)
from transformers.pipelines import PIPELINE_REGISTRY, TextGenerationPipeline, pipeline

from grazier.engines.llm import LLMEngine, register_engine
from grazier.utils.huggingface import check_huggingface_model_files_are_local
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
    def __init__(
        self,
        model: str,
        device: Optional[str] = None,
        override_check_models: bool = False,
        quantize: bool = False,
    ):
        super().__init__(device=device)

        # Setup quantization
        if quantize:
            nf4_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.float16,
            )
        else:
            nf4_config = None

        self._model = AutoModelForCausalLM.from_pretrained(
            model,
            device_map="auto" if not self.device_specified else None,
            quantization_config=nf4_config,
            trust_remote_code=True,
        )
        self._tokenizer = AutoTokenizer.from_pretrained(
            model,
            trust_remote_code=True,
            padding_side="left",
        )
        self._tokenizer.pad_token_id = self._model.config.eos_token_id or self._tokenizer.eos_token_id

        self._generator = pipeline(
            "text-generation" if not override_check_models else "text-generation-no-check-models",
            model=self._model,
            tokenizer=self._tokenizer,
            framework="pt",
            device=self.device if self.device_specified else None,
            device_map="auto" if not self.device_specified else None,
            torch_dtype=torch.float16 if quantize else None,
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
            pad_token_id=self._tokenizer.pad_token_id,
            eos_token_id=self._tokenizer.eos_token_id,
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

    @staticmethod
    def requires_configuration() -> bool:
        # Check if there's an internet connection, if so, then we need to check if the model is available
        # locally. If not, then we can't use this engine.
        import requests

        try:
            requests.get("https://huggingface.co/")
            return False
        except requests.exceptions.ConnectionError:
            return True
        except Exception as e:
            logging.warning(f"Unexpected error when checking if HuggingFace is available: {e}")
            return True


@register_engine
@singleton
class GPTJ6B(HuggingFaceTextGenerationLMEngine):
    name = ("GPT-J 6B", "gptj-6B")

    def __init__(self, device: Optional[str] = None) -> None:
        super().__init__("EleutherAI/gpt-j-6B", device=device)

    @staticmethod
    def is_configured() -> bool:
        return check_huggingface_model_files_are_local("EleutherAI/gpt-j-6B")


@register_engine
@singleton
class GPT2(HuggingFaceTextGenerationLMEngine):
    name = ("GPT-2", "gpt2")

    def __init__(self, device: Optional[str] = None) -> None:
        super().__init__("gpt2", device=device)

    @staticmethod
    def is_configured() -> bool:
        return check_huggingface_model_files_are_local("gpt2")


@register_engine
@singleton
class GPT2Med(HuggingFaceTextGenerationLMEngine):
    name = ("GPT-2 Medium", "gpt2-med")

    def __init__(self, device: Optional[str] = None) -> None:
        super().__init__("gpt2-medium", device=device)

    @staticmethod
    def is_configured() -> bool:
        return check_huggingface_model_files_are_local("gpt2-medium")


@register_engine
@singleton
class GPT2Lg(HuggingFaceTextGenerationLMEngine):
    name = ("GPT-2 Large", "gpt2-lg")

    def __init__(self, device: Optional[str] = None) -> None:
        super().__init__("gpt2-large", device=device)

    @staticmethod
    def is_configured() -> bool:
        return check_huggingface_model_files_are_local("gpt2-large")


@register_engine
@singleton
class GPT2XL(HuggingFaceTextGenerationLMEngine):
    name = ("GPT-2 XL", "gpt2-xl")

    def __init__(self, device: Optional[str] = None) -> None:
        super().__init__("gpt2-xl", device=device)

    @staticmethod
    def is_configured() -> bool:
        return check_huggingface_model_files_are_local("gpt2-xl")


@register_engine
@singleton
class DistilGPT2(HuggingFaceTextGenerationLMEngine):
    name = ("DistilGPT-2", "distilgpt2")

    def __init__(self, device: Optional[str] = None) -> None:
        super().__init__("distilgpt2", device=device)

    @staticmethod
    def is_configured() -> bool:
        return check_huggingface_model_files_are_local("distilgpt2")


@register_engine
@singleton
class GPTNeo125M(HuggingFaceTextGenerationLMEngine):
    name = ("GPT-Neo 125M", "gptneo-125M")

    def __init__(self, device: Optional[str] = None) -> None:
        super().__init__("EleutherAI/gpt-neo-125M", device=device)

    @staticmethod
    def is_configured() -> bool:
        return check_huggingface_model_files_are_local("EleutherAI/gpt-neo-125M")


@register_engine
@singleton
class GPTNeo1B(HuggingFaceTextGenerationLMEngine):
    name = ("GPT-Neo 1.3B", "gptneo-1.3B")

    def __init__(self, device: Optional[str] = None) -> None:
        super().__init__("EleutherAI/gpt-neo-1.3B", device=device)

    @staticmethod
    def is_configured() -> bool:
        return check_huggingface_model_files_are_local("EleutherAI/gpt-neo-1.3B")


@register_engine
@singleton
class GPTNeo2B(HuggingFaceTextGenerationLMEngine):
    name = ("GPT-Neo 2.7B", "gptneo-2.7B")

    def __init__(self, device: Optional[str] = None) -> None:
        super().__init__("EleutherAI/gpt-neo-2.7B", device=device)

    @staticmethod
    def is_configured() -> bool:
        return check_huggingface_model_files_are_local("EleutherAI/gpt-neo-2.7B")


@register_engine
@singleton
class StableLMBase3B(HuggingFaceTextGenerationLMEngine):
    name = ("StableLM 3B", "stablelm-3b")

    def __init__(self, device: Optional[str] = "defer") -> None:
        super().__init__("stabilityai/stablelm-base-alpha-3b", device=device)

    @staticmethod
    def is_configured() -> bool:
        return check_huggingface_model_files_are_local("stabilityai/stablelm-base-alpha-3b")


@register_engine
@singleton
class StableLMBase7B(HuggingFaceTextGenerationLMEngine):
    name = ("StableLM 7B", "stablelm-7b")

    def __init__(self, device: Optional[str] = "defer") -> None:
        super().__init__("stabilityai/stablelm-base-alpha-7b", device=device)

    @staticmethod
    def is_configured() -> bool:
        return check_huggingface_model_files_are_local("stabilityai/stablelm-base-alpha-7b")


@register_engine
@singleton
class OPT125M(HuggingFaceTextGenerationLMEngine):
    name = ("OPT 125M", "opt-125M")

    def __init__(self, device: Optional[str] = None) -> None:
        super().__init__("facebook/opt-125m", device=device)

    @staticmethod
    def is_configured() -> bool:
        return check_huggingface_model_files_are_local("facebook/opt-125m")


@register_engine
@singleton
class OPT350M(HuggingFaceTextGenerationLMEngine):
    name = ("OPT 350M", "opt-350M")

    def __init__(self, device: Optional[str] = None) -> None:
        super().__init__("facebook/opt-350m", device=device)

    @staticmethod
    def is_configured() -> bool:
        return check_huggingface_model_files_are_local("facebook/opt-350m")


@register_engine
@singleton
class OPT1x3B(HuggingFaceTextGenerationLMEngine):
    name = ("OPT 1.3B", "opt-1.3b")

    def __init__(self, device: Optional[str] = None) -> None:
        super().__init__("facebook/opt-1.3b", device=device)

    @staticmethod
    def is_configured() -> bool:
        return check_huggingface_model_files_are_local("facebook/opt-1.3b")


@register_engine
@singleton
class OPT2x7B(HuggingFaceTextGenerationLMEngine):
    name = ("OPT 2.7B", "opt-2.7b")

    def __init__(self, device: Optional[str] = None) -> None:
        super().__init__("facebook/opt-2.7b", device=device)

    @staticmethod
    def is_configured() -> bool:
        return check_huggingface_model_files_are_local("facebook/opt-2.7b")


@register_engine
@singleton
class OPT6x7B(HuggingFaceTextGenerationLMEngine):
    name = ("OPT 6.7B", "opt-6.7b")

    def __init__(self, device: Optional[str] = None) -> None:
        super().__init__("facebook/opt-6.7b", device=device)

    @staticmethod
    def is_configured() -> bool:
        return check_huggingface_model_files_are_local("facebook/opt-6.7b")


@register_engine
@singleton
class OPT13B(HuggingFaceTextGenerationLMEngine):
    name = ("OPT 13B", "opt-13b")

    def __init__(self, device: Optional[str] = None) -> None:
        super().__init__("facebook/opt-13b", device=device)

    @staticmethod
    def is_configured() -> bool:
        return check_huggingface_model_files_are_local("facebook/opt-13b")


@register_engine
@singleton
class OPT30B(HuggingFaceTextGenerationLMEngine):
    name = ("OPT 30B", "opt-30b")

    def __init__(self, device: Optional[str] = None) -> None:
        super().__init__("facebook/opt-30b", device=device)

    @staticmethod
    def is_configured() -> bool:
        return check_huggingface_model_files_are_local("facebook/opt-30b")


@register_engine
@singleton
class OPT66B(HuggingFaceTextGenerationLMEngine):
    name = ("OPT 66B", "opt-66b")

    def __init__(self, device: Optional[str] = None) -> None:
        super().__init__("facebook/opt-66b", device=device)

    @staticmethod
    def is_configured() -> bool:
        return check_huggingface_model_files_are_local("facebook/opt-66b")


@register_engine
@singleton
class Falcon40B(HuggingFaceTextGenerationLMEngine):
    name = ("Falcon 40B", "falcon-40b")

    def __init__(self, device: Optional[str] = None) -> None:
        super().__init__("tiiuae/falcon-40b", device=device)

    @staticmethod
    def is_configured() -> bool:
        return check_huggingface_model_files_are_local("tiiuae/falcon-40b")


@register_engine
@singleton
class Falcon7B(HuggingFaceTextGenerationLMEngine):
    name = ("Falcon 7B", "falcon-7b")

    def __init__(self, device: Optional[str] = None) -> None:
        super().__init__("tiiuae/falcon-7b", device=device)

    @staticmethod
    def is_configured() -> bool:
        return check_huggingface_model_files_are_local("tiiuae/falcon-7b")


@register_engine
@singleton
class FalconRW7B(HuggingFaceTextGenerationLMEngine):
    name = ("Falcon RW 7B", "falcon-rw-7b")

    def __init__(self, device: Optional[str] = None) -> None:
        super().__init__("tiiuae/falcon-rw-7b", device=device, override_check_models=True)

    @staticmethod
    def is_configured() -> bool:
        return check_huggingface_model_files_are_local("tiiuae/falcon-rw-7b")


@register_engine
@singleton
class FalconRW1B(HuggingFaceTextGenerationLMEngine):
    name = ("Falcon RW 1B", "falcon-rw-1b")

    def __init__(self, device: Optional[str] = None) -> None:
        super().__init__("tiiuae/falcon-rw-1b", device=device, override_check_models=True)

    @staticmethod
    def is_configured() -> bool:
        return check_huggingface_model_files_are_local("tiiuae/falcon-rw-1b")


@register_engine
@singleton
class H2OGPT_GM_OASST1_EN_2048_FALCON7B(HuggingFaceTextGenerationLMEngine):
    name = ("H2OGPT_GM_OASST1_EN_2048_FALCON7B", "h2ogpt-gm-oasst1-en-2048-falcon-7b")

    def __init__(self, device: Optional[str] = None) -> None:
        super().__init__("h2oai/h2ogpt-gm-oasst1-en-2048-falcon-7b", device=device, override_check_models=True)

    @staticmethod
    def is_configured() -> bool:
        return check_huggingface_model_files_are_local("h2oai/h2ogpt-gm-oasst1-en-2048-falcon-7b")

    @property
    def prompt_prefix(self) -> str:
        return "<|prompt|>"

    @property
    def prompt_suffix(self) -> str:
        return "<|endoftext|><|answer|>"


@register_engine
@singleton
class LLama27B(HuggingFaceTextGenerationLMEngine):
    name = ("Llama-2 (7B)", "llama-2-7b")

    def __init__(self, device: Optional[str] = None) -> None:
        super().__init__("meta-llama/Llama-2-7b-hf", device=device, quantize=True)

    @staticmethod
    def is_configured() -> bool:
        return check_huggingface_model_files_are_local("meta-llama/Llama-2-7b-hf")


@register_engine
@singleton
class LLama213B(HuggingFaceTextGenerationLMEngine):
    name = ("Llama-2 (13B)", "llama-2-13b")

    def __init__(self, device: Optional[str] = None) -> None:
        super().__init__("meta-llama/Llama-2-13b-hf", device=device, quantize=True)

    @staticmethod
    def is_configured() -> bool:
        return check_huggingface_model_files_are_local("meta-llama/Llama-2-13b-hf")


@register_engine
@singleton
class LLama270B(HuggingFaceTextGenerationLMEngine):
    name = ("Llama-2 (70B)", "llama-2-70b")

    def __init__(self, device: Optional[str] = None) -> None:
        super().__init__("meta-llama/Llama-2-70b-hf", device=device, quantize=True)

    @staticmethod
    def is_configured() -> bool:
        return check_huggingface_model_files_are_local("meta-llama/Llama-2-70b-hf")
