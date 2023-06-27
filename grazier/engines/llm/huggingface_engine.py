
from typing import Any, List, Optional, Type

from transformers.pipelines import pipeline

from grazier.engines.llm import LLMEngine, register_engine
from grazier.utils.python import singleton


class HuggingFaceTextGenerationLMEngine(LLMEngine):
    def __init__(self, model: str, device: Optional[str] = None):
        super().__init__(device=device)
        self._generator = pipeline(
            "text-generation",
            model=model,
            framework="pt",
            device=self.device,
            trust_remote_code=True,
        )

    def call(
        self, prompt: str, n_completions: int = 1, temperature: Optional[float] = None, **kwargs: Any
    ) -> List[str]:
        if temperature is not None and temperature > 0:
            outputs = self._generator(
                prompt,
                max_new_tokens=256,
                min_new_tokens=10,
                num_return_sequences=n_completions,
                do_sample=True,
                temperature=temperature,
                return_full_text=False,
            )
        elif n_completions > 1:
            outputs = self._generator(
                prompt,
                max_new_tokens=256,
                min_new_tokens=10,
                num_return_sequences=n_completions,
                do_sample=True,
                return_full_text=False,
            )
        else:
            outputs = self._generator(
                prompt,
                max_new_tokens=256,
                min_new_tokens=10,
                num_return_sequences=n_completions,
                do_sample=False,
                return_full_text=False,
            )

        outputs = [g["generated_text"].strip() for g in outputs]  # type: ignore

        return outputs


    @staticmethod
    def from_hub_model(modelstr: str) -> Type['HuggingFaceTextGenerationLMEngine']:
        class _RemoteHFModel(HuggingFaceTextGenerationLMEngine):
            name = (modelstr, modelstr)
            def __init__(self, device: Optional[str] = None):
                super().__init__(modelstr, device=device)

        _cf_class = singleton(_RemoteHFModel)
        return _cf_class # type: ignore



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
    name = ("StableLM 3B", "stablelm-3B")
    def __init__(self, device: Optional[str] = "defer") -> None:
        super().__init__("stabilityai/stablelm-base-alpha-3b", device=device)


@register_engine
@singleton
class StableLMBase7B(HuggingFaceTextGenerationLMEngine):
    name = ("StableLM 7B", "stablelm-7B")
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