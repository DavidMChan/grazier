import logging

import click
from rich import print


@click.group()
def main() -> None:
    pass


@main.command()
@click.option("--configured", is_flag=True, help="Only list configured engines")
def list_engines(
    configured: bool,
) -> None:
    """List available engines."""
    from grazier.engines.chat import LM_CHAT_ENGINES_CLI
    from grazier.engines.llm import LM_ENGINES_CLI

    # Types here are super broken because of the wrapped decorator
    configured_models_llm, unconfigured_models_llm = [], []
    for k, v in sorted(LM_ENGINES_CLI.items()):  # type: ignore
        engine_requires_configuration = v.requires_configuration()
        engine_is_configured = not engine_requires_configuration or v.is_configured()
        if configured and not engine_is_configured:
            continue
        configured_models_llm.append((k, v)) if engine_is_configured else unconfigured_models_llm.append((k, v))

    # Types here are super broken because of the wrapped decorator
    configured_models_chat, unconfigured_models_chat = [], []
    for k, v in sorted(LM_CHAT_ENGINES_CLI.items()):  # type: ignore
        engine_requires_configuration = v.requires_configuration()
        engine_is_configured = not engine_requires_configuration or v.is_configured()
        if configured and not engine_is_configured:
            continue
        configured_models_chat.append((k, v)) if engine_is_configured else unconfigured_models_chat.append((k, v))

    print("Available Models (Completions API, Configured):")
    print("\n".join([f"\t [green]- {v.name[0]}/{k} [/green]" for k, v in configured_models_llm]))  # type: ignore
    if not configured:
        print("\n".join([f"\t [red]- {v.name[0]}/{k} (Unconfigured) [/red]" for k, v in unconfigured_models_llm]))  # type: ignore

    print()
    print("Available Models (Chat API):")
    print("\n".join([f"\t [green]- {v.name[0]}/{k} [/green]" for k, v in configured_models_chat]))  # type: ignore
    if not configured:
        print("\n".join([f"\t [red]- {v.name[0]}/{k} (Unconfigured) [/red]" for k, v in unconfigured_models_chat]))  # type: ignore


@main.command()
@click.argument("engine")
@click.argument("prompt")
def complete(
    engine: str,
    prompt: str,
) -> None:
    from grazier import LLMEngine

    _e = LLMEngine.from_string(engine)
    print(_e(prompt, n_completions=2)[0])


@main.command()
@click.argument("engine")
def configure(
    engine: str,
) -> None:
    """List available engines."""
    from grazier.engines.chat import LM_CHAT_ENGINES_CLI
    from grazier.engines.llm import LM_ENGINES_CLI

    if engine.lower() in LM_ENGINES_CLI:
        LM_ENGINES_CLI[engine.lower()].configure()
    elif engine.lower() in LM_CHAT_ENGINES_CLI:
        LM_CHAT_ENGINES_CLI[engine.lower()].configure()
    else:
        logging.error(f"Engine {engine} not found.")


@main.command()
def reset_credential_store():
    from grazier.utils.secrets import reset_credential_store

    reset_credential_store()
