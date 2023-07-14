import click


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

    print("Available Models (Completions API):")

    # Types here are super broken because of the wrapped decorator
    for k, v in sorted(LM_ENGINES_CLI.items()):  # type: ignore
        engine_is_configured = v.is_configured()
        if configured and not engine_is_configured:
            continue
        print(f"\t - {v.name[0]}/{k} (Configured: {engine_is_configured})")  # type: ignore

    print("Available Models (Chat API):")
    # Types here are super broken because of the wrapped decorator
    # Types here are super broken because of the wrapped decorator
    for k, v in sorted(LM_CHAT_ENGINES_CLI.items()):  # type: ignore
        engine_is_configured = v.is_configured()
        if configured and not engine_is_configured:
            continue
        print(f"\t - {v.name[0]}/{k} (Configured: {engine_is_configured})")  # type: ignore


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
