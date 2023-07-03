import click


@click.group()
def main() -> None:
    pass


@main.command()
def list_engines() -> None:
    """List available engines."""
    from grazier.engines.chat import LM_CHAT_ENGINES_CLI
    from grazier.engines.llm import LM_ENGINES_CLI

    print("Available LLM engines (Completion):")

    # Types here are super broken because of the wrapped decorator
    for k, v in sorted(LM_ENGINES_CLI.items()):  # type: ignore
        print(f"\t - {v.name[0]}/{k}")  # type: ignore

    print("Available LLM engines (Chat):")
    # Types here are super broken because of the wrapped decorator
    # Types here are super broken because of the wrapped decorator
    for k, v in sorted(LM_CHAT_ENGINES_CLI.items()):  # type: ignore
        print(f"\t - {v.name[0]}/{k}")  # type: ignore


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
