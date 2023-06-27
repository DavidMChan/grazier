
import click


@click.group()
def main() -> None:
    pass


@main.command()
def list_engines() -> None:
    """List available engines."""
    from grazier.engines.llm import LM_ENGINES_CLI
    from grazier.engines.chat import LM_CHAT_ENGINES_CLI

    print("Available LLM engines (Completion):")

    # Types here are super broken because of the wrapped decorator
    for cls in sorted([cls.name for cls in LM_ENGINES_CLI.values()]): # type: ignore
        print(f"\t - {cls[0]}/{cls[1]}") # type: ignore

    print("Available LLM engines (Chat):")
    # Types here are super broken because of the wrapped decorator
    for cls in sorted([cls.name for cls in LM_CHAT_ENGINES_CLI.values()]): # type: ignore
        print(f"\t - {cls[0]}/{cls[1]}") # type: ignore
