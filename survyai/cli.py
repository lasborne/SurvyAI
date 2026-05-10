"""
Command-line interface implemented on top of `SurvyAIAgentService`.

Keeps Click parsing here so `agent/` stays free of CLI concerns.
"""

from __future__ import annotations

import sys

import click

from survyai.version import __version__
from survyai.agent_service import SurvyAIAgentService
from survyai.capabilities import format_capabilities_summary, scan_machine_capabilities


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option(__version__, prog_name="SurvyAI")
def cli() -> None:
    """SurvyAI — surveying & CAD/GIS agent (CLI)."""


@cli.command("query")
@click.argument("query_text", type=str)
@click.option("-o", "--output", type=click.Path(), help="Optional path to save full JSON/text output.")
@click.option("-v", "--verbose", is_flag=True, help="Verbose logging to stderr.")
@click.option(
    "--interactive/--no-interactive",
    default=False,
    help="Prompt for permissions (e.g. internet search) in the terminal.",
)
@click.option(
    "--fallback",
    "use_fallback",
    is_flag=True,
    help="Use the configured fallback LLM instead of the primary.",
)
def query_command(
    query_text: str,
    output: str | None,
    verbose: bool,
    interactive: bool,
    use_fallback: bool,
) -> None:
    """Send a single natural-language task to the agent."""
    if verbose:
        import logging

        logging.getLogger().setLevel(logging.DEBUG)

    service = SurvyAIAgentService(eager_init=True)
    result = service.run_task(
        query_text,
        use_fallback_llm=use_fallback,
        interactive=interactive,
    )

    text_out = result.response
    if not result.success and result.error:
        text_out = f"{text_out}\n\n(Error: {result.error})" if text_out else f"(Error: {result.error})"

    click.echo(text_out)

    if output:
        import json
        from pathlib import Path

        Path(output).write_text(json.dumps(result.raw, indent=2), encoding="utf-8")


@cli.command("gui")
def gui_command() -> None:
    """Launch the SurvyAI Windows desktop application (PySide6)."""
    from survyai.gui.main import run_gui

    raise SystemExit(run_gui())


@cli.command("version")
def version_command() -> None:
    """Print SurvyAI version."""
    click.echo(__version__)


@cli.command("test")
@click.option(
    "--init-agent",
    is_flag=True,
    help="Also initialize the full agent (requires valid API keys).",
)
def test_command(init_agent: bool) -> None:
    """Print machine capability summary; optionally verify agent startup."""
    caps = scan_machine_capabilities()
    click.echo(format_capabilities_summary(caps))
    if init_agent:
        click.echo("\nInitializing agent...")
        SurvyAIAgentService(eager_init=True)
        click.echo("Agent initialized OK.")


def main() -> None:
    # Convenience: `python -m cli "your question"` without typing the `query` subcommand
    if len(sys.argv) > 1 and sys.argv[1] not in (
        "query",
        "test",
        "version",
        "gui",
        "--help",
        "-h",
    ):
        if sys.argv[1] not in (
            "-o",
            "--output",
            "-v",
            "--verbose",
            "--interactive",
            "--no-interactive",
            "--fallback",
        ):
            sys.argv.insert(1, "query")
    cli()


if __name__ == "__main__":
    main()
