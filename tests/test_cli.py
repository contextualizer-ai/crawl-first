"""Basic tests for crawl-first CLI."""

from click.testing import CliRunner

from crawl_first.cli import main


def test_cli_help() -> None:
    """Test that CLI help works."""
    runner = CliRunner()
    result = runner.invoke(main, ["--help"])
    assert result.exit_code == 0
    assert "crawl-first" in result.output.lower()


def test_cli_requires_input() -> None:
    """Test that CLI requires either biosample-id or input-file."""
    runner = CliRunner()
    with runner.isolated_filesystem():
        result = runner.invoke(main, ["--email", "test@example.com"])
        assert result.exit_code != 0
        assert "Must specify either --biosample-id or --input-file" in result.output


def test_cli_requires_output() -> None:
    """Test that CLI requires either output-file or output-dir."""
    runner = CliRunner()
    with runner.isolated_filesystem():
        result = runner.invoke(
            main, ["--biosample-id", "nmdc:bsm-11-test", "--email", "test@example.com"]
        )
        assert result.exit_code != 0
        assert "Must specify either --output-file or --output-dir" in result.output
