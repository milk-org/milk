"""
Basic sanity for the two milk-cli Python wrappers:
    - CLI          -- persistent interactive REPL (live shell)
    - CLICommands  -- one-shot `milk-cli -c "a;b;c"` batch
"""

from __future__ import annotations

import pytest

from milk.cliwrap import CLI, CLICommands, HAVE_CLI

pytestmark = pytest.mark.skipif(
    not HAVE_CLI, reason="MILK compiled without CLI support"
)


@pytest.mark.timeout(10)
def test_liveshell_echo_roundtrip():
    cli = CLI(strip_ansi=True)
    try:
        assert cli.send_line("echo hello_shell") == "hello_shell"
    finally:
        cli.close()


@pytest.mark.timeout(10)
def test_liveshell_retcode_on_close():
    cli = CLI(strip_ansi=True)
    cli.send_line("echo warmup")
    cli.close()
    assert cli.retcode == 0


@pytest.mark.timeout(10)
def test_clicommands_echo():
    with CLICommands(["echo hello_batch"]) as result:
        assert "hello_batch" in result.stdout


@pytest.mark.timeout(10)
def test_clicommands_multiple():
    with CLICommands(["echo one", "echo two", "echo three"]) as result:
        assert "one" in result.stdout
        assert "two" in result.stdout
        assert "three" in result.stdout
