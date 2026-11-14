"""Fixtures for TUI regression tests (milk-CTRL and the legacy TUIs).

Purpose:     a TUI test drives the real programs in a terminal without touching the user's shm,
             history files or tmux sessions, and leaves no pane or process behind.
Does:        tui_env gives each test its own MILK_SHM_DIR (under the directory that
             tests/conftestaux/milk.py sets for the session), HOME, TMPDIR and terminal settings,
             also in os.environ so that the private tmux server inherits them, and skips the test
             without milk-CTRL or tmux; tmux_server starts that private server; entities makes
             streams, FPS and processinfo loops; milkctrl and legacy_pane start milk-CTRL or
             another TUI in a pane; milkctrl_pty runs milk-CTRL on a pty for byte-level and
             tty-state checks.
Used by:     legacy_tui_test.py, milkctrl_tui_test.py, unit_tui_test.py.
Uses:        libtmux (the tests are skipped without it); milkctrl_driver.py: TuiPane, MilkCtrl,
             PtyRun; entities.py; tests/cli/conftest.py: _install_dir, _isolated_shm_env.
See also:    docs/guide/milk-CTRL.md; docs/guide/fpsCTRL_reference.md.
"""

from __future__ import annotations

import contextlib
import os
import shutil
import uuid
from pathlib import Path

import pytest

from tests.cli.conftest import _install_dir, _isolated_shm_env
from tests.tui.entities import Entities
from tests.tui.milkctrl_driver import MilkCtrl, PtyRun, TuiPane

libtmux = pytest.importorskip("libtmux")

_PYDIR = Path(__file__).resolve().parents[2]


@pytest.fixture
def tui_env(tmp_path, monkeypatch):
    """(env, shm_dir) for one test; also applied to os.environ so the private tmux server inherits it."""
    if (
        "MILK_INSTALLDIR" not in os.environ
        or not (_install_dir() / "bin" / "milk-CTRL").exists()
    ):
        pytest.skip("MILK_INSTALLDIR/bin/milk-CTRL not found")
    if shutil.which("tmux") is None:
        pytest.skip("tmux not installed")
    for d in ("home", "tmp", "cwd"):
        (tmp_path / d).mkdir()
    env, shm_dir = _isolated_shm_env(
        {
            "HOME": str(tmp_path / "home"),
            "TMPDIR": str(tmp_path / "tmp"),
            "TERM": "xterm-256color",
            "COLORTERM": "truecolor",
            "LANG": "C.UTF-8",
            "PYTHONPATH": str(_PYDIR),
        }
    )
    for k in ("NO_COLOR", "MILK_CTRL_THEME", "TMUX"):
        env.pop(k, None)
        monkeypatch.delenv(k, raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    yield env, shm_dir
    shutil.rmtree(shm_dir, ignore_errors=True)


@pytest.fixture
def tmux_server(tui_env):
    server = libtmux.Server(socket_name=f"mctl-{uuid.uuid4().hex[:8]}")
    yield server
    try:
        server.kill()
    except Exception:
        pass


@pytest.fixture
def entities(tui_env):
    env, shm_dir = tui_env
    ent = Entities(env, shm_dir)
    yield ent
    ent.cleanup()


@pytest.fixture
def milkctrl(tui_env, tmux_server, tmp_path):
    """Factory: milkctrl(*args, x=200, y=50, env_extra=None) -> MilkCtrl running in a private tmux pane."""
    env, _ = tui_env
    panes: list[MilkCtrl] = []

    def launch(
        *args: str,
        x: int = 200,
        y: int = 50,
        env_extra: dict[str, str] | None = None,
        ready: bool = True,
    ) -> MilkCtrl:
        e = dict(env)
        if env_extra:
            e.update(env_extra)
        p = MilkCtrl(
            tmux_server,
            [str(_install_dir() / "bin" / "milk-CTRL"), *args],
            e,
            x=x,
            y=y,
            cwd=str(tmp_path / "cwd"),
        )
        panes.append(p)
        if ready:
            p.wait_for(r"F2:DASH")  # keys sent before raw mode would be flushed
        return p

    yield launch
    for p in panes:
        p.kill()


@pytest.fixture
def legacy_pane(tui_env, tmux_server):
    """Factory: legacy_pane('milk-streamCTRL', *args) -> TuiPane."""
    env, _ = tui_env
    panes: list[TuiPane] = []

    def launch(tool: str, *args: str, x: int = 200, y: int = 50) -> TuiPane:
        p = TuiPane(
            tmux_server,
            [str(_install_dir() / "bin" / tool), *args],
            dict(env),
            x=x,
            y=y,
        )
        panes.append(p)
        return p

    yield launch
    for p in panes:
        p.kill()


@pytest.fixture
def milkctrl_pty(tui_env):
    """Context-manager factory: with milkctrl_pty(*args) as run: ... (byte-level and tty-state checks)."""
    env, _ = tui_env

    @contextlib.contextmanager
    def start(
        *args: str,
        rows: int = 50,
        cols: int = 160,
        env_extra: dict[str, str] | None = None,
    ):
        e = dict(env)
        if env_extra:
            e.update(env_extra)
        run = PtyRun([str(_install_dir() / "bin" / "milk-CTRL"), *args], e, rows, cols)
        try:
            yield run
        finally:
            run.close()

    return start
