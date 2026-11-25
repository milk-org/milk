"""C5a: bash tab completion of moka and mokainit (click's shell completion).

Completions are driven in-process with click's ShellComplete, as bash would ask for them, and
end to end through click's env protocol (_MOKA_COMPLETE=bash_complete, COMP_WORDS, COMP_CWORD).
"""

from __future__ import annotations

import os
import subprocess
import sys

from click.shell_completion import ShellComplete
from click.testing import CliRunner

from moka.operate.cli import moka
from moka.operate.plans import PLANS
from moka.provision.cli import mokainit

FIXED = ("state", "plans", "tasks", "link", "unlink", "nuke", "completion")
# a local plan, and a local override of a built-in one (logs a WARNING when loaded)
LOCAL = (
    '\n[plans.probe]\nsteps = ["RunStop"]\naccepts = "any"\n'
    + '\n[plans.teardown]\nsteps = ["RunStop", "ReleaseRundir"]\naccepts = "any"\n'
)


def _complete(args: list[str], incomplete: str = "", cli=moka, prog: str = "moka"):
    """(type, value) of each candidate for `<prog> <args> <incomplete><TAB>`."""
    var = f"_{prog.upper()}_COMPLETE"
    items = ShellComplete(cli, {}, prog, var).get_completions(args, incomplete)
    return [(item.type, item.value) for item in items]


def _values(args: list[str], incomplete: str = "") -> list[str]:
    return [value for _, value in _complete(args, incomplete)]


def _env(words: list[str]) -> dict[str, str]:
    """click's bash protocol: the words of the line, the last one being completed."""
    return {
        "_MOKA_COMPLETE": "bash_complete",
        "COMP_WORDS": " ".join(words),
        "COMP_CWORD": str(len(words) - 1),
    }


def _bash(*words: str):
    """`moka <words><TAB>` through the env protocol, in-process."""
    return CliRunner().invoke(moka, [], env=_env(["moka", *words]), prog_name="moka")


def _broken(pp) -> None:
    with (pp.root_folder / "conf" / "conf.toml").open("a") as stream:
        stream.write("\ngarbage = = \n")


# ---- for free: commands, plans, options and choices ----


def test_commands_and_builtin_plans_complete(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # not a rootdir
    assert _values([], "mk") == ["mkcm", "mkpokes"]
    names = _values([])
    assert set(FIXED) <= set(names) and set(PLANS.names()) <= set(names)


def test_choices_and_options_of_plans_complete():
    assert set(_values(["mkcm"])) == {"tsvd", "tikhonov", "pinv"}
    (types,) = [p.choices for p in PLANS["mkpokes"].params if p.choices]
    assert _values(["mkpokes"]) == list(types) and types
    options = _values(["mkcm"], "-")
    for opt in ("-n", "--nmodes", "-r", "--reg", "-p", "--pipe"):
        assert opt in options


def test_env_protocol_prints_type_value_lines(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = _bash("mk")
    assert result.exit_code == 0, result.output
    assert result.output.splitlines() == ["plain,mkcm", "plain,mkpokes"]


# ---- the plans of the pipeline of -p, else cwd ----


def test_local_plans_complete_with_pipe_or_in_the_rootdir(
    make_tmp_pipeline, monkeypatch
):
    pp = make_tmp_pipeline(LOCAL)
    assert "probe" in _values(["-p", str(pp.root_folder)])
    assert _values(["-p", str(pp.root_folder)], "pr") == ["probe"]
    monkeypatch.chdir(pp.root_folder)
    assert "probe" in _values([])


def test_a_broken_pipeline_completes_the_builtin_commands_silently(make_tmp_pipeline):
    pp = make_tmp_pipeline(LOCAL)
    _broken(pp)
    result = _bash("-p", str(pp.root_folder), "")
    assert result.exit_code == 0 and result.stderr == ""
    expected = sorted([*FIXED, *PLANS.names()])
    assert result.output.splitlines() == [f"plain,{name}" for name in expected]
    result = _bash(
        "-p", str(pp.root_folder), "nope", ""
    )  # an unknown command: no traceback
    assert result.exit_code == 0 and result.exception is None and result.stderr == ""


# ---- completers: compute units, linkages, plan names, pipes ----


def test_compute_units_complete_once(pp):
    rd = str(pp.root_folder)
    assert _values(["runstart", "-p", rd]) == ["delay0", "delay1"]
    assert _values(["runstart", "-p", rd, "delay0"]) == ["delay1"]
    assert _values(["runstop", "-p", rd], "delay1") == ["delay1"]


def test_linkages_and_plan_names_complete(make_tmp_pipeline):
    pp = make_tmp_pipeline(LOCAL)
    rd = str(pp.root_folder)
    linkages = list(pp.config.systemlinkage)
    assert linkages and _values(["link", "-p", rd]) == linkages
    assert _values(["unlink", "-p", rd]) == [*linkages, "all"]
    assert _values(["tasks", "-p", rd]) == pp.plans.names()
    assert _values(["-p", rd, "tasks"], "pr") == ["probe"]  # -p before the command


def test_pipe_completes_short_names_of_the_rootdirs_of_cwd(pp, tmp_path, monkeypatch):
    (tmp_path / "fake-rootdir").mkdir()  # not a rootdir
    monkeypatch.chdir(tmp_path)
    assert pp.root_folder.parent == tmp_path
    short = pp.root_folder.name.removesuffix("-rootdir")
    assert _complete(["-p"]) == [("plain", short)]
    assert _complete(["state", "-p"], short[:2]) == [("plain", short)]
    assert _complete(["-p"], "/some/pa") == [("dir", "/some/pa")]
    assert _complete(["-p"], "zz") == [("dir", "zz")]  # nothing matches: a folder


def test_a_failing_completer_offers_nothing(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert _values(["runstart", "-p", "/nope"]) == []
    result = _bash("runstart", "-p", "/nope", "")
    assert result.exit_code == 0 and result.stderr == ""
    assert result.output == "\n"  # click echoes the empty list of candidates


# ---- silence ----


def test_completion_is_silent_but_commands_still_warn(make_tmp_pipeline, pp_env):
    """A subprocess: in-process, pytest's handlers on the root logger would hide Python's
    last-resort handler, which is what prints a WARNING at each TAB."""
    pp = make_tmp_pipeline(LOCAL)  # its teardown overrides the built-in one
    code = "from moka.operate.cli import moka; moka(prog_name='moka')"
    env = {**os.environ, **_env(["moka", "-p", str(pp.root_folder), ""])}
    done = subprocess.run(
        [sys.executable, "-c", code], env=env, capture_output=True, text=True
    )
    assert done.returncode == 0 and done.stderr == ""
    assert "plain,probe" in done.stdout.splitlines() and "WARNING" not in done.stdout
    result = CliRunner().invoke(moka, ["plans", "-p", str(pp.root_folder)])
    assert result.exit_code == 0, result.output
    assert "Local plan 'teardown' overrides another one." in result.stderr


# ---- the completion command ----


def test_completion_prints_the_bash_script(tmp_path):
    for cli, prog in ((moka, "moka"), (mokainit, "mokainit")):
        result = CliRunner().invoke(cli, ["completion", "bash"], prog_name=prog)
        assert result.exit_code == 0, result.output
        assert result.stderr == "" and "rootdir:" not in result.output
        script = result.output
        assert f"_{prog.upper()}_COMPLETE=bash_complete" in script
        assert any(
            line.split()[:1] == ["complete"] and line.split()[-1] == prog
            for line in script.splitlines()
        )
        path = tmp_path / f"{prog}.bash"
        path.write_text(script)
        assert subprocess.run(["bash", "-n", str(path)]).returncode == 0
    assert CliRunner().invoke(moka, ["completion", "zsh"]).exit_code == 2


def test_completion_names_the_console_script_it_runs_as(tmp_path):
    """As a console script sets argv[0]: the env var and the complete line follow it."""
    for prog, module in (("moka", "operate"), ("mokainit", "provision")):
        code = (
            f"import sys; sys.argv[0] = '/usr/bin/{prog}'; "
            f"from moka.{module}.cli import {prog}; {prog}()"
        )
        done = subprocess.run(
            [sys.executable, "-c", code, "completion", "bash"],
            capture_output=True,
            text=True,
            cwd=tmp_path,
        )
        assert done.returncode == 0 and done.stderr == ""
        assert f"_{prog.upper()}_COMPLETE=bash_complete" in done.stdout
        assert f"-F _{prog}_completion {prog}\n" in done.stdout


def test_completion_needs_no_pipeline(make_tmp_pipeline, monkeypatch):
    pp = make_tmp_pipeline(LOCAL)
    monkeypatch.chdir(pp.root_folder)  # a rootdir whose plans would warn when loaded
    result = CliRunner().invoke(moka, ["completion", "bash"])
    assert result.exit_code == 0 and result.stderr == ""
    assert "_MOKA_COMPLETE" in result.output and "rootdir:" not in result.output
    assert _values(["completion"]) == ["bash"]


def test_completion_is_a_fixed_command(make_tmp_pipeline):
    result = CliRunner().invoke(moka, ["-h"])
    lines = [line.strip() for line in result.output.splitlines()]
    assert "completion SHELL" in lines
    pp = make_tmp_pipeline(
        '\n[plans.completion]\nsteps = ["RunStop"]\naccepts = "any"\n'
    )
    result = CliRunner().invoke(moka, ["-h", "-p", str(pp.root_folder)])
    assert result.exit_code != 0
    text = result.output + str(result.exception)
    assert "clash" in text and "completion" in text


# ---- mokainit ----


def test_mokainit_paths_complete_as_folders():
    items = _complete(["init"], "", cli=mokainit, prog="mokainit")
    assert items == [("dir", "")]
    assert ("plain", "completion") in _complete([], "", cli=mokainit, prog="mokainit")
