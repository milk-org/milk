"""C2: the help of plan commands (`moka <plan> -h`, the rows of `moka -h`)."""

from __future__ import annotations

import click
import pytest
from click.testing import CliRunner

from moka.operate.cli import moka
from moka.operate.plans import PLANS
from moka.operate.tasks import TASKS


def _invoke(*args: str):
    return CliRunner().invoke(moka, list(args))


def _text(result) -> str:
    return result.output + ("" if result.exception is None else str(result.exception))


def test_mkcm_help_says_what_the_plan_takes_and_does():
    result = _invoke("mkcm", "-h")
    assert result.exit_code == 0, _text(result)
    out = result.output
    assert PLANS["mkcm"].help in out
    assert "FLAVOR is one of tsvd, tikhonov, pinv (default: tsvd)." in out
    assert "Keep at most NMODES modes." in out
    assert "required for tikhonov" in out
    lines = [line.strip() for line in out.splitlines()]
    assert "Steps:" in lines
    (step,) = [x for x in lines if x.startswith("1. MakeCM")]
    assert "Writes calib/CMmodes" in step
    assert "Runs on a cold or a deployed pipeline. Stops at the first failure." in out
    assert "FLAVOR is one of" in lines[lines.index("Steps:") - 2]  # not rewrapped


def test_a_bad_choice_is_a_usage_error_naming_the_choices():
    for args, choice in ((("mkcm", "nope"), "tsvd"), (("mkpokes", "nope"), "zonal")):
        result = _invoke(*args)
        assert result.exit_code == 2 and choice in result.output, _text(result)


def test_choices_leave_the_signature_and_the_defaults_alone():
    reg = next(p for p in PLANS["mkcm"].params if p.name == "reg")
    assert reg.default is None
    ctx = click.Context(moka)
    (option,) = [
        o
        for o in moka.get_command(ctx, "mkcm").params
        if isinstance(o, click.Option) and "-r" in o.opts
    ]
    assert option.default is None
    assert "mkcm [-p PIPE] [FLAVOR] [-n NMODES] [-r REG]" in _invoke("-h").output


def test_every_builtin_is_described():
    for name in PLANS.names():
        plan = PLANS[name]
        assert plan.help.isascii() and plan.help, name
        for p in plan.params:
            assert p.positional or (p.help and p.help.isascii()), (name, p.name)
        out = _invoke(name, "-h").output
        assert out.isascii() and plan.help in out, name
    for name in TASKS.names():
        doc = TASKS[name].__dict__.get("__doc__")
        assert doc and doc.isascii(), name


HELPED = (
    '\n[plans.probe]\nsteps = ["ReleaseRundir"]\naccepts = "any"\nhelp = "Probe it."\n'
    + '\n[plans.plain]\nsteps = ["ReleaseRundir"]\n'
    + '\n[plans.notask]\nsteps = ["Nope"]\n'
)


def test_a_toml_plan_help_reaches_both_helps(make_tmp_pipeline, pp_env):
    pp = make_tmp_pipeline(HELPED)
    root = str(pp.root_folder)
    listing = _invoke("-h", "-p", root).output
    assert "Probe it." in listing and "Run the plain plan." in listing
    assert "Probe it." in _invoke("probe", "-h", "-p", root).output
    assert "Run the plain plan." in _invoke("plain", "-h", "-p", root).output


def test_a_plan_with_an_unknown_task_still_has_help(make_tmp_pipeline, pp_env):
    pp = make_tmp_pipeline(HELPED)
    result = _invoke("notask", "-h", "-p", str(pp.root_folder))
    assert result.exit_code == 0, _text(result)
    assert "1. Nope" in result.output
