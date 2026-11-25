"""C4: a plan run tells its progress (moka.plan log records); the CLI renders them on stderr.

Fake tasks: Already (success() true), Ran (runs), Refuse (can() false), Fail (runs, success()
false), Boom (forward() raises a non-MilkError). `pp` is the cold pipelinebasic.
"""

from __future__ import annotations

import io
import logging
import re
import sys
import threading

import click
import pytest
from click.testing import CliRunner

from moka import ui
from moka.errors import BestEffortError, TaskCannotRunError, TaskDidNotSucceedError
from moka.operate.cli import moka
from moka.operate.plans import PLANS, Plan, Step, run_plan
from moka.operate.tasks import TASKS, Task
from tests.conftestaux.polling import poll_until_or_fail


class _Fake(Task):
    done = False
    allowed = True
    works = True

    def can(self):
        self.failure_reason = "needs `deploy` first."
        return self.allowed

    def forward(self):
        self.done = self.works

    def success(self):
        if not self.done:
            self.failure_reason = "still off."
        return self.done


class _Boom(_Fake):
    def forward(self):
        raise ValueError("boom\nsecond line")


FAKES = {
    "Already": type("Already", (_Fake,), {"done": True}),
    "Ran": type("Ran", (_Fake,), {}),
    "Refuse": type("Refuse", (_Fake,), {"allowed": False}),
    "Fail": type("Fail", (_Fake,), {"works": False}),
    "Boom": _Boom,
}


@pytest.fixture
def plan(monkeypatch):
    """plan(*labels, on_failure=, accepts=): register plan `probe` over the fake tasks."""
    monkeypatch.setattr(TASKS, "_items", dict(TASKS._items))
    monkeypatch.setattr(PLANS, "_items", dict(PLANS._items))
    for name, cls in FAKES.items():
        TASKS.register(name, cls)

    def register(*labels, on_failure="abort", accepts="any"):
        steps = tuple(Step(label) for label in labels)
        PLANS.register("probe", Plan("probe", steps, on_failure, accepts))

    return register


def _records(caplog) -> list[dict]:
    return [r.moka for r in caplog.records if r.name == "moka.plan"]


def _invoke(*args: str):
    return CliRunner().invoke(moka, list(args))


# ---- the log records of run() ----


def test_records_of_a_three_step_plan(pp, plan, caplog):
    plan("Already", "Ran", "Ran")
    with caplog.at_level(logging.INFO, logger="moka.plan"):
        run_plan(pp, "probe")
    records = _records(caplog)
    assert [r["event"] for r in records] == ["plan"] + ["start", "end"] * 3 + ["final"]
    first = records[0]
    assert first["pipeline"] == "basic" and first["state"] == "cold"
    assert first["labels"] == ["Already", "Ran", "Ran"]
    ends = records[2:-1:2]
    assert [(r["i"], r["n"], r["label"]) for r in ends] == [
        (1, 3, "Already"),
        (2, 3, "Ran"),
        (3, 3, "Ran"),
    ]
    assert [r["outcome"] for r in ends] == ["already", "done", "done"]
    final = records[-1]
    assert final["ok"] and final["failed"] == [] and final["seconds"] >= 0
    messages = [r.getMessage() for r in caplog.records if r.name == "moka.plan"]
    assert messages[0] == "probe (cold): Already -> Ran -> Ran"
    assert messages[2] == "probe [1/3] Already: already done"
    assert re.fullmatch(r"probe \[2/3\] Ran: done \(\d+\.\d s\)", messages[4])
    assert re.fullmatch(r"probe done in \d+\.\d s", messages[-1])


@pytest.mark.parametrize(
    "task, outcome, text, raised",
    [
        ("Already", "already", "already done", None),
        ("Ran", "done", r"done \(\d+\.\d s\)", None),
        ("Refuse", "refused", "refused: needs `deploy` first.", TaskCannotRunError),
        ("Fail", "failed", r"failed \(\d+\.\d s\): still off.", TaskDidNotSucceedError),
        ("Boom", "failed", r"failed \(\d+\.\d s\): boom", ValueError),
    ],
)
def test_one_record_per_outcome(pp, plan, caplog, task, outcome, text, raised):
    plan(task)
    with caplog.at_level(logging.INFO, logger="moka.plan"):
        if raised is None:
            run_plan(pp, "probe")
        else:
            with pytest.raises(raised) as info:
                run_plan(pp, "probe")
    end, final = _records(caplog)[-2:]
    assert end["event"] == "end" and end["outcome"] == outcome
    assert re.fullmatch(text, end["text"])
    assert final["ok"] is (raised is None)
    if raised is ValueError:
        assert str(info.value) == "boom\nsecond line"  # unchanged
        assert end["reason"] == "boom"
    if raised is not None:
        assert final["failed"] == [task]
        assert final["text"] == f"probe: stopped at step 1/1 ({task})"


def test_best_effort_tells_every_step_then_the_failures(pp, plan, caplog):
    plan("Fail", "Ran", "Refuse", on_failure="best_effort")
    with caplog.at_level(logging.INFO, logger="moka.plan"):
        with pytest.raises(BestEffortError) as info:
            run_plan(pp, "probe")
    assert [label for label, _ in info.value.failures] == ["Fail", "Refuse"]
    assert [type(e) for _, e in info.value.failures] == [
        TaskDidNotSucceedError,
        TaskCannotRunError,
    ]
    records = _records(caplog)
    ends = [r for r in records if r["event"] == "end"]
    assert [r["outcome"] for r in ends] == ["failed", "done", "refused"]
    assert records[-1]["failed"] == ["Fail", "Refuse"] and not records[-1]["ok"]
    assert records[-1]["text"] == "probe: 2 of 3 steps failed (Fail, Refuse)"


def test_the_api_prints_nothing_without_a_handler(pp, plan, capsys):
    plan("Ran", "Fail", on_failure="best_effort")
    with pytest.raises(BestEffortError):
        run_plan(pp, "probe")
    assert capsys.readouterr() == ("", "")


# ---- the CLI ----


def test_a_plan_run_is_told_on_stderr(pp, plan):
    plan("Already", "Ran")
    result = _invoke("probe", "-p", str(pp.root_folder))
    assert result.exit_code == 0, result.output
    assert result.stdout == ""
    lines = result.stderr.splitlines()
    assert lines[:3] == ["> moka probe", f"rootdir: {pp.root_folder}", "basic: cold"]
    assert lines[3] == "  [1/2] Already ... [ -- ] already done"
    assert re.fullmatch(r"  \[2/2\] Ran \.{7} \[ ok \] done \(\d+\.\d s\)", lines[4])
    assert re.fullmatch(r"\[ ok \] probe done in \d+\.\d s", lines[5])
    assert len(lines) == 6


def test_a_refused_step_is_told_once(pp, plan):
    plan("Refuse", "Ran")
    result = _invoke("probe", "-p", str(pp.root_folder))
    assert result.exit_code == 1
    assert result.stderr.splitlines()[3:] == [
        "  [1/2] Refuse ... [ NO ] refused: needs `deploy` first.",
        "[FAIL] probe: stopped at step 1/2 (Refuse)",
    ]
    assert result.output.count("needs `deploy` first.") == 1
    assert "Error" not in result.output


def test_a_failed_step_and_a_failed_best_effort_plan(pp, plan):
    plan("Fail", "Boom", "Ran", on_failure="best_effort")
    result = _invoke("probe", "-p", str(pp.root_folder))
    assert result.exit_code == 1
    lines = result.stderr.splitlines()[3:]
    assert re.fullmatch(
        r"  \[1/3\] Fail \.{3} \[FAIL\] failed \(\d+\.\d s\): still off\.", lines[0]
    )
    assert re.fullmatch(
        r"  \[2/3\] Boom \.{3} \[FAIL\] failed \(\d+\.\d s\): boom", lines[1]
    )
    assert (
        lines[2] == "[FAIL] probe: stopped at step 2/3 (Boom)"
    )  # not a MilkError: stops
    assert "Error: ValueError: boom" in result.stderr  # and is printed as before
    assert "[3/3]" not in result.stderr


def test_a_best_effort_failure_is_one_verdict(pp, plan):
    plan("Fail", "Ran", "Refuse", on_failure="best_effort")
    result = _invoke("probe", "-p", str(pp.root_folder))
    assert result.exit_code == 1
    lines = result.stderr.splitlines()[3:]
    assert [line.split("] ", 1)[1][:6] for line in lines[:3]] == [
        "Fail .",
        "Ran ..",
        "Refuse",
    ]
    assert lines[3:] == ["[FAIL] probe: 2 of 3 steps failed (Fail, Refuse)"]


def test_a_gate_refusal_says_the_state_and_why(pp, plan):
    plan("Ran", accepts="consistent")
    result = _invoke("probe", "-p", str(pp.root_folder))
    assert result.exit_code == 1 and result.stdout == ""
    assert result.stderr.splitlines() == [
        "> moka probe",
        f"rootdir: {pp.root_folder}",
        "basic: cold",
        "[FAIL] Pipeline basic is cold; run deploy.",
        "pipeline: basic",
    ]


@pytest.mark.parametrize("args", [("state",), ("plans",), ("tasks",), ("link",)])
def test_read_only_commands_keep_stdout_and_say_nothing_on_stderr(pp, args):
    result = _invoke(*args, "-p", str(pp.root_folder))
    assert result.exit_code == 0, result.output
    assert result.stderr == "" and result.stdout == result.output
    assert result.stdout.splitlines()[0] == f"rootdir: {pp.root_folder}"
    if args != ("link",):
        assert result.stdout.splitlines()[1] == "basic: cold"


def test_acting_commands_print_nothing_on_stdout(pp, paper, shm_dir_cleanafter):
    from moka.operate.tasks import ClaimRundir

    for name in ("otherloop_output", "input_remote"):
        (shm_dir_cleanafter / f"{name}.im.shm").write_bytes(b"")
    pp.task_do(ClaimRundir)
    paper["basic_delay0"] = paper["basic_delay1"] = (True, False)
    for args in (("link", "shared"), ("unlink", "shared"), ("nuke", "--yes")):
        result = _invoke(*args, "-p", str(pp.root_folder))
        assert result.exit_code == 0, (args, result.output)
        assert result.stdout == "", (args, result.output)
        assert result.stderr.startswith(
            f"> moka {args[0]}\nrootdir: {pp.root_folder}\n"
        )


def _digits(text: str) -> str:
    return re.sub(r"\d+\.\d s", "N s", text)


@pytest.mark.parametrize("labels", [("Already", "Ran"), ("Refuse",), ("Fail", "Ran")])
def test_narration_is_ascii_and_colored_only_on_demand(pp, plan, labels):
    plan(*labels, on_failure="best_effort")
    plain = _invoke("probe", "-p", str(pp.root_folder))
    assert plain.stderr.isascii()
    assert "\r" not in plain.stderr and "\x1b" not in plain.stderr
    colored = _invoke("--color=always", "probe", "-p", str(pp.root_folder))
    assert "\x1b[" in colored.stderr
    assert _digits(click.unstyle(colored.stderr)) == _digits(plain.stderr)


# ---- the handler on a terminal ----


class _Terminal(io.StringIO):
    def isatty(self) -> bool:
        return True


@pytest.fixture
def replay(pp, plan, caplog, monkeypatch):
    """The records of a run of plan probe (Ran), and a PlanHandler on a fake terminal."""
    plan("Ran")
    with caplog.at_level(logging.INFO, logger="moka.plan"):
        run_plan(pp, "probe")
    monkeypatch.setenv("NO_COLOR", "1")
    monkeypatch.setattr(sys, "stderr", _Terminal())
    monkeypatch.setattr(ui.PlanHandler, "tick", 0.001)
    records = [r for r in caplog.records if r.name == "moka.plan"]
    handler = ui.PlanHandler()
    yield records, handler, sys.stderr
    handler.close()


def _threads_back_to(count: int) -> None:
    poll_until_or_fail(lambda: threading.active_count() == count, 5)


def test_a_running_step_counts_its_seconds_then_shows_its_outcome(replay):
    (first, start, end, final), handler, stream = replay
    threads = threading.active_count()
    handler.handle(first)
    handler.handle(start)
    assert stream.getvalue() == "basic: cold\n  [1/1] Ran ..."  # open, no newline
    poll_until_or_fail(
        lambda: re.search(r"\r  \[1/1\] Ran \.\.\. \d+ s$", stream.getvalue()), 5
    )
    handler.handle(end)
    _threads_back_to(threads)
    text = stream.getvalue()
    assert text.endswith(f"\r\x1b[K  [1/1] Ran ... [ ok ] {end.moka['text']}\n")
    handler.handle(final)
    assert stream.getvalue().endswith(f"[ ok ] {final.moka['text']}\n")


def test_another_record_closes_the_open_step_line(replay):
    (first, start, end, _), handler, stream = replay
    threads = threading.active_count()
    handler.handle(first)
    handler.handle(start)
    handler.handle(
        logging.makeLogRecord(
            {
                "name": "moka",
                "levelno": logging.WARNING,
                "levelname": "WARNING",
                "msg": "hey",
            }
        )
    )
    _threads_back_to(threads)
    handler.handle(end)
    lines = stream.getvalue().split("\n")
    assert lines[2] == "WARNING: hey"
    assert lines[1].startswith("  [1/1] Ran ...")  # the open line, then its counter
    assert lines[3] == f"  [1/1] Ran ... [ ok ] {end.moka['text']}"  # whole, no \r


def test_close_stops_the_counter(replay):
    (first, start, _, _), handler, stream = replay
    threads = threading.active_count()
    handler.handle(first)
    handler.handle(start)
    assert threading.active_count() == threads + 1
    handler.close()
    _threads_back_to(threads)


def test_off_a_terminal_a_start_prints_nothing(replay, monkeypatch):
    (first, start, end, final), _, _ = replay
    stream = io.StringIO()
    handler = ui.PlanHandler()
    handler.setStream(stream)
    threads = threading.active_count()
    for record in (first, start):
        handler.handle(record)
    assert stream.getvalue() == "basic: cold\n" and threading.active_count() == threads
    handler.handle(end)
    handler.handle(final)
    assert "\r" not in stream.getvalue() and "\x1b" not in stream.getvalue()
    handler.close()
