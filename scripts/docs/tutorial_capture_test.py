"""Tests of tutorial_capture.py: no milk, tmux or live system needed.

python -m pytest -p no:cacheprovider scripts/docs
"""

import importlib.util
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location(
    "tutorial_capture", Path(__file__).with_name("tutorial_capture.py")
)
tc = importlib.util.module_from_spec(_spec)
sys.modules["tutorial_capture"] = tc
_spec.loader.exec_module(tc)

GOOD = """\
# Page

<!-- tut:run id=mk -->
```bash
mkdir -p ~/AOloop2 && cd ~/AOloop2
echo hello
```

<!-- tut:out id=mk -->
```text title="Output"
old output
```

<!-- tut:run id=refused expect=1 -->
```bash
moka mkcm -p aoloop
```

!!! tip "Look around"

    <!-- tut:py id=close-loop -->
    ```python
    print("x")
    ```

<!-- tut:tui id=fpsctrl cmd="milk-fpsCTRL -n aoloop_mfilt" size=150x16 keys="Right Down" -->
![fpsCTRL](img/moka_tutorial/fpsctrl.svg)

<!-- tut:fig id=mlat -->
![mlat](img/moka_tutorial/mlat.png)

<!-- tut:manual id=edit-loopgain -->
```bash
vi aoloop-conf/fpsdetails.toml
```

<!-- tut:manual -->
```python
viewer()
```

Trailing text.
"""


@pytest.fixture(autouse=True)
def no_calibration_checks(monkeypatch):
    """GOOD has none of the real page's calibration ids."""
    monkeypatch.setattr(tc, "CALIBRATION_CHECKS", {})


def write_page(tmp_path, text, images=("fpsctrl.svg", "mlat.png")):
    page = tmp_path / "moka_tutorial.md"
    page.write_text(text)
    img = tmp_path / "img/moka_tutorial"
    img.mkdir(parents=True, exist_ok=True)
    for name in images:
        (img / name).write_bytes(b"x")
    return page


# ---------------------------------------------------------------- parser


def test_parser_six_kinds():
    page = tc.parse(GOOD)
    assert page.errors == []
    kinds = [(m.kind, m.id) for m in page.markers]
    assert kinds == [
        ("run", "mk"),
        ("out", "mk"),
        ("run", "refused"),
        ("py", "close-loop"),
        ("tui", "fpsctrl"),
        ("fig", "mlat"),
        ("manual", "edit-loopgain"),
        ("manual", None),
    ]
    by_id = {m.id: m for m in page.markers if m.kind != "out"}
    assert by_id["refused"].attrs["expect"] == "1"
    assert by_id["fpsctrl"].attrs["cmd"] == "milk-fpsCTRL -n aoloop_mfilt"
    assert by_id["fpsctrl"].attrs["size"] == "150x16"
    assert by_id["fpsctrl"].attrs["keys"] == "Right Down"
    assert by_id["fpsctrl"].image == "img/moka_tutorial/fpsctrl.svg"
    assert by_id["close-loop"].fence.lang == "python"
    assert by_id["close-loop"].fence.indent == "    "
    assert tc.block_code(page, by_id["close-loop"]) == 'print("x")\n'


def blank_form(text):
    """GOOD with one blank line (container-indented in the admonition) after each marker."""
    out = []
    for ln in text.splitlines(keepends=True):
        out.append(ln)
        if ln.lstrip().startswith("<!-- tut:"):
            out.append("    \n" if ln.startswith("    ") else "\n")
    return "".join(out)


def test_parser_blank_line_form():
    text = blank_form(GOOD)
    page = tc.parse(text)
    assert page.errors == []
    ref = tc.parse(GOOD)
    assert [(m.kind, m.id) for m in page.markers] == [
        (m.kind, m.id) for m in ref.markers
    ]
    assert all(m.fence or m.image for m in page.markers)
    by_id = {m.id: m for m in page.markers if m.kind != "out"}
    assert by_id["close-loop"].fence.indent == "    "
    assert by_id["fpsctrl"].image == "img/moka_tutorial/fpsctrl.svg"
    assert tc.block_code(page, by_id["close-loop"]) == 'print("x")\n'


def test_check_blank_line_form(tmp_path):
    page = write_page(tmp_path, blank_form(GOOD))
    assert tc.check_page(page.read_text(), tmp_path) == []


def test_check_two_blank_lines_or_content_is_error(tmp_path):
    for gap in ("\n\n", "\ntext\n"):
        text = "<!-- tut:py id=p -->\n" + gap + "```python\nx\n```\n"
        errors = tc.check_page(text, tmp_path)
        assert any("not followed by a python block" in e for e in errors)


def test_parser_syntax_errors():
    text = (
        "<!-- tut:bogus id=a -->\n<!-- tut:run id=a color=red -->\n<!-- tut:run -->\n"
    )
    errors = tc.parse(text).errors
    assert any("unknown marker tut:bogus" in e for e in errors)
    assert any("does not take 'color=red'" in e for e in errors)
    assert any("has no id" in e for e in errors)


def test_markers_inside_fences_are_ignored():
    text = "```text\n<!-- tut:run id=x -->\n```\n"
    assert tc.parse(text).markers == []


# ---------------------------------------------------------------- --check


def test_check_good_page(tmp_path):
    page = write_page(tmp_path, GOOD)
    assert tc.check_page(page.read_text(), tmp_path) == []


def test_check_block_without_marker(tmp_path):
    text = GOOD + "\n```bash\nls\n```\n"
    page = write_page(tmp_path, text)
    errors = tc.check_page(page.read_text(), tmp_path)
    assert errors == [f"line {GOOD.count(chr(10)) + 2}: bash block without marker"]


def test_check_duplicate_id(tmp_path):
    text = GOOD + "\n<!-- tut:run id=mk -->\n```bash\nls\n```\n"
    errors = tc.check_page(write_page(tmp_path, text).read_text(), tmp_path)
    assert len(errors) == 1 and "duplicate id mk" in errors[0]


def test_check_out_without_run(tmp_path):
    text = GOOD + "\n<!-- tut:out id=nope -->\n```text\n```\n"
    errors = tc.check_page(write_page(tmp_path, text).read_text(), tmp_path)
    assert len(errors) == 1 and "tut:out id=nope has no tut:run" in errors[0]


def test_check_calibration_id_without_run(tmp_path, monkeypatch):
    monkeypatch.setattr(tc, "CALIBRATION_CHECKS", {"mk": False, "gone": True})
    errors = tc.check_page(write_page(tmp_path, GOOD).read_text(), tmp_path)
    assert errors == ["no tut:run id=gone (a key of CALIBRATION_CHECKS)"]


def test_check_missing_image(tmp_path):
    page = write_page(tmp_path, GOOD, images=("fpsctrl.svg",))
    errors = tc.check_page(page.read_text(), tmp_path)
    assert len(errors) == 1 and "missing image img/moka_tutorial/mlat.png" in errors[0]


def test_check_wrong_target(tmp_path):
    text = GOOD + "\n<!-- tut:py id=p2 -->\n```bash\nls\n```\n"
    errors = tc.check_page(write_page(tmp_path, text).read_text(), tmp_path)
    assert len(errors) == 1 and "not followed by a python block" in errors[0]


def test_check_cli(tmp_path, monkeypatch, capsys):
    page = write_page(tmp_path, GOOD + "\n```python\nx\n```\n")
    monkeypatch.setattr(tc, "PAGE", page)
    monkeypatch.setattr(tc, "REPO", tmp_path)
    assert tc.main(["--check"]) == 1
    assert "python block without marker" in capsys.readouterr().err
    page.write_text(GOOD)
    assert tc.main(["--check"]) == 0


# ---------------------------------------------------------------- pure helpers


def test_rewrite_in_place():
    new = tc.rewrite(GOOD, {"mk": "hello\nworld\n"})
    assert new == GOOD.replace("old output\n", "hello\nworld\n")
    assert tc.rewrite(new, {"mk": "hello\nworld\n"}) == new


def test_rewrite_keeps_indent_and_other_bytes():
    text = "a\r\n\n    <!-- tut:out id=x -->\n    ```text\n    old\n    ```\nb  \n"
    new = tc.rewrite(text, {"x": "one\n\ntwo"})
    assert (
        new
        == "a\r\n\n    <!-- tut:out id=x -->\n    ```text\n    one\n\n    two\n    ```\nb  \n"
    )
    assert tc.rewrite(new, {"x": "one\n\ntwo"}) == new


def test_rewrite_blank_line_form():
    text = blank_form(GOOD)
    new = tc.rewrite(text, {"mk": "hello\nworld\n"})
    assert new == text.replace("old output\n", "hello\nworld\n")
    assert tc.rewrite(new, {"mk": "hello\nworld\n"}) == new


def test_normalize_order():
    home = Path("/home/someone")
    shm = Path("/home/someone/shm/tutorial_capture")
    root = "/home/someone/src/milk-standalone/"
    text = (
        "wrote /home/someone/AOloop2/x.fits (0.8 s)\n"
        'Cannot open shm file "/home/someone/shm/tutorial_capture/dm07disp.im.shm"\n'
        "clone /home/someone/src/milk-standalone/milk-examples/pipelines\n"
    )
    assert tc.normalize(text, home, shm, root) == (
        "wrote ~/AOloop2/x.fits (0.8 s)\n"
        'Cannot open shm file "/milk/shm/dm07disp.im.shm"\n'
        "clone ~/src/milk/milk-examples/pipelines\n"
    )


def test_plain_drops_escapes_and_trailing_spaces():
    text = (
        "\x1b[1;35mWARNING   \x1b[0;m\n"
        '\x1b[1;35m (PID 12) Cannot open shm file "/milk/shm/x.im.shm"\n'
        "  \x1b[0;m\n"
        "  [1/4] ClaimRundir ... [ ok ] done (0.0 s)\n"
    )
    assert tc.plain(text) == (
        "WARNING\n"
        ' (PID 12) Cannot open shm file "/milk/shm/x.im.shm"\n'
        "\n"
        "  [1/4] ClaimRundir ... [ ok ] done (0.0 s)\n"
    )


def test_mask_pids():
    text = ' (PID 2541931) Cannot open shm file "/milk/shm/x.im.shm"\nPID 12 (PID x)\n'
    assert tc.mask_pids(text) == (
        ' (PID n) Cannot open shm file "/milk/shm/x.im.shm"\nPID 12 (PID x)\n'
    )


def test_calibration_refusal():
    assert tc.calibration_refusal(5.7e4, None, False) is None
    assert tc.calibration_refusal(5.7e4, 2.7, True) is None
    assert "light is off" in tc.calibration_refusal(-300.0, 2.9, True)
    assert "light is off" in tc.calibration_refusal(300.0, None, False)
    assert "no result" in tc.calibration_refusal(5.7e4, None, True)
    for bad in (-2.34, 6.13):
        assert "outside 1.5 to 4.5" in tc.calibration_refusal(5.7e4, bad, True)


def test_milk_root_refusal(tmp_path):
    assert "not set" in tc.milk_root_refusal(None)
    assert "not set" in tc.milk_root_refusal("")
    assert "does not exist" in tc.milk_root_refusal(str(tmp_path))
    (tmp_path / "milk-examples/pipelines").mkdir(parents=True)
    assert tc.milk_root_refusal(str(tmp_path)) is None


def test_start_refusals(tmp_path):
    home, shm = tmp_path / "home", tmp_path / "shm" / "tutorial_capture"
    home.mkdir()
    assert tc.start_refusals(home, shm) == []
    (home / "AOloop2").mkdir()
    shm.mkdir(parents=True)
    assert tc.start_refusals(home, shm) == [
        f"{home / 'AOloop2'} exists",
        f"{shm} exists",
    ]


# ---------------------------------------------------------------- ANSI to SVG

ANSI = (
    "\x1b[1m\x1b[7m fpsCTRL \x1b[0m <a&b>\n"
    "\x1b[31mred\x1b[39m \x1b[92mbright\x1b[0m\n"
    "\x1b[38;5;208morange\x1b[48;5;240mgray-bg\x1b[0m\n"
    "\x1b[38;2;10;200;30mtrue\x1b[48;2;1;2;3mcolor\x1b[m \x1b[?25l\x1b(B\x07end\n"
)


def test_ansi_to_svg_parses():
    svg = tc.ansi_to_svg(ANSI, 40, 4)
    root = ET.fromstring(svg)
    ns = "{http://www.w3.org/2000/svg}"
    texts = [t.text for t in root.iter(f"{ns}text")]
    assert " fpsCTRL " in texts and " <a&b>" in texts and "end" in texts
    fills = {t.text: t.get("fill") for t in root.iter(f"{ns}text")}
    assert fills["red"] == "#cd0000"
    assert fills["bright"] == "#00ff00"
    assert fills["orange"] == "#ff8700"
    assert fills["true"] == "#0ac81e"
    bold = [t for t in root.iter(f"{ns}text") if t.text == " fpsCTRL "][0]
    assert bold.get("font-weight") == "bold"
    rect_fills = {r.get("fill") for r in root.iter(f"{ns}rect")}
    # Reverse video: the default foreground becomes the background.
    assert {tc._FG, "#585858", "#010203"} <= rect_fills


def test_ansi_to_svg_blocks_from_milk_termview():
    # Two image rows and the status lines of a real milk-termview capture.
    ansi = Path(__file__).with_name("tutorial_capture_termview.ansi").read_text()
    root = ET.fromstring(tc.ansi_to_svg(ansi, 80, 5))
    ns = "{http://www.w3.org/2000/svg}"
    texts = "".join(t.text for t in root.iter(f"{ns}text"))
    assert not set("\u2580\u2584\u2588") & set(texts)
    assert "Image: " in texts and "tvtest" in texts and "Val: [0 : 1]" in texts
    rects = list(root.iter(f"{ns}rect"))
    # Lower halves: 8.5 high, at mid-line of rows 0 and 1, in the 24-bit colours.
    lower = [r for r in rects if r.get("height") == "8.5"]
    assert {r.get("y") for r in lower} == {"16.5", "33.5"}
    assert len({r.get("fill") for r in lower}) > 10
    assert "#ffffff" in {r.get("fill") for r in rects if r.get("height") == "17.0"}


def test_ansi_to_svg_merges_block_runs():
    svg = tc.ansi_to_svg(
        "\x1b[38;2;1;2;3m\x1b[48;2;4;5;6m\u2584\u2584\u2584\u2580\n", 4, 1
    )
    rects = [
        (r.get("y"), r.get("width"), r.get("height"), r.get("fill"))
        for r in ET.fromstring(svg).iter("{http://www.w3.org/2000/svg}rect")
    ]
    assert rects[1:] == [
        ("8.0", "33.6", "17.0", "#040506"),
        ("16.5", "25.2", "8.5", "#010203"),
        ("8.0", "8.4", "8.5", "#010203"),
    ]


# ---------------------------------------------------------------- Shell (plain bash)


def test_shell_carries_cd_and_first_failure(tmp_path):
    shell = tc.Shell(tmp_path, dict(os.environ))
    try:
        assert shell.run(f"cd {tmp_path}/..")[0] == 0
        assert shell.run("pwd") == (0, f"{tmp_path.parent}\n")
        assert shell.run("false; true")[0] == 1
        assert shell.run("echo ok") == (0, "ok\n")
    finally:
        shell.close()


def test_shell_exit_raises(tmp_path):
    shell = tc.Shell(tmp_path, dict(os.environ))
    try:
        with pytest.raises(tc.CaptureError, match="the shell exited"):
            shell.run("echo bye\nexit 1")
    finally:
        shell.close()


# ---------------------------------------------------------------- --run, faked


class FakeSystem:
    def __init__(self, fail=None, fail_start=False, fail_close=False, wfs_sum=5.7e4):
        self.fail = fail
        self.wfs_sum = wfs_sum
        self.fail_start, self.fail_close = fail_start, fail_close
        self.calls = []
        self.closed = False

    def __call__(self, cwd, workdir, env):
        self.cwd, self.workdir, self.env = cwd, workdir, env
        self.shm = Path(env["MILK_SHM_DIR"])
        # The private folder exists while the system runs.
        self.shm_existed = self.shm.is_dir()
        if self.fail_start:
            workdir.mkdir()
            raise RuntimeError("start failed")
        return self

    def bash(self, code):
        self.calls.append(("bash", code))
        self.workdir.mkdir(exist_ok=True)
        if "mkcm" in code:
            return 1, "[ NO ] refused\n"
        if self.fail and self.fail in code:
            return 2, "boom\n"
        return 0, f"{self.workdir}/done {self.shm}/s.im.shm\n"

    def python(self, code):
        self.calls.append(("python", code))
        return 0

    def tui(self, cmd, cols, rows, keys):
        self.calls.append(("tui", cmd, keys))
        return "\x1b[32mok\x1b[0m\n"

    def figure(self, fig_id, dest):
        self.calls.append(("figure", fig_id))
        dest.write_bytes(b"png")

    def manual(self, action_id):
        self.calls.append(("manual", action_id))

    def calibration(self):
        self.calls.append(("calibration",))
        return self.wfs_sum, 2.7

    def close(self):
        # Teardown runs while the private folder still exists.
        assert self.shm.is_dir()
        self.closed = True
        if self.fail_close:
            raise RuntimeError("close failed")


class FakeTmux:
    """Records the env it stops; checks it runs after close, before the removals."""

    def __init__(self, fake):
        self.fake = fake
        self.env = None

    def __call__(self, env):
        assert self.fake.closed or self.fake.fail_start
        assert Path(env["TMUX_TMPDIR"]).is_dir() and self.fake.shm.is_dir()
        self.env = env


@pytest.fixture
def setup(tmp_path):
    home, shm = tmp_path / "home", tmp_path / "shm"
    root = tmp_path / "milk"
    home.mkdir()
    shm.mkdir()
    (root / "milk-examples/pipelines").mkdir(parents=True)
    (tmp_path / "docs").mkdir()
    return write_page(tmp_path / "docs", GOOD), home, shm, str(root)


def run(setup, fake, **kw):
    page, home, shm, root = setup
    kw.setdefault("tmux_stopper", FakeTmux(fake))
    tc.run(page, home, shm, root, system_factory=fake, **kw)


def test_run_writes_all(setup, monkeypatch):
    page, home, shm, _ = setup
    monkeypatch.setenv("TMUX", "/tmp/tmux-1000/default,1,0")
    fake = FakeSystem()
    tmux = FakeTmux(fake)
    run(setup, fake, tmux_stopper=tmux)
    assert fake.closed and not (home / "AOloop2").exists()
    assert fake.shm == shm / "tutorial_capture" and fake.shm_existed
    assert not fake.shm.exists()
    # A private tmux socket folder, stopped and removed at the end.
    assert "TMUX" not in fake.env and tmux.env is fake.env
    assert not Path(fake.env["TMUX_TMPDIR"]).exists()
    assert [c[0] for c in fake.calls] == [
        "bash",
        "bash",
        "python",
        "tui",
        "figure",
        "manual",
    ]
    assert ("tui", "milk-fpsCTRL -n aoloop_mfilt", ["Right", "Down"]) in fake.calls
    text = page.read_text()
    assert text == GOOD.replace("old output\n", "~/AOloop2/done /milk/shm/s.im.shm\n")
    svg = (page.parent / "img/moka_tutorial/fpsctrl.svg").read_text()
    ET.fromstring(svg)
    assert (page.parent / "img/moka_tutorial/mlat.png").read_bytes() == b"png"


def test_run_all_or_none(setup):
    page, home, shm, _ = setup
    fake = FakeSystem(fail="echo hello")
    with pytest.raises(tc.CaptureError, match="exit code 2"):
        run(setup, fake)
    assert fake.closed and not (home / "AOloop2").exists()
    assert fake.shm_existed and not fake.shm.exists()
    assert page.read_text() == GOOD
    assert (page.parent / "img/moka_tutorial/fpsctrl.svg").read_bytes() == b"x"


@pytest.mark.parametrize("wfs_sum", [5.7e4, 300.0])
def test_run_calibration_check(setup, wfs_sum, monkeypatch):
    monkeypatch.setattr(tc, "CALIBRATION_CHECKS", {"measure-mlat": True})
    page = setup[0]
    text = GOOD.replace("tut:run id=mk ", "tut:run id=measure-mlat ").replace(
        "tut:out id=mk ", "tut:out id=measure-mlat "
    )
    page.write_text(text)
    fake = FakeSystem(wfs_sum=wfs_sum)
    if wfs_sum < tc.MIN_WFS_SUM:
        with pytest.raises(tc.CaptureError, match="measure-mlat: bad calibration"):
            run(setup, fake)
        assert page.read_text() == text
    else:
        run(setup, fake)
        assert "done" in page.read_text()
    assert [c[0] for c in fake.calls][:2] == ["bash", "calibration"]


def test_run_refuses_to_start(setup):
    page, home, shm, _ = setup
    (shm / "tutorial_capture").mkdir()
    fake = FakeSystem()
    with pytest.raises(tc.CaptureError, match="refusing to start"):
        run(setup, fake)
    assert fake.calls == [] and page.read_text() == GOOD
    # Not created by this run: left in place.
    assert (shm / "tutorial_capture").is_dir()


@pytest.mark.parametrize("where", ["fail_start", "fail_close"])
def test_run_cleans_up_when_system_fails(setup, where):
    page, home, _, _ = setup
    fake = FakeSystem(**{where: True})
    with pytest.raises(RuntimeError, match="failed"):
        run(setup, fake)
    assert not fake.shm.exists() and not (home / "AOloop2").exists()
    assert not Path(fake.env["TMUX_TMPDIR"]).exists()
    assert page.read_text() == GOOD


def test_run_workdir_removal_failure_keeps_other_removals(setup, monkeypatch, tmp_path):
    page, home, _, _ = setup
    # The image staging folder lands here, so its removal can be checked.
    monkeypatch.setattr(tc.tempfile, "tempdir", str(tmp_path))
    fake = FakeSystem()
    rmtree = tc.shutil.rmtree

    def failing_rmtree(path, *a, **kw):
        if Path(path) == home / "AOloop2":
            raise PermissionError("read-only")
        rmtree(path, *a, **kw)

    monkeypatch.setattr(tc.shutil, "rmtree", failing_rmtree)
    with pytest.raises(PermissionError):
        run(setup, fake)
    assert not fake.shm.exists() and not Path(fake.env["TMUX_TMPDIR"]).exists()
    assert not list(tmp_path.glob("tut_images_*"))
    assert page.read_text() == GOOD


def test_run_needs_milk_root(setup, tmp_path):
    page, home, shm, _ = setup
    for root in (None, str(tmp_path / "nowhere")):
        with pytest.raises(tc.CaptureError, match="MILK_ROOT|does not exist"):
            tc.run(page, home, shm, root, system_factory=FakeSystem())
    assert not (shm / "tutorial_capture").exists()


def test_run_from(setup):
    page, home, shm, _ = setup
    (home / "AOloop2").mkdir()
    fake = FakeSystem()
    with pytest.raises(tc.CaptureError, match="--from needs .*tutorial_capture"):
        run(setup, fake, start="close-loop")
    (shm / "tutorial_capture").mkdir()
    tmux = FakeTmux(fake)
    run(setup, fake, start="close-loop", tmux_stopper=tmux)
    assert fake.cwd == home / "AOloop2"
    assert fake.env["MILK_SHM_DIR"] == str(shm / "tutorial_capture")
    assert fake.env["MILK_PROC_DIR"] == fake.env["MILK_SHM_DIR"]
    # The caller's tmux: no private socket folder, nothing stopped.
    assert fake.env.get("TMUX_TMPDIR") == os.environ.get("TMUX_TMPDIR")
    assert tmux.env is None
    assert [c[0] for c in fake.calls] == ["python", "tui", "figure", "manual"]
    # Not created by this run: left in place.
    assert (home / "AOloop2").is_dir() and (shm / "tutorial_capture").is_dir()
    assert page.read_text() == GOOD
