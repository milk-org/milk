"""
Integration tests for the milk.bash GNU Bash integration layer (Phase 1).
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

from milk.cliwrap import HAVE_CLI

pytestmark = pytest.mark.skipif(
    not HAVE_CLI, reason="MILK compiled without CLI support"
)


def _find_milk_installdir() -> Path:
    env_dir = os.environ.get("MILK_INSTALLDIR")
    if env_dir and (Path(env_dir) / "bin" / "milk.bash").exists():
        return Path(env_dir)
    repo_root = Path(__file__).resolve().parents[3]
    for cand in sorted((repo_root / "_install").glob("milk-*"), reverse=True):
        if (cand / "bin" / "milk.bash").exists():
            return cand
    return Path("/usr/local/milk")


def _run_bash(
    script: str, extra_env: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    install_dir = _find_milk_installdir()
    shm_dir = tempfile.mkdtemp(prefix="milk_bash_test_", dir="/dev/shm")
    try:
        env = os.environ.copy()
        env["MILK_INSTALLDIR"] = str(install_dir)
        env["PATH"] = f"{install_dir / 'bin'}:{env.get('PATH', '')}"
        env["LD_LIBRARY_PATH"] = (
            f"{install_dir / 'lib'}:{env.get('LD_LIBRARY_PATH', '')}"
        )
        env["MILK_SHM_DIR"] = shm_dir
        if extra_env:
            env.update(extra_env)

        full_script = f'source "{install_dir / "bin" / "milk.bash"}"\n{script}\n'
        return subprocess.run(
            ["/bin/bash", "-e", "-c", full_script],
            env=env,
            text=True,
            capture_output=True,
            timeout=20,
        )
    finally:
        shutil.rmtree(shm_dir, ignore_errors=True)


def test_builtin_loaded_and_commands_registered():
    res = _run_bash("""
        [[ "$(type -t milkcall)" == "builtin" ]]
        [[ "$(type -t milkget)" == "builtin" ]]
        [[ "$(type -t @mk2Dim)" == "function" ]]
        [[ "$(type -t @imstats)" == "function" ]]
        echo "OK"
        """)
    assert res.returncode == 0, res.stderr
    assert "OK" in res.stdout


def test_get_set_and_dcvar_sync():
    res = _run_bash("""
        @mk2Drnd "@S:im1" 32 48
        [[ "$(@get s.im1.xsize)" == "32" ]]
        @get -v ys s.im1.ysize
        [[ "$ys" == "48" ]]
        @imstats im1 >/dev/null
        [[ "$imsize0" == "32" ]]
        [[ "$imsize1" == "48" ]]
        [[ -n "$vp50" ]]
        @set s.im1.kw.EXPTIME 0.0125 "Exposure time [s]"
        @set s.im1.kw.FILTER "H-band" "Filter name"
        [[ "$(@get s.im1.kw.FILTER)" == "H-band" ]]
        echo "OK"
        """)
    assert res.returncode == 0, res.stderr
    assert "OK" in res.stdout


def test_calc_inplace_slices_and_functions():
    res = _run_bash("""
        @mk2Dim "@S:im1" 16 16
        c0=$(@get s.im1.cnt0)
        @calc "im1 = 5.0"
        c1=$(@get s.im1.cnt0)
        (( c1 > c0 ))
        @calc "im1 += 3.0"
        [[ "$(@calc "im1[0,0]")" == "8" ]]

        x0=2; x1=6; cx=8; r=2
        @calc "im1[x0:x1, cx-r:cx+r] = 20.0"
        [[ "$(@calc "im1[4,8]")" == "20" ]]
        [[ "$(@calc "im1[0,0]")" == "8" ]]
        [[ "$(@calc "imedian(im1)")" == "8" ]]

        @mk2Dim "@S:mA" 2 2
        @mk2Dim "@S:mB" 2 2
        @calc "mA = 2.0"
        @calc "mB = 4.0"
        @calc "mC = matmul(mA, mB)"
        [[ "$(@calc "mC[0,0]")" == "16" ]]
        echo "OK"
        """)
    assert res.returncode == 0, res.stderr
    assert "OK" in res.stdout


def test_at_test_and_exists():
    res = _run_bash("""
        @mk2Drnd "@S:im1" 32 32
        @calc "im1 = im1 * 10.0"
        @exists s.im1
        ! @exists s.nonexistent_stream
        @test "irms(im1) > 0.5 && s.im1.xsize == 32 && s.im1.dtype == \\"FLOAT\\""
        ! @test "s.im1.xsize == 999"
        echo "OK"
        """)
    assert res.returncode == 0, res.stderr
    assert "OK" in res.stdout


def test_waitfor_and_grab():
    res = _run_bash("""
        @mk2Dim "@S:live" 8 8
        rc=0
        @waitfor s.live --timeout 0.04 || rc=$?
        [[ "$rc" -eq 124 ]]

        (
            for i in 1 2 3; do
                sleep 0.03
                @calc "live += 1.0"
            done
        ) &
        @grab s.live 3 "@S:burst" --timeout 2.0
        wait
        [[ "$(@get s.burst.xsize)" == "8" ]]
        [[ "$(@get s.burst.ysize)" == "8" ]]
        [[ "$(@get s.burst.zsize)" == "3" ]]
        echo "OK"
        """)
    assert res.returncode == 0, res.stderr
    assert "OK" in res.stdout


def test_defer_xargs_and_custom_sigil():
    res = _run_bash("""
        @mk2Dim "@S:s1" 4 4
        @mk2Dim "@S:s2" 4 4
        printf "s1\\ns2\\n" | xargs -n1 @imstats >/dev/null
        @defer "echo cleanup_1"
        @defer "echo cleanup_2"
        echo "main_body"
        """)
    assert res.returncode == 0, res.stderr
    lines = [
        ln.strip()
        for ln in res.stdout.strip().splitlines()
        if ln.strip() in ("main_body", "cleanup_1", "cleanup_2")
    ]
    assert lines == ["main_body", "cleanup_2", "cleanup_1"]

    res_comma = _run_bash(
        """
        ,mk2Dim "@S:im_c" 10 12
        [[ "$(,get s.im_c.ysize)" == "12" ]]
        echo "COMMA_OK"
        """,
        extra_env={"MILKCLI_SIGIL": ","},
    )
    assert res_comma.returncode == 0, res_comma.stderr
    assert "COMMA_OK" in res_comma.stdout


def test_milk_cli_tier2_and_native_fallback(tmp_path: Path):
    install_dir = _find_milk_installdir()
    milk_cli = install_dir / "bin" / "milk-cli"
    shm_dir = tempfile.mkdtemp(prefix="milk_cli_t2_", dir="/dev/shm")
    try:
        env = os.environ.copy()
        env["MILK_INSTALLDIR"] = str(install_dir)
        env["PATH"] = f"{install_dir / 'bin'}:{env.get('PATH', '')}"
        env["LD_LIBRARY_PATH"] = (
            f"{install_dir / 'lib'}:{env.get('LD_LIBRARY_PATH', '')}"
        )
        env["MILK_SHM_DIR"] = shm_dir

        script_file = tmp_path / "tier2_test.milk"
        script_file.write_text(
            "mem.mk2Dim @S:t2im 14 18\n"
            "@ a = 2.5 + 3.5\n"
            "@ t2im = a * 2.0\n"
            "@ tot = itot(t2im)\n"
            'echo "T2_RESULT=${a}:${tot}:$(@get s.t2im.ysize):@s.t2im.xsize"\n'
        )
        res = subprocess.run(
            [str(milk_cli), "-s", str(script_file)],
            env=env,
            text=True,
            capture_output=True,
            timeout=20,
        )
        assert res.returncode == 0, res.stderr
        assert "T2_RESULT=6:3024:18:14" in res.stdout

        # Verify MILKCLI_ENGINE=native fallback still works
        env_native = dict(env)
        env_native["MILKCLI_ENGINE"] = "native"
        native_script = tmp_path / "native_test.milk"
        native_script.write_text("nv = 7 + 8\n" "echo NATIVE_OK=$nv\n")
        res_nat = subprocess.run(
            [str(milk_cli), "-s", str(native_script)],
            env=env_native,
            text=True,
            capture_output=True,
            timeout=20,
        )
        assert res_nat.returncode == 0, res_nat.stderr
        assert "NATIVE_OK=15" in res_nat.stdout
    finally:
        shutil.rmtree(shm_dir, ignore_errors=True)


def test_milk_script_upgrade(tmp_path: Path):
    install_dir = _find_milk_installdir()
    upgrader = install_dir / "bin" / "milk-script-upgrade"
    env = os.environ.copy()
    env["MILK_INSTALLDIR"] = str(install_dir)
    env["PATH"] = f"{install_dir / 'bin'}:{env.get('PATH', '')}"
    env["LD_LIBRARY_PATH"] = f"{install_dir / 'lib'}:{env.get('LD_LIBRARY_PATH', '')}"

    legacy_file = tmp_path / "legacy_sample.milk"
    legacy_file.write_text("mem.mk2Dim im1 16 16\n" "a = 2 + 3\n" "echo $a\n")

    # --check returns non-zero when legacy constructs are present
    res_chk = subprocess.run(
        [str(upgrader), "--check", str(legacy_file)],
        env=env,
        text=True,
        capture_output=True,
        timeout=15,
    )
    assert res_chk.returncode == 1

    # --write rewrites in-place and creates .bak
    res_wr = subprocess.run(
        [str(upgrader), "--write", str(legacy_file)],
        env=env,
        text=True,
        capture_output=True,
        timeout=15,
    )
    assert res_wr.returncode == 0, res_wr.stderr
    assert (tmp_path / "legacy_sample.milk.bak").exists()
    upgraded_text = legacy_file.read_text()
    assert "@mem.mk2Dim im1 16 16" in upgraded_text
    assert '@calc "a = 2 + 3"' in upgraded_text
