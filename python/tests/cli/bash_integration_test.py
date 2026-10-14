"""
Comprehensive integration tests for the milk.bash GNU Bash integration layer
and the milk-cli bash engine (Tier-1 and Tier-2 syntax).
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
            timeout=25,
        )
    finally:
        shutil.rmtree(shm_dir, ignore_errors=True)


def _run_milk_cli(
    script: str,
    tmp_path: Path,
    mode: str = "script",
    extra_env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    install_dir = _find_milk_installdir()
    milk_cli = install_dir / "bin" / "milk-cli"
    shm_dir = tempfile.mkdtemp(prefix="milk_cli_test_", dir="/dev/shm")
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

        if mode == "cmd":
            cmd = [str(milk_cli), "-c", script]
            return subprocess.run(
                cmd, env=env, text=True, capture_output=True, timeout=25
            )
        if mode == "stdin":
            return subprocess.run(
                [str(milk_cli)],
                input=script + "\nexit\n",
                env=env,
                text=True,
                capture_output=True,
                timeout=25,
            )
        script_file = tmp_path / "test_script.milk"
        script_file.write_text(script + "\n")
        return subprocess.run(
            [str(milk_cli), "-s", str(script_file)],
            env=env,
            text=True,
            capture_output=True,
            timeout=25,
        )
    finally:
        shutil.rmtree(shm_dir, ignore_errors=True)


def test_builtin_loaded_and_commands_registered():
    res = _run_bash("""
        for b in milkcall milkget milkset milkcalc milktest milkexists milkwaitfor milkgrab milkcomplete milkshutdown; do
            [[ "$(type -t "$b")" == "builtin" ]]
        done
        for f in @ @calc @test @get @set @exists @running @waitfor @waitany @grab @defer @on @source @h @hc @hm @hf @streamcat @streamwrite @unload; do
            [[ "$(type -t "$f")" == "function" ]]
        done
        [[ "$(type -t @mk2Dim)" == "function" ]]
        [[ "$(type -t @mem.mk2Dim)" == "function" ]]
        [[ "$(type -t @imstats)" == "function" ]]
        [[ "$(type -t @info.imstats)" == "function" ]]

        # Standard bash help must remain untouched, while @h / @hm / @hc invoke milk help
        help echo | grep -qi "echo"
        @h >/dev/null
        @hm >/dev/null
        @hc mk2Dim >/dev/null
        echo "OK"
        """)
    assert res.returncode == 0, res.stderr
    assert "OK" in res.stdout


def test_stream_modifiers_all_types():
    res = _run_bash("""
        # Location modifiers: @S: (shared) and @L: (local)
        @mk2Dim "@S:sh_im" 8 8
        @mk2Dim "@L:loc_im" 8 8
        [[ "$(@get s.sh_im.shared)" == "1" ]]
        [[ "$(@get s.loc_im.shared)" == "0" ]]

        # Explicit @T<type>: datatype modifiers
        @mk2Dim "@TF32:im_f32" 8 8
        @mk2Dim "@TF64:im_f64" 8 8
        @mk2Dim "@TF16:im_f16" 8 8
        @mk2Dim "@TU8:im_u8" 8 8
        @mk2Dim "@TU16:im_u16" 8 8
        @mk2Dim "@TU32:im_u32" 8 8
        @mk2Dim "@TU64:im_u64" 8 8
        @mk2Dim "@TI8:im_i8" 8 8
        @mk2Dim "@TI16:im_i16" 8 8
        @mk2Dim "@TI32:im_i32" 8 8
        @mk2Dim "@TI64:im_i64" 8 8
        @mk2Dim "@TC64:im_c64" 8 8
        @mk2Dim "@TC128:im_c128" 8 8

        [[ "$(@get s.im_f32.dtype)" == "FLOAT" ]]
        [[ "$(@get s.im_f32.typename)" == "FLT32" ]]
        [[ "$(@get s.im_f64.dtype)" == "DOUBLE" ]]
        [[ "$(@get s.im_f64.typename)" == "FLT64" ]]
        [[ "$(@get s.im_f16.dtype)" == "HALF" ]]
        [[ "$(@get s.im_f16.typename)" == "FLT16" ]]
        [[ "$(@get s.im_u8.dtype)" == "UINT8" ]]
        [[ "$(@get s.im_u16.dtype)" == "UINT16" ]]
        [[ "$(@get s.im_u32.dtype)" == "UINT32" ]]
        [[ "$(@get s.im_u64.dtype)" == "UINT64" ]]
        [[ "$(@get s.im_i8.dtype)" == "INT8" ]]
        [[ "$(@get s.im_i16.dtype)" == "INT16" ]]
        [[ "$(@get s.im_i32.dtype)" == "INT32" ]]
        [[ "$(@get s.im_i64.dtype)" == "INT64" ]]
        [[ "$(@get s.im_c64.dtype)" == "COMPLEX_FLOAT" ]]
        [[ "$(@get s.im_c64.typename)" == "CPLX32" ]]
        [[ "$(@get s.im_c128.dtype)" == "COMPLEX_DOUBLE" ]]
        [[ "$(@get s.im_c128.typename)" == "CPLX64" ]]

        # Combined location + datatype modifiers
        @mk2Dim "@STF64:sh_f64" 10 12
        @mk2Dim "@LTU16:loc_u16" 10 12
        [[ "$(@get s.sh_f64.shared)" == "1" ]]
        [[ "$(@get s.sh_f64.dtype)" == "DOUBLE" ]]
        [[ "$(@get s.loc_u16.shared)" == "0" ]]
        [[ "$(@get s.loc_u16.dtype)" == "UINT16" ]]

        # Invalid datatype modifier must fail cleanly
        ! @mk2Dim "@TX99:bad_im" 8 8 2>/dev/null
        echo "OK"
        """)
    assert res.returncode == 0, res.stderr
    assert "OK" in res.stdout


def test_get_set_fps_kw_and_dcvar_sync():
    res = _run_bash("""
        @mk2Drnd "@S:im1" 32 48
        [[ "$(@get s.im1.xsize)" == "32" ]]
        [[ "$(@get s.im1.size0)" == "32" ]]
        [[ "$(@get s.im1.ysize)" == "48" ]]
        [[ "$(@get s.im1.size1)" == "48" ]]
        [[ "$(@get s.im1.zsize)" == "1" ]]
        [[ "$(@get s.im1.naxis)" == "2" ]]
        [[ "$(@get s.im1.nelement)" == "1536" ]]
        [[ "$(@get s.im1.dtype)" == "FLOAT" ]]
        [[ "$(@get s.im1.shared)" == "1" ]]

        # Zero-fork @get -v into bash variable
        @get -v ys s.im1.ysize
        [[ "$ys" == "48" ]]

        # Direct entity invocation via command fallback
        [[ "$(@s.im1.xsize)" == "32" ]]

        # dcvar sync back into bash variables after @imstats
        @imstats im1 >/dev/null
        [[ "$imsize0" == "32" ]]
        [[ "$imsize1" == "48" ]]
        [[ -n "$vp50" ]]
        [[ -n "$vrms" ]]

        # FITS keyword read/write (int, float, string, stdin pipe, wildcard)
        @set s.im1.kw.NFRAMES 100 "Frame count"
        @set s.im1.kw.EXPTIME 0.0125 "Exposure time [s]"
        @set s.im1.kw.FILTER "H-band" "Filter name"
        [[ "$(@get s.im1.kw.NFRAMES)" == "100" ]]
        [[ "$(@get s.im1.kw.EXPTIME)" == "0.0125" ]]
        [[ "$(@get s.im1.kw.FILTER)" == "H-band" ]]

        echo "0.025" | @set s.im1.kw.EXPTIME -
        [[ "$(@get s.im1.kw.EXPTIME)" == "0.025" ]]

        kwlist="$(@get "s.im1.kw.*")"
        [[ "$kwlist" == *"NFRAMES"* && "$kwlist" == *"EXPTIME"* && "$kwlist" == *"FILTER"* ]]

        # FPS creation via standalone fpsinit, @exists, @get, @set
        milk-fpsexec-imggen-mkrndim -n testfps fpsinit >/dev/null 2>&1
        @exists f.testfps
        fps_params="$(@get "f.testfps.*")"
        [[ "$fps_params" == *"xsize"* ]]
        @set f.testfps.xsize 256
        [[ "$(@get f.testfps.xsize)" == "256" ]]
        @f.testfps.ysize=512
        [[ "$(@get f.testfps.ysize)" == "512" ]]
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
        @calc "im1 *= 2.0"
        @calc "im1 -= 4.0"
        @calc "im1 /= 1.5"
        [[ "$(@calc "im1[0,0]")" == "8" ]]
        [[ "$(@get s.im1.shared)" == "1" ]]

        # Single-pixel read & compound write
        @calc "im1[1, 2] = 40.0"
        @calc "im1[1, 2] += 2.0"
        [[ "$(@calc "im1[1, 2]")" == "42" ]]

        # Variable and arithmetic-expression slice bounds on LHS and RHS
        @calc "im1 = 8.0"
        x0=2; x1=6; cx=8; r=2
        @calc "im1[x0:x1, cx-r:cx+r] = 20.0"
        [[ "$(@calc "im1[4,8]")" == "20" ]]
        [[ "$(@calc "im1[0,0]")" == "8" ]]
        [[ "$(@calc "imedian(im1)")" == "8" ]]

        # Sub-image extraction with variable bounds and reduction on sliced view
        @calc "sub = im1[x0:x1, cx-r:cx+r]"
        [[ "$(@get s.sub.xsize)" == "5" ]]
        [[ "$(@get s.sub.ysize)" == "5" ]]
        [[ "$(@calc "itot(sub)")" == "500" ]]
        [[ "$(@calc "itot(im1[x0:x1, cx-r:cx+r])")" == "500" ]]

        # Passing variable slice bounds to CLI commands (@info.imgstat / @imstats)
        # 1) Double quotes (bash expands variables)
        @info.imgstat "im1[${x0}:${x1},${x0}:${x1}]" >/dev/null
        [[ "$imsize0" == "5" && "$imsize1" == "5" ]]
        # 2) Single quotes (milk expression parser evaluates synced bash variables & math)
        @info.imgstat 'im1[x0:x1, cx-r:cx+r]' >/dev/null
        [[ "$imsize0" == "5" && "$imsize1" == "5" ]]
        [[ "$vmin" == "20" && "$vmax" == "20" ]]

        # Entity reads and writes inside @calc
        @calc "area = s.im1.xsize * @s.im1.ysize"
        [[ "$area" == "256" ]]
        @calc "s.im1.kw.GAIN = 2.5 * 4"
        [[ "$(@get s.im1.kw.GAIN)" == "10" ]]

        # Matrix multiplication (2D x 2D and 2D x 1D)
        @mk2Dim "@S:mA" 2 2
        @mk2Dim "@S:mB" 2 2
        @calc "mA = 2.0"
        @calc "mB = 4.0"
        @calc "mC = matmul(mA, mB)"
        [[ "$(@calc "mC[0,0]")" == "16" ]]
        [[ "$(@calc "itot(mC)")" == "64" ]]
        echo "OK"
        """)
    assert res.returncode == 0, res.stderr
    assert "OK" in res.stdout


def test_at_test_and_exists():
    res = _run_bash("""
        @mk2Drnd "@S:im1" 32 32
        @calc "im1 = im1 * 10.0"
        @set s.im1.kw.FILTER "H-band"
        @set s.im1.kw.EXPTIME 0.02

        @exists s.im1
        ! @exists s.nonexistent_stream
        ! @exists f.nonexistent_fps
        ! @exists p.nonexistent_proc

        # Test both bare s. and @s. entity syntax, numeric & string comparisons
        @test 'irms(im1) > 0.5 && s.im1.xsize == 32 && s.im1.dtype == "FLOAT"'
        @test '@s.im1.xsize == 32 && @s.im1.dtype == "FLOAT" && @s.im1.kw.FILTER == "H-band"'
        @test 's.im1.kw.EXPTIME > 0.01 && s.im1.kw.EXPTIME < 0.05'

        # False condition returns exit code 1
        rc=0
        @test "s.im1.xsize == 999" || rc=$?
        [[ "$rc" -eq 1 ]]

        # Syntax error returns exit code 2
        rc2=0
        @test "1 + * / ==" 2>/dev/null || rc2=$?
        [[ "$rc2" -eq 2 ]]
        echo "OK"
        """)
    assert res.returncode == 0, res.stderr
    assert "OK" in res.stdout


def test_waitfor_waitany_grab_on_and_source(tmp_path: Path):
    tier2_lib = tmp_path / "helper_lib.milk"
    tier2_lib.write_text(
        "mem.mk2Dim @S:src_im 6 8\n"
        "@ src_im = $1 * 3.0\n"
        "@ src_tot = itot(src_im)\n"
    )

    res = _run_bash(f"""
        @mk2Dim "@S:live" 8 8
        @mk2Dim "@S:live2" 8 8

        # Timeout returns 124
        rc=0
        @waitfor s.live --timeout 0.04 || rc=$?
        [[ "$rc" -eq 124 ]]

        # Condition wait (boolean expression and key=val)
        (
            sleep 0.04
            @set s.live.kw.READY 1
            @calc "live += 1.0"
        ) &
        @waitfor "s.live.kw.READY=1" --timeout 2.0
        @waitfor "s.live.cnt0 >= 1" --timeout 2.0
        wait

        # @waitany detects whichever stream updates first
        (
            sleep 0.04
            @calc "live2 += 5.0"
        ) &
        updated="$(@waitany live live2 --timeout 2.0)"
        wait
        [[ "$updated" == "live2" ]]

        # @grab captures N frames into a 3D cube
        (
            for i in 1 2 3; do
                sleep 0.03
                @calc "live = i * 10.0"
            done
        ) &
        @grab s.live 3 "@S:burst" --timeout 2.0
        wait
        [[ "$(@get s.burst.xsize)" == "8" ]]
        [[ "$(@get s.burst.ysize)" == "8" ]]
        [[ "$(@get s.burst.zsize)" == "3" ]]
        [[ "$(@calc "burst[0,0,0]")" == "10" ]]
        [[ "$(@calc "burst[0,0,1]")" == "20" ]]
        [[ "$(@calc "burst[0,0,2]")" == "30" ]]

        # @on reactive watcher
        marker_file="{tmp_path}/on_marker.txt"
        @on s.live "echo fired >> $marker_file"
        [[ -n "$(@on --list)" ]]
        sleep 0.03
        @calc "live += 1.0"
        sleep 0.05
        @on --clear
        [[ -z "$(@on --list)" ]]
        [[ -f "$marker_file" ]]

        # @source translates and sources a Tier-2 .milk script into plain bash
        @source "{tier2_lib}" 5
        [[ "$(@get s.src_im.ysize)" == "8" ]]
        [[ "$src_tot" == "720" ]]
        echo "OK"
        """)
    assert res.returncode == 0, res.stderr
    assert "OK" in res.stdout


def test_pipelines_streamio_defer_xargs_and_sigil():
    res = _run_bash("""
        # Concurrent milk commands in a pipeline: @streamcat | awk | @streamwrite
        @mk2Dim "@S:s1" 4 4
        @mk2Dim "@S:s2" 4 4
        @calc "s1 = 3.0"
        @streamcat s1 | awk '{print $1 * 4}' | @streamwrite s2
        [[ "$(@calc "s2[0,0]")" == "12" ]]
        [[ "$(@calc "itot(s2)")" == "192" ]]

        # Pipe-clean @listim (one stream name per line when stdout is not a tty)
        names="$(@listim | sort)"
        [[ "$names" == *"s1"* && "$names" == *"s2"* ]]

        # External tool invocation via libexec/milk-bin/@<cmd> symlinks
        printf "s1\\ns2\\n" | xargs -n1 @imstats >/dev/null

        # Bash arrays and associative arrays with milk commands
        declare -a streams=(s1 s2)
        declare -A totals=()
        for s in "${streams[@]}"; do
            totals[$s]="$(@calc "itot($s)")"
        done
        [[ "${totals[s1]}" == "48" ]]
        [[ "${totals[s2]}" == "192" ]]

        # LIFO @defer cleanup stack
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

    # Custom sigil MILKCLI_SIGIL=,
    res_comma = _run_bash(
        """
        ,mk2Dim "@S:im_c" 10 12
        ,calc "im_c = 7.0"
        [[ "$(,get s.im_c.ysize)" == "12" ]]
        ,test "itot(im_c) == 840 && s.im_c.xsize == 10"
        echo "COMMA_OK"
        """,
        extra_env={"MILKCLI_SIGIL": ","},
    )
    assert res_comma.returncode == 0, res_comma.stderr
    assert "COMMA_OK" in res_comma.stdout


def test_milk_cli_tier2_comprehensive(tmp_path: Path):
    res_s = _run_milk_cli(
        """
        mem.mk2Dim @S:t2im 14 18
        @ a = 2.5 + 3.5
        @ t2im = a * 2.0
        @ t2im[1:3, 2:4] = 20.0
        @ tot = itot(t2im)

        # Inline @{<expr>} and unquoted/quoted @s.<name>.<prop>
        val=@{a * 10 + t2im[2, 3]}
        nx=@s.t2im.xsize
        @s.t2im.kw.OBSERVER="Subaru"
        @s.t2im.kw.EXPTIME = 0.05
        @ s.t2im.kw.GAIN = a * 1.5

        # Test @test with @s. inside both single and double quotes
        @test '@s.t2im.dtype == "FLOAT" && @s.t2im.kw.OBSERVER == "Subaru"'
        @test "@s.t2im.dtype == \\"FLOAT\\" && @s.t2im.kw.GAIN == 9"

        # Test [ -S <stream> ] shorthand
        if [ -S t2im ] && [ ! -S no_such_stream ]; then
            echo "T2_OK=${a}:${tot}:${val}:${nx}:@s.t2im.ysize:@s.t2im.kw.OBSERVER.:@s.t2im.kw.EXPTIME:@s.t2im.kw.GAIN"
        fi
        """,
        tmp_path,
        mode="script",
    )
    assert res_s.returncode == 0, res_s.stderr
    # 14*18 = 252 pixels; 9 pixels in [1:3, 2:4] are 20.0 (+8 each over 12.0): 252*12 + 9*8 = 3024 + 72 = 3096
    assert "T2_OK=6:3096:80:14:18:Subaru.:0.05:9" in res_s.stdout

    # Test milk-cli -c one-liner mode
    res_c = _run_milk_cli(
        'mem.mk2Dim @S:cim 10 20; @ cim = 4; echo "CMD_OK=@s.cim.ysize:@{itot(cim)}"',
        tmp_path,
        mode="cmd",
    )
    assert res_c.returncode == 0, res_c.stderr
    assert "CMD_OK=20:800" in res_c.stdout

    # Test milk-cli stdin mode
    res_in = _run_milk_cli(
        'mem.mk2Dim @S:inim 6 7\n@ inim = 5\necho "STDIN_OK=@s.inim.xsize:@{itot(inim)}"',
        tmp_path,
        mode="stdin",
    )
    assert res_in.returncode == 0, res_in.stderr
    assert "STDIN_OK=6:210" in res_in.stdout

    # Verify MILKCLI_ENGINE=native fallback still works
    res_nat = _run_milk_cli(
        "nv = 7 + 8\necho NATIVE_OK=$nv",
        tmp_path,
        mode="script",
        extra_env={"MILKCLI_ENGINE": "native"},
    )
    assert res_nat.returncode == 0, res_nat.stderr
    assert "NATIVE_OK=15" in res_nat.stdout


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
