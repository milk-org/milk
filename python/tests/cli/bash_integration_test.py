"""
Comprehensive integration tests for the milk.bash GNU Bash integration layer
and the milk-cli bash engine (milk.bash primitives and milk-cli shorthands).
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

from milk.cliwrap import CLICommands, HAVE_CLI, MILK_CLI_EXEC

pytestmark = [
    pytest.mark.skipif(not HAVE_CLI, reason="MILK compiled without CLI support"),
    pytest.mark.timeout(25),
]


def _install_dir() -> Path:
    return Path(os.environ["MILK_INSTALLDIR"]).resolve()


def _isolated_shm_env(
    extra_env: dict[str, str] | None = None,
) -> tuple[dict[str, str], str]:
    """Build a subprocess env rooted inside conftest's spoofed MILK_SHM_DIR."""
    install_dir = _install_dir()
    base_shm = os.environ.get("MILK_SHM_DIR", "/tmp")
    os.makedirs(base_shm, exist_ok=True)
    shm_dir = tempfile.mkdtemp(prefix="milk_bash_", dir=base_shm)

    env = os.environ.copy()
    env["MILK_INSTALLDIR"] = str(install_dir)
    env["PATH"] = f"{install_dir / 'bin'}:{env.get('PATH', '')}"
    env["LD_LIBRARY_PATH"] = f"{install_dir / 'lib'}:{env.get('LD_LIBRARY_PATH', '')}"
    env["MILK_SHM_DIR"] = shm_dir
    env["MILK_PROC_DIR"] = shm_dir
    if extra_env:
        env.update(extra_env)
    return env, shm_dir


def _run_bash(
    script: str, extra_env: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    install_dir = _install_dir()
    env, shm_dir = _isolated_shm_env(extra_env)
    try:
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
    assert MILK_CLI_EXEC is not None
    env, shm_dir = _isolated_shm_env(extra_env)
    try:
        if mode == "cmd":
            return subprocess.run(
                [MILK_CLI_EXEC, "-c", script],
                env=env,
                text=True,
                capture_output=True,
                timeout=25,
            )
        if mode == "stdin":
            return subprocess.run(
                [MILK_CLI_EXEC],
                input=script + "\nexit\n",
                env=env,
                text=True,
                capture_output=True,
                timeout=25,
            )
        script_file = tmp_path / "test_script.milk"
        script_file.write_text(script + "\n")
        return subprocess.run(
            [MILK_CLI_EXEC, "-s", str(script_file)],
            env=env,
            text=True,
            capture_output=True,
            timeout=25,
        )
    finally:
        shutil.rmtree(shm_dir, ignore_errors=True)


def test_builtin_loaded_and_commands_registered():
    res = _run_bash("""
        for b in milkcall milkget milkset milkcalc milktest milkexists milkwaitfor milkgrab milkcomplete milkshutdown milkserver milklog; do
            [[ "$(type -t "$b")" == "builtin" ]]
        done
        for f in @ @calc @test @get @set @exists @running @waitfor @waitany @grab @defer @on @source @server @log @h @hb @hc @hm @hf @streamcat @streamwrite @unload; do
            [[ "$(type -t "$f")" == "function" ]]
        done
        [[ "$(type -t @mk2Dim)" == "function" ]]
        [[ "$(type -t @mem.mk2Dim)" == "function" ]]
        [[ "$(type -t @imstats)" == "function" ]]
        [[ "$(type -t @info.imstats)" == "function" ]]

        # Standard bash help must remain untouched, while @h / @hb / @hm / @hc invoke milk help
        help echo | grep -qi "echo"
        h_out="$(@h)"
        [[ "${h_out}" == *"source milk.bash"* ]]
        [[ "${h_out}" == *"@hb"* ]]
        hb_out="$(@hb)"
        [[ "${hb_out}" == *"BASH-ENHANCED MODE"* ]]
        [[ "${hb_out}" == *"@waitfor"* ]]
        [[ "${hb_out}" == *"@server"* ]]
        [[ "${hb_out}" == *"@log"* ]]
        @h bash | grep -q "BASH-ENHANCED MODE"
        @hm >/dev/null
        @hc mk2Dim >/dev/null
        @hc @mk2Dim >/dev/null

        # Verify visible [milk] prompt label, startup banner, and complete @-command removal on @unload
        shopt -s extglob
        PS1='user@host:$ '
        unset _MILK_ORIG_PS1
        banner_out="$(MILK_BANNER=1 source "${MILK_INSTALLDIR}/bin/milk.bash")"
        [[ "${banner_out}" == *"milk v"*"bash integration loaded"* ]]
        [[ "${banner_out}" == *"for help"* ]]
        source "${MILK_INSTALLDIR}/bin/milk.bash"
        [[ "$(type -t @)" == "function" ]]
        [[ "${PS1}" == *"[milk]"*"user@host:$ " ]]
        ps1_once="${PS1}"
        source "${MILK_INSTALLDIR}/bin/milk.bash"
        [[ "${PS1}" == "${ps1_once}" ]]
        @unload
        [[ "${PS1}" == 'user@host:$ ' ]]
        [[ -z "$(compgen -c "@")" ]]
        [[ -z "$(type -t @mk2Dim)" ]]
        set +e
        @mk2Dim 2>/dev/null
        rc_after_unload=$?
        set -e
        [[ "${rc_after_unload}" -eq 127 ]]
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
    shorthand_lib = tmp_path / "helper_lib.milk"
    shorthand_lib.write_text(
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

        # @source translates and sources a milk-cli shorthand (.milk) script into plain bash
        @source "{shorthand_lib}" 5
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


def test_milk_cli_shorthands_comprehensive(tmp_path: Path):
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

        # Test @exists s.<stream>
        if @exists s.t2im && ! @exists s.no_such_stream; then
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

    # Verify milk.cliwrap.CLICommands works with milk.bash and milk-cli shorthand syntax
    with CLICommands(
        [
            "mem.mk2Dim @S:wrap_im 8 8",
            "@ wrap_im = 6",
            'echo "CLIWRAP_OK=@s.wrap_im.xsize:@{itot(wrap_im)}"',
        ]
    ) as res_wrap:
        assert res_wrap.returncode == 0, res_wrap.stderr
        assert "CLIWRAP_OK=8:384" in res_wrap.stdout


def test_adversarial_slicing_1d_2d_3d_and_compound():
    res = _run_bash("""
        # 1. 1D-axis, 2D, and 3D slicing + single-voxel indexing
        @mk2Dim "@S:s1d" 16 1
        @calc "s1d = 1"
        @calc "s1d[2:5] = 10"
        @calc "s1d[0, 0] = 100"
        @calc "s1d[-1, 0] = 200"
        [[ "$(@calc 'itot(s1d)')" == "350" ]]
        [[ "$(@calc 's1d[0, 0] + s1d[3, 0] + s1d[15, 0]')" == "310" ]]

        @mk2Dim "@S:s2d" 8 8
        @calc "s2d = 0"
        @calc "x0 = 1"
        @calc "x1 = 3"
        @calc "cx = 2"
        @calc "r = 1"
        # Spaces and parentheses inside slice bounds
        @calc "s2d[ x0 : x1 , (cx - r) : (cx + r) ] = 4"
        [[ "$(@calc 'itot(s2d)')" == "36" ]]

        # Compound slice scalar operations (+=, -=, *=, /=)
        @calc "s2d[1:3, 1:3] += 6"
        [[ "$(@calc 's2d[2, 2]')" == "10" ]]
        @calc "s2d[1:3, 1:3] -= 2"
        [[ "$(@calc 's2d[2, 2]')" == "8" ]]
        @calc "s2d[1:3, 1:3] *= 3"
        [[ "$(@calc 's2d[2, 2]')" == "24" ]]
        @calc "s2d[1:3, 1:3] /= 4"
        [[ "$(@calc 's2d[2, 2]')" == "6" ]]

        # Slice-to-slice copy and compound image-to-slice +=
        @calc "s2d[4:6, 4:6] = s2d[1:3, 1:3]"
        @calc "s2d[4:6, 4:6] += s2d[1:3, 1:3]"
        [[ "$(@calc 's2d[5, 5]')" == "12" ]]
        [[ "$(@calc 'itot(s2d)')" == "162" ]]

        # Negative indices and negative slices
        @calc "s2d[-1, -1] = 50"
        [[ "$(@calc 's2d[7, 7]')" == "50" ]]
        [[ "$(@calc 's2d[-1, -1]')" == "50" ]]
        @calc "s2d[-2:-1, -2:-1] = 7"
        [[ "$(@calc 'itot(s2d[6:7, 6:7])')" == "28" ]]

        # Strided and binned slices on RHS
        @mk2Dim "@S:grid" 8 8
        @calc "grid = 4"
        @calc "strided = grid[0:7:2, 0:7:2]"
        [[ "$(@get s.strided.xsize)" == "4" ]]
        [[ "$(@get s.strided.ysize)" == "4" ]]
        [[ "$(@calc 'itot(strided)')" == "64" ]]
        @calc "binned = grid[0:7:2b, 0:7:2b]"
        [[ "$(@get s.binned.xsize)" == "4" ]]
        [[ "$(@get s.binned.ysize)" == "4" ]]

        # 3D cube slicing and single-voxel indexing
        @mk3Dim "@S:cube" 4 4 4
        @calc "cube = 1"
        @calc "cube[1:2, 1:2, 1:2] = 10"
        @calc "cube[0, 0, 3] = 50"
        # 64 voxels: 8 voxels at 10 (+9 each = +72), 1 voxel at 50 (+49), total = 64 + 72 + 49 = 185
        [[ "$(@calc 'itot(cube)')" == "185" ]]
        [[ "$(@calc 'cube[2, 2, 2] + cube[0, 0, 3]')" == "60" ]]
        @calc "plane2 = cube[*, *, 2]"
        [[ "$(@calc 'itot(plane2)')" == "52" ]]

        # CLI command (@info.imgstat) with valid expression slice vs adversarial invalid slices
        @info.imgstat 's2d[ x0 : x1 , (cx - r) : (cx + r) ]' >/dev/null

        set +e
        @calc "s2d[999, 0]" 2>/dev/null
        rc_oob_pos=$?
        @calc "s2d[-999, 0]" 2>/dev/null
        rc_oob_neg=$?
        @calc "s2d[6:2, 0:3]" 2>/dev/null
        rc_inv=$?
        @calc "s2d[0:2, 0:2] = grid" 2>/dev/null
        rc_dim=$?
        @calc "s2d[undef_var:3, 0:3]" 2>/dev/null
        rc_undef=$?
        @calc "s2d[0:3, 0:3" 2>/dev/null
        rc_unclosed=$?
        @info.imgstat 's2d[undef_var:3, 0:3]' >/dev/null 2>&1
        rc_cmd_undef=$?
        @info.imgstat 's2d[999:1000, 0:3]' >/dev/null 2>&1
        rc_cmd_oob=$?
        @info.imgstat 's2d[-999, 0]' >/dev/null 2>&1
        rc_cmd_neg=$?
        @info.imgstat 's2d[6:2, 0:3]' >/dev/null 2>&1
        rc_cmd_inv=$?
        set -e

        [[ "$rc_oob_pos" -ne 0 ]]
        [[ "$rc_oob_neg" -ne 0 ]]
        [[ "$rc_inv" -ne 0 ]]
        [[ "$rc_dim" -ne 0 ]]
        [[ "$rc_undef" -ne 0 ]]
        [[ "$rc_unclosed" -ne 0 ]]
        [[ "$rc_cmd_undef" -ne 0 ]]
        [[ "$rc_cmd_oob" -ne 0 ]]
        [[ "$rc_cmd_neg" -ne 0 ]]
        [[ "$rc_cmd_inv" -ne 0 ]]

        echo "ADV_SLICE_OK"
        """)
    assert res.returncode == 0, f"stdout={res.stdout}\nstderr={res.stderr}"
    assert "ADV_SLICE_OK" in res.stdout


def test_adversarial_multi_datatype_calc_and_streamio():
    res = _run_bash("""
        for spec in \\
            "TF32:FLOAT:f32" \\
            "TF64:DOUBLE:f64" \\
            "TUI8:UINT8:ui8" \\
            "TSI8:INT8:si8" \\
            "TUI16:UINT16:ui16" \\
            "TSI16:INT16:si16" \\
            "TUI32:UINT32:ui32" \\
            "TSI32:INT32:si32" \\
            "TUI64:UINT64:ui64" \\
            "TSI64:INT64:si64"; do
            IFS=: read -r mod dt name <<< "$spec"
            @mk2Dim "@S:@${mod}:${name}" 4 4
            [[ "$(@get s.${name}.dtype)" == "$dt" ]]

            # Scalar fill, compound +=, *=, single-pixel & slice writes
            @calc "${name} = 3"
            @calc "${name} += 2"
            @calc "${name}[0:1, 0:1] *= 2"
            @calc "${name}[3, 3] = 20"
            # 16 pixels: 4 in [0:1,0:1] are 10, 1 at [3,3] is 20, remaining 11 are 5 -> 40 + 20 + 55 = 115
            [[ "$(@calc "itot(${name})")" == "115" ]]
            [[ "$(@calc "${name}[0, 0] + ${name}[3, 3]")" == "30" ]]
            [[ "$(@calc "imedian(${name})")" == "5" ]]

            # Round-trip via @streamcat | awk | @streamwrite
            @streamcat "$name" | awk '{print $1 + 1}' | @streamwrite "$name"
            # Adding 1 to all 16 pixels increases total by 16: 115 + 16 = 131
            [[ "$(@calc "itot(${name})")" == "131" ]]
        done
        echo "ADV_DTYPE_OK"
        """)
    assert res.returncode == 0, f"stdout={res.stdout}\nstderr={res.stderr}"
    assert "ADV_DTYPE_OK" in res.stdout


def test_adversarial_quoting_heredocs_and_bash_constructs(tmp_path: Path):
    res = _run_milk_cli(
        """
        mem.mk2Dim @S:h_im 6 6
        @ h_im = 5
        @ x0 = 0
        @ x1 = 3
        @ h_im[ x0 + 1 : x1 - 1 , 0 : 1 ] = 15

        # Unquoted slice with spaces inside [...] passed to milk-cli shorthand command
        info.imgstat h_im[ x0 + 1 : x1 - 1 , 0 : 1 ] >/dev/null

        # Email addresses and literal strings containing @s. / @f. mid-word must not be mangled
        email="contact@s.example.com"
        fmail="admin@f.domain.org"
        [[ "$email" == "contact@s.example.com" ]]
        [[ "$fmail" == "admin@f.domain.org" ]]

        # Bash arrays, ${#arr[@]}, associative arrays, and regex [[ =~ ]]
        arr=(alpha beta gamma delta)
        [[ "${#arr[@]}" == "4" ]]
        declare -A map
        map[key1]="val_@s.not_expanded"
        [[ "${map[key1]}" == "val_@s.not_expanded" ]]
        num_str="324"
        [[ "$num_str" =~ ^[0-9]+$ ]]

        # Multiple @{ ... } on a single line with nested parentheses
        combo="@{(2 + 3) * 4}:@{itot(h_im) + (h_im[1, 0] * 2)}"
        # 36 pixels at 5 = 180; 4 pixels in [1:2, 0:1] at 15 (+10 each = +40) -> itot = 220; + 30 = 250
        [[ "$combo" == "20:250" ]]

        # Single-quoted heredoc containing fake milk commands must not be translated
        lit_doc=$(cat << 'EOF'
mem.mk2Dim should_not_run 4 4
@ a = 999
@s.h_im.xsize
EOF
)
        [[ ! -S should_not_run ]]
        [[ "$lit_doc" == *"@s.h_im.xsize"* ]]

        # Unquoted heredoc preserves literal lines without executing them as commands
        exp_doc=$(cat << EOF
size_is=@s.h_im.xsize
EOF
)
        [[ "$exp_doc" == "size_is=6" ]]

        # Bash function + case statement + subshell
        run_step() {
            local mode="$1"
            case "$mode" in
                add)
                    @ h_im += 2
                    ;;
                *)
                    return 1
                    ;;
            esac
        }
        run_step add
        sub_tot=$(@calc "itot(h_im)")
        # 220 + 36*2 = 292
        [[ "$sub_tot" == "292" ]]
        echo "ADV_BASH_OK"
        """,
        tmp_path,
        mode="script",
    )
    assert res.returncode == 0, f"stdout={res.stdout}\nstderr={res.stderr}"
    assert "ADV_BASH_OK" in res.stdout


def test_adversarial_error_recovery_and_fault_isolation():
    res = _run_bash("""
        @mk2Dim "@S:ok_im" 4 4
        @calc "ok_im = 10"
        @mk2Dim "@S:zero_im" 4 4
        @calc "zero_im = 0"
        @calc "keep_var = 42"

        set +e
        # 1. Division by zero across all @calc paths
        @calc "1 / 0" 2>/dev/null
        rc_div1=$?
        @calc "keep_var /= 0" 2>/dev/null
        rc_div2=$?
        @calc "ok_im /= 0" 2>/dev/null
        rc_div3=$?
        @calc "ok_im[0:1, 0:1] /= 0" 2>/dev/null
        rc_div4=$?
        @calc "ok_im[0, 0] /= 0" 2>/dev/null
        rc_div5=$?
        @calc "ok_im /= zero_im" 2>/dev/null
        rc_div6=$?

        # 2. Unknown commands, invalid stream modifiers, non-existent entities, waitfor timeout
        @no_such_milk_cmd_xyz 2>/dev/null
        rc_unk=$?
        @mk2Dim "@TX99:bad_mod" 4 4 2>/dev/null
        rc_mod1=$?
        @mk2Dim "@EN:bad_mod2" 4 4 2>/dev/null
        rc_mod2=$?
        @get s.nonexistent_stream_xyz.xsize 2>/dev/null
        rc_get=$?
        @set s.nonexistent_stream_xyz.kw.KEY 1 2>/dev/null
        rc_set=$?
        @waitfor --timeout 0.05 s.ok_im.cnt0 "> 999999" 2>/dev/null
        rc_wait=$?
        set -e

        [[ "$rc_div1" -ne 0 ]]
        [[ "$rc_div2" -ne 0 ]]
        [[ "$rc_div3" -ne 0 ]]
        [[ "$rc_div4" -ne 0 ]]
        [[ "$rc_div5" -ne 0 ]]
        [[ "$rc_div6" -ne 0 ]]
        [[ "$rc_unk" -eq 127 ]]
        [[ "$rc_mod1" -ne 0 ]]
        [[ "$rc_mod2" -ne 0 ]]
        [[ "$rc_get" -ne 0 ]]
        [[ "$rc_set" -ne 0 ]]
        [[ "$rc_wait" -eq 124 ]]

        # 3. Verify server survived all errors with state intact, then run 50 rapid calls
        [[ "$(@calc 'keep_var')" == "42" ]]
        [[ "$(@calc 'itot(ok_im)')" == "160" ]]
        for ((i = 0; i < 50; i++)); do
            @calc "ok_im += 1"
        done
        # 160 + 50 * 16 = 960
        [[ "$(@calc 'itot(ok_im)')" == "960" ]]
        echo "ADV_RECOVERY_OK"
        """)
    assert res.returncode == 0, f"stdout={res.stdout}\nstderr={res.stderr}"
    assert "ADV_RECOVERY_OK" in res.stdout


def test_interactive_syndiag_valid_milk_syntax(tmp_path):
    """Verify Tree-sitter syndiag never flags valid milk.bash / milk-cli syntax as errors."""
    import ctypes
    import fcntl
    import pty
    import select
    import struct
    import termios
    import time

    # 1. Direct C API verification of cli_ts_get_diagnostic and cli_ts_is_incomplete
    lib_path = _install_dir() / "lib" / "libCLIcore.so"
    if lib_path.exists():
        lib = ctypes.CDLL(str(lib_path))

        class Diag(ctypes.Structure):
            _fields_ = [
                ("severity", ctypes.c_int),
                ("start_byte", ctypes.c_uint32),
                ("end_byte", ctypes.c_uint32),
                ("message", ctypes.c_char * 128),
                ("token", ctypes.c_char * 64),
            ]

        lib.cli_ts_init()
        valid_lines = [
            "mem.mk2Dim @S:im1 128 128",
            "mem.mk2Dim @L:im1 128 128",
            "mem.mk2Dim @STF64:im_dbl 64 64",
            "mem.mk3Dim @STC100:cube 32 32 10",
            "@ a = 2.5 + 3.5",
            "@ im1[10:20, 10:20] = 500",
            "im1[10:20, 10:20] = 500",
            "im1 += 10",
            "a = 2.5 + 3.5",
            "val=@{im1[10, 10]}",
            'echo "center=@{im1[64, 64]}"',
            '@s.im1.kw.OBSERVER = "Subaru"',
            "@f.testfps.gain = 0.25",
            "im1 |> saveFITS out.fits",
        ]
        for line in valid_lines:
            d = Diag()
            lib.cli_ts_get_diagnostic(line.encode(), ctypes.byref(d))
            assert (
                d.severity == 0
            ), f"False-positive diagnostic on {line!r}: sev={d.severity} msg={d.message.decode()!r}"
            inc = lib.cli_ts_is_incomplete(line.encode())
            assert inc == 0, f"False-positive incomplete on {line!r}"

        invalid_lines = [
            ('echo "unclosed', 1),
            ("for i in 1 2; do echo $i", 1),
            ("echo ) bad", 2),
            ("val=@{im1[10, 10]", 1),
        ]
        for line, expected_sev in invalid_lines:
            d = Diag()
            lib.cli_ts_get_diagnostic(line.encode(), ctypes.byref(d))
            assert (
                d.severity == expected_sev
            ), f"Expected severity {expected_sev} on {line!r}, got {d.severity} ({d.message.decode()!r})"

    # 2. Interactive PTY verification: typing "mem.mk2Dim @S:im1 128 128" must not show "[syntax error"
    env, shm_dir = _isolated_shm_env({"HOME": str(tmp_path)})
    master_fd, slave_fd = pty.openpty()
    winsize = struct.pack("HHHH", 24, 120, 0, 0)
    fcntl.ioctl(slave_fd, termios.TIOCSWINSZ, winsize)

    cli_bin = MILK_CLI_EXEC or str(_install_dir() / "bin" / "milk-cli")
    proc = subprocess.Popen(
        [cli_bin],
        stdin=slave_fd,
        stdout=slave_fd,
        stderr=slave_fd,
        env=env,
        close_fds=True,
    )
    os.close(slave_fd)

    out_chunks: list[bytes] = []

    def drain(duration: float = 0.25) -> None:
        deadline = time.time() + duration
        while time.time() < deadline:
            r, _, _ = select.select([master_fd], [], [], 0.05)
            if r:
                try:
                    data = os.read(master_fd, 4096)
                    if data:
                        out_chunks.append(data)
                except OSError:
                    break

    try:
        drain(0.6)
        for cmd in [
            "mem.mk2Dim @S:im1 128 128",
            "@ im1[10:20, 10:20] = 500",
            'echo "PIX=@{im1[15, 15]}"',
            "exit",
        ]:
            os.write(master_fd, (cmd + "\n").encode())
            drain(0.25)
        proc.wait(timeout=5)
    finally:
        os.close(master_fd)
        if proc.poll() is None:
            proc.kill()
        shutil.rmtree(shm_dir, ignore_errors=True)

    full_out = b"".join(out_chunks).decode("utf-8", errors="replace")
    assert (
        "[syntax error" not in full_out
    ), f"Unexpected syntax error in PTY output:\n{full_out}"
    assert "PIX=500" in full_out, f"Expected PIX=500 in PTY output:\n{full_out}"


def test_server_management_and_multi_shell_sharing():
    res = _run_bash("""
        trap '@server stop --all 2>/dev/null || true' EXIT

        # 1. Before any milk command runs, @server status shows IDLE without starting the server
        st0="$(@server status)"
        [[ "${st0}" == *"IDLE"* ]]
        [[ "${st0}" == *"private (per-shell)"* ]]

        # 2. Running a command starts the private per-shell server
        @mk2Dim "@L:loc_priv" 8 8
        @calc "loc_priv = 2"
        st1="$(@server status)"
        [[ "${st1}" == *"RUNNING"* ]]
        [[ "${st1}" == *"1 local images"* ]]

        # 3. Start and switch to a persistent named server 'ao_test_srv'
        PS1='user@host:$ '
        @server start ao_test_srv
        [[ "${_MILK_SERVER_NAME}" == "ao_test_srv" ]]
        [[ "${PS1}" == *"[milk:ao_test_srv]"* ]]

        # Verify local image from private server is isolated from 'ao_test_srv'
        st_ao0="$(@server status)"
        [[ "${st_ao0}" == *"0 local images"* ]]

        # Create state inside 'ao_test_srv'
        @mk2Dim "@L:ao_priv_im" 6 6
        @calc "ao_priv_im = 5"
        @calc "shared_ao_var = 777"

        # 4. Spawn a second independent bash shell connected via 'source milk.bash --server ao_test_srv'
        sub_out="$(bash -e -c '
            PS1="sub$ "
            source "${MILK_INSTALLDIR}/bin/milk.bash" --server ao_test_srv
            [[ "${PS1}" == *"[milk:ao_test_srv]"* ]]
            echo "VAR=$(@calc shared_ao_var)"
            echo "TOT=$(@calc \"itot(ao_priv_im)\")"
            # Unloading in a client shell must NOT kill the shared named server
            @unload
        ')"
        [[ "${sub_out}" == *"VAR=777"* ]]
        [[ "${sub_out}" == *"TOT=180"* ]]

        # 5. Verify 'ao_test_srv' is still alive after client shell unloaded & exited
        [[ "$(@calc 'shared_ao_var')" == "777" ]]
        ls_json="$(@server list --json)"
        [[ "${ls_json}" == *'"name":"ao_test_srv"'* ]]
        [[ "${ls_json}" == *'"mode":"named"'* ]]

        # 6. Switch back to local private server and verify original local image is preserved
        @server local
        [[ -z "${_MILK_SERVER_NAME:-}" ]]
        [[ "${PS1}" == *"[milk]"*"user@host:$ " ]]
        [[ "$(@calc 'itot(loc_priv)')" == "128" ]]

        # 7. Stop named server
        @server stop ao_test_srv
        st_stopped="$(@server status ao_test_srv)"
        [[ "${st_stopped}" == *"IDLE"* ]]

        # 8. Optional tmux-backed server test if tmux is available
        if command -v tmux >/dev/null 2>&1; then
            @server start ao_tmux_srv --tmux -d
            st_tmux="$(@server status ao_tmux_srv)"
            [[ "${st_tmux}" == *"RUNNING"* ]]
            [[ "${st_tmux}" == *"named (tmux)"* ]]
            [[ "${st_tmux}" == *"milk_srv_ao_tmux_srv"* ]]
            @server stop ao_tmux_srv
        fi

        echo "SERVER_MGMT_OK"
        """)
    assert res.returncode == 0, f"stdout={res.stdout}\nstderr={res.stderr}"
    assert "SERVER_MGMT_OK" in res.stdout


def test_server_command_and_output_ring_buffer_log(tmp_path):
    res = _run_bash("""
        @mk2Dim "@S:log_im" 8 8
        @calc "log_im = 3"
        @imstats log_im >/dev/null
        @set s.log_im.kw.CAM "SCExAO" "Camera name"

        # Trigger a deliberate error to test error logging
        set +e
        @calc "1 / 0" 2>/dev/null
        set -e

        # 1. Default @log includes command text, durations, and captured stdout/stderr
        log_out="$(@log)"
        [[ "${log_out}" == *"mk2Dim @S:log_im 8 8"* ]]
        [[ "${log_out}" == *"@= log_im = 3"* ]]
        [[ "${log_out}" == *"imstats log_im"* ]]
        [[ "${log_out}" == *"Image size (->imsize0...)"* ]]
        [[ "${log_out}" == *"@set s.log_im.kw.CAM = SCExAO"* ]]
        [[ "${log_out}" == *"@= 1 / 0"* ]]

        # 2. Regex search across command and output (--grep)
        grep_out="$(@log --grep 'rms per pixel')"
        [[ "${grep_out}" == *"imstats log_im"* ]]
        [[ "${grep_out}" != *"mk2Dim @S:log_im"* ]]

        # 3. Error filter (--errors)
        err_out="$(@log --errors)"
        [[ "${err_out}" == *"@= 1 / 0"* ]]
        [[ "${err_out}" != *"@= log_im = 3"* ]]

        # 4. JSON output (--json -n 2)
        json_out="$(@log -n 2 --json)"
        [[ "${json_out}" == *'"cmd":"@set s.log_im.kw.CAM = SCExAO"'* ]]
        [[ "${json_out}" == *'"cmd":"@= 1 / 0"'* ]]
        [[ "${json_out}" == *'"rc":2'* ]]

        # 5. Clear log (--clear)
        @log --clear
        empty_json="$(@log --json)"
        [[ -z "${empty_json}" ]]
        echo "SRVLOG_OK"
        """)
    assert res.returncode == 0, f"stdout={res.stdout}\nstderr={res.stderr}"
    assert "SRVLOG_OK" in res.stdout

    # Also verify @server and @log work inside native milk-cli shorthand mode
    res_cli = _run_milk_cli(
        """
        mem.mk2Dim @L:cli_im 8 8
        @ cli_im = 9
        @server status
        @log --json
        """,
        tmp_path,
        mode="script",
    )
    assert res_cli.returncode == 0, f"stdout={res_cli.stdout}\nstderr={res_cli.stderr}"
    assert "in-process" in res_cli.stdout
    assert '"cmd":"@= cli_im = 9"' in res_cli.stdout


def test_command_hygiene_help_topics_and_fps_lifecycle():
    res = _run_bash("""
        # Phase 1: Excluded shell builtins, REPL-only commands, internal help topics, and duplicate aliases
        dump_out="$("${MILK_INSTALLDIR}/bin/milk-cli" --dump-commands)"
        comp_out="$(milkcomplete "@" cmd)"
        for excl in cd pwd echo printf read shift export unset vars alias unalias aliases \\
                    time watch exit quit exitCLI history searchhist fhist ghistory lhistory \\
                    savehistory savescript setprompt bookmark sessionlog synhl syndiag \\
                    cliindent showmatch helprl fhelp fparam help help-readline \\
                    list-streams list-fps fpsset; do
            ! grep -qx "${excl}" <<< "${dump_out}"
            ! grep -qx "@${excl}" <<< "${comp_out}"
            [[ -z "$(type -t "@${excl}")" ]]
        done

        # Retained commands must still be present
        for kept in mk2Dim imstats listim fpslist proclist procstat procctl procwait; do
            grep -qx "${kept}" <<< "${dump_out}"
            grep -qx "@${kept}" <<< "${comp_out}"
            [[ "$(type -t "@${kept}")" == "function" ]]
        done

        # Phase 2: @hc and @h support milk.bash @ primitives, short command names, and pillar topics
        for prim in waitfor waitany get set calc test exists running grab defer on source server log streamcat streamwrite unload; do
            hc1="$(@hc "@${prim}")"
            [[ "${hc1}" == *"@${prim}"* ]]
            [[ "${hc1}" != *"No command matching"* ]]
            hc2="$(@hc "${prim}")"
            [[ "${hc2}" == *"@${prim}"* ]]
        done
        [[ "$(@hc mk2Dim)" == *"mk2Dim"* ]]
        [[ "$(@hc imstats)" == *"imstats"* ]]
        [[ "$(@h @waitfor)" == *"@waitfor"* ]]
        [[ "$(@h streams)" == *"SHARED MEMORY STREAMS"* ]]
        [[ "$(@h fps)" == *"FUNCTION PARAMETER STRUCTURE"* ]]
        [[ "$(@h procinfo)" == *"PROCESSINFO TELEMETRY"* ]]
        [[ "$(@h seq)" == *"FPS SEQUENCER"* ]]

        # milk-help and milk-stream-help updated syntax
        mhelp="$("${MILK_INSTALLDIR}/bin/milk-help")"
        [[ "${mhelp}" == *"milk.bash"* ]]
        shelp="$("${MILK_INSTALLDIR}/bin/milk-stream-help")"
        [[ "${shelp}" == *"@S:mystream"* ]]
        [[ "${shelp}" == *"mystream[0:10, 0:10]"* ]]
        [[ "${shelp}" != *"mystream@S:0:10"* ]]

        # Phase 3: FPS metadata reads, optional '=' in @set, and FPS lifecycle control
        milk-fpsexec-imggen-mkrndim -n testfps_lc fpsinit >/dev/null 2>&1
        [[ "$(@get f.testfps_lc.conf)" == "OFF" ]]
        [[ "$(@get f.testfps_lc.run)" == "OFF" ]]
        [[ "$(@get f.testfps_lc.status)" == "IDLE" ]]
        (( $(@get f.testfps_lc.nbparam) > 0 ))
        [[ "$(@get f.testfps_lc.list)" == *"xsize"* ]]

        # Optional '=' separator in @set
        @set f.testfps_lc.xsize = 64
        [[ "$(@get f.testfps_lc.xsize)" == "64" ]]
        @set f.testfps_lc.conf = stop
        @set f.testfps_lc.run = stop
        @set f.testfps_lc.confstop
        @set f.testfps_lc.runstop

        # Verify FPS attribute completion includes lifecycle & metadata attributes
        fps_comp="$(milkcomplete "f.testfps_lc." arg)"
        [[ "${fps_comp}" == *"f.testfps_lc.status"* ]]
        [[ "${fps_comp}" == *"f.testfps_lc.conf"* ]]
        [[ "${fps_comp}" == *"f.testfps_lc.run"* ]]
        [[ "${fps_comp}" == *"f.testfps_lc.nbparam"* ]]

        echo "HYGIENE_HELP_FPS_OK"
        """)
    assert res.returncode == 0, f"stdout={res.stdout}\nstderr={res.stderr}"
    assert "HYGIENE_HELP_FPS_OK" in res.stdout
