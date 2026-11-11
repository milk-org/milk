"""milk in bash: keywords, entities, calculator and @{...}, in plain bash and in milk-cli.

Purpose:     a script must behave the same in any bash that sourced milk.bash and in milk-cli,
             which runs its lines in a bash co-process; these tests run the language end to end.
Does:        checks what milk.bash defines (builtins, keyword functions, module commands), @help,
             the banner, the [milk] prompt label and @unload; that the prompt shows a watched value
             or a session name without running it; that usage errors exit 2; stream modifiers (@S:
             @L: @T<type>:); @get and @set on streams, FITS keywords and FPSs; @calc on images,
             slices and entities; @test, @exists, @waitfor, @waitany, @grab, @on, @source, @defer
             and pipelines; @{...} in milk-cli -s, -c and piped input, and through CLICommands; calc
             mode; that milk.bash loads nothing without libmilkbash.so; that waits release the
             stream's semaphore slots.
Covers:      scripts/milk.bash; the libmilkbash.so builtins and the client code they call
             (src/cli/libmilkscript: bash/milk_bash_builtins.c, milk_bash_hooks.c,
             milk_bash_defer.c, client/, ipc/milk_waitfor.c); the session server (server/);
             milk-cli's bash co-process (bash/milk_bash_coproc.c, milk_bash_exec.c);
             milk.cliwrap.CLICommands.
Uses:        conftest.py: _run_bash, _run_milk_cli, milk_bash, value, _install_dir.
See also:    docs/guide/scripting.md; docs/guide/sessions.md, "Using milk from Plain Bash";
             bash_translation_test.py.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import os
import re
import shutil
import subprocess

from milk.cliwrap import CLICommands, HAVE_CLI
from tests.cli.conftest import (
    _run_bash,
    _run_milk_cli,
    _install_dir,
    milk_bash,
    value,
)

pytestmark = [
    pytest.mark.skipif(not HAVE_CLI, reason="MILK compiled without CLI support"),
    pytest.mark.timeout(25),
]


def test_builtin_loaded_and_commands_registered():
    res = _run_bash("""
        for b in milkcall milkget milkset milkcalc milktest milkexists milkwaitfor milkgrab milkcomplete milkshutdown milksession milklog milkdefer; do
            [[ "$(type -t "$b")" == "builtin" ]]
        done
        for f in @calc @test @get @set @exists @ls @waitfor @waitany @sleep @grab @log @defer; do
            [[ "$(type -t "$f")" == "builtin" ]]
        done
        for f in @running @replay @on @source @session @help @pipe @module @config @unload; do
            [[ "$(type -t "$f")" == "function" ]]
        done
        [[ "$(type -t mem.mk2Dim)" == "function" ]]
        [[ "$(type -t info.imstats)" == "function" ]]

        # Standard bash help must remain untouched, while @help and <cmd> -h invoke milk help
        help echo | grep -qi "echo"
        h_out="$(@help)"
        [[ "${h_out}" == *"source milk.bash"* ]]
        [[ "${h_out}" == *"@help <topic>"* ]]
        hb_out="$(@help bash)"
        [[ "${hb_out}" == *"BASH-ENHANCED MODE"* ]]
        [[ "${hb_out}" == *"@waitfor"* ]]
        [[ "${hb_out}" == *"@session"* ]]
        [[ "${hb_out}" == *"@log"* ]]
        @help bash | grep -q "BASH-ENHANCED MODE"
        @help mem >/dev/null
        @help mem.mk2Dim >/dev/null
        mem.mk2Dim -h >/dev/null

        # Verify visible [milk] prompt label, startup banner, and complete @-command removal on @unload
        shopt -s extglob
        PS1='user@host:$ '
        unset _MILK_ORIG_PS1
        banner_out="$(MILK_BANNER=1 source "${MILK_INSTALLDIR}/bin/milk.bash")"
        [[ "${banner_out}" == *"milk v"*"bash integration loaded"* ]]
        [[ "${banner_out}" == *"for help"* ]]
        source "${MILK_INSTALLDIR}/bin/milk.bash"
        [[ "$(type -t @calc)" == "builtin" && "$(type -t @running)" == "function" && -z "$(type -t @)" ]]
        [[ "${PS1}" == *"[milk]"*"user@host:$ " ]]
        ps1_once="${PS1}"
        source "${MILK_INSTALLDIR}/bin/milk.bash"
        [[ "${PS1}" == "${ps1_once}" ]]
        @unload
        [[ "${PS1}" == 'user@host:$ ' ]]
        [[ -z "$(compgen -c "@")" ]]
        [[ -z "$(type -t mem.mk2Dim)" ]]
        set +e
        mem.mk2Dim 2>/dev/null
        rc_after_unload=$?
        set -e
        [[ "${rc_after_unload}" -eq 127 ]]
        echo "OK"
        """)
    assert res.returncode == 0, res.stderr
    assert "OK" in res.stdout


def test_prompt_shows_data_and_never_runs_it(tmp_path):
    """What the prompt shows of a watched value or a session name is data: bash expands PS1, and a
    `$(...)` or a backslash in it must stay text, with promptvars and without. The record of the
    shell's session is read only if it is a regular file of ours."""
    res = milk_bash(
        r"""
mem.mk2Dim "@S:pim" 4 4 >/dev/null
@set s.pim.kw.NOTE='$(touch run_kw)'
@config prompt watch s.pim.kw.NOTE
_milk_apply_ps1; echo "kw=${PS1@P}"
@config prompt clear
rec="${TMPDIR:-/tmp}/.milk_active_sock_${_MILK_OV_ID}"; sock="$MILKCLI_SOCK"
printf '%s' '@milk-cli-0-$(touch run_name)\w' > "$rec"
_milk_apply_ps1; echo "name=${PS1@P}"
shopt -u promptvars; _milk_apply_ps1; echo "novars=${PS1@P}"; shopt -s promptvars
rm -f "$rec"; unset _MILK_SERVER_NAME; export MILKCLI_SOCK="$sock"
printf '%s' '@milk-cli-0-linked' > target; ln -s "$PWD/target" "$rec"
_milk_apply_ps1; echo "link=[${_MILK_SERVER_NAME:-}]"
rm -f "$rec"
echo "ran=$(ls | grep -c '^run_')"
""",
        tmp_path,
    )
    out = res.stdout + res.stderr
    assert "NOTE=$(touch run_kw)" in value(res.stdout, "kw"), out
    assert r"milk:$(touch run_name)\w" in value(res.stdout, "name"), out
    assert r"milk:$(touch run_name)\w" in value(res.stdout, "novars"), out
    assert value(res.stdout, "link") == "[]", out
    assert value(res.stdout, "ran") == "0", out


def test_usage_errors_exit_2(tmp_path):
    """A keyword called with a missing or surplus argument says how it is used and exits 2
    (docs/guide/scripting.md, "Exit Status"); `-h` is help for the keywords bash overrides.
    """
    cases = {
        "set": "@set",
        "calc": "@calc",
        "test": "@test",
        "exists": "@exists",
        "exists2": "@exists s.a s.b",
        "running": "@running",
        "source": "@source",
        "on": "@on",
        "on1": "@on s.a",
        "onstop": "@on --stop",
        "watch": "@config prompt watch",
        "unwatch": "@config prompt unwatch",
        "preset": "@config prompt preset",
        "prompt": "@config prompt nosuchword",
    }
    script = "".join(f'{cmd} 2>{k}.err; echo "{k}=$?"\n' for k, cmd in cases.items())
    res = milk_bash(
        script + '@restore -h | grep -qi "restore" && echo "help=ok"', tmp_path
    )
    out = res.stdout + res.stderr
    for k in cases:
        assert value(res.stdout, k) == "2", f"{cases[k]}: {out}"
        assert "sage:" in (tmp_path / f"{k}.err").read_text(), f"{cases[k]}: {out}"
    assert value(res.stdout, "help") == "ok", out


def test_stream_modifiers_all_types():
    res = _run_bash("""
        # Location modifiers: @S: (shared) and @L: (local)
        mem.mk2Dim "@S:sh_im" 8 8
        mem.mk2Dim "@L:loc_im" 8 8
        [[ "$(@get s.sh_im.shared)" == "1" && "$(@get s.loc_im.shared)" == "0" ]]

        # Explicit @T<type>: datatype modifiers
        for spec in \\
            F32:FLOAT:FLT32 F64:DOUBLE:FLT64 F16:HALF:FLT16 \\
            U8:UINT8: U16:UINT16: U32:UINT32: U64:UINT64: \\
            I8:INT8: I16:INT16: I32:INT32: I64:INT64: \\
            C64:COMPLEX_FLOAT:CPLX32 C128:COMPLEX_DOUBLE:CPLX64; do
            IFS=: read -r mod dt tn <<< "$spec"
            im="im_${mod,,}"
            mem.mk2Dim "@T${mod}:${im}" 8 8
            [[ "$(@get "s.${im}.dtype")" == "$dt" ]]
            [[ -z "$tn" || "$(@get "s.${im}.typename")" == "$tn" ]]
        done

        # Combined location + datatype modifiers
        mem.mk2Dim "@STF64:sh_f64" 10 12
        mem.mk2Dim "@LTU16:loc_u16" 10 12
        [[ "$(@get s.sh_f64.shared)" == "1" && "$(@get s.sh_f64.dtype)" == "DOUBLE" ]]
        [[ "$(@get s.loc_u16.shared)" == "0" && "$(@get s.loc_u16.dtype)" == "UINT16" ]]

        # Invalid datatype modifier must fail cleanly
        ! mem.mk2Dim "@TX99:bad_im" 8 8 2>/dev/null
        echo "OK"
        """)
    assert res.returncode == 0, res.stderr
    assert "OK" in res.stdout


def test_get_set_fps_kw_and_dcvar_sync():
    res = _run_bash("""
        imgen.mkrndim "@S:im1" 32 48
        for kv in xsize:32 size0:32 ysize:48 size1:48 zsize:1 naxis:2 nelement:1536 dtype:FLOAT shared:1; do
            [[ "$(@get "s.im1.${kv%%:*}")" == "${kv#*:}" ]]
        done

        # Zero-fork @get -v into bash variable and dcvar sync after info.imstats
        @get -v ys s.im1.ysize
        [[ "$ys" == "48" ]]
        info.imstats im1 >/dev/null
        [[ "$imsize0" == "32" && "$imsize1" == "48" && -n "$vp50" && -n "$vrms" ]]

        # FITS keyword read/write (int, float, string, stdin pipe, wildcard)
        @set s.im1.kw.NFRAMES 100 "Frame count"
        @set s.im1.kw.EXPTIME 0.0125 "Exposure time [s]"
        @set s.im1.kw.FILTER "H-band" "Filter name"
        [[ "$(@get s.im1.kw.NFRAMES)" == "100" && "$(@get s.im1.kw.EXPTIME)" == "0.0125" && "$(@get s.im1.kw.FILTER)" == "H-band" ]]

        echo "0.025" | @set s.im1.kw.EXPTIME -
        [[ "$(@get s.im1.kw.EXPTIME)" == "0.025" ]]

        kwlist="$(@get "s.im1.kw.*")"
        [[ "$kwlist" == *"NFRAMES"* && "$kwlist" == *"EXPTIME"* && "$kwlist" == *"FILTER"* ]]

        # FPS creation via standalone fpsinit, @exists, @get, @set
        milk-fpsexec-imggen-mkrndim -n testfps fpsinit >/dev/null 2>&1
        @exists f.testfps
        [[ "$(@get "f.testfps.*")" == *"xsize"* ]]
        @set f.testfps.xsize 256
        @set f.testfps.ysize=512
        [[ "$(@get f.testfps.xsize)" == "256" && "$(@get f.testfps.ysize)" == "512" ]]
        echo "OK"
        """)
    assert res.returncode == 0, res.stderr
    assert "OK" in res.stdout


def test_calc_inplace_slices_and_functions():
    res = _run_bash("""
        mem.mk2Dim "@S:im1" 16 16
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

        # Passing variable slice bounds to CLI commands (info.imstats)
        # 1) Double quotes (bash expands variables)
        info.imstats "im1[${x0}:${x1},${x0}:${x1}]" >/dev/null
        [[ "$imsize0" == "5" && "$imsize1" == "5" ]]
        # 2) Single quotes: the expression parser reads calc variables (the @calc above named x0 x1 cx r)
        info.imstats 'im1[x0:x1, cx-r:cx+r]' >/dev/null
        [[ "$imsize0" == "5" && "$imsize1" == "5" ]]
        [[ "$vmin" == "20" && "$vmax" == "20" ]]

        # Entity reads and writes inside @calc
        @calc -v area "s.im1.xsize * s.im1.ysize"
        [[ "$area" == "256" ]]
        @calc "s.im1.kw.GAIN = 2.5 * 4"
        [[ "$(@get s.im1.kw.GAIN)" == "10" ]]

        # Matrix multiplication (2D x 2D and 2D x 1D)
        mem.mk2Dim "@S:mA" 2 2
        mem.mk2Dim "@S:mB" 2 2
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
        imgen.mkrndim "@S:im1" 32 32
        @calc "im1 = im1 * 10.0"
        @set s.im1.kw.FILTER "H-band"
        @set s.im1.kw.EXPTIME 0.02

        @exists s.im1
        ! @exists s.nonexistent_stream
        ! @exists f.nonexistent_fps
        ! @exists p.nonexistent_proc

        # Test both bare s. and @s. entity syntax, numeric & string comparisons
        @test 'irms(im1) > 0.5 && s.im1.xsize == 32 && s.im1.dtype == "FLOAT"'
        @test 's.im1.xsize == 32 && s.im1.dtype == "FLOAT" && s.im1.kw.FILTER == "H-band"'
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
        'mem.mk2Dim @S:src_im 6 8\n@calc "src_im = $1 * 3.0"\n@calc -v src_tot "itot(src_im)"\n'
    )

    res = _run_bash(f"""
        mem.mk2Dim "@S:live" 8 8
        mem.mk2Dim "@S:live2" 8 8

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


def test_pipelines_streamio_defer_and_xargs():
    res = _run_bash("""
        # Concurrent milk commands in a pipeline: @grab --stdout | awk | @replay --stdin
        mem.mk2Dim "@S:s1" 4 4
        mem.mk2Dim "@S:s2" 4 4
        @calc "s1 = 3.0"
        @grab --stdout s1 | awk '{print $1 * 4}' | @replay --stdin s2 -
        [[ "$(@calc "s2[0,0]")" == "12" ]]
        [[ "$(@calc "itot(s2)")" == "192" ]]

        # Pipe-clean mem.listim (one stream name per line when stdout is not a tty)
        names="$(mem.listim | sort)"
        [[ "$names" == *"s1"* && "$names" == *"s2"* ]]

        # External tool invocation via the libexec/milk/bin/<module.cmd> links
        printf "s1\\ns2\\n" | xargs -n1 info.imstats >/dev/null

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


def test_milk_cli_shorthands_comprehensive(tmp_path: Path):
    res_s = _run_milk_cli(
        """
        mem.mk2Dim @S:t2im 14 18
        @calc -v a "2.5 + 3.5"
        @calc "t2im = a * 2.0"
        @calc "t2im[1:3, 2:4] = 20.0"
        @calc -v tot "itot(t2im)"

        # Inline @{<expr>} and unquoted/quoted @s.<name>.<prop>
        val=@{a * 10 + t2im[2, 3]}
        nx=@{s.t2im.xsize}
        @set s.t2im.kw.OBSERVER="Subaru"
        @set s.t2im.kw.EXPTIME=0.05
        @calc "s.t2im.kw.GAIN = a * 1.5"

        # Test @test with @s. inside both single and double quotes
        @test 's.t2im.dtype == "FLOAT" && s.t2im.kw.OBSERVER == "Subaru"'
        @test "s.t2im.dtype == \\"FLOAT\\" && s.t2im.kw.GAIN == 9"

        # Test @exists s.<stream>
        if @exists s.t2im && ! @exists s.no_such_stream; then
            echo "T2_OK=${a}:${tot}:${val}:${nx}:@{s.t2im.ysize}:@{s.t2im.kw.OBSERVER}.:@{s.t2im.kw.EXPTIME}:@{s.t2im.kw.GAIN}"
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
        'mem.mk2Dim @S:cim 10 20; @calc "cim = 4"; echo "CMD_OK=@{s.cim.ysize}:@{itot(cim)}"',
        tmp_path,
        mode="cmd",
    )
    assert res_c.returncode == 0, res_c.stderr
    assert "CMD_OK=20:800" in res_c.stdout

    # Test milk-cli stdin mode
    res_in = _run_milk_cli(
        'mem.mk2Dim @S:inim 6 7\n@calc "inim = 5"\necho "STDIN_OK=@{s.inim.xsize}:@{itot(inim)}"',
        tmp_path,
        mode="stdin",
    )
    assert res_in.returncode == 0, res_in.stderr
    assert "STDIN_OK=6:210" in res_in.stdout

    # Verify milk.cliwrap.CLICommands works with milk.bash and milk-cli shorthand syntax
    with CLICommands(
        [
            "mem.mk2Dim @S:wrap_im 8 8",
            '@calc "wrap_im = 6"',
            'echo "CLIWRAP_OK=@{s.wrap_im.xsize}:@{itot(wrap_im)}"',
        ]
    ) as res_wrap:
        assert res_wrap.returncode == 0, res_wrap.stderr
        assert "CLIWRAP_OK=8:384" in res_wrap.stdout


def test_calc_repl_inproc_and_bash_sync(tmp_path: Path):
    script = """
bx=40
by=100
@calc
a = 1.5 + 2.5
b = a * 3
c = bx + 2
bx = bx + 10
d = bx + by
e = $bx + 5
exit
bx=7
@calc
f = bx * 3
exit
echo "CALC_REPL_OK=@{a}:@{b}:@{c}:@{d}:@{e}:@{f}"
"""
    for mode in ("stdin", "script"):
        res = _run_milk_cli(script, tmp_path, mode=mode)
        assert res.returncode == 0, res.stderr
        assert "CALC_REPL_OK=4:12:42:150:45:21" in res.stdout


@pytest.mark.timeout(120)
def test_milk_bash_stops_without_its_builtins(tmp_path):
    (tmp_path / "bin").mkdir()
    shutil.copy(_install_dir() / "bin" / "milk.bash", tmp_path / "bin")
    (tmp_path / "bin" / "milk-cli").write_text("#!/bin/sh\n")
    (tmp_path / "bin" / "milk-cli").chmod(0o755)
    res = subprocess.run(
        [
            "bash",
            "-c",
            f'source "{tmp_path}/bin/milk.bash"; echo "rc=$? kw=$(type -t @get)"',
        ],
        capture_output=True,
        text=True,
        timeout=20,
        env={**os.environ, "MILK_INSTALLDIR": str(tmp_path)},
    )
    assert value(res.stdout, "rc") == "1 kw=", res.stdout + res.stderr
    assert (
        "libmilkbash.so is missing" in res.stderr and "milk is not loaded" in res.stderr
    ), res.stderr


@pytest.mark.timeout(120)
def test_semaphore_slots_released_after_waits(tmp_path):
    """Six concurrent `@waitfor` on one stream, then none of the stream's reader slots is still held."""
    res = milk_bash(
        'mem.mk2Dim "@S:x" 4 4 >/dev/null\n'
        'for i in 1 2 3 4 5 6; do ( @waitfor s.x --timeout 0.6 >/dev/null 2>&1; echo "w=$?" ) & done\n'
        "sleep 0.2; mem.imseminfo x > during.txt 2>&1; wait\n"
        "mem.imseminfo x > after.txt 2>&1\n"
        '@waitfor s.x --timeout 0.2 >/dev/null 2>&1; echo "last=$?"\n'
        "mem.imseminfo x > final.txt 2>&1",
        tmp_path,
    )
    assert res.stdout.count("w=124") == 6, res.stdout + res.stderr

    def held(path: Path) -> list[int]:
        pids = []
        for ln in path.read_text().splitlines():
            m = re.match(r"\s*(\d+)\s+(\d+)\s+(-?\d+)\s+(-?\d+)\s*$", ln)
            if m and int(m.group(4)) > 0:
                pids.append(int(m.group(4)))
        return pids

    assert held(tmp_path / "during.txt"), (tmp_path / "during.txt").read_text()
    assert held(tmp_path / "after.txt") == [], (tmp_path / "after.txt").read_text()
    assert held(tmp_path / "final.txt") == [], (tmp_path / "final.txt").read_text()


def test_advanced_multi_subsystem_ao_pipeline(tmp_path: Path):
    """End-to-end multi-operation script combining:
    - `@calc` mode blocks (with comments, `quit` and `exit`), `$((...))` bash arithmetic inside `@calc`,
    - complex wavefront construction (`expi`, `cplx`), 2D FFT/IFFT (`fft`, `ifft`, `fftshift`),
    - spatial filtering (`gauss`, `boxmean`, `medfilt3`), radial profile (`rprof`),
    - linear algebra (`svd`, `pinv`, `@` matmul, `einsum`, `solve`),
    - shared-memory stream properties & keywords (`s.<name>.kw.*`),
    - and a live `milk-fpsexec-milk-calc` background unit using `hist()` and `ema()` with `@waitfor`.
    """
    script = """
    N=$((16 + 16))
    mem.mk2Dim @S:wf_phs $N $N
    mem.mk2Dim @S:wf_amp $N $N
    mem.mk2Dim @S:dm_cmd $N $N
    mem.mk2Dim @S:dm_ema $N $N

    # 1. Populate pupil amplitude and aberration phase in a @calc mode block ended by 'quit'
    @calc
    # Pupil mask and synthetic Zernike-like phase screen
    r_norm = ir(wf_amp) / (N * 0.4)
    theta  = itheta(wf_amp)
    wf_amp = where(r_norm <= 1.0, 1.0, 0.0)
    wf_phs = wf_amp * (0.6 * (2.0 * pow(r_norm, 2) - 1.0) * cos(theta) + 0.3 * sin(2.0 * theta))
    s.wf_phs.kw.WAVELEN = 1.65
    s.wf_phs.kw.INSTRUM = "SCExAO"
    quit

    # 2. Complex field, PSF via FFT, inverse FFT round-trip, and spatial/radial diagnostics
    @calc
    u_pup = wf_amp * expi(wf_phs)
    u_foc = fft(u_pup)
    psf   = pow(abs(u_foc), 2)
    psf_s = fftshift(psf)
    u_rec = ifft(u_foc)
    rec_err = imax(abs(real(u_rec) - real(u_pup)))
    psf_sm  = gauss(psf_s, 1.2)
    psf_rp  = rprof(psf_s)
    exit

    # 3. Modal response matrix calibration & SVD pseudo-inverse control law
    @calc
    # Build 4 synthetic influence modes (flattened 64-element sensor vectors x 4 modes = [64 x 4])
    m0 = reshape(ix(zeros(8, 8)) / 4.0, 64, 1)
    m1 = reshape(iy(zeros(8, 8)) / 4.0, 64, 1)
    m2 = reshape((pow(ix(zeros(8, 8)), 2) - pow(iy(zeros(8, 8)), 2)) / 16.0, 64, 1)
    m3 = reshape(ix(zeros(8, 8)) * iy(zeros(8, 8)) / 16.0, 64, 1)
    Rmat = vstack(m0, m1, m2, m3)
    U, svals, Vt = svd(Rmat)
    Cmat = pinv(Rmat, 1e-4)
    # Test target vector recovery: y = Rmat @ c_true -> c_est = Cmat @ y
    c_true = reshape([0.5, -0.25, 0.75, -0.5], 4, 1)
    y_meas = Rmat @ c_true
    c_est  = Cmat @ y_meas
    ctrl_err = imax(abs(c_est - c_true))
    gram_mat = einsum("ki,kj->ij", Rmat, Rmat)
    gram_tr  = itot(gram_mat * eye(4))
    exit

    # 4. Run a background milk.calc unit with temporal hist() and ema() on dm_cmd
    milk-fpsexec-milk-calc -n ao_telemetry -t dm_cmd -e "dm_ema = ema(dm_cmd, 0.5); dm_diff = dm_cmd - hist(dm_cmd, 1)" fpsinit >/dev/null 2>&1
    milk-fpsexec-milk-calc -n ao_telemetry run >/dev/null 2>&1 &
    unit_pid=$!
    @defer "kill -INT $unit_pid 2>/dev/null || true; milk-fps-rm ao_telemetry >/dev/null 2>&1 || true"

    sleep 0.25
    for k in 1 2 3 4; do
        @calc "dm_cmd = wf_phs * $((k * 2))"
        sleep 0.05
    done
    @waitfor "s.dm_ema.cnt0 >= 2" --timeout 5.0
    kill -INT $unit_pid 2>/dev/null || true
    wait $unit_pid 2>/dev/null || true

    echo "AO_OK=@{rec_err < 1e-4}:@{ctrl_err < 1e-4}:@{s.wf_phs.kw.INSTRUM}:@{s.wf_phs.kw.WAVELEN}:@{itot(psf_rp) > 0}:@{gram_tr > 0}:@{s.dm_ema.xsize}"
    """
    res = _run_milk_cli(script, tmp_path, mode="script")
    assert res.returncode == 0, res.stdout + res.stderr
    assert "AO_OK=1:1:SCExAO:1.65:1:1:32" in res.stdout
