"""
Comprehensive integration tests for the milk.bash GNU Bash integration layer
and the milk-cli bash engine (milk.bash primitives and milk-cli shorthands).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from milk.cliwrap import CLICommands, HAVE_CLI
from tests.cli.conftest import _run_bash, _run_milk_cli

pytestmark = [
    pytest.mark.skipif(not HAVE_CLI, reason="MILK compiled without CLI support"),
    pytest.mark.timeout(25),
]


def test_builtin_loaded_and_commands_registered():
    res = _run_bash("""
        for b in milkcall milkget milkset milkcalc milktest milkexists milkwaitfor milkgrab milkcomplete milkshutdown milkserver milklog milkdefer; do
            [[ "$(type -t "$b")" == "builtin" ]]
        done
        for f in @calc @test @get @set @exists @ls @waitfor @waitany @sleep @grab @log @defer; do
            [[ "$(type -t "$f")" == "builtin" ]]
        done
        for f in @running @replay @on @source @server @help @pipe @module @config @unload; do
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
        [[ "${hb_out}" == *"@server"* ]]
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
