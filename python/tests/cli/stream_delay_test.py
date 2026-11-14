"""Tests for `mem.streamdelay` (`milk-fpsexec-mem-streamdelay`) and calc's `streamdelay(...)`.

Covers:
  - Shared `milk_stream_delay_*` ring-buffer state engine (`COREMOD_memory/stream_delay.{h,c}`)
  - Tier 3 standalone unit `milk-fpsexec-mem-streamdelay`:
    * Discrete & fractional frame-unit (`unit=1`) and time-unit (`unit=0`) delays
    * Causal temporal window filters (`timeavemode` = 1 boxcar, 2 linear, 3 exp, 4 gauss)
    * Arbitrary 1-D FIR tap weights (`weights_name`), including signed differentiators `[1, -1]`
    * 1-D, 2-D, and 3-D stream shapes and `UINT16` -> `FLOAT32` (`out_float=ON`) conversion
  - Tier 2 built-in calculator operator `streamdelay(...)` (`src/calc/ext/mc_ext_builtin.c`):
    * `--list` catalog visibility in `@calc` and `milk-fpsexec-milk-calc`
    * Discrete & fractional frame delays, time-unit delays (including `UINT16` `writetime` preservation)
    * Window filters, optional `weights` port (positional & keyword), and `:label` FPS isolation
"""

from __future__ import annotations

import numpy as np
import pytest

from milk.cliwrap import HAVE_CLI
from tests.cli.conftest import milk_bash, value

pytestmark = [
    pytest.mark.skipif(not HAVE_CLI, reason="MILK compiled without CLI support"),
    pytest.mark.timeout(60),
]


def test_calc_streamdelay_catalog(tmp_path):
    """Verify `streamdelay` appears in both `@calc --list` and `milk-fpsexec-milk-calc --list`."""
    res = milk_bash(
        """
@calc --list streamdelay | grep -q "streamdelay" && echo "bash_list=ok"
milk-fpsexec-milk-calc --list streamdelay | grep -q "Tier 2 operator" && echo "unit_list=ok"
milk-fpsexec-milk-calc --list --json streamdelay | grep -q '"name": "streamdelay"' && echo "json_list=ok"
""",
        tmp_path,
    )
    assert [value(res.stdout, k) for k in ("bash_list", "unit_list", "json_list")] == [
        "ok",
        "ok",
        "ok",
    ], (
        res.stdout + res.stderr
    )


def test_calc_streamdelay_frames_interp_window_and_fir(tmp_path):
    """Verify `streamdelay(...)` in `@calc` for discrete & fractional frame delays,
    causal window filters, FIR tap weights (positional and keyword), and 1D/2D/3D shapes.
    """
    res = milk_bash(
        """
# 1. Discrete integer frame delay (delay=2, unit=1): zero pre-warmup then exact 2-step lag
mem.mk2Dim "@S:in2d" 4 4 >/dev/null
for v in 10 20 30 40 50; do
    @calc "in2d = $v; d2 = streamdelay:int2(in2d, unit=1, delay=2)"
    echo -n "$(@get s.d2.mean),"
done | sed 's/,$//' | xargs -I{} echo "seq_d2={}"

# 2. Fractional linear frame delay (delay=1.25, unit=1, interp=1): (1-0.25)*x[n-1] + 0.25*x[n-2]
for v in 10 20 40 80; do
    @calc "in2d = $v; df = streamdelay:frac(in2d, unit=1, delay=1.25, interp=1)"
done
# At v=80: n-1 is 40, n-2 is 20 -> 0.75*40 + 0.25*20 = 35
echo "frac_val=$(@get s.df.mean)"

# 3. Causal boxcar window average (avemode=1, unit=1, delay=0, window=2 -> mean of last 3 frames)
for v in 10 20 30 60; do
    @calc "in2d = $v; wb = streamdelay:box(in2d, unit=1, delay=0, avemode=1, window=2)"
done
# Last 3 frames at v=60: (60 + 30 + 20) / 3 = 36.666668 -> check 3 * mean == 110
echo "box_ok=$(@calc 'abs(imean(wb) * 3.0 - 110.0) < 1e-4')"

# 4. Causal linear taper window (avemode=2, unit=1, delay=0, window=2): weights [1.0, 0.5, 0.0] -> (2*x[n] + x[n-1])/3
for v in 10 20 50; do
    @calc "in2d = $v; wl = streamdelay:lin(in2d, unit=1, delay=0, avemode=2, window=2)"
done
# (1.0*50 + 0.5*20) / 1.5 = 40
echo "lin_val=$(@get s.wl.mean)"

# 5. Arbitrary 1D FIR tap weights [1.0, -1.0] (temporal differentiator, sum(w) == 0!)
@calc "taps_diff = [1.0, -1.0]"
for v in 10 15 27; do
    @calc "in2d = $v; diff_out = streamdelay:fir1(in2d, taps_diff, unit=1, delay=0)"
done
# 27 - 15 = 12
echo "fir_diff=$(@get s.diff_out.mean)"

# Keyword port binding weights=taps_w with [0.5, 0.3, 0.2]
@calc "taps_w = [0.5, 0.3, 0.2]"
for v in 10 20 40; do
    @calc "in2d = $v; fir_kw = streamdelay:fir2(in2d, weights=taps_w, unit=1, delay=0)"
done
# 0.5*40 + 0.3*20 + 0.2*10 = 28
echo "fir_kw=$(@get s.fir_kw.mean)"

# 6. 1D and 3D tensor support
@calc "S:in1d = zeros(8)"
mem.mk3Dim "@S:in3d" 3 3 2 >/dev/null
for v in 7 14; do
    @calc "S:in1d = $v; out1d = streamdelay:s1(in1d, unit=1, delay=1)"
    @calc "S:in3d = $v; out3d = streamdelay:s3(in3d, unit=1, delay=1)"
done
echo "s1_ok=$(@get s.out1d.naxis):$(@get s.out1d.xsize):$(@get s.out1d.mean)"
echo "s3_ok=$(@get s.out3d.naxis):$(@get s.out3d.zsize):$(@get s.out3d.mean)"
""",
        tmp_path,
    )
    assert value(res.stdout, "seq_d2") == "0,0,10,20,30", res.stdout + res.stderr
    assert float(value(res.stdout, "frac_val")) == pytest.approx(35.0, abs=1e-4)
    assert value(res.stdout, "box_ok") == "1", res.stdout + res.stderr
    assert float(value(res.stdout, "lin_val")) == pytest.approx(40.0, abs=1e-4)
    assert float(value(res.stdout, "fir_diff")) == pytest.approx(12.0, abs=1e-4)
    assert float(value(res.stdout, "fir_kw")) == pytest.approx(28.0, abs=1e-4)
    assert value(res.stdout, "s1_ok") == "1:8:7", res.stdout + res.stderr
    assert value(res.stdout, "s3_ok") == "3:2:7", res.stdout + res.stderr


def test_calc_unit_streamdelay_labels_and_time_mode(tmp_path):
    """Verify `milk-fpsexec-milk-calc` exposes `:label` FPS parameters for multiple
    `streamdelay` instances, supports live `@set` tuning, and preserves `writetime` on `UINT16`.
    """
    res = milk_bash(
        r"""
stop_unit() {
    kill -INT "$1" 2>/dev/null
    for _ in $(seq 60); do kill -0 "$1" 2>/dev/null || break; sleep 0.03; done
    wait "$1" 2>/dev/null || true
}

mem.mk2Dim "@S:clk_in" 4 4 >/dev/null
@calc "clk_in = 0"

# Two labeled instances in one calc unit -> exposes .wfs.* and .dm.* in FPS
cat > "$MILK_SHM_DIR/sd_dual.calc" <<'EOF'
w_out = streamdelay:wfs(clk_in, unit=1)
d_out = streamdelay:dm(clk_in, unit=1)
EOF
milk-fpsexec-milk-calc -n u_sd -t clk_in -f "$MILK_SHM_DIR/sd_dual.calc" fpsinit >/dev/null
@set f.u_sd.wfs.delay=1 f.u_sd.dm.delay=2
echo "fps_delays=$(@get f.u_sd.wfs.delay):$(@get f.u_sd.dm.delay)"

milk-fpsexec-milk-calc -n u_sd run >/dev/null 2>&1 &
u_pid=$!
for _ in $(seq 80); do [ "$(@get p.u_sd.loopcnt 2>/dev/null || echo 0)" -ge 0 ] && [ -e "$MILK_SHM_DIR/proc.u_sd."* ] 2>/dev/null && break; sleep 0.02; done
sleep 0.1

for v in 100 200 300; do
    c_prev=$(@get s.d_out.cnt0 2>/dev/null || echo 0)
    @calc "S:clk_in = $v"
    for _ in $(seq 60); do
        [ "$(@get s.d_out.cnt0 2>/dev/null || echo 0)" -gt "$c_prev" ] && break
        sleep 0.02
    done
done
# At v=300: w_out (delay=1) == 200, d_out (delay=2) == 100
echo "run1=$(@get s.w_out.mean):$(@get s.d_out.mean)"

# Live tune wfs.delay from 1 -> 0.5 with interp=1 without restarting unit!
@set f.u_sd.wfs.delay=0.5 f.u_sd.wfs.interp=1
c_prev=$(@get s.w_out.cnt0)
@calc "S:clk_in = 500"
for _ in $(seq 60); do
    [ "$(@get s.w_out.cnt0)" -gt "$c_prev" ] && break
    sleep 0.02
done
# At v=500: 0.5*500 + 0.5*300 = 400
echo "run2=$(@get s.w_out.mean)"
stop_unit "$u_pid"

# Time-unit delay on a UINT16 shared stream (tests ensure_f32_im writetime preservation)
@calc "S:u16_cam = u16(zeros(4, 4))"
@calc "S:u16_cam = 10; u16_d = streamdelay:tu16(u16_cam, unit=0, delay=0.05)"
sleep 0.08
@calc "S:u16_cam = 90; u16_d = streamdelay:tu16(u16_cam, unit=0, delay=0.05)"
# Frame 10 is ~80ms old (>= 50ms), Frame 90 is 0ms old -> discrete causal output is 10
echo "u16_time=$(@get s.u16_d.mean)"
""",
        tmp_path,
    )
    assert value(res.stdout, "fps_delays") == "1:2", res.stdout + res.stderr
    assert value(res.stdout, "run1") == "200:100", res.stdout + res.stderr
    assert float(value(res.stdout, "run2")) == pytest.approx(400.0, abs=1e-4)
    assert float(value(res.stdout, "u16_time")) == pytest.approx(10.0, abs=1e-4)


def test_mem_streamdelay_unit_modes(tmp_path):
    """Verify standalone `milk-fpsexec-mem-streamdelay` in frame mode, time mode,
    fractional interpolation, causal window filtering, and live 1D FIR `.weights_name`.
    """
    res = milk_bash(
        r"""
stop_unit() {
    kill -INT "$1" 2>/dev/null
    for _ in $(seq 60); do kill -0 "$1" 2>/dev/null || break; sleep 0.03; done
    wait "$1" 2>/dev/null || true
}

# Create 3D input stream (tests non-2D support in mem.streamdelay!)
mem.mk3Dim "@S:sd_in" 4 4 2 >/dev/null
@calc "S:sd_in = 5"

milk-fpsexec-mem-streamdelay -n sd_u fpsinit >/dev/null
@set f.sd_u.in_name=sd_in f.sd_u.out_name=sd_out f.sd_u.unit=1 f.sd_u.delayfr=1.5 \
     f.sd_u.interp=1 f.sd_u.out_float=ON \
     f.sd_u.procinfo.triggersname=sd_in f.sd_u.procinfo.triggermode=3 \
     f.sd_u.procinfo.loopcntMax=-1

milk-fpsexec-mem-streamdelay -n sd_u run >/dev/null 2>&1 &
sd_pid=$!
for _ in $(seq 80); do [ -e "$MILK_SHM_DIR/sd_out.im.shm" ] && break; sleep 0.02; done
sleep 0.08

for v in 10 20 60; do
    c_prev=$(@get s.sd_out.cnt0 2>/dev/null || echo 0)
    @calc "S:sd_in = $v"
    for _ in $(seq 60); do
        [ "$(@get s.sd_out.cnt0 2>/dev/null || echo 0)" -gt "$c_prev" ] && break
        sleep 0.02
    done
done
# At v=60: lag 1.5 frames between age 1 (20) and age 2 (10) -> 0.5*20 + 0.5*10 = 15
echo "f_interp=$(@get s.sd_out.mean)"
echo "out_zsize=$(@get s.sd_out.zsize)"
echo "lag_stat=$(@get f.sd_u.status.framelag)"

# Live causal boxcar window filter (timeavemode=1, avewidth=1.0 at delayfr=0)
@set f.sd_u.delayfr=0.0 f.sd_u.interp=0 f.sd_u.option.timeavemode=1 f.sd_u.option.avewidth=1.0
c_prev=$(@get s.sd_out.cnt0)
@calc "S:sd_in = 80"
for _ in $(seq 60); do
    [ "$(@get s.sd_out.cnt0)" -gt "$c_prev" ] && break
    sleep 0.02
done
# Mean of age 0 (80) and age 1 (60) -> (80 + 60) / 2 = 70
echo "live_box=$(@get s.sd_out.mean)"

# Live-attach 1D FIR tap weights stream [1.0, -1.0] at delayfr=0 while unit is running!
@calc "S:sd_taps = [1.0, -1.0]"
@set f.sd_u.option.timeavemode=0 f.sd_u.weights_name=sd_taps
c_prev=$(@get s.sd_out.cnt0)
@calc "S:sd_in = 100"
for _ in $(seq 60); do
    [ "$(@get s.sd_out.cnt0)" -gt "$c_prev" ] && break
    sleep 0.02
done
# 100 - 80 = 20
echo "live_fir=$(@get s.sd_out.mean)"

# Live switch to time-unit discrete delay (delaysec=0.04, weights cleared)
@set f.sd_u.weights_name=null f.sd_u.unit=0 f.sd_u.delaysec=0.04 f.sd_u.interp=0
@calc "S:sd_in = 77"
sleep 0.06
c_prev=$(@get s.sd_out.cnt0)
@calc "S:sd_in = 999"
for _ in $(seq 60); do
    [ "$(@get s.sd_out.cnt0)" -gt "$c_prev" ] && break
    sleep 0.02
done
# Frame 77 is ~60ms old (>= 40ms), Frame 999 is 0ms old -> output is 77
echo "time_dly=$(@get s.sd_out.mean)"

stop_unit "$sd_pid"
""",
        tmp_path,
    )
    assert float(value(res.stdout, "f_interp")) == pytest.approx(15.0, abs=1e-4), (
        res.stdout + res.stderr
    )
    assert value(res.stdout, "out_zsize") == "2", res.stdout + res.stderr
    assert int(value(res.stdout, "lag_stat")) >= 1, res.stdout + res.stderr
    assert float(value(res.stdout, "live_box")) == pytest.approx(70.0, abs=1e-4), (
        res.stdout + res.stderr
    )
    assert float(value(res.stdout, "live_fir")) == pytest.approx(20.0, abs=1e-4), (
        res.stdout + res.stderr
    )
    assert float(value(res.stdout, "time_dly")) == pytest.approx(77.0, abs=1e-4), (
        res.stdout + res.stderr
    )
