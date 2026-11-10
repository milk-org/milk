#!/usr/bin/env bash
# Adaptive optics loop made of milk compute units only (no cacao), on a simulated WFS whose pixels
# are the wavefront phase. See README.md.
#
#   ./aoloop.sh [--default-shm] [--fused] start
#                         make the streams, compute the control matrix, start the units (loop open),
#                         print MILK_SHM_DIR
#   ./aoloop.sh close     close the loop
#   ./aoloop.sh open      open the loop and flatten the correction
#   ./aoloop.sh measure   residual open and closed, camera-to-DM latency, time per unit
#   ./aoloop.sh stop      stop the units, remove the streams and the work directory
#
# The streams go to a MILK_SHM_DIR of their own in the work directory, or with --default-shm to
# the usual one ($MILK_SHM_DIR, else /milk/shm), where milk-CTRL sees them. With --fused, one unit
# runs the whole loop iteration instead of one unit per step. The other commands use the directory
# and the layout that start chose.
set -u

DMNX=12      # DM: DMNX x DMNX actuators
WFSNX=24     # WFS: WFSNX x WFSNX pixels, 2 per actuator
NM=100       # controlled modes, of DMNX^2 (the actuators outside the pupil are barely seen)
RATE=2000    # camera frame rate, Hz
NTURB=35     # turbulence modes: cosine modes up to 5 cycles across the pupil, piston left out
NACT=$((DMNX * DMNX))
NWFS=$((WFSNX * WFSNX))
SIM="wfssim"                                              # test hardware
SPLIT="dmtrig dmcomb acquwfs wfs2modes mfilt modes2dm"    # the AO loop, one unit per step
FUSED="aoloop"                                            # the AO loop, one unit (--fused)
STEPS="acquwfs wfs2modes mfilt modes2dm dmcomb"           # the programs that the fused unit runs
MFILT="loopON:int=0 gain:float=0.5 mult:float=0.99 limit:float=10"
DMCOMB="maxstroke:float=1"
STREAMS="dmch0 dmch1 dmch2 dmch3 dm dmtrig wfspup IF Tmodes tamp tcoef wfsim wfsdark wfsref wfss
         CM M2A mgain mval mint tcam latency"

here=$(cd "$(dirname "$0")" && pwd)
work=${TMPDIR:-/tmp}/milk-aoloop-$(id -u)
shm=$work/shm fused=0
while :; do
    case ${1:-} in
    --default-shm)
        shm=${MILK_SHM_DIR:-/milk/shm}
        [[ -d $shm ]] || { echo "$shm does not exist"; exit 1; } ;;
    --fused) fused=1 ;;
    *) break ;;
    esac
    shift
done
if [[ ${1:-} == start && -f $work/shmdir ]]; then
    "$here/${0##*/}" stop >/dev/null          # end the previous run, in whichever directory
elif [[ -f $work/shmdir ]]; then
    shm=$(<"$work/shmdir")
    [[ -f $work/fused ]] && fused=1
fi
if ((fused)); then LOOP=$FUSED FILT=$FUSED; else LOOP=$SPLIT FILT=mfilt; fi
export MILK_SHM_DIR=$shm MILK_QUIET=1
# a private run keeps its units' status files (p.<unit>) there too: units are found by name
[[ $shm == "$work/shm" ]] && export MILK_PROC_DIR=$shm
mkdir -p "$MILK_SHM_DIR" "$work" && cd "$work" || exit 1
source milk.bash || exit 1

# unit NAME TRIGGER [PARAMS [PERIOD [PROGRAMS]]]: start the calc unit NAME, triggered by the stream
# TRIGGER, or with none every PERIOD seconds. It runs the programs PROGRAMS of units/ one after the
# other (default: NAME). FPS strings hold 63 characters, so they are copied into a short path
# relative to the work directory.
unit() {
    local name=$1 trig=$2 params=${3:-} loopd=${4:+-loopd $4} p
    milk-fps-rm -f -x "$name" >/dev/null 2>&1
    for p in ${5:-$name}; do cat "$here/units/$p.calc" || exit 1; done >"units/$name.calc" &&
        milk-fpsexec-milk-calc -n "$name" ${trig:+-t $trig} ${params:+-p "$params"} -f "units/$name.calc" \
            $loopd fpsinit >/dev/null &&
        milk-fpsexec-milk-calc -n "$name" $loopd runstart >/dev/null || { echo "$name: failed to start"; exit 1; }
}

start() {
    rmunits
    mkdir -p units
    echo "$MILK_SHM_DIR" >shmdir
    ((fused)) && touch fused
    echo "MILK_SHM_DIR=$MILK_SHM_DIR"
    [[ -n ${MILK_PROC_DIR:-} ]] && echo "MILK_PROC_DIR=$MILK_PROC_DIR"
    local c=$(((WFSNX - 1) / 2)).5 r=$((WFSNX / 2)) p=$((WFSNX / DMNX)) m=$((NM - 1))
    local xy="x = 0:$((WFSNX - 1)), y = 0:$((WFSNX - 1))"
    local inpup="sq(x - $c) + sq(y - $c) < $r * $r"
    @calc "def sq(v) = v * v" || exit 1

    # DM channels (0: flat, 1: AO loop, 2 and 3: free) and their sum
    @calc "S:dmch0 = zeros($DMNX, $DMNX); S:dmch1 = dmch0; S:dmch2 = dmch0; S:dmch3 = dmch0; S:dm = dmch0" || exit 1

    # model of the simulated WFS: pupil, Gaussian influence functions (one per actuator, p pixels
    # apart), turbulence modes and their amplitudes (steeper with spatial frequency)
    @calc "S:wfspup = [$inpup for $xy]" || exit 1
    @calc "S:IF = [($inpup) * exp(-(sq(x - $p*(z % $DMNX) - $p/2 + 0.5) + sq(y - $p*floor(z / $DMNX) - $p/2 + 0.5)) / 2) \
                   for $xy, z = 0:$((NACT - 1))]" || exit 1
    @calc "S:Tmodes = [($inpup) * cos(pi*floor((z+1)/6)*(x+0.5)/$WFSNX) * cos(pi*((z+1)%6)*(y+0.5)/$WFSNX) \
                       for $xy, z = 0:$((NTURB - 1))]" || exit 1
    @calc "S:tamp = [0.3 * pow(sq(floor((z+1)/6)) + sq((z+1)%6), -11/12) for z = 0:$((NTURB - 1))]; S:tcoef = tamp * 0" || exit 1
    @calc "S:wfsim = zeros($WFSNX, $WFSNX)" || exit 1
    unit wfssim "" "ar:float=0.995 tgain:float=1 noise:float=0.01 nact:int=$NACT" "$(echo "1 / $RATE" | bc -l)"

    # calibration, computed from the model: the response matrix of a phase WFS is IF itself.
    # Modes are its right singular vectors: M2A maps mode coefficients to DM shapes, CM maps WFS
    # frames to mode coefficients (U / s, truncated to NM modes)
    @calc "S:wfsdark = zeros($WFSNX, $WFSNX); S:wfsref = reshape(IF @ reshape(dmch0, $NACT), $WFSNX, $WFSNX)" || exit 1
    @calc "U, s, Vt = svd(reshape(IF, $NWFS, $NACT))" || exit 1
    @calc "S:M2A = reshape(transpose(Vt[0:$m, *]), $DMNX, $DMNX, $NM); \
           S:CM = reshape(U[*, 0:$m], $WFSNX, $WFSNX, $NM) / reshape(s[0:$m], 1, 1, $NM)" || exit 1
    @calc "S:wfss = wfsdark; S:mval = zeros($NM); S:mint = mval; S:mgain = mval + 1" || exit 1

    if ((fused)); then
        # AO loop in one unit, triggered by the camera: the same programs, one after the other.
        # Writes to dmch0, dmch2 and dmch3 reach dm at the next camera frame.
        unit $FUSED wfsim "$MFILT $DMCOMB" "" "$STEPS"
    else
        # AO loop, each unit triggered by the previous one's output:
        #   wfsim -> acquwfs -> wfss -> wfs2modes -> mval -> mfilt -> mint -> modes2dm -> dmch1
        #   dmch0..3 -> dmtrig (trigmerge) -> dmcomb -> dm -> (WFS camera)
        milk-fps-rm -f -x dmtrig >/dev/null 2>&1
        milk-fpsexec-mem-trigmerge -n dmtrig fpsinit dmch0 dmch1 dmtrig .in2=dmch2 .in3=dmch3 >/dev/null &&
            milk-fpsexec-mem-trigmerge -n dmtrig -loops runstart >/dev/null 2>&1 || { echo "dmtrig: failed to start"; exit 1; }
        unit dmcomb dmtrig "$DMCOMB"
        unit acquwfs wfsim
        unit wfs2modes wfss
        unit mfilt mval "$MFILT"
        unit modes2dm mint
        @calc "dmch0 = dmch0"             # a first DM write: dmcomb writes dm
    fi
    @waitfor s.dmch1 10 -t 5 || { echo "the loop does not run"; exit 1; }
    echo "running: $SIM (test hardware) and $LOOP; loop open"
}

# rms N: RMS of the residual phase over the next N frames of wfss, inside the pupil, into $rms
rms() {
    @grab s.wfss "$1" @L:_res >/dev/null || exit 1
    @calc -v rms "sqrt(itot(_res * _res) / ($1 * itot(wfspup)))" || exit 1
}

measure() {
    local u n=$((RATE * 2)) ro mis
    declare -A mis0
    for u in $SIM $LOOP; do
        @get -v mis p.$u.missedframes
        mis0[$u]=$mis
    done
    open
    @waitfor s.wfss $((RATE / 2)) -t 5 || exit 1
    rms $n
    ro=$rms
    close
    @waitfor s.wfss $((RATE / 2)) -t 5 || exit 1
    rms $n
    echo "residual RMS over $n frames (phase units): loop open $ro, closed $rms"

    @calc "S:tcam = f64(zeros(1)); S:latency = tcam" || exit 1
    unit tcam wfsim
    unit latency dm
    @grab s.latency "$n" @L:_lat >/dev/null || exit 1
    @calc -v lmed,l90 "imedian(_lat) * 1e6; perc(_lat, 0.9) * 1e6" || exit 1
    milk-fps-rm -f -x tcam >/dev/null 2>&1
    milk-fps-rm -f -x latency >/dev/null 2>&1
    printf 'camera frame to DM write: median %.0f us, 90%% below %.0f us\n' "$lmed" "$l90"

    printf '%-10s %9s %10s %10s %7s\n' unit "rate (Hz)" "busy (us)" "max (us)" missed
    for u in $SIM $LOOP; do
        @get hz=p.$u.loopfreq ex=p.$u.exectime exm=p.$u.exec_max mis=p.$u.missedframes
        printf '%-10s %9.0f %10.1f %10.1f %7s\n' "$u" "$hz" "$ex" "$exm" $((mis - ${mis0[$u]}))
    done
}

close() { @set f.$FILT.loopON=1; }
open() { @set f.$FILT.loopON=0 && @calc "mint = 0 * mint"; }

rmunits() {
    local u
    for u in tcam latency $SPLIT $FUSED $SIM; do milk-fps-rm -f -x "$u" >/dev/null 2>&1; done
}

stop() {
    local s
    rmunits
    for s in $STREAMS; do milk-stream-rm "$s" </dev/null >/dev/null 2>&1; done
    @session prune >/dev/null 2>&1        # the session files that exited shells left
    rm -rf "$work"
    echo stopped
}

case ${1:-} in
start | close | open | measure | stop) "$1" ;;
*) sed -n '2,16p' "$here/${0##*/}" | cut -c3- && exit 2 ;;
esac
