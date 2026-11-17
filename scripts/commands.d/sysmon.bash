# SPDX-FileCopyrightText: 2026 Olivier Guyon et al
#
# SPDX-License-Identifier: LGPL-3.0-or-later
#
# sysmon.bash - Headless CPU/GPU/NUMA/SHM profiling and compute report commands
# Loaded automatically from $MILK_INSTALLDIR/share/milk/commands.d/sysmon.bash

declare -g  SYSMON_PID=""
declare -g  SYSMON_OUTDIR=""
declare -g  SYSMON_STOP_FILE=""
declare -g  SYSMON_PHASES_FILE=""
declare -g  SYSMON_JSON=""
declare -g  SYSMON_CSV=""

# Start background milk-sysmon telemetry collector into <outdir>
sysmon.start() {
    @args 'outdir:str:output directory for telemetry files' \
          'rate:float=5.0:sampling rate [Hz]' -- "$@"

    mkdir -p "${outdir}" || return 1
    SYSMON_OUTDIR="$(cd "${outdir}" && pwd -P)" || return 1
    SYSMON_STOP_FILE="${SYSMON_OUTDIR}/.sysmon.stop"
    SYSMON_PHASES_FILE="${SYSMON_OUTDIR}/compute_phases.tsv"
    SYSMON_JSON="${SYSMON_OUTDIR}/compute_telemetry.json"
    SYSMON_CSV="${SYSMON_OUTDIR}/compute_timeseries.csv"

    rm -f "${SYSMON_STOP_FILE}"
    : > "${SYSMON_PHASES_FILE}"

    milk-sysmon \
        -r "${rate}" \
        -o "${SYSMON_JSON}" \
        -c "${SYSMON_CSV}" \
        -p "${SYSMON_PHASES_FILE}" \
        -s "${SYSMON_STOP_FILE}" \
        >/dev/null 2>&1 &
    SYSMON_PID=$!
    @defer --exit sysmon.stop
}

# Record a phase boundary marker in the active telemetry timeline
sysmon.mark() {
    @args 'phase:str:short phase identifier' \
          'label:str=:human-readable phase description' -- "$@"

    [[ -n "${SYSMON_PHASES_FILE:-}" ]] || return 0
    local ts desc="${label:-${phase}}"
    ts="$(date +%s.%N)"
    printf '%s\t%s\t%s\n' "${ts}" "${phase}" "${desc}" >> "${SYSMON_PHASES_FILE}"
}

# Stop background milk-sysmon and flush JSON + CSV outputs cleanly
sysmon.stop() {
    @args -- "$@"

    if [[ -z "${SYSMON_PID:-}" ]]; then
        return 0
    fi
    if [[ -n "${SYSMON_STOP_FILE:-}" ]]; then
        : > "${SYSMON_STOP_FILE}" 2>/dev/null || true
    fi
    if kill -0 "${SYSMON_PID}" 2>/dev/null; then
        local _i
        for _i in {1..40}; do
            kill -0 "${SYSMON_PID}" 2>/dev/null || break
            sleep 0.05
        done
        if kill -0 "${SYSMON_PID}" 2>/dev/null; then
            kill -TERM "${SYSMON_PID}" 2>/dev/null || true
        fi
        wait "${SYSMON_PID}" 2>/dev/null || true
    fi
    rm -f "${SYSMON_STOP_FILE:-}"
    SYSMON_PID=""
    return 0
}

# Generate Compute Performance Report (COMPUTE_REPORT.md + JSON + plots) via milk-perf-report
sysmon.report() {
    @args 'outdir:str=:output directory (default: active sysmon outdir)' \
          'ao_json:str=:optional AO optical metrics JSON path' \
          'bench_txt:str=:optional cacaosim-benchmark table path' \
          'calib_stg1:str=:optional stage-1 calib_timings.json path' \
          'calib_stg2:str=:optional stage-2 calib_timings.json path' \
          'compare_json:str=:optional modular vs fused comparison JSON' -- "$@"

    sysmon.stop || true

    local target_dir="${outdir:-${SYSMON_OUTDIR:-.}}"
    local telem="${target_dir}/compute_telemetry.json"
    local csv="${target_dir}/compute_timeseries.csv"
    if [[ ! -f "${telem}" ]]; then
        echo "sysmon.report: telemetry JSON not found: ${telem}" >&2
        return 1
    fi

    local -a cmd=(
        milk-perf-report
        --telemetry "${telem}"
        --csv "${csv}"
        --outdir "${target_dir}"
    )
    [[ -n "${ao_json}" && -f "${ao_json}" ]]           && cmd+=(--ao-metrics "${ao_json}")
    [[ -n "${bench_txt}" && -f "${bench_txt}" ]]       && cmd+=(--microbench "${bench_txt}")
    [[ -n "${calib_stg1}" && -f "${calib_stg1}" ]]     && cmd+=(--calib-stg1 "${calib_stg1}")
    [[ -n "${calib_stg2}" && -f "${calib_stg2}" ]]     && cmd+=(--calib-stg2 "${calib_stg2}")
    [[ -n "${compare_json}" && -f "${compare_json}" ]] && cmd+=(--compare-modes "${compare_json}")
    "${cmd[@]}"
}
