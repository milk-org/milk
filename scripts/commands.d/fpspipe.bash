# SPDX-FileCopyrightText: 2026 Olivier Guyon et al
#
# SPDX-License-Identifier: LGPL-3.0-or-later
#
# fpspipe.bash - High-speed FPS compute-unit pipeline lifecycle commands
# Loaded automatically from $MILK_INSTALLDIR/share/milk/commands.d/fpspipe.bash

declare -ga FPSPIPE_UNITS=()
declare -ga FPSPIPE_STREAMS=()
declare -ga FPSPIPE_CREATED_UNITS=()
declare -ga FPSPIPE_CREATED_STREAMS=()
declare -gi FPSPIPE_COMMITTED=1

__fpspipe_track_unit() {
    local target="$1" existing
    for existing in "${FPSPIPE_UNITS[@]}"; do
        [[ "${existing}" == "${target}" ]] && return 0
    done
    FPSPIPE_UNITS+=("${target}")
}

__fpspipe_track_stream() {
    local target="$1" existing
    for existing in "${FPSPIPE_STREAMS[@]}"; do
        [[ "${existing}" == "${target}" ]] && return 0
    done
    FPSPIPE_STREAMS+=("${target}")
}

__fpspipe_rollback_uncommitted() {
    (( ${FPSPIPE_COMMITTED:-1} )) && return 0
    FPSPIPE_COMMITTED=1
    if (( ${#FPSPIPE_CREATED_UNITS[@]} )); then
        milk-fps-runstop -q "${FPSPIPE_CREATED_UNITS[@]}" >/dev/null 2>&1 || true
        milk-fps-confstop -q "${FPSPIPE_CREATED_UNITS[@]}" >/dev/null 2>&1 || true
        milk-fps-rm -f -q --ignore-missing "${FPSPIPE_CREATED_UNITS[@]}" >/dev/null 2>&1 || true
    fi
    if (( ${#FPSPIPE_CREATED_STREAMS[@]} )); then
        milk-stream-rm -q --ignore-missing "${FPSPIPE_CREATED_STREAMS[@]}" >/dev/null 2>&1 || true
    fi
    return 0
}

__fpspipe_resolve_numa() {
    local req="${1:-auto}" node="-1" p v c n sib first_sib cpulist=""
    __FPSPIPE_NUMA_NODE="-1"
    __FPSPIPE_NUMA_CPULIST=""
    case "${req}" in
        -1|off|OFF|none|NONE|"") return 0 ;;
        auto|AUTO|gpu|GPU|-2)
            for p in /sys/bus/pci/devices/*; do
                [[ -r "${p}/vendor" && -r "${p}/class" && -r "${p}/numa_node" ]] || continue
                read -r v < "${p}/vendor" || continue
                [[ "${v}" == "0x10de" || "${v}" == "0x1002" ]] || continue
                read -r c < "${p}/class" || continue
                case "${c}" in 0x0300*|0x0302*|0x0380*) ;; *) continue ;; esac
                read -r n < "${p}/numa_node" || continue
                if [[ "${n}" =~ ^[0-9]+$ ]]; then
                    node="${n}"
                    [[ "${v}" == "0x10de" ]] && break
                fi
            done
            if (( node < 0 )) && [[ -d /sys/devices/system/node/node1 ]]; then
                node=0
            fi
            ;;
        *[!0-9]*) return 0 ;;
        *) node="${req}" ;;
    esac
    (( node >= 0 )) || return 0
    [[ -r "/sys/devices/system/node/node${node}/cpulist" ]] || return 0
    read -r cpulist < "/sys/devices/system/node/node${node}/cpulist" || return 0
    if [[ "${cpulist}" == *,* ]]; then
        local first_part="${cpulist%%,*}"
        local first_cpu="${first_part%%-*}"
        local sib_file="/sys/devices/system/cpu/cpu${first_cpu}/topology/thread_siblings_list"
        if [[ -r "${sib_file}" ]]; then
            read -r sib < "${sib_file}"
            first_sib="${sib#*,}"
            if [[ "${first_sib}" != "${sib}" && "${first_part}" == *-* ]]; then
                local hi="${first_part#*-}"
                if (( ${first_sib%%,*} > hi )); then
                    cpulist="${first_part}"
                fi
            fi
        fi
    fi
    __FPSPIPE_NUMA_NODE="${node}"
    __FPSPIPE_NUMA_CPULIST="${cpulist}"
}

# Initialize FPS pipeline workspace, environment, and uncommitted-exit rollback
fpspipe.workdir() {
    @args 'name:str:pipeline name' \
          'rootdir:str:pipeline root directory' \
          'prefix:str=:stream prefix (e.g. aol0_)' \
          'app:str=milk:FPS keyword application tag' \
          'restart:bool=0:restart active units during deployment' \
          'numanode:str=:NUMA node policy (empty=MILK_NUMANODE or auto, -1/off)' \
          'ompwait:str=:OpenMP wait policy (empty=MILK_OMP_WAIT_POLICY or PASSIVE)' -- "$@"

    local eff_omp="${ompwait:-${MILK_OMP_WAIT_POLICY:-PASSIVE}}"
    case "${eff_omp}" in
        PASSIVE|passive|1)
            export MILK_OMP_WAIT_POLICY="PASSIVE" OMP_WAIT_POLICY="PASSIVE" GOMP_SPINCOUNT="0"
            ;;
        ACTIVE|active|2)
            export MILK_OMP_WAIT_POLICY="ACTIVE" OMP_WAIT_POLICY="ACTIVE"
            unset GOMP_SPINCOUNT
            ;;
        off|OFF|none|NONE)
            unset MILK_OMP_WAIT_POLICY
            ;;
    esac

    local eff_numa="${numanode:-${MILK_NUMANODE:-auto}}"
    __fpspipe_resolve_numa "${eff_numa}"
    if [[ -n "${__FPSPIPE_NUMA_CPULIST}" ]] && (( __FPSPIPE_NUMA_NODE >= 0 )); then
        export MILK_NUMANODE="${__FPSPIPE_NUMA_NODE}"
        if command -v taskset >/dev/null 2>&1; then
            taskset -a -pc "${__FPSPIPE_NUMA_CPULIST}" "$$" >/dev/null 2>&1 || true
        fi
    else
        export MILK_NUMANODE="-1"
    fi

    mkdir -p "${rootdir}" || return 1
    rootdir="$(cd "${rootdir}" && pwd -P)" || return 1

    local rundir="${rootdir}/${name}-rundir"
    local logdir="${rootdir%/*}/logdir-${name}"
    mkdir -p "${rootdir}/conf" "${rundir}" "${logdir}" || return 1
    ln -sfn "${logdir}" "${rundir}/logdir" || return 1

    export FPSPIPE_NAME="${name}"
    export FPSPIPE_ROOTDIR="${rootdir}"
    export FPSPIPE_RUNDIR="${rundir}"
    export FPSPIPE_PREFIX="${prefix}"
    export FPSPIPE_APP="${app}"
    export FPSPIPE_RESTART="${restart}"
    export FPS_KEYWORDARRAY=":${app}:${name}:${prefix%_}:"

    FPSPIPE_UNITS=()
    FPSPIPE_STREAMS=()
    FPSPIPE_CREATED_UNITS=()
    FPSPIPE_CREATED_STREAMS=()
    FPSPIPE_COMMITTED=0

    cd "${FPSPIPE_RUNDIR}" || return 1
    @defer --exit __fpspipe_rollback_uncommitted
}

# Load FITS file (.fits, .fits.fz, .fits.gz) into shared-memory stream
fpspipe.loadfits() {
    @args 'sfx:str:stream suffix or full stream name' \
          'src:str:source FITS path or stem' \
          'force:bool=0:reload even when stream already exists' -- "$@"

    local sname="${sfx}"
    if [[ -n "${FPSPIPE_PREFIX:-}" && "${sfx}" != "${FPSPIPE_PREFIX}"* ]]; then
        sname="${FPSPIPE_PREFIX}${sfx}"
    fi
    local sfx_short="${sname#"${FPSPIPE_PREFIX:-}"}"

    local cand resolved="" base
    for base in "${src}" "${FPSPIPE_ROOTDIR:+${FPSPIPE_ROOTDIR}/${src}}"; do
        [[ -n "${base}" ]] || continue
        for cand in "${base}" "${base}.fits" "${base}.fits.fz" "${base}.fits.gz"; do
            if [[ -f "${cand}" ]]; then
                resolved="${cand}"
                break 2
            fi
        done
    done
    if [[ -z "${resolved}" ]]; then
        echo "fpspipe.loadfits: FITS file not found for '${src}'" >&2
        return 1
    fi

    __fpspipe_track_stream "${sname}"

    local abs_resolved ext=".fits"
    abs_resolved="$(realpath "${resolved}")" || return 1
    case "${abs_resolved}" in
        *.fits.fz) ext=".fits.fz" ;;
        *.fits.gz) ext=".fits.gz" ;;
    esac

    local existed=0
    if @exists "s.${sname}"; then
        existed=1
        if (( ! force )); then
            if [[ -d "${FPSPIPE_ROOTDIR:-}/conf" ]]; then
                ln -sfn "${abs_resolved}" "${FPSPIPE_ROOTDIR}/conf/shmim.${sfx_short}${ext}"
                printf '%s\n' "${abs_resolved}" > "${FPSPIPE_ROOTDIR}/conf/shmim.${sfx_short}.name.txt"
            fi
            return 0
        fi
    fi

    # Stage a short path (< 63 chars for FPTYPE_FILENAME) via conf/ or /tmp symlink
    local load_arg="${resolved}" tmp_link=""
    if [[ -d "${FPSPIPE_ROOTDIR:-}/conf" ]]; then
        ln -sfn "${abs_resolved}" "${FPSPIPE_ROOTDIR}/conf/shmim.${sfx_short}${ext}" || return 1
        if [[ "${PWD}" == "${FPSPIPE_RUNDIR:-}" ]]; then
            load_arg="../conf/shmim.${sfx_short}${ext}"
        fi
    fi
    if (( ${#load_arg} > 63 )); then
        tmp_link="/tmp/.fp_$$${ext}"
        ln -sfn "${abs_resolved}" "${tmp_link}" || return 1
        load_arg="${tmp_link}"
    fi

    local rc=0
    if (( existed )); then
        # Update existing SHM stream in-place so live readers keep their mapping and receive semaphores
        if iofits.loadfits "${load_arg}" "@L:__fpspipe_tmp" 2 >/dev/null; then
            @calc "@S:${sname} = @L:__fpspipe_tmp" || rc=1
            mem.delete "@L:__fpspipe_tmp" >/dev/null 2>&1 || true
        else
            rc=1
        fi
    else
        if iofits.loadfits "${load_arg}" "@S:${sname}" 2 >/dev/null; then
            FPSPIPE_CREATED_STREAMS+=("${sname}")
        else
            rc=1
        fi
    fi
    if [[ -n "${tmp_link}" ]]; then
        rm -f "${tmp_link}"
    fi
    (( rc == 0 )) || return "${rc}"

    if [[ -d "${FPSPIPE_ROOTDIR:-}/conf" ]]; then
        printf '%s\n' "${abs_resolved}" > "${FPSPIPE_ROOTDIR}/conf/shmim.${sfx_short}.name.txt"
        local nx=0 ny=0 nz=0 nax=0
        if @get nax="s.${sname}.naxis" nx="s.${sname}.xsize" ny="s.${sname}.ysize" nz="s.${sname}.zsize" 2>/dev/null; then
            if (( nax >= 3 )); then
                printf '%s %s %s \n' "${nx}" "${ny}" "${nz}" > "${FPSPIPE_ROOTDIR}/conf/shmim.${sfx_short}.imsize.txt"
            else
                printf '%s %s \n' "${nx}" "${ny}" > "${FPSPIPE_ROOTDIR}/conf/shmim.${sfx_short}.imsize.txt"
            fi
        fi
    fi
    return 0
}

# Save a shared-memory stream to a FITS file in-process via iofits.saveFITS
fpspipe.savefits() {
    @args 'sfx:str:stream suffix or full stream name' \
          'dst:str:destination FITS path (with or without .fits)' -- "$@"

    local sname="${sfx}"
    if ! @exists "s.${sname}"; then
        if [[ -n "${FPSPIPE_PREFIX:-}" && "${sfx}" != "${FPSPIPE_PREFIX}"* ]] &&
           @exists "s.${FPSPIPE_PREFIX}${sfx}"; then
            sname="${FPSPIPE_PREFIX}${sfx}"
        else
            echo "fpspipe.savefits: stream '${sfx}' not found in SHM" >&2
            return 1
        fi
    fi

    local abs_dst out_dir out_base
    abs_dst="$(realpath -m "${dst}")" || return 1
    if [[ "${abs_dst}" != *.fits && "${abs_dst}" != *.fits.gz && "${abs_dst}" != *.fits.fz ]]; then
        abs_dst="${abs_dst}.fits"
    fi
    out_dir="$(dirname "${abs_dst}")"
    out_base="$(basename "${abs_dst}")"
    mkdir -p "${out_dir}" || return 1

    # iofits.saveFITS truncates .out_fname at FUNCTION_PARAMETER_STRMAXLEN (63 chars),
    # so invoke from out_dir (restoring PWD) and pass "!${out_base}".
    local old_pwd="${PWD}" rc=0
    cd "${out_dir}" || return 1
    iofits.saveFITS "@S:${sname}" "!${out_base}" >/dev/null || rc=$?
    cd "${old_pwd}" || true
    return "${rc}"
}

# Initialize FPS compute units concurrently from a manifest
fpspipe.init() {
    @args 'manifest:str=-:unit manifest file (- for stdin)' \
          'restart:bool=0:stop and re-initialize active units' -- "$@"

    local content line unit _rest
    if [[ "${manifest}" == "-" ]]; then
        content="$(cat)"
    else
        if [[ ! -f "${manifest}" ]]; then
            echo "fpspipe.init: cannot open manifest '${manifest}'" >&2
            return 1
        fi
        content="$(<"${manifest}")"
    fi

    while IFS= read -r line || [[ -n "${line}" ]]; do
        line="${line#"${line%%[![:space:]]*}"}"
        [[ -z "${line}" || "${line}" == \#* ]] && continue
        read -r unit _rest <<< "${line}"
        if [[ -n "${unit}" ]]; then
            __fpspipe_track_unit "${unit}"
        fi
    done <<< "${content}"

    local -a flags=(-q --print-created)
    if (( restart || ${FPSPIPE_RESTART:-0} )); then
        flags+=(--restart)
    fi

    local created_out cu
    created_out="$(milk-fps-init "${flags[@]}" -b - <<< "${content}")" || return $?
    while IFS= read -r cu; do
        if [[ -n "${cu}" ]]; then
            FPSPIPE_CREATED_UNITS+=("${cu}")
        fi
    done <<< "${created_out}"
    return 0
}

# Apply batch FPS parameter updates in a single C transaction
fpspipe.setparams() {
    @args 'batch:str=-:parameter batch file (- for stdin)' \
          'writerun:bool=0:update FPFLAG_WRITERUN parameters on running units' \
          'force:bool=0:override FPFLAG_WRITERUN guard on running units' -- "$@"

    local -a flags=(-q)
    if (( force || ${FPSPIPE_RESTART:-0} )); then
        flags+=(-f)
    elif (( writerun )); then
        flags+=(--respect-writerun)
    else
        flags+=(--skip-running)
    fi
    milk-fps-set "${flags[@]}" -b "${batch}"
}

# Mark pipeline deployment committed so exit rollback is disarmed
fpspipe.commit() {
    @args -- "$@"
    FPSPIPE_COMMITTED=1
}

# Print run/conf status table for tracked pipeline FPS units
fpspipe.status() {
    @args -- "$@"
    if (( ${#FPSPIPE_UNITS[@]} == 0 )); then
        milk-fps-list
        return $?
    fi

    printf '%-28s %-10s %-10s %8s %8s\n' "UNIT" "RUN" "CONF" "RUNPID" "CONFPID"
    local unit rstate cstate rpid cpid
    for unit in "${FPSPIPE_UNITS[@]}"; do
        if @exists "f.${unit}"; then
            @get rstate="f.${unit}.status" cstate="f.${unit}.conf" \
                 rpid="f.${unit}.runpid" cpid="f.${unit}.confpid"
            printf '%-28s %-10s %-10s %8s %8s\n' "${unit}" "${rstate}" "${cstate}" "${rpid}" "${cpid}"
        else
            printf '%-28s %-10s %-10s %8s %8s\n' "${unit}" "missing" "-" "-" "-"
        fi
    done
}

# Stop and remove tracked pipeline FPS units and shared-memory streams
fpspipe.teardown() {
    @args 'clean_rundir:bool=1:remove fps.*.{confdir,datadir} in rundir' -- "$@"

    FPSPIPE_COMMITTED=1
    if (( ${#FPSPIPE_UNITS[@]} )); then
        milk-fps-runstop -q "${FPSPIPE_UNITS[@]}" >/dev/null 2>&1 || true
        milk-fps-confstop -q "${FPSPIPE_UNITS[@]}" >/dev/null 2>&1 || true
        milk-fps-rm -f -q --ignore-missing "${FPSPIPE_UNITS[@]}" || return 1
    fi
    if (( ${#FPSPIPE_STREAMS[@]} )); then
        milk-stream-rm -q --ignore-missing "${FPSPIPE_STREAMS[@]}" || return 1
    fi
    if (( clean_rundir )) && [[ -n "${FPSPIPE_RUNDIR:-}" && -d "${FPSPIPE_RUNDIR}" ]]; then
        rm -rf "${FPSPIPE_RUNDIR}"/fps.*.confdir "${FPSPIPE_RUNDIR}"/fps.*.datadir
    fi
    FPSPIPE_UNITS=()
    FPSPIPE_STREAMS=()
    FPSPIPE_CREATED_UNITS=()
    FPSPIPE_CREATED_STREAMS=()
}
