#include "perfbench.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <sys/stat.h>
#include <time.h>

/**
 * @brief Run a shell command, discarding output.
 *
 * @param fmt  printf-style format string
 * @return     exit code of command, -1 on error
 */
int run_cmd(const char *fmt, ...)
{
    char    buf[MAX_CMD];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);
    return system(buf);
}

/**
 * @brief Resolve the process SHM directory.
 *
 * Checks MILK_SHM_DIR env var, then /milk/shm, then /tmp.
 */
void resolve_procdir(bench_cfg_t *cfg)
{
    const char *env = getenv("MILK_SHM_DIR");
    if (env && strlen(env) > 0)
    {
        strncpy(cfg->procdir, env, sizeof(cfg->procdir) - 1);
        return;
    }

    struct stat st;
    if (stat("/milk/shm", &st) == 0 && S_ISDIR(st.st_mode))
    {
        strncpy(cfg->procdir, "/milk/shm", sizeof(cfg->procdir) - 1);
        return;
    }

    strncpy(cfg->procdir, "/tmp", sizeof(cfg->procdir) - 1);
}

/**
 * @brief Get SHM directory (same as MILK_SHM_DIR).
 */
void resolve_shmdir(char *shmdir, size_t sz)
{
    const char *env = getenv("MILK_SHM_DIR");
    if (env && strlen(env) > 0)
    {
        strncpy(shmdir, env, sz - 1);
        return;
    }

    struct stat st;
    if (stat("/milk/shm", &st) == 0 && S_ISDIR(st.st_mode))
    {
        strncpy(shmdir, "/milk/shm", sz - 1);
        return;
    }

    strncpy(shmdir, "/tmp", sz - 1);
}

/**
 * @brief Get short git commit hash.
 */
void resolve_git_commit(bench_cfg_t *cfg)
{
    FILE *fp = popen("git rev-parse --short HEAD 2>/dev/null", "r");
    if (!fp)
    {
        strncpy(cfg->git_commit, "unknown", sizeof(cfg->git_commit) - 1);
        return;
    }
    if (!fgets(cfg->git_commit, sizeof(cfg->git_commit) - 1, fp))
    {
        strncpy(cfg->git_commit, "unknown", sizeof(cfg->git_commit) - 1);
    }
    else
    {
        /* strip trailing newline */
        cfg->git_commit[strcspn(cfg->git_commit, "\n")] = '\0';
    }
    pclose(fp);
}

/**
 * @brief Get executable size in bytes.
 */
int64_t exe_size(const char *exe)
{
    /* find full path */
    char cmd[MAX_CMD];
    snprintf(cmd, sizeof(cmd), "command -v %s 2>/dev/null", exe);
    FILE *fp = popen(cmd, "r");
    if (!fp)
    {
        return 0;
    }
    char path[MAX_PATH] = { 0 };
    if (!fgets(path, sizeof(path) - 1, fp))
    {
        pclose(fp);
        return 0;
    }
    pclose(fp);
    path[strcspn(path, "\n")] = '\0';

    if (strlen(path) == 0)
    {
        return 0;
    }

    struct stat st;
    if (stat(path, &st) != 0)
    {
        return 0;
    }
    return (int64_t) st.st_size;
}

/**
 * @brief Extract MILK_BUILD sentinel string from binary.
 */
void read_build_tags(const char *exe, char *out, size_t outsz)
{
    out[0] = '\0';

    /* Resolve full path */
    char cmd[MAX_CMD];
    snprintf(cmd, sizeof(cmd), "command -v '%s' 2>/dev/null", exe);
    FILE *fp = popen(cmd, "r");
    if (!fp)
    {
        return;
    }
    char path[MAX_PATH] = { 0 };
    if (!fgets(path, sizeof(path) - 1, fp))
    {
        pclose(fp);
        return;
    }
    pclose(fp);
    path[strcspn(path, "\n")] = '\0';
    if (strlen(path) == 0)
    {
        return;
    }

    /* Extract the sentinel via strings(1) */
    snprintf(cmd, sizeof(cmd),
             "strings '%s' 2>/dev/null"
             " | grep 'MILK_BUILD:'",
             path);
    fp = popen(cmd, "r");
    if (!fp)
    {
        return;
    }
    char raw[512] = { 0 };
    if (!fgets(raw, sizeof(raw) - 1, fp))
    {
        pclose(fp);
        out[0] = '\0';
        return;
    }
    pclose(fp);
    raw[strcspn(raw, "\n")] = '\0';

    /* Locate payload after "MILK_BUILD:" prefix */
    char *payload = strstr(raw, "MILK_BUILD:");
    if (!payload)
    {
        return;
    }
    payload += strlen("MILK_BUILD:");

    /* Build a compact human-readable summary */
    char   summary[256] = { 0 };
    size_t slen         = 0;

    if (strstr(payload, "OPT=3"))
    {
        slen += (size_t) snprintf(summary + slen, sizeof(summary) - slen, "O3 ");
    }
    if (strstr(payload, "PGO=USE"))
    {
        slen += (size_t) snprintf(summary + slen, sizeof(summary) - slen, "PGO ");
    }
    else if (strstr(payload, "PGO=GENERATE"))
    {
        slen += (size_t) snprintf(summary + slen, sizeof(summary) - slen, "PGO-instr ");
    }
    if (strstr(payload, "LTO=STATIC"))
    {
        slen += (size_t) snprintf(summary + slen, sizeof(summary) - slen, "LTO-static ");
    }
    else if (strstr(payload, "LTO=1"))
    {
        slen += (size_t) snprintf(summary + slen, sizeof(summary) - slen, "LTO ");
    }

    /* Extract architecture field */
    {
        char *ap = strstr(payload, "ARCH=");
        if (ap)
        {
            ap += 5;
            char   arch[32] = { 0 };
            size_t ai       = 0;
            while (*ap && *ap != ',' && ai < 31)
            {
                arch[ai++] = *ap++;
            }
            slen += (size_t) snprintf(summary + slen, sizeof(summary) - slen, "[%s]", arch);
        }
    }

    if (slen == 0)
    {
        snprintf(summary, sizeof(summary), "default (no PGO/LTO)");
    }

    /* Trim trailing space */
    while (slen > 0 && summary[slen - 1] == ' ')
    {
        summary[--slen] = '\0';
    }

    snprintf(out, outsz, "%s", summary);
}

/* ================================================================
 * Unique FPS name
 * ============================================================= */

void make_fpsname(char *out, size_t sz)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    unsigned long seed = (unsigned long) ts.tv_nsec ^ (unsigned long) ts.tv_sec;
    snprintf(out, sz, "pb%07lu", seed % 10000000UL);
}

/* ================================================================
 * FPS lifecycle helpers
 * ============================================================= */

/**
 * @brief Run milk-fps-set for one parameter.
 */
void fps_set(const char *fpsname, const char *param_tag, const char *value)
{
    run_cmd("milk-fps-set %s.%s %s"
            " >/dev/null 2>&1",
            fpsname, param_tag, value);
}

/**
 * @brief Initialize FPS and configure procinfo.
 */
void fps_setup(bench_cfg_t *cfg)
{
    /* fpsinit with -procinfo flag */
    run_cmd("%s %s:fpsinit -procinfo"
            " >/dev/null 2>&1",
            cfg->fpsexec, cfg->fpsname);

    /* confstep */
    run_cmd("%s %s:confstep"
            " >/dev/null 2>&1",
            cfg->fpsexec, cfg->fpsname);

    /* enable processinfo */
    fps_set(cfg->fpsname, "procinfo.enabled", "ON");
    fps_set(cfg->fpsname, "procinfo.MeasureTiming", "ON");
    /* triggermode 0 = IMMEDIATE */
    fps_set(cfg->fpsname, "procinfo.triggermode", "0");

    /* apply extra positional args if any */
    if (cfg->fpsargs[0] != '\0')
    {
        run_cmd("%s %s:set %s"
                " >/dev/null 2>&1",
                cfg->fpsexec, cfg->fpsname, cfg->fpsargs);
    }
}

/**
 * @brief Auto-create missing SHM streams.
 */
void fps_create_streams(bench_cfg_t *cfg)
{
    char shmdir[MAX_PATH];
    resolve_shmdir(shmdir, sizeof(shmdir));

    char cmd[MAX_CMD];
    snprintf(cmd, sizeof(cmd),
             "%s %s:fps 2>/dev/null"
             " | sed 's/\\x1b\\[[0-9;]*m//g'"
             " | awk '$3==\"STREAMNAME\" && NF>=8"
             " {print $4}'",
             cfg->fpsexec, cfg->fpsname);

    FILE *fp = popen(cmd, "r");
    if (!fp)
    {
        return;
    }

    char sname[256];
    while (fgets(sname, sizeof(sname), fp))
    {
        sname[strcspn(sname, "\n")] = '\0';
        if (strlen(sname) == 0)
        {
            continue;
        }

        char impath[MAX_PATH + 256 + 32];
        snprintf(impath, sizeof(impath), "%s/%s.im.shm", shmdir, sname);

        struct stat st;
        if (stat(impath, &st) != 0)
        {
            printf("  Creating stream: %s (32x32)\n", sname);
            run_cmd("milk-perfbench-mkstream"
                    " %s 32 32 >/dev/null 2>&1",
                    sname);
        }
        else
        {
            printf("  Stream exists: %s\n", sname);
        }
    }
    pclose(fp);
}

/**
 * @brief Cleanup: remove FPS SHM files.
 */
void fps_cleanup(bench_cfg_t *cfg)
{
    char shmdir[MAX_PATH];
    resolve_shmdir(shmdir, sizeof(shmdir));

    run_cmd("rm -f '%s/fps.%s'*.shm"
            " '%s/%s.fps.datadir'"
            " 2>/dev/null",
            shmdir, cfg->fpsname, shmdir, cfg->fpsname);
}
