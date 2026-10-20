// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/* CTest milkctrl_unit_ansi: pure checks of milk-CTRL's terminal layer (ASCII-only source). */

#include "overview_ansi.h"

#include <fcntl.h>
#include <locale.h>
#include <sys/wait.h>

static int failures = 0;

#define CHECK(c)                                                         \
    do                                                                   \
    {                                                                    \
        if (!(c))                                                        \
        {                                                                \
            fprintf(stderr, "%s:%d: FAIL %s\n", __FILE__, __LINE__, #c); \
            failures++;                                                  \
        }                                                                \
    } while (0)

/* Printable-cluster and sanitizer cases (robust-10) */
static void test_sanitize(void)
{
    CHECK(ov_utf8_cluster_printable("abc", 3) == 1);
    CHECK(ov_utf8_cluster_printable("\x1b[31m", 5) == 0);
    CHECK(ov_utf8_cluster_printable("\xc2\x9b", 2) == 0); /* C1 CSI */
    CHECK(ov_utf8_cluster_printable("\x7f", 1) == 0);
    CHECK(ov_utf8_cluster_printable("\xff", 1) == 0);
    CHECK(ov_utf8_cluster_printable("\xe6\xb5\x81", 3) == 1); /* U+6D41 */
    CHECK(ov_utf8_cluster_printable("\xe6\xb5", 2) == 0);     /* truncated */
    CHECK(ov_utf8_cluster_printable("\xe2\x96\xb2", 3) == 1); /* U+25B2 */

    char s1[] = "a\x1b[2Jb";
    ov_str_sanitize(s1);
    CHECK(strcmp(s1, "a?[2Jb") == 0);
    char s2[] = "ok\xc2\x9bx";
    ov_str_sanitize(s2);
    CHECK(strcmp(s2, "ok??x") == 0);
    char s3[] = "fine \xe2\x96\xb2";
    ov_str_sanitize(s3);
    CHECK(strcmp(s3, "fine \xe2\x96\xb2") == 0);
}

/* A control byte reaching the shadow grid is stored as '?' */
static void test_append_cluster(void)
{
    ov_buf_reset_size(10, 20);
    ov__cursor_row = 1;
    ov__cursor_col = 1;
    ov_buf_append_cluster("\x1b", 1, 1);
    CHECK(strcmp(ov__shadow[0][0].ch, "?") == 0);
    ov_buf_append_cluster("A", 1, 1);
    CHECK(strcmp(ov__shadow[0][1].ch, "A") == 0);
}

/* With a non-UTF-8 user locale, widths still come from C.UTF-8 (gap-13) */
static void test_locale(void)
{
    setenv("LC_ALL", "C", 1);
    unsetenv("LANG");
    int utf8 = ov_term_init_locale();
    CHECK(utf8 == 0);
    if (setlocale(LC_CTYPE, NULL) == NULL || strstr(setlocale(LC_CTYPE, NULL), "UTF-8") == NULL)
    {
        fprintf(stderr, "skip: C.UTF-8 locale not available\n");
        return;
    }
    CHECK(ov_str_display_width("\xe6\xb5\x81") == 2);
}

/* Wide glyph invariants, cluster widths and fitting (robust-16) */
static void test_widths(void)
{
    int b = 0, w = 0;

    ov_buf_reset_size(10, 20);
    ov__cursor_row = 1;
    ov__cursor_col = 1;
    ov_buf_append_cluster("\xe6\xb5\x81", 3, 2);
    CHECK(ov__shadow[0][0].width == 2 && ov__shadow[0][1].width == 0);
    ov__cursor_col = 2; /* overwrite the right half */
    ov_buf_append_cluster("A", 1, 1);
    CHECK(ov__shadow[0][0].width == 1 && strcmp(ov__shadow[0][0].ch, " ") == 0);
    CHECK(strcmp(ov__shadow[0][1].ch, "A") == 0);

    ov__cursor_col = 5;
    ov_buf_append_cluster("\xe6\xb5\x81", 3, 2);
    ov__cursor_col = 5; /* narrow glyph over the left half */
    ov_buf_append_cluster("B", 1, 1);
    CHECK(ov__shadow[0][5].width == 1 && strcmp(ov__shadow[0][5].ch, " ") == 0);

    CHECK(ov_utf8_next_cluster("e\xcc\x81x", 4, &b, &w) && b == 3 && w == 1); /* combining acute */
    CHECK(ov_utf8_next_cluster("\xe2\x9a\xa0", 3, &b, &w) && w == 1);         /* U+26A0 is narrow */

    int cols = 0;
    CHECK(ov_utf8_fit("ab\xe6\xb5\x81"
                      "c",
                      6, 3, &cols) == 2 &&
          cols == 2);
    CHECK(ov_utf8_fit("ab\xe6\xb5\x81"
                      "c",
                      6, 4, &cols) == 5 &&
          cols == 4);
}

/* After a glyph whose width terminals disagree on, the next cell gets an explicit cursor move */
static void test_delta_resync(void)
{
    int  p[2];
    char out[4096] = { 0 };

    CHECK(pipe(p) == 0);
    ov__tty_fd = p[1];
    ov_buf_reset_size(1, 5);
    ov_buf_force_clear();
    ov__cursor_row = 1;
    ov__cursor_col = 1;
    ov_buf_append_cluster("\xe2\x8f\xb8", 3, 1); /* U+23F8 */
    ov_buf_append_cluster("Z", 1, 1);
    ov_buf_flush_delta(1, 5);
    ssize_t n = read(p[0], out, sizeof(out) - 1);
    CHECK(n > 0 && strstr(out, "\033[1;2HZ") != NULL);
    close(p[0]);
    close(p[1]);
    ov__tty_fd = STDOUT_FILENO;
}

/* Output larger than the frame buffer is flushed, not dropped (gap-14) */
static void test_overflow(void)
{
    char     chunk[100];
    uint64_t before = ov__total_bytes_rendered;
    int      total  = 3 * 1024 * 1024;

    memset(chunk, 'x', sizeof(chunk));
    ov__tty_fd = open("/dev/null", O_WRONLY);
    for (int sent = 0; sent < total; sent += (int) sizeof(chunk))
    {
        ov_buf_append(chunk, (int) sizeof(chunk));
    }
    ov_buf_flush_internal();
    CHECK(ov__total_bytes_rendered - before == (uint64_t) ((total + 99) / 100 * 100));
    close(ov__tty_fd);
    ov__tty_fd = STDOUT_FILENO;
}

/* Feed @seq through a pipe on stdin and return the first key it decodes to */
static int key_of(int wfd, const char *seq)
{
    if (write(wfd, seq, strlen(seq)) < 0)
    {
        return -2;
    }
    return ov_get_key();
}

/* Key parser: decode table, mouse bursts, split sequences (robust-11, gap-6) */
static void test_keys(void)
{
    int p[2];
    CHECK(pipe(p) == 0);
    int saved = dup(STDIN_FILENO);
    dup2(p[0], STDIN_FILENO);

    CHECK(key_of(p[1], "\x1b[A") == OV_KEY_UP);
    CHECK(key_of(p[1], "\x1b[11~") == OV_KEY_F1);
    CHECK(key_of(p[1], "\x1b[14~") == OV_KEY_F4);
    CHECK(key_of(p[1], "\x1b[[C") == OV_KEY_F3);
    CHECK(key_of(p[1], "\x1bOP") == OV_KEY_F1);
    CHECK(key_of(p[1], "\x1b[17~") == OV_KEY_F6);
    CHECK(key_of(p[1], "\x1b[1;5C") == OV_KEY_CTRL_RIGHT);
    CHECK(key_of(p[1], "\x1b[<0;10;5M") == OV_KEY_MOUSE_CLICK && ov_mouse_col == 10 &&
          ov_mouse_row == 5);
    CHECK(key_of(p[1], "\x1b[<0;10;5m") == OV_KEY_MOUSE_RELEASE);
    CHECK(key_of(p[1], "\x1b[2;3X") == OV_KEY_NONE); /* unknown CSI: consumed whole */
    CHECK(ov_get_key() == OV_KEY_NONE);
    CHECK(key_of(p[1], "a") == 'a');

    /* 30 hover reports then 'q': no parameter byte may come out as a key */
    char burst[1024] = "";
    for (int i = 0; i < 30; i++)
    {
        strcat(burst, "\x1b[<35;120;40M");
    }
    strcat(burst, "q");
    CHECK(write(p[1], burst, strlen(burst)) == (ssize_t) strlen(burst));
    int k, moves = 0, other = 0;
    while ((k = ov_get_key()) != OV_KEY_NONE)
    {
        if (k == OV_KEY_MOUSE_MOVE)
        {
            moves++;
        }
        else if (k != 'q')
        {
            other++;
        }
    }
    CHECK(moves == 30 && other == 0);

    /* A report split across two writes 5 ms apart decodes once complete */
    CHECK(write(p[1], "\x1b[<35;4", 7) == 7);
    pid_t pid = fork();
    if (pid == 0)
    {
        usleep(5000);
        _exit(write(p[1], "0;2M", 4) == 4 ? 0 : 1);
    }
    CHECK(ov_get_key() == OV_KEY_MOUSE_MOVE && ov_hover_col == 40 && ov_hover_row == 2);
    waitpid(pid, NULL, 0);

    /* Bracketed paste: printable bytes flagged, CR and escape sequences dropped (robust-7) */
    const char *paste = "\x1b[200~q\r\x1b[Ay\x1b[201~x";
    CHECK(write(p[1], paste, strlen(paste)) == (ssize_t) strlen(paste));
    CHECK(ov_get_key() == ('q' | OV_KEY_PASTED));
    CHECK(ov_get_key() == ('y' | OV_KEY_PASTED));
    CHECK(ov_get_key() == 'x');
    CHECK(ov_get_key() == OV_KEY_NONE);

    dup2(saved, STDIN_FILENO);
    close(saved);
    close(p[0]);
    close(p[1]);
}

int main(void)
{
    test_sanitize();
    test_append_cluster();
    test_locale();
    test_widths();
    test_delta_resync();
    test_overflow();
    test_keys();
    if (failures == 0)
    {
        printf("all ok\n");
    }
    return failures;
}
