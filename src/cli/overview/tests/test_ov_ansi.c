// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/* CTest milkctrl_unit_ansi: pure checks of milk-CTRL's terminal layer (ASCII-only source). */

#include "overview_ansi.h"

#include <locale.h>

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

int main(void)
{
    test_sanitize();
    test_append_cluster();
    test_locale();
    if (failures == 0)
    {
        printf("all ok\n");
    }
    return failures;
}
