// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/* CTest milkctrl_unit_ui: view table, layout helpers, key table and dispatch, unit formatting. */

#include "overview_data_internal.h"
#include "overview_render_internal.h"
#include "overview_input_keys.h"

#include <stdlib.h>

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

/* View descriptors: unique F-keys and CLI names, entry focus on a shown panel, cycling wraps */
static void test_view_descs(void)
{
    for (int a = 0; a < OV_VIEW_COUNT; a++)
    {
        CHECK(ov_view_by_fkey(ov_views[a].fkey) == a);
        CHECK(ov_view_by_name(ov_views[a].cli_name) == a);
        CHECK(ov_views[a].default_focus < 0 ||
              ov_view_has_panel((ov_view_t) a, (ov_focus_t) ov_views[a].default_focus));
        CHECK(ov_view_cycle(ov_view_cycle(a, +1), -1) == a);
    }
    CHECK(ov_view_by_fkey(1) == -1 && ov_view_by_name("nope") == -1);
    CHECK(ov_view_cycle(0, -1) == OV_VIEW_COUNT - 1 && ov_view_cycle(OV_VIEW_COUNT - 1, 1) == 0);
    CHECK(strcmp(ov_view_label(OV_VIEW_LOOPS), "LOOPS") == 0);
    static OV_LAYOUT lay;
    memset(&lay, 0, sizeof(lay));
    lay.focus = OV_FOCUS_FPS;
    ov_set_view(&lay, OV_VIEW_DASHBOARD); /* DASH keeps the focus */
    CHECK(lay.view == OV_VIEW_DASHBOARD && lay.focus == OV_FOCUS_FPS);
    ov_set_view(&lay, OV_VIEW_LOOPS);
    CHECK(lay.view == OV_VIEW_LOOPS && lay.focus == OV_FOCUS_GRAPH);
}

/* Loop list height: one helper replaces the four copies of the DASH formula; F7 uses its whole pane */
static void test_loop_list_rows(void)
{
    static OV_LAYOUT lay;
    memset(&lay, 0, sizeof(lay));
    for (int h = 0; h < 60; h++)
    {
        lay.r_graph.height = h + 3;
        lay.view           = OV_VIEW_DASHBOARD;
        lay.subtab         = OV_SUB_LOOPS;
        CHECK(ov_loop_list_rows(&lay) == ((h >= 6) ? ((h > 8) ? (h / 2) : 3) : h));
        CHECK(ov_subtab_page(&lay) == ov_loop_list_rows(&lay));
        lay.subtab = OV_SUB_CONN;
        CHECK(ov_subtab_page(&lay) == h);
        lay.view = OV_VIEW_LOOPS;
        CHECK(ov_loop_list_rows(&lay) == ((h < 1) ? 1 : h));
    }
}

/* Key table: every entry is documented somewhere; status-line hints are distinct */
static void test_keymap(void)
{
    for (int i = 0; i < ov_nb_keys; i++)
    {
        CHECK(ov_keys[i].display != NULL && ov_keys[i].label != NULL);
        CHECK(ov_keys[i].section >= 0 || ov_keys[i].group >= 0);
        CHECK(ov_keys[i].section < HS_COUNT && ov_keys[i].group < OV_KG_COUNT);
        for (int j = i + 1; ov_keys[i].footer != NULL && j < ov_nb_keys; j++)
        {
            CHECK(ov_keys[j].footer == NULL || strcmp(ov_keys[i].footer, ov_keys[j].footer) != 0);
        }
    }
    char buf[96];
    CHECK(ov_key_footer(buf, sizeof(buf)) == strlen(buf));
    CHECK(strstr(buf, "TAB") != NULL && strstr(buf, "q") != NULL);
    CHECK(ov_key_footer(buf, 4) == 3); /* truncated, still terminated */
}

static int calls[4];

/** row handlers for test_key_dispatch: record the call; row 1 declines */
static int h0(int key, OV_LAYOUT *lay, const OV_MODEL *m)
{
    (void) key, (void) lay, (void) m;
    return ++calls[0], 1;
}
static int h1(int key, OV_LAYOUT *lay, const OV_MODEL *m)
{
    (void) key, (void) lay, (void) m;
    return ++calls[1], 0;
}
static int h2(int key, OV_LAYOUT *lay, const OV_MODEL *m)
{
    (void) key, (void) lay, (void) m;
    return ++calls[2], 2;
}

/* Key dispatch: context needed / forbidden, first match, a declining row falls through, wildcard */
static void test_key_dispatch(void)
{
    static const OV_KEYBIND t[] = {
        { { 'x' }, OV_CTX_CTRL, 0, h0 },
        { { 'x', 'y' }, 0, OV_CTX_F5, h1 },
        { { 'x' }, 0, 0, h2 },
        { { OV_KEY_ANY }, OV_CTX_F5, 0, h0 },
    };
    static OV_LAYOUT lay;
    memset(&lay, 0, sizeof(lay));
    lay.view = OV_VIEW_DASHBOARD;
    CHECK(ov_key_dispatch(t, 4, 'x', &lay, NULL) == 2 && calls[1] == 1 && calls[2] == 1);
    lay.ctrl_mode = 1;
    CHECK(ov_key_dispatch(t, 4, 'x', &lay, NULL) == 1 && calls[0] == 1);
    CHECK(ov_key_dispatch(t, 4, 'y', &lay, NULL) == 0 && calls[1] == 2);
    lay.view      = OV_VIEW_FPS;
    lay.ctrl_mode = 0;
    CHECK(ov_key_dispatch(t, 4, 'q', &lay, NULL) == 1 && calls[0] == 2);
    CHECK(ov_key_dispatch(t, 4, 'y', &lay, NULL) == 1 && calls[1] == 2); /* F5: row 1 forbidden */
    unsigned c = ov_key_ctx(&lay, NULL);
    CHECK((c & OV_CTX_F5) && !(c & OV_CTX_MODEL) && (c & OV_CTX_FOCUS_LIST));
}


/* Durations keep at most 7 columns */
static void test_fmt_dt(void)
{
    static const struct
    {
        int64_t     ns;
        const char *want;
    } g[] = { { -1, "-" },
              { 850, "850ns" },
              { 12345, "12.3us" },
              { 1250000, "1.25ms" },
              { 999000000, "999ms" },
              { 2400000000LL, "2.40s" },
              { 123456789000LL, "123s" } };
    char buf[16];
    for (size_t i = 0; i < sizeof(g) / sizeof(g[0]); i++)
    {
        CHECK(strcmp(ov_fmt_dt_ns(buf, sizeof(buf), g[i].ns), g[i].want) == 0);
        CHECK(strlen(buf) <= 7);
    }
}


int main(void)
{
    test_view_descs();
    test_loop_list_rows();
    test_keymap();
    test_key_dispatch();
    test_fmt_dt();
    if (failures == 0)
    {
        printf("all ok\n");
    }
    return failures;
}
