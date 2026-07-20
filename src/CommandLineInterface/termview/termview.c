// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "termview.h"
#include "CommandLineInterface/TUItools.h"
#include "ImageStreamIO/ImageStreamIO.h"
#include <math.h>
#include <ncurses.h>
#include <stdlib.h>
#include <string.h>

static int                loop = 1;
static short unsigned int wrow, wcol;

// Character set for intensity mapping (dark to light)
static const char *charset     = " .:-=+*#%@";
static int         charset_len = 10;

// Internal state
static termview_options_t current_options;
static bool               has_color_support = false;
static bool               has_256_color     = false;

// Zoom and Pan state
static double view_zoom     = 1.0;
static double view_center_x = 0.5;
static double view_center_y = 0.5;

// Range state (cached for locking)
static double current_min_val = 0.0;
static double current_max_val = 1.0;

// Color pair ranges
#define COLOR_BASE_HEAT 20
#define COLOR_BASE_COLD 30
#define COLOR_BASE_JET 40
#define NB_COLORS_MAP 6

// 256-color mode ranges (64 steps each)
// Shifted to fit within 256 COLOR_PAIRS limit
#define STEPS_256 64
#define BASE_256_HEAT 50
#define BASE_256_COLD 114
#define BASE_256_JET 178

static double get_pixel_value(IMAGE *img, int x, int y)
{
    long idx = y * img->md[0].size[0] + x;

    // Boundary check
    if (x < 0 || x >= img->md[0].size[0] || y < 0 || y >= img->md[0].size[1])
    {
        return 0.0;
    }

    switch (img->md[0].datatype)
    {
    case _DATATYPE_UINT8:
        return (double) img->array.UI8[idx];
    case _DATATYPE_INT8:
        return (double) img->array.SI8[idx];
    case _DATATYPE_UINT16:
        return (double) img->array.UI16[idx];
    case _DATATYPE_INT16:
        return (double) img->array.SI16[idx];
    case _DATATYPE_UINT32:
        return (double) img->array.UI32[idx];
    case _DATATYPE_INT32:
        return (double) img->array.SI32[idx];
    case _DATATYPE_UINT64:
        return (double) img->array.UI64[idx];
    case _DATATYPE_INT64:
        return (double) img->array.SI64[idx];
    case _DATATYPE_FLOAT:
        return (double) img->array.F[idx];
    case _DATATYPE_DOUBLE:
        return (double) img->array.D[idx];
    default:
        return 0.0;
    }
}

// Convert normalized RGB to 256-color index (6x6x6 cube starts at 16)
static int rgb_to_256(float r, float g, float b)
{
    if (r < 0.0f)
    {
        r = 0.0f;
    }
    if (r > 1.0f)
    {
        r = 1.0f;
    }
    if (g < 0.0f)
    {
        g = 0.0f;
    }
    if (g > 1.0f)
    {
        g = 1.0f;
    }
    if (b < 0.0f)
    {
        b = 0.0f;
    }
    if (b > 1.0f)
    {
        b = 1.0f;
    }

    int ir = (int) (r * 5.0 + 0.5);
    int ig = (int) (g * 5.0 + 0.5);
    int ib = (int) (b * 5.0 + 0.5);

    if (ir > 5)
    {
        ir = 5;
    }
    if (ig > 5)
    {
        ig = 5;
    }
    if (ib > 5)
    {
        ib = 5;
    }

    return 16 + (ir * 36) + (ig * 6) + ib;
}

static void get_heat_color(float v, float *r, float *g, float *b)
{
    *r = *g = *b = 0.0;
    if (v < 0.0)
    {
        v = 0.0;
    }
    if (v > 1.0)
    {
        v = 1.0;
    }

    if (v < 0.25)
    { // Black to Blue
        *b = v * 4.0;
    }
    else if (v < 0.5)
    { // Blue to Cyan
        *b = 1.0;
        *g = (v - 0.25) * 4.0;
    }
    else if (v < 0.75)
    { // Cyan to Yellow
        *b = 1.0 - (v - 0.5) * 4.0;
        *g = 1.0;
        *r = (v - 0.5) * 4.0;
    }
    else
    { // Yellow to Red
        *g = 1.0 - (v - 0.75) * 4.0;
        *r = 1.0;
    }
}

static void get_cold_color(float v, float *r, float *g, float *b)
{
    *r = *g = *b = 0.0;
    if (v < 0.0)
    {
        v = 0.0;
    }
    if (v > 1.0)
    {
        v = 1.0;
    }

    // Black -> Blue -> Cyan -> White
    if (v < 0.33)
    {
        *b = v * 3.0;
    }
    else if (v < 0.66)
    {
        *b = 1.0;
        *g = (v - 0.33) * 3.0;
    }
    else
    {
        *b = 1.0;
        *g = 1.0;
        *r = (v - 0.66) * 3.0;
    }
}

static void get_jet_color(float v, float *r, float *g, float *b)
{
    *r = *g = *b = 0.0;
    if (v < 0.0)
    {
        v = 0.0;
    }
    if (v > 1.0)
    {
        v = 1.0;
    }

    // Blue -> Cyan -> Green -> Yellow -> Red
    if (v < 0.25)
    {
        *b = 1.0;
        *g = v * 4.0;
    }
    else if (v < 0.5)
    {
        *b = 1.0 - (v - 0.25) * 4.0;
        *g = 1.0;
    }
    else if (v < 0.75)
    {
        *g = 1.0;
        *r = (v - 0.5) * 4.0;
    }
    else
    {
        *g = 1.0 - (v - 0.75) * 4.0;
        *r = 1.0;
    }
}

static void init_colormaps()
{
    if (!has_colors())
    {
        return;
    }

    if (COLORS >= 256)
    {
        has_256_color = true;
        float r, g, b;
        for (int i = 0; i < STEPS_256; i++)
        {
            float v = (float) i / (STEPS_256 - 1);

            // Heat
            get_heat_color(v, &r, &g, &b);
            init_pair(BASE_256_HEAT + i, COLOR_WHITE, rgb_to_256(r, g, b));

            // Cold
            get_cold_color(v, &r, &g, &b);
            init_pair(BASE_256_COLD + i, COLOR_WHITE, rgb_to_256(r, g, b));

            // Jet
            get_jet_color(v, &r, &g, &b);
            init_pair(BASE_256_JET + i, COLOR_WHITE, rgb_to_256(r, g, b));
        }
    }
    else
    {
        // Fallback standard colors
        // HEAT
        init_pair(COLOR_BASE_HEAT + 0, COLOR_WHITE, COLOR_BLUE);
        init_pair(COLOR_BASE_HEAT + 1, COLOR_BLACK, COLOR_CYAN);
        init_pair(COLOR_BASE_HEAT + 2, COLOR_BLACK, COLOR_GREEN);
        init_pair(COLOR_BASE_HEAT + 3, COLOR_BLACK, COLOR_YELLOW);
        init_pair(COLOR_BASE_HEAT + 4, COLOR_WHITE, COLOR_RED);
        init_pair(COLOR_BASE_HEAT + 5, COLOR_WHITE, COLOR_MAGENTA);

        // COLD
        init_pair(COLOR_BASE_COLD + 0, COLOR_WHITE, COLOR_BLACK);
        init_pair(COLOR_BASE_COLD + 1, COLOR_WHITE, COLOR_BLUE);
        init_pair(COLOR_BASE_COLD + 2, COLOR_BLACK, COLOR_CYAN);
        init_pair(COLOR_BASE_COLD + 3, COLOR_BLACK, COLOR_WHITE);
        init_pair(COLOR_BASE_COLD + 4, COLOR_BLACK, COLOR_WHITE);
        init_pair(COLOR_BASE_COLD + 5, COLOR_BLACK, COLOR_WHITE);

        // JET
        init_pair(COLOR_BASE_JET + 0, COLOR_WHITE, COLOR_BLUE);
        init_pair(COLOR_BASE_JET + 1, COLOR_BLACK, COLOR_CYAN);
        init_pair(COLOR_BASE_JET + 2, COLOR_BLACK, COLOR_GREEN);
        init_pair(COLOR_BASE_JET + 3, COLOR_BLACK, COLOR_YELLOW);
        init_pair(COLOR_BASE_JET + 4, COLOR_WHITE, COLOR_RED);
        init_pair(COLOR_BASE_JET + 5, COLOR_WHITE, COLOR_RED);
    }
}

static int compare_doubles(const void *a, const void *b)
{
    double arg1 = *(const double *) a;
    double arg2 = *(const double *) b;
    if (arg1 < arg2)
    {
        return -1;
    }
    if (arg1 > arg2)
    {
        return 1;
    }
    return 0;
}

static double get_input_double(const char *prompt)
{
    int r = wrow / 2;
    int c = wcol / 2 - 15;

    // Clear prompt area
    attron(A_REVERSE);
    for (int i = 0; i < 30; i++)
    {
        mvaddch(r, c + i, ' ');
    }
    mvprintw(r, c, "%s: ", prompt);
    attroff(A_REVERSE);
    refresh();

    // Temporarily switch to blocking mode and enable echo
    nodelay(stdscr, FALSE);
    echo();
    curs_set(1);

    char buf[32];
    mvgetnstr(r, c + strlen(prompt) + 2, buf, 30);

    // Restore settings
    noecho();
    curs_set(0);
    nodelay(stdscr, TRUE);

    return atof(buf);
}

errno_t termview_screen(const char *imagename, termview_options_t options)
{
    IMAGE img;
    ImageStreamIO_read_sharedmem_image_toIMAGE(imagename, &img);

    if (img.md == NULL)
    {
        printf("Error: Could not connect to image %s\n", imagename);
        return 0;
    }

    current_options = options;

    // Initialize TUI
    TUI_set_screenprintmode(SCREENPRINT_NCURSES);
    TUI_init_terminal(&wrow, &wcol);

    if (has_colors())
    {
        has_color_support = true;
        init_colormaps();
    }

    double *display_buffer = NULL;
    int     buffer_size    = 0;

    while (loop)
    {
        // Handle input
        int ch = get_singlechar_nonblock();
        switch (ch)
        {
        case 'q':
            loop = 0;
            break;
        case KEY_RESIZE:
            getmaxyx(stdscr, wrow, wcol);
            clear();
            refresh();
            break;
        case 'c':
            current_options.colormap = (current_options.colormap + 1) % COLORMAP_NB;
            break;
        case 's':
            current_options.scale = (current_options.scale + 1) % SCALE_NB;
            break;
        case 'r':
            current_options.range = (current_options.range + 1) % RANGE_NB;
            if (current_options.range == RANGE_MANUAL)
            {
                current_options.range_locked = true;
            }
            else
            {
                current_options.range_locked = false;
            }
            break;
        case 'l': // Lock range
            current_options.range_locked = !current_options.range_locked;
            if (current_options.range_locked)
            {
                current_options.range      = RANGE_MANUAL;
                current_options.manual_min = current_min_val;
                current_options.manual_max = current_max_val;
            }
            else
            {
                current_options.range = RANGE_MINMAX; // Default back to auto
            }
            break;
        case 'm': // Manual range
            current_options.manual_min   = get_input_double("Min Val");
            current_options.manual_max   = get_input_double("Max Val");
            current_options.range        = RANGE_MANUAL;
            current_options.range_locked = true;
            break;
        // Zoom keys
        case '+':
        case '=':
            view_zoom *= 1.2;
            break;
        case '-':
        case '_':
            view_zoom /= 1.2;
            if (view_zoom < 0.1)
            {
                view_zoom = 0.1;
            }
            break;
        case '0':
            view_zoom     = 1.0;
            view_center_x = 0.5;
            view_center_y = 0.5;
            break;
        // Pan keys
        case KEY_LEFT:
            view_center_x -= 0.1 / view_zoom;
            if (view_center_x < 0.0)
            {
                view_center_x = 0.0;
            }
            break;
        case KEY_RIGHT:
            view_center_x += 0.1 / view_zoom;
            if (view_center_x > 1.0)
            {
                view_center_x = 1.0;
            }
            break;
        case KEY_UP:
            view_center_y -= 0.1 / view_zoom;
            if (view_center_y < 0.0)
            {
                view_center_y = 0.0;
            }
            break;
        case KEY_DOWN:
            view_center_y += 0.1 / view_zoom;
            if (view_center_y > 1.0)
            {
                view_center_y = 1.0;
            }
            break;
        }

        if (loop == 0)
        {
            break;
        }

        // Display
        erase();

        uint32_t xsize = img.md[0].size[0];
        uint32_t ysize = img.md[0].size[1];

        // Layout:
        // Image Area: [0..wrow-4] x [0..wcol-12]
        // Colorbar:   [0..wrow-4] x [wcol-10..wcol]

        int bar_width     = 12;
        int disp_rows     = wrow - 4;
        int disp_cols     = wcol - bar_width - 1;
        int bar_col_start = wcol - bar_width;

        if (disp_rows <= 0 || disp_cols <= 0)
        {
            usleep(100000);
            continue;
        }

        // Reallocate buffer if needed
        if (disp_rows * disp_cols > buffer_size)
        {
            buffer_size    = disp_rows * disp_cols;
            display_buffer = (double *) realloc(display_buffer, buffer_size * sizeof(double));
        }

        // Subsample
        double view_w_img = (double) xsize / view_zoom;
        double view_h_img = (double) ysize / view_zoom;
        double start_x    = view_center_x * xsize - view_w_img / 2.0;
        double start_y    = view_center_y * ysize - view_h_img / 2.0;
        double step_x     = view_w_img / disp_cols;
        double step_y     = view_h_img / disp_rows;

        int buf_idx = 0;
        for (int i = 0; i < disp_rows; i++)
        {
            for (int j = 0; j < disp_cols; j++)
            {
                int    img_y = (int) (start_y + i * step_y);
                int    img_x = (int) (start_x + j * step_x);
                double val   = 0.0;
                if (img_x >= 0 && img_x < xsize && img_y >= 0 && img_y < ysize)
                {
                    val = get_pixel_value(&img, img_x, img_y);
                }
                display_buffer[buf_idx++] = val;
            }
        }
        int num_pixels = buf_idx;

        // Compute Stats
        double min_val = 0.0, max_val = 1.0;

        if (current_options.range == RANGE_MANUAL)
        {
            min_val = current_options.manual_min;
            max_val = current_options.manual_max;
        }
        else if (current_options.range == RANGE_MINMAX)
        {
            min_val = 1e20;
            max_val = -1e20;
            for (int k = 0; k < num_pixels; k++)
            {
                if (display_buffer[k] < min_val)
                {
                    min_val = display_buffer[k];
                }
                if (display_buffer[k] > max_val)
                {
                    max_val = display_buffer[k];
                }
            }
        }
        else
        {
            double *sorted_buf = (double *) malloc(num_pixels * sizeof(double));
            memcpy(sorted_buf, display_buffer, num_pixels * sizeof(double));
            qsort(sorted_buf, num_pixels, sizeof(double), compare_doubles);
            double p_low = 0.0, p_high = 1.0;
            switch (current_options.range)
            {
            case RANGE_01_99:
                p_low  = 0.01;
                p_high = 0.99;
                break;
            case RANGE_05_95:
                p_low  = 0.05;
                p_high = 0.95;
                break;
            case RANGE_10_90:
                p_low  = 0.10;
                p_high = 0.90;
                break;
            default:
                break;
            }
            min_val = sorted_buf[(int) (p_low * (num_pixels - 1))];
            max_val = sorted_buf[(int) (p_high * (num_pixels - 1))];
            free(sorted_buf);
        }
        if (max_val <= min_val)
        {
            max_val = min_val + 1.0; // Avoid div/0
        }

        current_min_val = min_val;
        current_max_val = max_val;

        // Render Image
        buf_idx = 0;
        for (int i = 0; i < disp_rows; i++)
        {
            for (int j = 0; j < disp_cols; j++)
            {
                int  img_y     = (int) (start_y + i * step_y);
                int  img_x     = (int) (start_x + j * step_x);
                bool in_bounds = (img_x >= 0 && img_x < xsize && img_y >= 0 && img_y < ysize);

                if (!in_bounds)
                {
                    buf_idx++;
                    continue;
                }

                double val      = display_buffer[buf_idx++];
                double norm_val = 0.0;

                // Scale
                if (val < min_val)
                {
                    val = min_val;
                }
                if (val > max_val)
                {
                    val = max_val;
                }
                double v_offset = val - min_val;
                double range    = max_val - min_val;

                switch (current_options.scale)
                {
                case SCALE_LINEAR:
                    norm_val = v_offset / range;
                    break;
                case SCALE_SQRT:
                    norm_val = sqrt(v_offset) / sqrt(range);
                    break;
                case SCALE_LOG:
                    norm_val = log(v_offset + 1.0) / log(range + 1.0);
                    break;
                default:
                    norm_val = v_offset / range;
                }
                if (norm_val < 0)
                {
                    norm_val = 0;
                }
                if (norm_val > 1)
                {
                    norm_val = 1;
                }

                // Draw Pixel
                if (current_options.colormap == COLORMAP_GREYSCALE || !has_color_support)
                {
                    int char_idx = (int) (norm_val * (charset_len - 1));
                    mvaddch(i, j, charset[char_idx]);
                }
                else
                {
                    int pair = 0;
                    if (has_256_color)
                    {
                        int base = BASE_256_HEAT;
                        if (current_options.colormap == COLORMAP_COLD)
                        {
                            base = BASE_256_COLD;
                        }
                        else if (current_options.colormap == COLORMAP_JET)
                        {
                            base = BASE_256_JET;
                        }
                        int idx = (int) (norm_val * (STEPS_256 - 1));
                        pair    = base + idx;
                    }
                    else
                    {
                        int base = COLOR_BASE_HEAT;
                        int n    = NB_COLORS_MAP;
                        if (current_options.colormap == COLORMAP_COLD)
                        {
                            base = COLOR_BASE_COLD;
                            n    = NB_COLORS_MAP;
                        } // Fixed cold range
                        else if (current_options.colormap == COLORMAP_JET)
                        {
                            base = COLOR_BASE_JET;
                        }
                        int idx = (int) (norm_val * (n - 1));
                        pair    = base + idx;
                    }
                    attron(COLOR_PAIR(pair));
                    mvaddch(i, j, ' ');
                    attroff(COLOR_PAIR(pair));
                }
            }
        }

        // Render Colorbar
        if (has_color_support && current_options.colormap != COLORMAP_GREYSCALE)
        {
            for (int i = 0; i < disp_rows; i++)
            {
                // Invert y: high value at top
                double norm_val = 1.0 - (double) i / (disp_rows - 1);

                int pair = 0;
                if (has_256_color)
                {
                    int base = BASE_256_HEAT;
                    if (current_options.colormap == COLORMAP_COLD)
                    {
                        base = BASE_256_COLD;
                    }
                    else if (current_options.colormap == COLORMAP_JET)
                    {
                        base = BASE_256_JET;
                    }
                    int idx = (int) (norm_val * (STEPS_256 - 1));
                    pair    = base + idx;
                }
                else
                {
                    int base = COLOR_BASE_HEAT;
                    int n    = NB_COLORS_MAP;
                    if (current_options.colormap == COLORMAP_COLD)
                    {
                        base = COLOR_BASE_COLD;
                        n    = NB_COLORS_MAP;
                    } // Fixed cold range
                    else if (current_options.colormap == COLORMAP_JET)
                    {
                        base = COLOR_BASE_JET;
                    }
                    int idx = (int) (norm_val * (n - 1));
                    pair    = base + idx;
                }

                attron(COLOR_PAIR(pair));
                mvaddch(i, bar_col_start, ' ');
                mvaddch(i, bar_col_start + 1, ' ');
                attroff(COLOR_PAIR(pair));

                // Labels
                if (i == 0)
                {
                    mvprintw(i, bar_col_start + 3, "%.2g", max_val);
                }
                if (i == disp_rows / 2)
                {
                    mvprintw(i, bar_col_start + 3, "%.2g", (min_val + max_val) / 2);
                }
                if (i == disp_rows - 1)
                {
                    mvprintw(i, bar_col_start + 3, "%.2g", min_val);
                }
            }
        }

        // Draw info
        int         info_row = wrow - 3;
        const char *cmap_str = "GREY";
        switch (current_options.colormap)
        {
        case COLORMAP_HEAT:
            cmap_str = "HEAT";
            break;
        case COLORMAP_COLD:
            cmap_str = "COLD";
            break;
        case COLORMAP_JET:
            cmap_str = "JET";
            break;
        default:
            break;
        }
        const char *scale_str = "LIN";
        switch (current_options.scale)
        {
        case SCALE_SQRT:
            scale_str = "SQRT";
            break;
        case SCALE_LOG:
            scale_str = "LOG";
            break;
        default:
            break;
        }
        const char *range_str = "MINMAX";
        if (current_options.range_locked)
        {
            range_str = "LOCKED";
        }
        else
        {
            switch (current_options.range)
            {
            case RANGE_01_99:
                range_str = "1-99%";
                break;
            case RANGE_05_95:
                range_str = "5-95%";
                break;
            case RANGE_10_90:
                range_str = "10-90%";
                break;
            default:
                break;
            }
        }

        mvprintw(info_row, 0, "Image: %s [%d x %d] Type: %d", img.md[0].name, xsize, ysize,
                 img.md[0].datatype);
        mvprintw(info_row + 1, 0, "Val: [%.4g : %.4g] Zoom: %.2fx", min_val, max_val, view_zoom);
        mvprintw(info_row + 2, 0, "[C]map: %s  [S]cale: %s  [R]ange: %s  (l:Lock m:Man q:Quit)",
                 cmap_str, scale_str, range_str);

        refresh();
        usleep(50000);
    }

    if (display_buffer)
    {
        free(display_buffer);
    }
    TUI_exit();
    ImageStreamIO_closeIm(&img);

    return 0;
}
