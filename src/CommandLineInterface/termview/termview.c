#include "termview.h"
#include <ncurses.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "CommandLineInterface/TUItools.h"
#include "ImageStreamIO/ImageStreamIO.h"

static int loop = 1;
static short unsigned int wrow, wcol;

// Character set for intensity mapping (dark to light)
static const char *charset = " .:-=+*#%@";
static int charset_len = 10;

// Internal state
static termview_options_t current_options;
static bool has_color_support = false;

// Color pair ranges
#define COLOR_BASE_HEAT 20
#define COLOR_BASE_COLD 30
#define COLOR_BASE_JET 40
#define NB_COLORS_MAP 6

static double get_pixel_value(IMAGE *img, int x, int y) {
    long idx = y * img->md[0].size[0] + x;
    
    // Boundary check
    if (x < 0 || x >= img->md[0].size[0] || y < 0 || y >= img->md[0].size[1]) {
        return 0.0;
    }

    switch(img->md[0].datatype) {
        case _DATATYPE_UINT8:  return (double) img->array.UI8[idx];
        case _DATATYPE_INT8:   return (double) img->array.SI8[idx];
        case _DATATYPE_UINT16: return (double) img->array.UI16[idx];
        case _DATATYPE_INT16:  return (double) img->array.SI16[idx];
        case _DATATYPE_UINT32: return (double) img->array.UI32[idx];
        case _DATATYPE_INT32:  return (double) img->array.SI32[idx];
        case _DATATYPE_UINT64: return (double) img->array.UI64[idx];
        case _DATATYPE_INT64:  return (double) img->array.SI64[idx];
        case _DATATYPE_FLOAT:  return (double) img->array.F[idx];
        case _DATATYPE_DOUBLE: return (double) img->array.D[idx];
        default: return 0.0;
    }
}

static void init_colormaps() {
    if (!has_colors()) return;

    // HEAT: Blue -> Cyan -> Green -> Yellow -> Red -> Magenta
    init_pair(COLOR_BASE_HEAT + 0, COLOR_WHITE, COLOR_BLUE);
    init_pair(COLOR_BASE_HEAT + 1, COLOR_BLACK, COLOR_CYAN);
    init_pair(COLOR_BASE_HEAT + 2, COLOR_BLACK, COLOR_GREEN);
    init_pair(COLOR_BASE_HEAT + 3, COLOR_BLACK, COLOR_YELLOW);
    init_pair(COLOR_BASE_HEAT + 4, COLOR_WHITE, COLOR_RED);
    init_pair(COLOR_BASE_HEAT + 5, COLOR_WHITE, COLOR_MAGENTA);

    // COLD: White -> Cyan -> Blue -> Black (Simulated with background)
    init_pair(COLOR_BASE_COLD + 0, COLOR_BLACK, COLOR_WHITE);
    init_pair(COLOR_BASE_COLD + 1, COLOR_BLACK, COLOR_CYAN);
    init_pair(COLOR_BASE_COLD + 2, COLOR_WHITE, COLOR_BLUE);
    init_pair(COLOR_BASE_COLD + 3, COLOR_WHITE, COLOR_BLACK);
    // Fill remaining to avoid crash if accessed, map to darkest
    init_pair(COLOR_BASE_COLD + 4, COLOR_WHITE, COLOR_BLACK);
    init_pair(COLOR_BASE_COLD + 5, COLOR_WHITE, COLOR_BLACK);

    // JET: Blue -> Cyan -> Green -> Yellow -> Red
    init_pair(COLOR_BASE_JET + 0, COLOR_WHITE, COLOR_BLUE);
    init_pair(COLOR_BASE_JET + 1, COLOR_BLACK, COLOR_CYAN);
    init_pair(COLOR_BASE_JET + 2, COLOR_BLACK, COLOR_GREEN);
    init_pair(COLOR_BASE_JET + 3, COLOR_BLACK, COLOR_YELLOW);
    init_pair(COLOR_BASE_JET + 4, COLOR_WHITE, COLOR_RED);
    init_pair(COLOR_BASE_JET + 5, COLOR_WHITE, COLOR_RED); // Repeat max
}

static int compare_doubles(const void *a, const void *b) {
    double arg1 = *(const double *)a;
    double arg2 = *(const double *)b;
    if (arg1 < arg2) return -1;
    if (arg1 > arg2) return 1;
    return 0;
}

errno_t termview_screen(const char *imagename, termview_options_t options)
{
    IMAGE img;
    ImageStreamIO_read_sharedmem_image_toIMAGE(imagename, &img);

    if (img.md == NULL) {
        printf("Error: Could not connect to image %s\n", imagename);
        return 0;
    }

    current_options = options;

    // Initialize TUI
    TUI_set_screenprintmode(SCREENPRINT_NCURSES);
    TUI_init_terminal(&wrow, &wcol);

    if (has_colors()) {
        has_color_support = true;
        init_colormaps();
    }

    double *display_buffer = NULL;
    int buffer_size = 0;

    while(loop) {
        // Handle input
        int ch = get_singlechar_nonblock();
        switch(ch) {
            case 'q': loop = 0; break;
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
                break;
        }

        if (loop == 0) break;

        // Display
        erase();
        
        uint32_t xsize = img.md[0].size[0];
        uint32_t ysize = img.md[0].size[1];
        
        // Reserve rows for info
        int disp_rows = wrow - 4;
        int disp_cols = wcol;
        
        if (disp_rows <= 0 || disp_cols <= 0) {
            usleep(100000);
            continue;
        }

        // Reallocate buffer if needed
        if (disp_rows * disp_cols > buffer_size) {
            buffer_size = disp_rows * disp_cols;
            display_buffer = (double*)realloc(display_buffer, buffer_size * sizeof(double));
        }

        // Subsample
        double step_x = 1.0;
        double step_y = 1.0;
        if (xsize > disp_cols) step_x = (double)xsize / disp_cols;
        if (ysize > disp_rows) step_y = (double)ysize / disp_rows;

        int buf_idx = 0;
        for (int i = 0; i < disp_rows; i++) {
            for (int j = 0; j < disp_cols; j++) {
                int img_y = (int)(i * step_y);
                int img_x = (int)(j * step_x);
                double val = 0.0;
                if (img_x < xsize && img_y < ysize) {
                    val = get_pixel_value(&img, img_x, img_y);
                }
                display_buffer[buf_idx++] = val;
            }
        }
        int num_pixels = buf_idx;

        // Compute Min/Max based on Range Mode
        double min_val = 0.0, max_val = 1.0;
        
        if (current_options.range == RANGE_MINMAX) {
            min_val = 1e20;
            max_val = -1e20;
            for(int k=0; k<num_pixels; k++) {
                if(display_buffer[k] < min_val) min_val = display_buffer[k];
                if(display_buffer[k] > max_val) max_val = display_buffer[k];
            }
        } else {
            // Percentiles require sorting
            double *sorted_buf = (double*)malloc(num_pixels * sizeof(double));
            memcpy(sorted_buf, display_buffer, num_pixels * sizeof(double));
            qsort(sorted_buf, num_pixels, sizeof(double), compare_doubles);
            
            double p_low = 0.0, p_high = 1.0;
            switch(current_options.range) {
                case RANGE_01_99: p_low = 0.01; p_high = 0.99; break;
                case RANGE_05_95: p_low = 0.05; p_high = 0.95; break;
                case RANGE_10_90: p_low = 0.10; p_high = 0.90; break;
                default: break;
            }
            min_val = sorted_buf[(int)(p_low * (num_pixels-1))];
            max_val = sorted_buf[(int)(p_high * (num_pixels-1))];
            free(sorted_buf);
        }

        if (max_val <= min_val) max_val = min_val + 1.0; // Avoid div/0

        // Apply Scaling
        // We apply scaling to the normalization limits, not the data first, 
        // to properly handle ranges. Actually, it's better to scale data then normalize.
        // But for display, we map val -> [0,1].
        // Linear: (v - min)/(max - min)
        // Sqrt: (sqrt(v - min))/(sqrt(max - min)) -- assumes v >= min
        // Log: log(v - min + 1) / log(max - min + 1)

        // Render
        buf_idx = 0;
        for (int i = 0; i < disp_rows; i++) {
            for (int j = 0; j < disp_cols; j++) {
                double val = display_buffer[buf_idx++];
                double norm_val = 0.0;

                // Clip to range
                if (val < min_val) val = min_val;
                if (val > max_val) val = max_val;

                double v_offset = val - min_val;
                double range = max_val - min_val;

                switch(current_options.scale) {
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

                if (norm_val < 0) norm_val = 0;
                if (norm_val > 1) norm_val = 1;

                // Colormap rendering
                if (current_options.colormap == COLORMAP_GREYSCALE || !has_color_support) {
                    int char_idx = (int)(norm_val * (charset_len - 1));
                    mvaddch(i, j, charset[char_idx]);
                } else {
                    int base_pair = COLOR_BASE_HEAT;
                    int num_colors_in_map = NB_COLORS_MAP;
                    
                    if (current_options.colormap == COLORMAP_COLD) {
                        base_pair = COLOR_BASE_COLD;
                        num_colors_in_map = 4; // Cold map has 4 colors defined
                    } else if (current_options.colormap == COLORMAP_JET) {
                        base_pair = COLOR_BASE_JET;
                    }

                    int color_idx = (int)(norm_val * (num_colors_in_map - 1));
                    int pair = base_pair + color_idx;
                    
                    attron(COLOR_PAIR(pair));
                    mvaddch(i, j, ' ');
                    attroff(COLOR_PAIR(pair));
                }
            }
        }

        // Draw info
        int info_row = wrow - 3;
        
        const char* cmap_str = "GREY";
        switch(current_options.colormap) {
            case COLORMAP_HEAT: cmap_str = "HEAT"; break;
            case COLORMAP_COLD: cmap_str = "COLD"; break;
            case COLORMAP_JET: cmap_str = "JET"; break;
            default: break;
        }
        
        const char* scale_str = "LIN";
        switch(current_options.scale) {
            case SCALE_SQRT: scale_str = "SQRT"; break;
            case SCALE_LOG: scale_str = "LOG"; break;
            default: break;
        }

        const char* range_str = "MINMAX";
        switch(current_options.range) {
            case RANGE_01_99: range_str = "1-99%"; break;
            case RANGE_05_95: range_str = "5-95%"; break;
            case RANGE_10_90: range_str = "10-90%"; break;
            default: break;
        }

        mvprintw(info_row, 0, "Image: %s [%d x %d] Type: %d", img.md[0].name, xsize, ysize, img.md[0].datatype);
        mvprintw(info_row + 1, 0, "Val: [%.4g : %.4g]", min_val, max_val);
        mvprintw(info_row + 2, 0, "[C]map: %s  [S]cale: %s  [R]ange: %s  (q:quit)", cmap_str, scale_str, range_str);

        refresh();
        usleep(50000);
    }

    if (display_buffer) free(display_buffer);
    TUI_exit();
    ImageStreamIO_closeIm(&img);

    return 0;
}
