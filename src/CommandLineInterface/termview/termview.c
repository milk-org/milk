#include "termview.h"
#include <ncurses.h>
#include <math.h>
#include "CommandLineInterface/TUItools.h"
#include "ImageStreamIO/ImageStreamIO.h"

static int loop = 1;
static short unsigned int wrow, wcol;

// Character set for intensity mapping (dark to light)
static const char *charset = " .:-=+*#%@";
static int charset_len = 10;

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

errno_t termview_screen(const char *imagename)
{
    IMAGE img;
    ImageStreamIO_read_sharedmem_image_toIMAGE(imagename, &img);

    if (img.md == NULL) {
        printf("Error: Could not connect to image %s\n", imagename);
        return 0;
    }

    // Initialize TUI
    TUI_set_screenprintmode(SCREENPRINT_NCURSES);
    TUI_init_terminal(&wrow, &wcol);

    while(loop) {
        // Handle input
        int ch = get_singlechar_nonblock();
        if (ch == 'q') {
            loop = 0;
        } else if (ch == KEY_RESIZE) {
            getmaxyx(stdscr, wrow, wcol);
            clear();
            refresh();
        }

        // Display
        erase();
        
        uint32_t xsize = img.md[0].size[0];
        uint32_t ysize = img.md[0].size[1];
        
        // Compute min/max for scaling
        double min_val = 1e20; // Initialize with large value
        double max_val = -1e20; // Initialize with small value
        int first = 1;

        // Simple subsampling/cropping to fit screen
        // Reserve rows for info
        int disp_rows = wrow - 4;
        int disp_cols = wcol;
        
        double step_x = 1.0;
        double step_y = 1.0;
        
        if (xsize > disp_cols) step_x = (double)xsize / disp_cols;
        if (ysize > disp_rows) step_y = (double)ysize / disp_rows;
        
        // Find min/max in the visible/sampled area
        for (int i = 0; i < disp_rows; i++) {
            for (int j = 0; j < disp_cols; j++) {
                int img_y = (int)(i * step_y);
                int img_x = (int)(j * step_x);
                
                if (img_x < xsize && img_y < ysize) {
                    double val = get_pixel_value(&img, img_x, img_y);
                    if (first) {
                        min_val = val;
                        max_val = val;
                        first = 0;
                    } else {
                        if (val < min_val) min_val = val;
                        if (val > max_val) max_val = val;
                    }
                }
            }
        }
        
        // Avoid division by zero
        if (max_val == min_val) max_val = min_val + 1.0;

        // Draw image
        for (int i = 0; i < disp_rows; i++) {
            for (int j = 0; j < disp_cols; j++) {
                int img_y = (int)(i * step_y);
                int img_x = (int)(j * step_x);
                
                if (img_x < xsize && img_y < ysize) {
                    double val = get_pixel_value(&img, img_x, img_y);
                    int char_idx = (int)((val - min_val) / (max_val - min_val) * (charset_len - 1));
                    if (char_idx < 0) char_idx = 0;
                    if (char_idx >= charset_len) char_idx = charset_len - 1;
                    
                    mvaddch(i, j, charset[char_idx]);
                }
            }
        }

        // Draw info
        int info_row = wrow - 3;
        mvprintw(info_row, 0, "Image: %s [%d x %d] Type: %d", img.md[0].name, xsize, ysize, img.md[0].datatype);
        mvprintw(info_row + 1, 0, "Min: %.4g  Max: %.4g", min_val, max_val);
        mvprintw(info_row + 2, 0, "Counter: %lu  (q to quit)", img.md[0].cnt0);

        refresh();
        usleep(50000);
    }

    TUI_exit();
    ImageStreamIO_closeIm(&img);

    return 0;
}
