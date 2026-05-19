/**
 * @file termview.c
 * @brief Termview module (TrueColor Standalone Rewrite)
 */

#include "termview.h"
#include "termview_ansi.h"
#include "ImageStreamIO/ImageStreamIO.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <unistd.h>
#include <termios.h>
#include <sys/ioctl.h>
#include <sys/select.h>
#include <signal.h>
#include <time.h>
#include <stdbool.h>

/* -----------------------------------------------------------------------
 * Typedefs & Constants
 * ----------------------------------------------------------------------- */
typedef struct {
    uint8_t r, g, b;
} rgb_t;

#define TV_KEY_UP    0x1001
#define TV_KEY_DOWN  0x1002
#define TV_KEY_LEFT  0x1003
#define TV_KEY_RIGHT 0x1004
#define TV_KEY_MOUSE 0x1005

typedef struct {
    int button; // 0: Left, 1: Middle, 2: Right, 64: ScrollUp, 65: ScrollDown
    int x; // 1-indexed
    int y; // 1-indexed
    int is_press; // 1 for press/drag, 0 for release
    int is_drag; // 1 if drag
} tv_mouse_event_t;

static tv_mouse_event_t last_mouse_event = {0};

/* -----------------------------------------------------------------------
 * Global State
 * ----------------------------------------------------------------------- */
static int loop = 1;
static unsigned short wrow = 0, wcol = 0;
static int force_redraw = 1;
static volatile sig_atomic_t tv_resize_flag = 0;

static termview_options_t current_options;
static double view_zoom = 1.0;
static double view_center_x = 0.5;
static double view_center_y = 0.5;

static double current_min_val = 0.0;
static double current_max_val = 1.0;

/* -----------------------------------------------------------------------
 * Grid-based Rendering State
 * ----------------------------------------------------------------------- */
typedef struct {
    uint8_t bg_r, bg_g, bg_b;
    uint8_t fg_r, fg_g, fg_b;
    char ch[4]; // utf-8 char (half-block is 3 bytes, full-block 3 bytes)
} tv_cell_t;

static rgb_t colormap_lut[1024];
static int lut_colormap = -1;
static int lut_scale = -1;
static double lut_min = -1.0;
static double lut_max = -1.0;

static rgb_t get_colormap_color(termview_colormap_t cmap, double v) {
    if (v < 0.0) v = 0.0;
    if (v > 1.0) v = 1.0;
    
    rgb_t c = {0, 0, 0};
    if (cmap == COLORMAP_GREYSCALE) {
        c.r = c.g = c.b = (uint8_t)(v * 255.0f);
    } else {
        float r = 0.0f, g = 0.0f, b = 0.0f;
        if (cmap == COLORMAP_HEAT) {
            if (v < 0.25f) { b = v * 4.0f; }
            else if (v < 0.5f) { b = 1.0f; g = (v - 0.25f) * 4.0f; }
            else if (v < 0.75f) { b = 1.0f - (v - 0.5f) * 4.0f; g = 1.0f; r = (v - 0.5f) * 4.0f; }
            else { g = 1.0f - (v - 0.75f) * 4.0f; r = 1.0f; }
        } else if (cmap == COLORMAP_COLD) {
            if (v < 0.33f) { b = v * 3.0f; }
            else if (v < 0.66f) { b = 1.0f; g = (v - 0.33f) * 3.0f; }
            else { b = 1.0f; g = 1.0f; r = (v - 0.66f) * 3.0f; }
        } else if (cmap == COLORMAP_JET) {
            if (v < 0.25f) { b = 1.0f; g = v * 4.0f; }
            else if (v < 0.5f) { b = 1.0f - (v - 0.25f) * 4.0f; g = 1.0f; }
            else if (v < 0.75f) { g = 1.0f; r = (v - 0.5f) * 4.0f; }
            else { g = 1.0f - (v - 0.75f) * 4.0f; r = 1.0f; }
        } else if (cmap == COLORMAP_INFERNO) {
            if (v < 0.33f) { r = v * 1.5f; b = v * 2.0f; }
            else if (v < 0.66f) { r = 0.5f + (v-0.33f)*1.5f; g = (v-0.33f)*1.5f; b = 0.66f - (v-0.33f)*2.0f; }
            else { r = 1.0f; g = 0.5f + (v-0.66f)*1.5f; }
        } else if (cmap == COLORMAP_VIRIDIS) {
            if (v < 0.33f) { r = 0.2f; g = v * 1.5f; b = 0.4f + v * 1.2f; }
            else if (v < 0.66f) { r = 0.2f; g = 0.5f + (v-0.33f)*1.0f; b = 0.8f - (v-0.33f)*1.5f; }
            else { r = 0.2f + (v-0.66f)*2.4f; g = 0.83f + (v-0.66f)*0.5f; b = 0.3f - (v-0.66f)*0.9f; }
        } else if (cmap == COLORMAP_MAGMA) {
            if (v < 0.25f) { r = v*2.0f; g = 0.0f; b = v*2.0f; }
            else if (v < 0.5f) { r = 0.5f + (v-0.25f)*2.0f; g = 0.0f; b = 0.5f - (v-0.25f)*2.0f; }
            else if (v < 0.75f) { r = 1.0f; g = (v-0.5f)*2.0f; b = 0.0f; }
            else { r = 1.0f; g = 0.5f + (v-0.75f)*2.0f; b = (v-0.75f)*4.0f; }
        } else if (cmap == COLORMAP_PLASMA) {
            if (v < 0.33f) { r = v*1.5f; g = 0.0f; b = 0.5f + v*1.5f; }
            else if (v < 0.66f) { r = 0.5f + (v-0.33f)*1.5f; g = (v-0.33f)*1.5f; b = 1.0f - (v-0.33f)*1.5f; }
            else { r = 1.0f; g = 0.5f + (v-0.66f)*1.5f; b = 0.5f - (v-0.66f)*1.5f; }
        } else if (cmap == COLORMAP_BONE) {
            if (v < 0.33f) { r = v*1.5f; g = v*1.5f; b = v*2.0f; }
            else if (v < 0.66f) { r = 0.5f + (v-0.33f)*1.0f; g = 0.5f + (v-0.33f)*1.5f; b = 0.66f + (v-0.33f)*1.0f; }
            else { r = 0.83f + (v-0.66f)*0.5f; g = 1.0f; b = 1.0f; }
        } else if (cmap == COLORMAP_RAINBOW) {
            float h = (1.0f - v) * 5.0f; 
            int i = (int)h;
            float f = h - i;
            float q = 1.0f - f;
            switch (i) {
                case 0: r = 1.0f; g = f; b = 0.0f; break;
                case 1: r = q; g = 1.0f; b = 0.0f; break;
                case 2: r = 0.0f; g = 1.0f; b = f; break;
                case 3: r = 0.0f; g = q; b = 1.0f; break;
                case 4: r = f; g = 0.0f; b = 1.0f; break;
                case 5: r = 1.0f; g = 0.0f; b = 1.0f; break;
                default: r = 1.0f; g = 0.0f; b = 1.0f; break;
            }
        } else if (cmap == COLORMAP_TURBO) {
            float k1 = 0.1357f + v * (4.5974f - v * (42.3277f - v * (130.5887f - v * (150.5666f - v * 58.1375f))));
            float k2 = 0.0914f + v * (2.1941f + v * (4.8429f - v * (14.1850f - v * (4.2773f + v * 2.8251f))));
            float k3 = 0.1066f + v * (12.6419f - v * (60.5820f - v * (110.3627f - v * (89.9031f - v * 27.3482f))));
            r = k1; g = k2; b = k3;
        } else if (cmap == COLORMAP_OCEAN) {
            r = v * 0.5f; g = v * 0.8f; b = 0.2f + v * 0.8f;
        } else if (cmap == COLORMAP_COPPER) {
            r = v * 1.25f; g = v * 0.7812f; b = v * 0.4975f;
        } else if (cmap == COLORMAP_SPRING) {
            r = 1.0f; g = v; b = 1.0f - v;
        } else if (cmap == COLORMAP_SUMMER) {
            r = v; g = 0.5f + v * 0.5f; b = 0.4f;
        } else if (cmap == COLORMAP_AUTUMN) {
            r = 1.0f; g = v; b = 0.0f;
        } else if (cmap == COLORMAP_WINTER) {
            r = 0.0f; g = v; b = 1.0f - v * 0.5f;
        }
        if(r>1.0f) r=1.0f; if(r<0.0f) r=0.0f;
        if(g>1.0f) g=1.0f; if(g<0.0f) g=0.0f;
        if(b>1.0f) b=1.0f; if(b<0.0f) b=0.0f;
        c.r = (uint8_t)(r * 255.0f); c.g = (uint8_t)(g * 255.0f); c.b = (uint8_t)(b * 255.0f);
    }
    return c;
}

static void build_colormap_lut(termview_colormap_t cmap, termview_scale_t scale) {
    for (int i = 0; i < 1024; i++) {
        double v = i / 1023.0;
        if (scale == SCALE_SQRT) v = sqrt(v);
        else if (scale == SCALE_LOG) v = log(v * 9.0 + 1.0) / log(10.0);
        else if (scale == SCALE_LOG_STRONG) v = log(v * 99.0 + 1.0) / log(100.0);
        else if (scale == SCALE_LOG_EXTREME) v = log(v * 999.0 + 1.0) / log(1000.0);
        else if (scale == SCALE_ASINH) v = asinh(v * 10.0) / asinh(10.0);
        else if (scale == SCALE_SQUARED) v = v * v;
        else if (scale == SCALE_CUBED) v = v * v * v;
        
        colormap_lut[i] = get_colormap_color(cmap, v);
    }
    lut_colormap = cmap;
    lut_scale = scale;
}


/* -----------------------------------------------------------------------
 * Terminal Management
 * ----------------------------------------------------------------------- */
static struct termios tv_orig_termios;

static void tv_handle_sigwinch(int sig)
{
    (void)sig;
    tv_resize_flag = 1;
}

static void tv_exit_raw(void)
{
    tcsetattr(STDIN_FILENO, TCSAFLUSH, &tv_orig_termios);
}

static void tv_enter_raw(void)
{
    tcgetattr(STDIN_FILENO, &tv_orig_termios);
    atexit(tv_exit_raw);
    struct termios raw = tv_orig_termios;
    raw.c_iflag &= ~(BRKINT | ICRNL | INPCK | ISTRIP | IXON);
    raw.c_oflag &= ~(OPOST);
    raw.c_cflag |= (CS8);
    raw.c_lflag &= ~(ECHO | ICANON | IEXTEN | ISIG);
    raw.c_cc[VMIN] = 0;
    raw.c_cc[VTIME] = 0;
    tcsetattr(STDIN_FILENO, TCSAFLUSH, &raw);

    struct sigaction sa;
    memset(&sa, 0, sizeof(sa));
    sa.sa_handler = tv_handle_sigwinch;
    sigaction(SIGWINCH, &sa, NULL);
}

static void tv_enter_alt_screen(void)
{
    if(write(STDOUT_FILENO, "\033[?1049h\033[?25l\033[?1002h\033[?1006h", 30) < 0) {} // alt screen + hide cursor + mouse track
}

static void tv_exit_alt_screen(void)
{
    if(write(STDOUT_FILENO, "\033[?1006l\033[?1002l\033[?1049l\033[?25h", 30) < 0) {} // exit alt screen + show cursor + no mouse track
}

static void tv_get_size(
    unsigned short *rows,
    unsigned short *cols)
{
    struct winsize ws;
    if (ioctl(STDOUT_FILENO, TIOCGWINSZ, &ws) == -1 || ws.ws_col == 0) {
        *rows = 24;
        *cols = 80;
    } else {
        *rows = ws.ws_row;
        *cols = ws.ws_col;
    }
}

static int tv_read_key(long timeout_us)
{
    fd_set readfds;
    FD_ZERO(&readfds);
    FD_SET(STDIN_FILENO, &readfds);

    struct timeval tv;
    tv.tv_sec = timeout_us / 1000000;
    tv.tv_usec = timeout_us % 1000000;

    int ready = select(STDIN_FILENO + 1, &readfds, NULL, NULL, &tv);
    if (ready == -1 || ready == 0) {
        return -1;
    }

    char seq[32];
    int nread = read(STDIN_FILENO, &seq[0], 1);
    if (nread != 1) return -1;

    if (seq[0] == 27) {
        if (read(STDIN_FILENO, &seq[1], 1) != 1) return 27;
        if (read(STDIN_FILENO, &seq[2], 1) != 1) return 27;

        if (seq[1] == '[') {
            if (seq[2] == '<') {
                // Parse SGR mouse: \033[<B;X;Y[M|m]
                int b = 0, x = 0, y = 0;
                int type = 0; // 'M' or 'm'
                int state = 0; // 0: b, 1: x, 2: y, 3: type
                for (int i = 3; i < 31; i++) {
                    if (read(STDIN_FILENO, &seq[i], 1) != 1) return 27;
                    if (seq[i] == ';') {
                        state++;
                    } else if (seq[i] == 'M' || seq[i] == 'm') {
                        type = seq[i];
                        break;
                    } else {
                        if (state == 0) b = b * 10 + (seq[i] - '0');
                        else if (state == 1) x = x * 10 + (seq[i] - '0');
                        else if (state == 2) y = y * 10 + (seq[i] - '0');
                    }
                }
                last_mouse_event.is_drag = ((b & 32) != 0);
                if (b & 64) {
                    last_mouse_event.button = b; // 64 or 65
                    last_mouse_event.is_drag = 0;
                } else {
                    last_mouse_event.button = b & 3; // 0: Left, 1: Middle, 2: Right
                }
                last_mouse_event.x = x;
                last_mouse_event.y = y;
                last_mouse_event.is_press = (type == 'M');
                return TV_KEY_MOUSE;
            } else {
                switch (seq[2]) {
                    case 'A': force_redraw = 1; return TV_KEY_UP;
                    case 'B': force_redraw = 1; return TV_KEY_DOWN;
                    case 'C': force_redraw = 1; return TV_KEY_RIGHT;
                    case 'D': force_redraw = 1; return TV_KEY_LEFT;
                }
            }
        }
        return 27;
    } else {
        return seq[0];
    }
}

/* -----------------------------------------------------------------------
 * Data Retrieval & Colormaps
 * ----------------------------------------------------------------------- */
static inline double get_pixel_value(IMAGE *img, int x, int y) {
    long idx = y * img->md[0].size[0] + x;
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



static inline int compare_doubles(const void *a, const void *b) {
    double arg1 = *(const double *)a;
    double arg2 = *(const double *)b;
    if (arg1 < arg2) return -1;
    if (arg1 > arg2) return 1;
    return 0;
}

/* -----------------------------------------------------------------------
 * Main Screen Loop
 * ----------------------------------------------------------------------- */

/* -----------------------------------------------------------------------
 * Termview Refactored Context
 * ----------------------------------------------------------------------- */
typedef struct {
    IMAGE *img;
    int target_fps;
    double current_cpu_usage;
    double current_stream_fps;
    uint64_t last_cnt0_for_fps;
    uint64_t last_cnt0;
    struct timespec last_time_real;
    struct timespec last_time_cpu;
    struct timespec last_render_real;
    
    int popup_type;
    struct timespec popup_expiry_time;

    int mouse_is_dragging;
    int roi_is_dragging;
    int last_mouse_x;
    int last_mouse_y;
    int roi_start_x;
    int roi_start_y;
    int roi_end_x;
    int roi_end_y;
    int input_mode;
    char input_buf[64];
    int input_len;

    double *display_buffer;
    int buffer_size;
    tv_cell_t *screen;
    tv_cell_t *prev_screen;
    int screen_size;
    char *frame_buffer;
    size_t frame_buffer_size;
    
    long timeout_us;
} tv_context_t;


static void termview_update_fps_stats(tv_context_t *ctx) {
    clock_gettime(CLOCK_MONOTONIC, &ctx->last_render_real);
    double real_diff = (ctx->last_render_real.tv_sec - ctx->last_time_real.tv_sec) + 
                       (ctx->last_render_real.tv_nsec - ctx->last_time_real.tv_nsec) / 1e9;
    if (real_diff >= 1.0) {
        struct timespec now_cpu;
        clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &now_cpu);
        double cpu_diff = (now_cpu.tv_sec - ctx->last_time_cpu.tv_sec) + 
                          (now_cpu.tv_nsec - ctx->last_time_cpu.tv_nsec) / 1e9;
        ctx->current_cpu_usage = (cpu_diff / real_diff) * 100.0;
        
        uint64_t current_cnt0 = ctx->img->md[0].cnt0;
        ctx->current_stream_fps = (current_cnt0 - ctx->last_cnt0_for_fps) / real_diff;
        ctx->last_cnt0_for_fps = current_cnt0;
        
        ctx->last_time_real = ctx->last_render_real;
        ctx->last_time_cpu = now_cpu;
    }
}

static void termview_handle_mouse_event(tv_context_t *ctx) {
    force_redraw = 1;
    int mx = last_mouse_event.x;
    int my = last_mouse_event.y;
    
    uint32_t xsize = ctx->img->md[0].size[0];
    uint32_t ysize = ctx->img->md[0].size[1];
    int bar_width = 12;
    int disp_char_rows = wrow - 4; // info bar is 4 rows
    int disp_cols = wcol - bar_width - 1;
    int disp_img_rows = disp_char_rows * 2;

    // Calculate mapping constants identically to the render loop
    double view_w_img = (double)xsize / view_zoom;
    double view_h_img = (double)ysize / view_zoom;
    double step = fmax(view_w_img / disp_cols, view_h_img / disp_img_rows);
    double center_img_x = view_center_x * xsize;
    double center_img_y = view_center_y * ysize;
    double center_disp_x = disp_cols / 2.0;
    double center_disp_y = disp_img_rows / 2.0;

    if (last_mouse_event.button == 64) {
        // Scroll Up -> Zoom in
        view_zoom *= 1.2;
    } else if (last_mouse_event.button == 65) {
        // Scroll Down -> Zoom out
        view_zoom /= 1.2;
        if (view_zoom < 0.1) view_zoom = 0.1;
    } else if (last_mouse_event.button == 0) { // Left Button
        if (last_mouse_event.is_press && !last_mouse_event.is_drag) {
            ctx->mouse_is_dragging = 1;
            ctx->last_mouse_x = mx;
            ctx->last_mouse_y = my;
        } else if (last_mouse_event.is_press && last_mouse_event.is_drag && ctx->mouse_is_dragging) {
            int dx = mx - ctx->last_mouse_x;
            int dy = my - ctx->last_mouse_y;
            
            double dx_pixels = -dx * step;
            double dy_pixels = -dy * 2 * step;
            
            view_center_x += dx_pixels / xsize;
            view_center_y += dy_pixels / ysize;
            
            if (view_center_x < 0.0) view_center_x = 0.0;
            if (view_center_x > 1.0) view_center_x = 1.0;
            if (view_center_y < 0.0) view_center_y = 0.0;
            if (view_center_y > 1.0) view_center_y = 1.0;
            
            ctx->last_mouse_x = mx;
            ctx->last_mouse_y = my;
        } else if (!last_mouse_event.is_press) {
            ctx->mouse_is_dragging = 0;
        }
    } else if (last_mouse_event.button == 2) { // Right Button
        if (last_mouse_event.is_press && !last_mouse_event.is_drag) {
            ctx->roi_is_dragging = 1;
            ctx->roi_start_x = mx;
            ctx->roi_start_y = my;
            ctx->roi_end_x = mx;
            ctx->roi_end_y = my;
        } else if (last_mouse_event.is_press && last_mouse_event.is_drag && ctx->roi_is_dragging) {
            ctx->roi_end_x = mx;
            ctx->roi_end_y = my;
        } else if (!last_mouse_event.is_press && ctx->roi_is_dragging) {
            ctx->roi_end_x = mx;
            ctx->roi_end_y = my;
            ctx->roi_is_dragging = 0;
            
            int min_x = (ctx->roi_start_x < ctx->roi_end_x) ? ctx->roi_start_x : ctx->roi_end_x;
            int max_x = (ctx->roi_start_x > ctx->roi_end_x) ? ctx->roi_start_x : ctx->roi_end_x;
            int min_y = (ctx->roi_start_y < ctx->roi_end_y) ? ctx->roi_start_y : ctx->roi_end_y;
            int max_y = (ctx->roi_start_y > ctx->roi_end_y) ? ctx->roi_start_y : ctx->roi_end_y;
            
            if (max_x - min_x >= 2 && max_y - min_y >= 2) {
                double roi_center_mx = (min_x + max_x) / 2.0;
                double roi_center_my = (min_y + max_y) / 2.0;
                
                double new_cx_pixel = center_img_x + (roi_center_mx - 1.0 - center_disp_x) * step;
                double roi_center_img_row = (roi_center_my - 1.0) * 2.0 + 0.5;
                double new_cy_pixel = center_img_y + (roi_center_img_row - center_disp_y) * step;
                
                double roi_w_char = max_x - min_x + 1;
                double roi_h_char = max_y - min_y + 1;
                double new_step_x = (roi_w_char * step) / disp_cols;
                double new_step_y = (roi_h_char * 2 * step) / disp_img_rows;
                double new_step = fmax(new_step_x, new_step_y);
                
                view_center_x = new_cx_pixel / xsize;
                view_center_y = new_cy_pixel / ysize;
                view_zoom *= (step / new_step);
                
                if (view_center_x < 0.0) view_center_x = 0.0;
                if (view_center_x > 1.0) view_center_x = 1.0;
                if (view_center_y < 0.0) view_center_y = 0.0;
                if (view_center_y > 1.0) view_center_y = 1.0;
            }
        }
    }
}

static void termview_handle_keyboard_event(tv_context_t *ctx, int ch) {
    switch(ch) {
        case 10: // ENTER key (LF)
        case 13: // ENTER key (CR)
        case 27: // ESC key
            if (ctx->popup_type > 0) {
                ctx->popup_type = 0;
                force_redraw = 1;
            }
            break;
        case 'q': force_redraw = 1; loop = 0; break;
        case 'c': force_redraw = 1; 
            current_options.colormap = (current_options.colormap + 1) % COLORMAP_NB; 
            ctx->popup_type = 1;
            clock_gettime(CLOCK_MONOTONIC, &ctx->popup_expiry_time);
            ctx->popup_expiry_time.tv_sec += 2;
            break;
        case 's': force_redraw = 1; 
            current_options.scale = (current_options.scale + 1) % SCALE_NB; 
            ctx->popup_type = 2;
            clock_gettime(CLOCK_MONOTONIC, &ctx->popup_expiry_time);
            ctx->popup_expiry_time.tv_sec += 2;
            break;
        case 'r': force_redraw = 1;
            current_options.range = (current_options.range + 1) % RANGE_NB;
            ctx->popup_type = 3;
            clock_gettime(CLOCK_MONOTONIC, &ctx->popup_expiry_time);
            ctx->popup_expiry_time.tv_sec += 2;
            break;
        case 'l': force_redraw = 1;
            current_options.range_locked = !current_options.range_locked;
            if (current_options.range_locked) {
                current_options.manual_min = current_min_val;
                current_options.manual_max = current_max_val;
            }
            break;
        case 'm': force_redraw = 1; ctx->input_mode = 1; ctx->input_len = 0; ctx->input_buf[0] = '\0'; break;
        case 'M': force_redraw = 1; ctx->input_mode = 2; ctx->input_len = 0; ctx->input_buf[0] = '\0'; break;
        case 'h': force_redraw = 1; current_options.flip_h = !current_options.flip_h; break;
        case 'v': force_redraw = 1; current_options.flip_v = !current_options.flip_v; break;
        case '+': force_redraw = 1;
        case '=': force_redraw = 1; view_zoom *= 1.2; break;
        case '-': force_redraw = 1;
        case '_': force_redraw = 1;
            view_zoom /= 1.2;
            if (view_zoom < 0.1) view_zoom = 0.1;
            break;
        case '0': force_redraw = 1;
            view_zoom = 1.0;
            view_center_x = 0.5;
            view_center_y = 0.5;
            break;
        case TV_KEY_LEFT: force_redraw = 1;
            view_center_x -= 0.1 / view_zoom;
            if (view_center_x < 0.0) view_center_x = 0.0;
            break;
        case TV_KEY_RIGHT: force_redraw = 1;
            view_center_x += 0.1 / view_zoom;
            if (view_center_x > 1.0) view_center_x = 1.0;
            break;
        case TV_KEY_UP: force_redraw = 1;
            if (ctx->popup_type == 1) {
                current_options.colormap = (current_options.colormap + COLORMAP_NB - 1) % COLORMAP_NB;
                clock_gettime(CLOCK_MONOTONIC, &ctx->popup_expiry_time);
                ctx->popup_expiry_time.tv_sec += 2;
            } else if (ctx->popup_type == 2) {
                current_options.scale = (current_options.scale + SCALE_NB - 1) % SCALE_NB;
                clock_gettime(CLOCK_MONOTONIC, &ctx->popup_expiry_time);
                ctx->popup_expiry_time.tv_sec += 2;
            } else if (ctx->popup_type == 3) {
                current_options.range = (current_options.range + RANGE_NB - 1) % RANGE_NB;
                clock_gettime(CLOCK_MONOTONIC, &ctx->popup_expiry_time);
                ctx->popup_expiry_time.tv_sec += 2;
            } else {
                view_center_y -= 0.1 / view_zoom;
                if (view_center_y < 0.0) view_center_y = 0.0;
            }
            break;
        case TV_KEY_DOWN: force_redraw = 1;
            if (ctx->popup_type == 1) {
                current_options.colormap = (current_options.colormap + 1) % COLORMAP_NB;
                clock_gettime(CLOCK_MONOTONIC, &ctx->popup_expiry_time);
                ctx->popup_expiry_time.tv_sec += 2;
            } else if (ctx->popup_type == 2) {
                current_options.scale = (current_options.scale + 1) % SCALE_NB;
                clock_gettime(CLOCK_MONOTONIC, &ctx->popup_expiry_time);
                ctx->popup_expiry_time.tv_sec += 2;
            } else if (ctx->popup_type == 3) {
                current_options.range = (current_options.range + 1) % RANGE_NB;
                clock_gettime(CLOCK_MONOTONIC, &ctx->popup_expiry_time);
                ctx->popup_expiry_time.tv_sec += 2;
            } else {
                view_center_y += 0.1 / view_zoom;
                if (view_center_y > 1.0) view_center_y = 1.0;
            }
            break;
        case '<': force_redraw = 1;
        case ',': force_redraw = 1;
            ctx->target_fps -= 1;
            if (ctx->target_fps < 1) ctx->target_fps = 1;
            break;
        case '>': force_redraw = 1;
        case '.': force_redraw = 1;
            ctx->target_fps += 1;
            if (ctx->target_fps > 120) ctx->target_fps = 120;
            break;
        case TV_KEY_MOUSE: 
            termview_handle_mouse_event(ctx);
            break;
    }
}

static void termview_render_image(tv_context_t *ctx, int disp_char_rows, int disp_cols, double center_img_x, double center_img_y, double sign_x, double sign_y, double step, uint32_t xsize, uint32_t ysize, double min_val, double max_val) {
#define IS_IN_BOUNDS(jj, yy) \
    ( ((int)(center_img_x + sign_x * ((jj) - center_disp_x) * step) >= 0) && \
      ((int)(center_img_x + sign_x * ((jj) - center_disp_x) * step) < (int)xsize) && \
      ((int)(center_img_y + sign_y * ((yy) - center_disp_y) * step) >= 0) && \
      ((int)(center_img_y + sign_y * ((yy) - center_disp_y) * step) < (int)ysize) )

    double center_disp_x = disp_cols / 2.0;
    double center_disp_y = (disp_char_rows * 2) / 2.0;

    for (int i = 0; i < disp_char_rows; i++) {
        for (int j = 0; j < disp_cols; j++) {
            int img_y_top = (int)(center_img_y + sign_y * ((i*2) - center_disp_y) * step);
            int img_x_top = (int)(center_img_x + sign_x * (j - center_disp_x) * step);
            bool in_bounds_top = (img_x_top >= 0 && img_x_top < (int)xsize && img_y_top >= 0 && img_y_top < (int)ysize);

            int img_y_bot = (int)(center_img_y + sign_y * ((i*2+1) - center_disp_y) * step);
            int img_x_bot = (int)(center_img_x + sign_x * (j - center_disp_x) * step);
            bool in_bounds_bot = (img_x_bot >= 0 && img_x_bot < (int)xsize && img_y_bot >= 0 && img_y_bot < (int)ysize);

            tv_cell_t *cell = &ctx->screen[i * wcol + j];

            if (!in_bounds_top && !in_bounds_bot) {
                bool near_top = IS_IN_BOUNDS(j-1, i*2) || IS_IN_BOUNDS(j+1, i*2) || IS_IN_BOUNDS(j, i*2-1) || IS_IN_BOUNDS(j, i*2+1);
                bool near_bot = IS_IN_BOUNDS(j-1, i*2+1) || IS_IN_BOUNDS(j+1, i*2+1) || IS_IN_BOUNDS(j, i*2) || IS_IN_BOUNDS(j, i*2+2);
                if (!near_top && !near_bot) continue; // left as space
            }

            double val_top = in_bounds_top ? ctx->display_buffer[(i*2)*disp_cols + j] : 0.0;
            double val_bot = in_bounds_bot ? ctx->display_buffer[(i*2+1)*disp_cols + j] : 0.0;

            double norm_top = (val_top - min_val) / (max_val - min_val);
            if (norm_top < 0) norm_top = 0; if (norm_top > 1) norm_top = 1;
            
            double norm_bot = (val_bot - min_val) / (max_val - min_val);
            if (norm_bot < 0) norm_bot = 0; if (norm_bot > 1) norm_bot = 1;

            rgb_t rgb_top = colormap_lut[(int)(norm_top * 1023)];
            rgb_t rgb_bot = colormap_lut[(int)(norm_bot * 1023)];

            if (!in_bounds_top) {
                if (IS_IN_BOUNDS(j-1, i*2) || IS_IN_BOUNDS(j+1, i*2) || IS_IN_BOUNDS(j, i*2-1) || IS_IN_BOUNDS(j, i*2+1)) {
                    rgb_top = (rgb_t){64, 64, 64}; // Dark gray border
                } else {
                    rgb_top = (rgb_t){0,0,0};
                }
            }

            if (!in_bounds_bot) {
                if (IS_IN_BOUNDS(j-1, i*2+1) || IS_IN_BOUNDS(j+1, i*2+1) || IS_IN_BOUNDS(j, i*2) || IS_IN_BOUNDS(j, i*2+2)) {
                    rgb_bot = (rgb_t){64, 64, 64}; // Dark gray border
                } else {
                    rgb_bot = (rgb_t){0,0,0};
                }
            }

            if (ctx->roi_is_dragging) {
                int rx_min = (ctx->roi_start_x < ctx->roi_end_x) ? ctx->roi_start_x : ctx->roi_end_x;
                int rx_max = (ctx->roi_start_x > ctx->roi_end_x) ? ctx->roi_start_x : ctx->roi_end_x;
                int ry_min = (ctx->roi_start_y < ctx->roi_end_y) ? ctx->roi_start_y : ctx->roi_end_y;
                int ry_max = (ctx->roi_start_y > ctx->roi_end_y) ? ctx->roi_start_y : ctx->roi_end_y;
                
                int term_x = j + 1;
                int term_y = i + 1;
                
                if (term_x >= rx_min && term_x <= rx_max && term_y >= ry_min && term_y <= ry_max) {
                    if (term_x == rx_min || term_x == rx_max || term_y == ry_min || term_y == ry_max) {
                        rgb_top = (rgb_t){255, 255, 0}; // Yellow border
                        rgb_bot = (rgb_t){255, 255, 0};
                    } else {
                        // Dim the inside of the ROI
                        rgb_top.r /= 2; rgb_top.g /= 2; rgb_top.b /= 2;
                        rgb_bot.r /= 2; rgb_bot.g /= 2; rgb_bot.b /= 2;
                    }
                }
            }

            cell->bg_r = rgb_top.r; cell->bg_g = rgb_top.g; cell->bg_b = rgb_top.b;
            cell->fg_r = rgb_bot.r; cell->fg_g = rgb_bot.g; cell->fg_b = rgb_bot.b;
            cell->ch[0] = (char)0xE2; cell->ch[1] = (char)0x96; cell->ch[2] = (char)0x84; cell->ch[3] = '\0';
        }
    }
#undef IS_IN_BOUNDS
}

static void termview_render_colorbar(tv_context_t *ctx, int disp_char_rows, int bar_col_start, double min_val, double max_val) {
    for (int i = 0; i < disp_char_rows; i++) {
        double norm_val = 1.0 - (double)i / (disp_char_rows - 1);
        rgb_t rgb = colormap_lut[(int)(norm_val * 1023)];
        
        tv_cell_t *c1 = &ctx->screen[i * wcol + bar_col_start - 1];
        tv_cell_t *c2 = &ctx->screen[i * wcol + bar_col_start];
        c1->fg_r = rgb.r; c1->fg_g = rgb.g; c1->fg_b = rgb.b;
        c1->bg_r = 0; c1->bg_g = 0; c1->bg_b = 0;
        c1->ch[0] = (char)0xE2; c1->ch[1] = (char)0x96; c1->ch[2] = (char)0x88; c1->ch[3] = '\0';
        
        c2->fg_r = rgb.r; c2->fg_g = rgb.g; c2->fg_b = rgb.b;
        c2->bg_r = 0; c2->bg_g = 0; c2->bg_b = 0;
        c2->ch[0] = (char)0xE2; c2->ch[1] = (char)0x96; c2->ch[2] = (char)0x88; c2->ch[3] = '\0';

        char label[32];
        int label_len = 0;
        if (i == 0) label_len = snprintf(label, sizeof(label), "%.2g", max_val);
        else if (i == disp_char_rows/2) label_len = snprintf(label, sizeof(label), "%.2g", (min_val+max_val)/2);
        else if (i == disp_char_rows-1) label_len = snprintf(label, sizeof(label), "%.2g", min_val);

        for(int k = 0; k < label_len; k++) {
            if (bar_col_start + 2 + k < wcol) {
                tv_cell_t *lc = &ctx->screen[i * wcol + bar_col_start + 2 + k];
                lc->ch[0] = label[k]; lc->ch[1] = '\0';
            }
        }
    }
}

static void termview_render_infobar(tv_context_t *ctx, uint32_t xsize, uint32_t ysize, double step, double min_val, double max_val) {
    int info_row = wrow - 3;
    
    const char* cmap_str = "GREYSCALE";
    switch(current_options.colormap) {
        case COLORMAP_HEAT: cmap_str = "HEAT"; break;
        case COLORMAP_COLD: cmap_str = "COLD"; break;
        case COLORMAP_JET: cmap_str = "JET"; break;
        case COLORMAP_INFERNO: cmap_str = "INFERNO"; break;
        case COLORMAP_VIRIDIS: cmap_str = "VIRIDIS"; break;
        case COLORMAP_MAGMA: cmap_str = "MAGMA"; break;
        case COLORMAP_PLASMA: cmap_str = "PLASMA"; break;
        case COLORMAP_BONE: cmap_str = "BONE"; break;
        case COLORMAP_RAINBOW: cmap_str = "RAINBOW"; break;
        case COLORMAP_TURBO: cmap_str = "TURBO"; break;
        case COLORMAP_OCEAN: cmap_str = "OCEAN"; break;
        case COLORMAP_COPPER: cmap_str = "COPPER"; break;
        case COLORMAP_SPRING: cmap_str = "SPRING"; break;
        case COLORMAP_SUMMER: cmap_str = "SUMMER"; break;
        case COLORMAP_AUTUMN: cmap_str = "AUTUMN"; break;
        case COLORMAP_WINTER: cmap_str = "WINTER"; break;
        default: break;
    }
    const char* scale_str = "LIN";
    switch(current_options.scale) {
        case SCALE_SQRT: scale_str = "SQRT"; break;
        case SCALE_LOG: scale_str = "LOG"; break;
        case SCALE_LOG_STRONG: scale_str = "LOG_STR"; break;
        case SCALE_LOG_EXTREME: scale_str = "LOG_EXT"; break;
        case SCALE_ASINH: scale_str = "ASINH"; break;
        case SCALE_SQUARED: scale_str = "SQUARED"; break;
        case SCALE_CUBED: scale_str = "CUBED"; break;
        default: break;
    }
    const char* range_str = "MINMAX";
    switch(current_options.range) {
        case RANGE_001_999: range_str = "0.1-99.9%"; break;
        case RANGE_005_995: range_str = "0.5-99.5%"; break;
        case RANGE_01_99: range_str = "1-99%"; break;
        case RANGE_05_95: range_str = "5-95%"; break;
        case RANGE_10_90: range_str = "10-90%"; break;
        case RANGE_15_85: range_str = "15-85%"; break;
        case RANGE_20_80: range_str = "20-80%"; break;
        default: break;
    }

    char info[256];
    int len;
    
    len = snprintf(info, sizeof(info), "Image: %s [%d x %d] Type: %d  |  CPU: %5.1f%%  UI: %3d FPS  Stream: %5.1f Hz", 
                       ctx->img->md[0].name, xsize, ysize, ctx->img->md[0].datatype, ctx->current_cpu_usage, ctx->target_fps, ctx->current_stream_fps);
    for(int k=0; k<len && k<wcol; k++) { 
        tv_cell_t *c = &ctx->screen[info_row * wcol + k];
        c->ch[0] = info[k]; 
        c->ch[1] = '\0';
        int name_len = strlen(ctx->img->md[0].name);
        if (k >= 7 && k < 7 + name_len) {
            c->fg_r = 0; c->fg_g = 255; c->fg_b = 255; // Cyan
        }
    }
    len = snprintf(info, sizeof(info), "Val: [%.4g : %.4g] Zoom: %.2fx (%.2f px/char H, %.2f px/char V)", min_val, max_val, view_zoom, step, step * 2.0);
    for(int k=0; k<len && k<wcol; k++) { ctx->screen[(info_row+1) * wcol + k].ch[0] = info[k]; ctx->screen[(info_row+1) * wcol + k].ch[1] = '\0'; }

    if (ctx->input_mode > 0) {
        len = snprintf(info, sizeof(info), "Enter Manual %s: %s_", ctx->input_mode == 1 ? "Min" : "Max", ctx->input_buf);
        for(int k=0; k<len && k<wcol; k++) { 
            tv_cell_t *c = &ctx->screen[(info_row+2) * wcol + k];
            c->ch[0] = info[k]; c->ch[1] = '\0';
            c->fg_r = 255; c->fg_g = 255; c->fg_b = 0; // Yellow text
        }
        for(int k=len; k<wcol; k++) {
            ctx->screen[(info_row+2) * wcol + k].ch[0] = ' '; ctx->screen[(info_row+2) * wcol + k].ch[1] = '\0';
        }
    } else {
        char info1[128];
        int len1 = snprintf(info1, sizeof(info1), "[C]map: %s  [S]cale: %s  [R]ange: %s [", cmap_str, scale_str, range_str);
        const char* lock_str = current_options.range_locked ? "LOCKED" : "AUTO";
        int lock_len = strlen(lock_str);
        char info2[256];
        int len2 = snprintf(info2, sizeof(info2), "%s%s]  (< >:FPS l:Lock m/M:MinMax ", info1, lock_str);
        
        const char* h_str = current_options.flip_h ? "h:FLIPH" : "h:fliph";
        const char* v_str = current_options.flip_v ? "v:FLIPV" : "v:flipv";
        
        len = snprintf(info, sizeof(info), "%s%s %s q:Quit)", info2, h_str, v_str);

        for(int k=0; k<len && k<wcol; k++) { 
            tv_cell_t *c = &ctx->screen[(info_row+2) * wcol + k];
            c->ch[0] = info[k]; c->ch[1] = '\0';
            if (k >= len1 && k < len1 + lock_len) {
                if (current_options.range_locked) {
                    c->fg_r = 255; c->fg_g = 50; c->fg_b = 50; // Red for Locked
                } else {
                    c->fg_r = 50; c->fg_g = 255; c->fg_b = 50; // Green for Auto
                }
            }
            if (current_options.flip_h && k >= len2 && k < len2 + (int)strlen(h_str)) {
                c->fg_r = 255; c->fg_g = 255; c->fg_b = 0; // Yellow
            }
            if (current_options.flip_v && k >= len2 + (int)strlen(h_str) + 1 && k < len2 + (int)strlen(h_str) + 1 + (int)strlen(v_str)) {
                c->fg_r = 255; c->fg_g = 255; c->fg_b = 0; // Yellow
            }
        }
        for(int k=len; k<wcol; k++) {
            ctx->screen[(info_row+2) * wcol + k].ch[0] = ' '; ctx->screen[(info_row+2) * wcol + k].ch[1] = '\0';
        }
    }
}

static void termview_render_popup(tv_context_t *ctx, int disp_char_rows, int disp_cols) {
    if (ctx->popup_type <= 0) return;
    int num_opts = 0;
    const char* title = "";
    const char* opts[32];
    int selected_idx = 0;
    int box_w = 16;
    if (ctx->popup_type == 1) {
        title = "COLORMAP";
        opts[0] = "GREYSCALE"; opts[1] = "HEAT"; opts[2] = "COLD"; opts[3] = "JET"; opts[4] = "INFERNO"; opts[5] = "VIRIDIS";
        opts[6] = "MAGMA"; opts[7] = "PLASMA"; opts[8] = "BONE";
        opts[9] = "RAINBOW"; opts[10] = "TURBO"; opts[11] = "OCEAN"; opts[12] = "COPPER"; opts[13] = "SPRING";
        opts[14] = "SUMMER"; opts[15] = "AUTUMN"; opts[16] = "WINTER";
        num_opts = COLORMAP_NB;
        selected_idx = current_options.colormap;
        box_w = 40; // Wider box to fit colormap preview
    } else if (ctx->popup_type == 2) {
        title = "SCALE";
        opts[0] = "LINEAR"; opts[1] = "SQRT"; opts[2] = "LOG"; opts[3] = "LOG_STRONG"; opts[4] = "LOG_EXTREME"; opts[5] = "ASINH"; opts[6] = "SQUARED"; opts[7] = "CUBED";
        num_opts = SCALE_NB;
        selected_idx = current_options.scale;
    } else if (ctx->popup_type == 3) {
        title = "RANGE";
        opts[0] = "MINMAX"; opts[1] = "0.1-99.9%"; opts[2] = "0.5-99.5%"; opts[3] = "1-99%"; opts[4] = "5-95%"; opts[5] = "10-90%"; opts[6] = "15-85%"; opts[7] = "20-80%";
        num_opts = RANGE_NB;
        selected_idx = current_options.range;
    }
    
    int box_h = num_opts + 2;
    int start_r = (disp_char_rows - box_h) / 2;
    int start_c = (disp_cols - box_w) / 2;
    if (start_r < 0) start_r = 0;
    if (start_c < 0) start_c = 0;
    
    for (int i = 0; i < box_h; i++) {
        int r = start_r + i;
        if (r >= disp_char_rows) break;
        for (int j = 0; j < box_w; j++) {
            int c = start_c + j;
            if (c >= disp_cols) break;
            
            tv_cell_t *cell = &ctx->screen[r * wcol + c];
            cell->bg_r = 50; cell->bg_g = 50; cell->bg_b = 50; // dark gray bg
            cell->fg_r = 255; cell->fg_g = 255; cell->fg_b = 255; // white fg
            cell->ch[0] = ' '; cell->ch[1] = '\0';
            
            if (i == 0) {
                int tlen = strlen(title);
                int tstart = (box_w - tlen) / 2;
                if (j >= tstart && j < tstart + tlen) {
                    cell->ch[0] = title[j - tstart];
                    cell->fg_r = 255; cell->fg_g = 255; cell->fg_b = 0; // Yellow title
                }
            } else if (i <= num_opts) {
                int opt_idx = i - 1;
                int is_sel = (opt_idx == selected_idx);
                if (is_sel) {
                    cell->bg_r = 200; cell->bg_g = 200; cell->bg_b = 200; // light gray selected
                    cell->fg_r = 0; cell->fg_g = 0; cell->fg_b = 0; // black fg
                }
                
                // Text portion
                if (j >= 2 && j - 2 < (int)strlen(opts[opt_idx])) {
                    cell->ch[0] = opts[opt_idx][j - 2];
                }
                
                // Compressed colormap rendering
                if (ctx->popup_type == 1 && j >= 16 && j < box_w - 2) {
                    int bar_len = box_w - 2 - 16;
                    double v = (double)(j - 16) / (double)(bar_len - 1);
                    rgb_t color = get_colormap_color((termview_colormap_t)opt_idx, v);
                    cell->bg_r = color.r; cell->bg_g = color.g; cell->bg_b = color.b;
                    cell->ch[0] = ' '; cell->ch[1] = '\0';
                }
            }
        }
    }
}

static void termview_flush_ansi(tv_context_t *ctx) {
    size_t fb_pos = 0;
    int cur_r = -1, cur_c = -1;
    int last_bg_r = -1, last_bg_g = -1, last_bg_b = -1;
    int last_fg_r = -1, last_fg_g = -1, last_fg_b = -1;

    for (int r = 0; r < wrow; r++) {
        for (int c = 0; c < wcol; c++) {
            tv_cell_t *cell = &ctx->screen[r * wcol + c];
            tv_cell_t *prev = &ctx->prev_screen[r * wcol + c];
            
            if (memcmp(cell, prev, sizeof(tv_cell_t)) != 0) {
                // Update cursor if not sequential
                if (cur_r != r || cur_c != c) {
                    tv_move(ctx->frame_buffer, &fb_pos, r + 1, c + 1);
                }
                
                // Update BG
                if (cell->bg_r != last_bg_r || cell->bg_g != last_bg_g || cell->bg_b != last_bg_b) {
                    tv_bg(ctx->frame_buffer, &fb_pos, cell->bg_r, cell->bg_g, cell->bg_b);
                    last_bg_r = cell->bg_r; last_bg_g = cell->bg_g; last_bg_b = cell->bg_b;
                }
                
                // Update FG
                if (cell->fg_r != last_fg_r || cell->fg_g != last_fg_g || cell->fg_b != last_fg_b) {
                    tv_fg(ctx->frame_buffer, &fb_pos, cell->fg_r, cell->fg_g, cell->fg_b);
                    last_fg_r = cell->fg_r; last_fg_g = cell->fg_g; last_fg_b = cell->fg_b;
                }

                // Write Char
                int k = 0;
                while(cell->ch[k] != '\0' && k < 4) {
                    ctx->frame_buffer[fb_pos++] = cell->ch[k++];
                }
                
                cur_r = r;
                cur_c = c + 1;
                *prev = *cell;
            }
        }
    }

    tv_reset(ctx->frame_buffer, &fb_pos);

    if (fb_pos > 0) {
        if(write(STDOUT_FILENO, ctx->frame_buffer, fb_pos) < 0) {}
    }
}

errno_t termview_screen(
    const char *imagename,
    termview_options_t options)
{
    IMAGE img;
    ImageStreamIO_read_sharedmem_image_toIMAGE(imagename, &img);

    if (img.md == NULL) {
        printf("Error: Could not connect to image %s\n", imagename);
        return 1;
    }

    current_options = options;

    tv_enter_raw();
    tv_enter_alt_screen();
    tv_get_size(&wrow, &wcol);

    tv_context_t context_instance = {0};
    tv_context_t *ctx = &context_instance;
    ctx->img = &img;
    ctx->target_fps = 20;
    ctx->last_cnt0 = (uint64_t)-1;
    ctx->last_cnt0_for_fps = img.md[0].cnt0;

    clock_gettime(CLOCK_MONOTONIC, &ctx->last_time_real);
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &ctx->last_time_cpu);
    ctx->last_render_real = ctx->last_time_real;

    while(loop) {
        if (tv_resize_flag) {
            tv_get_size(&wrow, &wcol);
            tv_resize_flag = 0;
            force_redraw = 1;
            if (ctx->prev_screen) {
                memset(ctx->prev_screen, 0, ctx->screen_size * sizeof(tv_cell_t));
            }
            if(write(STDOUT_FILENO, "\033[2J\033[H", 7) < 0) {}
        }

        long target_us = 1000000L / ctx->target_fps;
        struct timespec now_real;
        clock_gettime(CLOCK_MONOTONIC, &now_real);
        long elapsed_since_last = (now_real.tv_sec - ctx->last_render_real.tv_sec) * 1000000L + 
                                  (now_real.tv_nsec - ctx->last_render_real.tv_nsec) / 1000L;
        ctx->timeout_us = target_us - elapsed_since_last;
        if (ctx->timeout_us < 0) ctx->timeout_us = 0;

        if (ctx->popup_type > 0) {
            struct timespec now;
            clock_gettime(CLOCK_MONOTONIC, &now);
            long popup_remain_us = (ctx->popup_expiry_time.tv_sec - now.tv_sec) * 1000000L + 
                                   (ctx->popup_expiry_time.tv_nsec - now.tv_nsec) / 1000L;
            if (popup_remain_us <= 0) {
                ctx->popup_type = 0;
                force_redraw = 1;
            } else if (popup_remain_us < ctx->timeout_us) {
                ctx->timeout_us = popup_remain_us;
            }
        }

        int ch;
        int first_wait = 1;
        while (1) {
            long wait_us = first_wait ? ctx->timeout_us : 0;
            ch = tv_read_key(wait_us);
            first_wait = 0;
            if (ch == -1) break;

            if (ctx->input_mode > 0) {
                if (ch == 10 || ch == 13 || ch == 27) {
                    if (ch != 27 && ctx->input_len > 0) {
                        ctx->input_buf[ctx->input_len] = '\0';
                        if (ctx->input_mode == 1) {
                            current_options.manual_min = atof(ctx->input_buf);
                            current_options.range_locked = 1;
                        } else if (ctx->input_mode == 2) {
                            current_options.manual_max = atof(ctx->input_buf);
                            current_options.range_locked = 1;
                        }
                    }
                    ctx->input_mode = 0;
                    ctx->input_len = 0;
                    ctx->input_buf[0] = '\0';
                    force_redraw = 1;
                } else if (ch == 127 || ch == 8) {
                    if (ctx->input_len > 0) {
                        ctx->input_len--;
                        ctx->input_buf[ctx->input_len] = '\0';
                        force_redraw = 1;
                    }
                } else if (ch >= 32 && ch <= 126 && ctx->input_len < 63) {
                    ctx->input_buf[ctx->input_len++] = ch;
                    ctx->input_buf[ctx->input_len] = '\0';
                    force_redraw = 1;
                }
                continue;
            }

            termview_handle_keyboard_event(ctx, ch);
        }
        if (loop == 0) break;

        if (ctx->popup_type > 0) {
            struct timespec now;
            clock_gettime(CLOCK_MONOTONIC, &now);
            long popup_remain_us = (ctx->popup_expiry_time.tv_sec - now.tv_sec) * 1000000L + 
                                   (ctx->popup_expiry_time.tv_nsec - now.tv_nsec) / 1000L;
            if (popup_remain_us <= 0) {
                ctx->popup_type = 0;
                force_redraw = 1;
            }
        }
        
        termview_update_fps_stats(ctx);
        
        if (!force_redraw && img.md[0].cnt0 == ctx->last_cnt0) {
            continue;
        }
        ctx->last_cnt0 = img.md[0].cnt0;
        force_redraw = 0;

        uint32_t xsize = img.md[0].size[0];
        uint32_t ysize = img.md[0].size[1];

        int bar_width = 12;
        int disp_char_rows = wrow - 4;
        int disp_cols = wcol - bar_width - 1;
        int bar_col_start = wcol - bar_width;

        if (wrow * wcol > ctx->screen_size) {
            ctx->screen_size = wrow * wcol;
            ctx->screen = (tv_cell_t*)realloc(ctx->screen, ctx->screen_size * sizeof(tv_cell_t));
            ctx->prev_screen = (tv_cell_t*)realloc(ctx->prev_screen, ctx->screen_size * sizeof(tv_cell_t));
            memset(ctx->prev_screen, 0, ctx->screen_size * sizeof(tv_cell_t));
        }
        
        for (int i = 0; i < wrow * wcol; i++) {
            ctx->screen[i].bg_r = 0; ctx->screen[i].bg_g = 0; ctx->screen[i].bg_b = 0;
            ctx->screen[i].fg_r = 255; ctx->screen[i].fg_g = 255; ctx->screen[i].fg_b = 255;
            ctx->screen[i].ch[0] = ' '; ctx->screen[i].ch[1] = 0;
        }

        if (disp_char_rows <= 0 || disp_cols <= 0) {
            usleep(10000);
            continue;
        }

        int disp_img_rows = disp_char_rows * 2;

        if (disp_img_rows * disp_cols > ctx->buffer_size) {
            ctx->buffer_size = disp_img_rows * disp_cols;
            ctx->display_buffer = (double*)realloc(ctx->display_buffer, ctx->buffer_size * sizeof(double));
        }

        size_t needed_fb = tv_framebuf_size(wrow, wcol);
        if (needed_fb > ctx->frame_buffer_size) {
            ctx->frame_buffer_size = needed_fb;
            ctx->frame_buffer = (char*)realloc(ctx->frame_buffer, ctx->frame_buffer_size);
        }

        double view_w_img = (double)xsize / view_zoom;
        double view_h_img = (double)ysize / view_zoom;
        double step = fmax(view_w_img / disp_cols, view_h_img / disp_img_rows);
        
        double center_img_x = view_center_x * xsize;
        double center_img_y = view_center_y * ysize;
        double center_disp_x = disp_cols / 2.0;
        double center_disp_y = disp_img_rows / 2.0;

        double sign_y = current_options.flip_v ? -1.0 : 1.0;
        double sign_x = current_options.flip_h ? -1.0 : 1.0;

        int buf_idx = 0;
        if (img.md[0].datatype == _DATATYPE_FLOAT) {
            const float * restrict data = img.array.F;
            for (int i = 0; i < disp_img_rows; i++) {
                for (int j = 0; j < disp_cols; j++) {
                    int img_y = (int)(center_img_y + sign_y * (i - center_disp_y) * step);
                    int img_x = (int)(center_img_x + sign_x * (j - center_disp_x) * step);
                    if (img_x >= 0 && img_x < (int)xsize && img_y >= 0 && img_y < (int)ysize) {
                        ctx->display_buffer[buf_idx++] = (double)data[img_y * xsize + img_x];
                    } else {
                        ctx->display_buffer[buf_idx++] = 0.0;
                    }
                }
            }
        } else if (img.md[0].datatype == _DATATYPE_UINT16) {
            const uint16_t * restrict data = img.array.UI16;
            for (int i = 0; i < disp_img_rows; i++) {
                for (int j = 0; j < disp_cols; j++) {
                    int img_y = (int)(center_img_y + sign_y * (i - center_disp_y) * step);
                    int img_x = (int)(center_img_x + sign_x * (j - center_disp_x) * step);
                    if (img_x >= 0 && img_x < (int)xsize && img_y >= 0 && img_y < (int)ysize) {
                        ctx->display_buffer[buf_idx++] = (double)data[img_y * xsize + img_x];
                    } else {
                        ctx->display_buffer[buf_idx++] = 0.0;
                    }
                }
            }
        } else {
            for (int i = 0; i < disp_img_rows; i++) {
                for (int j = 0; j < disp_cols; j++) {
                    int img_y = (int)(center_img_y + sign_y * (i - center_disp_y) * step);
                    int img_x = (int)(center_img_x + sign_x * (j - center_disp_x) * step);
                    if (img_x >= 0 && img_x < (int)xsize && img_y >= 0 && img_y < (int)ysize) {
                        ctx->display_buffer[buf_idx++] = get_pixel_value(&img, img_x, img_y);
                    } else {
                        ctx->display_buffer[buf_idx++] = 0.0;
                    }
                }
            }
        }

        int num_pixels = buf_idx;

        double min_val = 0.0, max_val = 1.0;
        if (current_options.range_locked) {
            min_val = current_options.manual_min;
            max_val = current_options.manual_max;
        } else if (current_options.range == RANGE_MINMAX) {
            min_val = 1e20; max_val = -1e20;
            for(int k=0; k<num_pixels; k++) {
                if(ctx->display_buffer[k] < min_val) min_val = ctx->display_buffer[k];
                if(ctx->display_buffer[k] > max_val) max_val = ctx->display_buffer[k];
            }
        } else {
            double *sorted_buf = (double*)malloc(num_pixels * sizeof(double));
            memcpy(sorted_buf, ctx->display_buffer, num_pixels * sizeof(double));
            qsort(sorted_buf, num_pixels, sizeof(double), compare_doubles);
            double p_low = 0.0, p_high = 1.0;
            switch(current_options.range) {
                case RANGE_001_999: p_low = 0.001; p_high = 0.999; break;
                case RANGE_005_995: p_low = 0.005; p_high = 0.995; break;
                case RANGE_01_99: p_low = 0.01; p_high = 0.99; break;
                case RANGE_05_95: p_low = 0.05; p_high = 0.95; break;
                case RANGE_10_90: p_low = 0.10; p_high = 0.90; break;
                case RANGE_15_85: p_low = 0.15; p_high = 0.85; break;
                case RANGE_20_80: p_low = 0.20; p_high = 0.80; break;
                default: break;
            }
            min_val = sorted_buf[(int)(p_low * (num_pixels-1))];
            max_val = sorted_buf[(int)(p_high * (num_pixels-1))];
            free(sorted_buf);
        }
        if (max_val <= min_val) max_val = min_val + 1.0;

        current_min_val = min_val;
        current_max_val = max_val;

        if (current_options.colormap != lut_colormap || current_options.scale != lut_scale) {
            build_colormap_lut(current_options.colormap, current_options.scale);
        }

        termview_render_image(ctx, disp_char_rows, disp_cols, center_img_x, center_img_y, sign_x, sign_y, step, xsize, ysize, min_val, max_val);
        termview_render_colorbar(ctx, disp_char_rows, bar_col_start, min_val, max_val);
        termview_render_infobar(ctx, xsize, ysize, step, min_val, max_val);
        termview_render_popup(ctx, disp_char_rows, disp_cols);
        termview_flush_ansi(ctx);
    }

    tv_exit_alt_screen();
    tv_exit_raw();

    if (ctx->display_buffer) free(ctx->display_buffer);
    if (ctx->screen) free(ctx->screen);
    if (ctx->prev_screen) free(ctx->prev_screen);
    if (ctx->frame_buffer) free(ctx->frame_buffer);

    ImageStreamIO_closeIm(&img);
    return 0;
}
