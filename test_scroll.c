#include <stdio.h>
int main() {
    int scroll = 10;
    int count = 20;
    int page_h = 10;
    int dir = -3;
    scroll += dir;
    if (scroll < 0) scroll = 0;
    if (page_h > 0 && scroll > count - page_h) {
        scroll = count - page_h;
        if (scroll < 0) scroll = 0;
    }
    printf("scroll = %d\n", scroll);
    return 0;
}
