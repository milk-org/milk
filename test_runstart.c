#include "CLIcore.h"
#include "fps.h"

int main() {
    FPS fps;
    fps_connect("mkrnd", &fps, 0);
    functionparameter_RUNstart(&fps);
    fps_disconnect(&fps);
    return 0;
}
