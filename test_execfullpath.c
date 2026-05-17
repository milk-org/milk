#include "CLIcore_standalone.h"
#include "fps.h"
#include <stdio.h>

int main() {
    FPS fps;
    functionparameter_ReadSharedMem(&fps, "streamprocess");
    printf("Exec: '%s'\n", fps.md->execfullpath);
    functionparameter_FreeFPS(&fps);
    return 0;
}
