#include "CLIcore.h"
#include "fps.h"

int main() {
    FPS fps;
    functionparameter_ReadSharedMem(&fps, "streamprocess");
    functionparameter_CONFstart(&fps);
    functionparameter_FreeFPS(&fps);
    return 0;
}
