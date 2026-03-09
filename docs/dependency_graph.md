# Milk + Cacao Dependency Graph

> Generated from current CMakeLists.txt — 2026-03-09

---

## Core Stack

```mermaid
graph TD
    CFITSIO["cfitsio"]:::ext
    NCURSES["ncurses"]:::ext
    READLINE["readline"]:::ext

    ISIO["ImageStreamIO"]:::core
    PROCINFO["milkprocessinfo"]:::core
    FPS["milkfps"]:::core
    MILKDATA["milkdata"]:::core
    MILKTUI["milkTUI"]:::core

    PITUI["milkprocessinfoTUI"]:::fw
    FPSTUI["milkfpsTUI"]:::fw
    FPSCLI["milkfpsCLI"]:::fw
    FPSSTANDALONE["milkfpsStandalone"]:::fw

    ARITH["COREMODarith"]:::coremod
    IOFITS["COREMODiofits"]:::coremod
    MEMORY["COREMODmemory"]:::coremod
    TOOLS["COREMODtools"]:::coremod

    CLICORE["CLIcore"]:::fw

    FPSCTRL["milk-fpsCTRL"]:::exe
    PROCCTRL["milk-procCTRL"]:::exe
    STREAMCTRL["milk-streamCTRL"]:::exe
    MILKEXE["milk-fpsexec-*"]:::exe
    CACAOEXE["cacao-fpsexec-*"]:::exe
    FPSTOOLS["milk-fps-set/list/…"]:::exe

    ISIO -.->|headers only| CFITSIO
    PROCINFO --> ISIO
    FPS --> ISIO
    FPS --> PROCINFO
    MILKDATA --> FPS
    MILKDATA --> PROCINFO
    MILKTUI --> ISIO
    MILKTUI --> NCURSES

    IOFITS --> FPS
    IOFITS --> CFITSIO
    ARITH --> FPS
    ARITH --> IOFITS
    ARITH --> CFITSIO
    MEMORY --> FPS
    MEMORY --> CFITSIO
    TOOLS --> FPS

    PITUI --> PROCINFO
    PITUI --> MILKTUI
    FPSTUI --> FPS
    FPSTUI --> MILKTUI
    FPSCLI --> FPS
    FPSCLI --> CLICORE
    FPSSTANDALONE --> FPS
    FPSSTANDALONE --> MILKDATA

    CLICORE --> ARITH
    CLICORE --> IOFITS
    CLICORE --> MEMORY
    CLICORE --> TOOLS
    CLICORE --> FPS
    CLICORE --> MILKDATA
    CLICORE --> PROCINFO
    CLICORE --> MILKTUI
    CLICORE --> FPSTUI
    CLICORE --> PITUI
    CLICORE --> READLINE
    CLICORE --> NCURSES
    CLICORE --> CFITSIO

    FPSCTRL --> FPSTUI
    FPSCTRL --> CLICORE
    PROCCTRL --> PITUI
    STREAMCTRL --> CLICORE
    STREAMCTRL --> ISIO
    MILKEXE --> FPSSTANDALONE
    CACAOEXE --> FPSSTANDALONE
    FPSTOOLS --> FPSSTANDALONE

    classDef ext fill:#566573,stroke:#333,color:#fff
    classDef core fill:#1a5276,stroke:#123,color:#fff
    classDef fw fill:#2e86c1,stroke:#1a5,color:#fff
    classDef coremod fill:#1e8449,stroke:#145,color:#fff
    classDef exe fill:#b7950b,stroke:#a80,color:#000
```

## Plugins & Cacao

```mermaid
graph TD
    CLICORE["CLIcore"]:::fw
    OPENBLAS["OpenBLAS"]:::ext
    FFTW["FFTW"]:::ext
    GSL["GSL"]:::ext

    FFT["milkfft"]:::plugin
    LINALG["milklinalgebra"]:::plugin
    LINOPT["milklinoptimtools"]:::plugin
    STAT["milkstatistic"]:::plugin
    IMGGEN["milkimagegen"]:::plugin
    IMGBASIC["milkimagebasic"]:::plugin
    IMGFILT["milkimagefilter"]:::plugin
    IMGFMT["milkimageformat"]:::plugin
    INFO["milkinfo"]:::plugin
    ZERNIKE["milkZernikePolyn"]:::plugin
    LINARPRED["milklinARfilterPred"]:::plugin
    KDTREE["milkkdtree"]:::plugin
    IMREDUCE["milkimgreduce"]:::plugin
    PSF["milkpsf"]:::plugin
    CLUSTER["milkclustering"]:::plugin

    AOLOOP["cacaoAOloopControl"]:::cacao
    AODM["cacaoAOloopControlDM"]:::cacao
    AOIO["cacaoAOloopControlIOtools"]:::cacao
    AOACQ["cacaoAcquireCalib"]:::cacao
    AOPC["cacaoPredictiveControl"]:::cacao
    AOCT["cacaoCompTools"]:::cacao
    AOPT["cacaoPerfTest"]:::cacao
    COMPCALIB["cacaoComputeCalib"]:::cacao
    PYRWFS["cacaoPyramidWFS"]:::cacao

    FFT --> CLICORE
    FFT --> FFTW
    LINALG --> CLICORE
    LINALG --> OPENBLAS
    LINOPT --> CLICORE
    LINOPT --> GSL
    STAT --> CLICORE
    IMGGEN --> CLICORE
    IMGGEN --> STAT
    IMGBASIC --> CLICORE
    IMGFILT --> CLICORE
    IMGFMT --> CLICORE
    IMGFMT -.-> IMGFILT
    INFO --> CLICORE
    ZERNIKE --> CLICORE
    ZERNIKE --> IMGGEN
    LINARPRED --> CLICORE
    LINARPRED --> OPENBLAS
    KDTREE --> CLICORE
    IMREDUCE --> CLICORE
    PSF --> CLICORE
    CLUSTER --> CLICORE

    AOLOOP --> CLICORE
    AOLOOP --> LINOPT
    AODM --> CLICORE
    AODM --> FFT
    AODM --> IMGGEN
    AODM --> IMGFILT
    AODM --> AOLOOP
    AOIO --> CLICORE
    AOIO --> INFO
    AOIO --> AOLOOP
    AOACQ --> CLICORE
    AOACQ --> INFO
    AOACQ --> AOLOOP
    AOPC --> CLICORE
    AOPC --> LINOPT
    AOPC --> AOLOOP
    AOCT --> CLICORE
    AOCT --> AOLOOP
    AOPT --> CLICORE
    AOPT --> STAT
    AOPT --> AOLOOP
    COMPCALIB --> CLICORE
    COMPCALIB --> INFO
    COMPCALIB --> AOLOOP
    COMPCALIB --> OPENBLAS
    PYRWFS --> CLICORE
    PYRWFS --> INFO
    PYRWFS --> AOLOOP

    classDef ext fill:#566573,stroke:#333,color:#fff
    classDef fw fill:#2e86c1,stroke:#1a5,color:#fff
    classDef plugin fill:#7d3c98,stroke:#5a2,color:#fff
    classDef cacao fill:#d35400,stroke:#a00,color:#fff
```

## Standalone Build (USE_CLI=OFF)

```mermaid
graph TD
    CFITSIO["cfitsio"]:::ext
    ISIO["ImageStreamIO"]:::core
    PROCINFO["milkprocessinfo"]:::core
    FPS["milkfps"]:::core
    MILKDATA["milkdata"]:::core
    FPSSA["milkfpsStandalone"]:::core

    ARITH_C["COREMODarith_compute"]:::compute
    IOFITS_C["COREMODiofits_compute"]:::compute
    MEMORY_C["COREMODmemory_compute"]:::compute
    TOOLS_C["COREMODtools_compute"]:::compute

    FFT_C["milkfft_compute"]:::plugcomp
    IMGGEN_C["milkimagegen_compute"]:::plugcomp
    IMGBASIC_C["milkimagebasic_compute"]:::plugcomp
    IMGFILT_C["milkimagefilter_compute"]:::plugcomp
    STAT_C["milkstatistic_compute"]:::plugcomp

    MILKEXE["milk-fpsexec-*"]:::exe
    CACAOEXE["cacao-fpsexec-*"]:::exe
    CACAOEXE_P["cacao-fpsexec-*\n(plugins)"]:::exe
    FPSTOOLS["milk-fps-set/list/…"]:::exe

    ISIO -.->|headers only| CFITSIO
    PROCINFO --> ISIO
    FPS --> PROCINFO
    MILKDATA --> FPS
    FPSSA --> FPS
    FPSSA --> MILKDATA

    IOFITS_C --> CFITSIO
    IOFITS_C --> FPS
    ARITH_C --> FPS
    ARITH_C --> IOFITS_C
    MEMORY_C --> FPS
    TOOLS_C --> FPS

    FFT_C --> MEMORY_C
    FFT_C --> IOFITS_C
    IMGGEN_C --> MEMORY_C
    IMGGEN_C --> IOFITS_C
    IMGBASIC_C --> MEMORY_C
    IMGBASIC_C --> IOFITS_C
    IMGFILT_C --> MEMORY_C
    IMGFILT_C --> IOFITS_C
    STAT_C --> MEMORY_C
    STAT_C --> IOFITS_C
    IMGGEN_C --> STAT_C

    MILKEXE --> FPSSA
    MILKEXE --> ARITH_C
    MILKEXE --> MEMORY_C
    MILKEXE --> TOOLS_C
    MILKEXE --> IOFITS_C
    CACAOEXE --> FPSSA
    CACAOEXE --> ARITH_C
    CACAOEXE --> MEMORY_C
    CACAOEXE --> TOOLS_C
    CACAOEXE --> IOFITS_C
    CACAOEXE_P --> FPSSA
    CACAOEXE_P --> FFT_C
    CACAOEXE_P --> IMGGEN_C
    CACAOEXE_P --> IMGBASIC_C
    CACAOEXE_P --> IMGFILT_C
    FPSTOOLS --> FPSSA

    classDef ext fill:#566573,stroke:#333,color:#fff
    classDef core fill:#1a5276,stroke:#123,color:#fff
    classDef compute fill:#1e8449,stroke:#145,color:#fff
    classDef plugcomp fill:#7d3c98,stroke:#5a2,color:#fff
    classDef exe fill:#b7950b,stroke:#a80,color:#000
```

### Legend

| Color | Layer |
|---|---|
| ⚫ Grey | External libraries |
| 🔵 Dark blue | Core (ISIO → procinfo → FPS → milkdata) |
| 🔵 Light blue | Framework (CLIcore, fpsCLI/TUI, fpsStandalone) |
| 🟢 Green | COREMODs / _compute variants |
| 🟣 Purple | milk-extra plugins / _compute variants |
| 🟠 Orange | Cacao modules |
| 🟡 Yellow | Executables |

---

## Exhaustive Dependency Table

### Core Libraries

| Target | Links to | Optional |
|---|---|---|
| ImageStreamIO | *(none at link time)* | cfitsio (headers), CUDA |
| milkprocessinfo | ImageStreamIO | |
| milkprocessinfoTUI | milkprocessinfo, milkTUI, ncurses | |
| milkfps | ImageStreamIO, milkprocessinfo | |
| milkdata | ImageStreamIO, milkfps, milkprocessinfo | |
| milkTUI | ImageStreamIO, ncurses | |
| milkfpsStandalone | milkfps, milkdata | |
| milkfpsCLI | milkfps, CLIcore | |
| milkfpsTUI | milkfps, milkTUI, ncurses | |
| CLIcore | milkCOREMODarith, milkCOREMODiofits, milkCOREMODmemory, milkCOREMODtools, milkfps, milkdata, milkprocessinfo, milkTUI, milkfpsTUI, milkprocessinfoTUI, cfitsio, readline, ncurses | OpenMP, hwloc |

### COREMOD Libraries

| Target | Links to |
|---|---|
| milkCOREMODiofits | milkfps, cfitsio |
| milkCOREMODarith | milkfps, milkCOREMODiofits, cfitsio |
| milkCOREMODmemory | milkfps, cfitsio |
| milkCOREMODtools | milkfps |

### COREMOD _compute Variants (MILK_NO_CLI)

| Target | Links to |
|---|---|
| milkCOREMODiofits_compute | milkfps, cfitsio |
| milkCOREMODarith_compute | milkfps, milkCOREMODiofits_compute, cfitsio |
| milkCOREMODmemory_compute | milkfps, cfitsio |
| milkCOREMODtools_compute | milkfps |

### milk-extra Plugins (full / CLI)

| Target | Links to | Optional |
|---|---|---|
| milkfft | CLIcore, fftw3, fftw3f | |
| milklinalgebra | CLIcore, OpenBLAS | CUDA, MAGMA, MKL, lapacke |
| milklinoptimtools | CLIcore | GSL |
| milkimagegen | CLIcore, milkstatistic | |
| milkstatistic | CLIcore | |
| milkimagebasic | CLIcore | |
| milkimagefilter | CLIcore | |
| milkimageformat | CLIcore | |
| milkinfo | CLIcore | |
| milkZernikePolyn | CLIcore, milkimagegen | |
| milklinARfilterPred | CLIcore | OpenBLAS, MKL |
| milkkdtree | CLIcore | |
| milkimgreduce | CLIcore | |
| milkpsf | CLIcore | |
| milkclustering | CLIcore | |

### milk-extra Plugin _compute Variants (MILK_NO_CLI)

| Target | Links to |
|---|---|
| milkfft_compute | fftw3, fftw3f, milkCOREMODmemory_compute, milkCOREMODiofits_compute, cfitsio |
| milkimagebasic_compute | milkCOREMODmemory_compute, milkCOREMODiofits_compute, cfitsio |
| milkimagefilter_compute | milkCOREMODmemory_compute, milkCOREMODiofits_compute, cfitsio |
| milkimagegen_compute | milkstatistic_compute, milkCOREMODmemory_compute, milkCOREMODiofits_compute, cfitsio |
| milkstatistic_compute | milkCOREMODmemory_compute, milkCOREMODiofits_compute, cfitsio, m |

### Cacao Modules

| Target | Links to | Optional |
|---|---|---|
| cacaoAOloopControl | CLIcore, milklinoptimtools | |
| cacaoAOloopControlDM | CLIcore, milkfft, milkimagegen, milkimagefilter, cacaoAOloopControl | |
| cacaoAOloopControlIOtools | CLIcore, milkinfo, cacaoAOloopControl | |
| cacaoAOloopControlacquireCalib | CLIcore, milkinfo, cacaoAOloopControl | |
| cacaoAOloopControlPredCtrl | CLIcore, milklinoptimtools, cacaoAOloopControl | |
| cacaoAOloopControlCompTools | CLIcore, cacaoAOloopControl | |
| cacaoAOloopControlPerfTest | CLIcore, milkstatistic, cacaoAOloopControl | |
| cacaocomputeCalib | CLIcore, milkinfo, cacaoAOloopControl | CUDA, MAGMA, OpenBLAS, MKL, lapacke |
| cacaopyramidWFStools | CLIcore, milkinfo, cacaoAOloopControl | lapacke |

### Executables

| Target | Links to |
|---|---|
| milk-cli | CLIcore + all module libs |
| milk-fpsCTRL | milkfpsTUI, milkfps, milkprocessinfo, ImageStreamIO, CLIcore, ncurses |
| milk-procCTRL | milkprocessinfoTUI |
| milk-streamCTRL | CLIcore, ImageStreamIO |
| milk-fps-list/search/info/rm | milkfps, milkfpsStandalone, milkdata, milkprocessinfo, ImageStreamIO |
| milk-fps-set/track | milkfps, milkfpsStandalone, milkdata, milkprocessinfo, ImageStreamIO |
| milk-fps-conf*/run* | milkfps, milkfpsStandalone, milkdata, milkprocessinfo, ImageStreamIO |
| milk-fps-valkey | milkfps, milkprocessinfo, ImageStreamIO, valkey |
| stream-monproc-runner | milkinfo, CLIcore, ImageStreamIO |
| stream-monproc-disp | milkinfo, CLIcore, ImageStreamIO, ncurses |

### Standalone Executables via CMake Functions

| Function | Base link set |
|---|---|
| `add_milk_standalone()` | milkfps, milkfpsStandalone, milkdata, milkprocessinfo, ImageStreamIO, milkCOREMODmemory_compute, milkCOREMODtools_compute, milkCOREMODarith_compute, milkCOREMODiofits_compute, cfitsio |
| `add_cacao_standalone()` | same as above |
| `add_cacao_standalone_plugins()` | above + selected plugin _compute libs (default: all 4) |

`add_cacao_standalone_plugins()` accepts an optional list of plugins to link:

```cmake
add_cacao_standalone_plugins(name src.c)               # all 4 plugins
add_cacao_standalone_plugins(name src.c fft imagegen)   # selective
```

Valid plugin names: `fft`, `imagegen`, `imagefilter`, `imagebasic`.

> [!NOTE]
> `_compute` library variants are compiled with `MILK_NO_CLI` — they contain pure
> computation code with CLI registration stubs.
>
> Standalone executables do **not** link `${LIBNAME}` by default (which would pull in
> CLIcore). If a standalone needs module-lib symbols, add an explicit
> `target_link_libraries(... PUBLIC ${LIBNAME})` after the `add_*_standalone()` call.
>
> Currently **76 of 90** standalone executables are CLIcore-free. 14 exceptions are
> whitelisted in `milk-check-standalone-deps` (they require module-lib symbols
> for OpenBLAS, FFT, tree algorithms, etc.).

---

## Core Library Chain (Bottom-Up)

```
cfitsio (headers only)
  ╌╌ ImageStreamIO
       └─ milkprocessinfo
            └─ milkfps
                 └─ milkdata
                      ├─ milkfpsStandalone  (standalone path)
                      └─ CLIcore            (full CLI path)
```
