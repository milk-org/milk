<div class="md-hero" markdown>

# :telescope: milk

<p class="md-hero__tagline">
High-performance real-time image processing framework for Adaptive Optics and scientific computing.
Microsecond-latency pipelines through zero-copy shared memory.
</p>

<p class="md-hero__badges">
<a href="https://github.com/milk-org/milk">
<img alt="GitHub stars" src="https://img.shields.io/github/stars/milk-org/milk?style=flat-square&color=00bfa5">
</a>
<a href="https://github.com/milk-org/milk/blob/framework-dev/LICENSE">
<img alt="License" src="https://img.shields.io/github/license/milk-org/milk?style=flat-square&color=0097a7">
</a>
<a href="https://github.com/milk-org/milk/actions">
<img alt="Build" src="https://img.shields.io/github/actions/workflow/status/milk-org/milk/docs.yml?style=flat-square&label=docs&color=26a69a">
</a>
</p>

</div>

`milk` orchestrates many small compute units that communicate through zero-copy shared memory
tensors, enabling microsecond-latency data pipelines. The three pillars — **ImageStreamIO**,
**FPS**, and **processinfo** — live entirely in `/dev/shm/`.

---

## :rocket: Getting Started

<!-- prettier-ignore-start -->
<div class="grid cards" markdown>

- :material-download-circle:{ .lg .middle } **Install**

    ***

    Clone, build, and configure the milk framework.

    [⮕ Installation](install/compile.md)

- :material-layers-outline:{ .lg .middle } **Build Tiers**

    ***

    Engine → Core → Full: compile only what you need.

    [⮕ Build tiers](install/build_tiers.md)

- :material-console:{ .lg .middle } **CLI Overview**

    ***

    Interactive shell, standalone executables, and scripting basics.

    [⮕ CLI overview](cli/CLI_Overview.md)

- :material-help-circle-outline:{ .lg .middle } **FAQ**

    ***

    Common issues with builds, SHM, FPS, and CLI.

    [⮕ FAQ & Troubleshooting](faq.md)

</div>

<!-- prettier-ignore-end -->
<!-- note: prettier comments NEED a blank line just before -->

---

## :classical_building: Core Concepts

<!-- prettier-ignore-start -->
<div class="grid cards" markdown>

- :material-memory:{ .lg .middle } **Streams**

    ***

    Zero-copy shared memory tensors (`ImageStreamIO`).

    [⮕ Streams](streams.md)

- :material-tune-variant:{ .lg .middle } **FPS**

    ***

    Live parameter sync, state control, TUI dashboards.

    [⮕ FPS](fps.md)

- :material-heart-pulse:{ .lg .middle } **Process Info**

    ***

    Heartbeat telemetry, loop-rate profiling, health monitoring.

    [⮕ Process Info](procinfo.md)

- :material-sitemap-outline:{ .lg .middle } **Architecture**

    ***

    System overview, layered design, data flow diagrams.

    [⮕ Architecture](architecture.md)

</div>

<!-- prettier-ignore-end -->

---

## :hammer_and_wrench: Developer Guides

<!-- prettier-ignore-start -->
<div class="grid cards" markdown>

- :material-school-outline:{ .lg .middle } **Tutorial**

    ***

    Write your first compute module step by step.

    [⮕ Tutorial](developer/tutorial.md)

- :material-code-braces:{ .lg .middle } **Coding Standards**

    ***

    C style, line length, includes, Kernel-Doc.

    [⮕ Coding standards](developer/coding_standards.md)

- :material-puzzle-outline:{ .lg .middle } **Adding Plugins**

    ***

    Build modules that compile alongside the core.

    [⮕ Plugins](developer/plugins.md)

- :material-file-tree-outline:{ .lg .middle } **Template Code**

    ***

    Breakdown of `milk_module_example`.

    [⮕ Template source](developer/TemplateSourceCode.md)

</div>

<!-- prettier-ignore-end -->

---

## :bar_chart: Operations & Reference

<!-- prettier-ignore-start -->
<div class="grid cards" markdown>

- :material-speedometer:{ .lg .middle } **Performance**

    ***

    CPU pinning, RT scheduling, SIMD, BLAS, GPU.

    [⮕ Performance](performance.md)

- :material-chart-line:{ .lg .middle } **PGO & LTO**

    ***

    Profile-guided optimization + static link-time optimization for 15–40 % speedup.

    [⮕ PGO & LTO](pgo.md)

- :material-bug-outline:{ .lg .middle } **Debugging**

    ***

    GDB, tmux logs, procinfo diagnostics, common failures.

    [⮕ Debugging](debugging.md)

</div>

<!-- prettier-ignore-end -->

---

## :link: More Resources

- **[What's New](whatsnew.md)** — recent features and upgrades
- [CLI Syntax Reference](cli/CLIcore.md) · [Readline Keys](cli/helpreadline.md)
- [Scripts Reference](scripts.md) · [Python API](python.md) · [Valkey Integration](valkey.md)
- [Programmer's Guide](programmers_guide.md) · [Dependency Graph](dependency_graph.md) ·
  [fpsCTRL Reference](fpsCTRL_reference.md)
- [Working with Git](developer/WorkingWithGit.md) · [Code Assist Tools](code_assist.md)
- [All Markdown Files](Markdown_Index.md) · [Tag Index](tags.md)
