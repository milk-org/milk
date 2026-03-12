# Milk Documentation

Welcome to the `milk` documentation! 

`milk` provides a high-performance framework and tools for image processing and analysis, particularly suited for building real-time execution pipelines (such as Adaptive Optics loops) out of small modular units. The framework provides zero-copy tensor passing and instant parameter synchronization.

## 🚀 Getting Started

If you are new to the `milk` environment, follow these steps:

1. [**Install milk**](install/compile.md) — clone, build, and configure.
2. [**CLI Overview**](cli/CLI_Overview.md) — understand `milk-cli` and standalone executables.
3. [**Shared Memory Streams**](streams.md) — core concept: zero-copy image passing.
4. [**Function Processing System (FPS)**](fps.md) — core concept: process parameters and configuration.
5. [**Developer Tutorial**](developer/tutorial.md) — write your first compute module.

## 🏛️ Core Architecture

For a deep dive into how `milk` components interact at a system level, these documents explain the underlying structures.

- [**Software Architecture**](architecture.md): Top-level hierarchical overview of the system design, subsystems, and data flow.
- [**Programmer's Guide**](programmers_guide.md): The best starting point to understand the overall architecture, C API, and CMake setup.
- [**Dependency Graph**](dependency_graph.md): Visual map of module dependencies.
- [**FPS Standalone and CMD Modes**](FPS_Standalone_CMD_Modes.md): Execution context details for `milk-fpsexec-*` binaries.
- [**Process Info (`procinfo`)**](procinfo.md): Telemetry, heartbeat monitoring, and profiling.
- [**Debugging**](debugging.md): GDB, procinfo diagnostics, tmux log inspection, common failure patterns.
- [**Performance Tuning**](performance.md): CPU pinning, RT scheduling, shared memory, GPU acceleration.
- [**Profile-Guided Optimization (PGO)**](pgo.md): Build with runtime profiles for 10–30% speedup.

## 🛠️ Developer Guides

Guidelines and tutorials for writing your own compute modules or extending the CLI framework.

- [**Coding Standards**](developer/coding_standards.md): C coding conventions for `milk`.
- [**Adding Plugins**](developer/plugins.md): How to build modules that compile alongside the core.
- [**Template Source Code**](developer/TemplateSourceCode.md): Breakdown of the example C module.
- [**Loading Custom Modules**](developer/LoadingModules.md): Linking `.so` modules at runtime.
- [**Working With Git**](developer/WorkingWithGit.md): Branching and commit workflow.
- [**Documenting Code**](developer/DocumentingCode.md): In-code documentation standards.
- [**Developer Tutorial**](developer/tutorial.md): Step-by-step guide to writing your first module.
- [**Module Files Layout**](developer/ModuleFiles.md): Directory structure conventions.
- [**Python API**](python.md): Accessing streams from Python with `pyMilk` and numpy.
- [**Valkey Integration**](valkey.md): Multi-host FPS parameter sync via Valkey.
- [**Code Assist Tools**](code_assist.md): Agent rules and workflows for AI-assisted development.

## 📖 CLI & Tools Reference

- [**CLI Core Syntax**](cli/CLIcore.md): Argument parsing and command invocation rules.
- [**Readline Keys**](cli/helpreadline.md): Keyboard shortcuts inside the `milk` shell.
- [**Scripts Reference**](scripts.md): All `milk-*` shell scripts and utilities.
- [**Help Text**](cli/help.txt): Built-in help text reference.
- [**FAQ & Troubleshooting**](faq.md): Common issues and solutions.
- [**Automatically Generated Index**](Markdown_Index.md): Complete list of all Markdown files in the repository.

***
*Can't find what you're looking for? Check the [Automatically Generated Document Index](Markdown_Index.md) for all plugin READMEs.*

---
← [Documentation Index](index.md)
