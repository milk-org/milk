# Milk Documentation

Welcome to the `milk` documentation! 

`milk` provides a high-performance framework and tools for image processing and analysis, particularly suited for building real-time execution pipelines (such as Adaptive Optics loops) out of small modular units. The framework provides zero-copy tensor passing and instant parameter synchronization.

## 🚀 Getting Started

If you are new to the `milk` environment, start here to get the software installed and learn the basic concepts.

- [**Installation Guide**](install/compile.md)
- [**Command Line Interface (CLI) Overview**](cli/CLI_Overview.md)
- [**Shared Memory Streams**](streams.md): Core concept covering zero-copy image passing.
- [**Function Processing System (FPS)**](fps.md): Core concept covering process parameters and configuration.

## 🏛️ Core Architecture

For a deep dive into how `milk` components interact at a system level, these documents explain the underlying structures.

- [**Programmer's Guide**](programmers_guide.md): The best starting point to understand the overall architecture, C API, and CMake setup.
- [**Dependency Graph**](dependency_graph.md): Visual map of module dependencies.
- [**FPS Standalone and CMD Modes**](FPS_Standalone_CMD_Modes.md): Execution context details for `milk-fpsexec-*` binaries.
- [**Process Info (`procinfo`)**](procinfo.md): Telemetry, heartbeat monitoring, and profiling.

## 🛠️ Developer Guides

Guidelines and tutorials for writing your own compute modules or extending the CLI framework.

- [**Coding Standards**](developer/coding_standards.md): C coding conventions for `milk`.
- [**Adding Plugins**](developer/plugins.md): How to build modules that compile alongside the core.
- [**Template Source Code**](developer/TemplateSourceCode.md): Breakdown of the example C module.
- [**Loading Custom Modules**](developer/LoadingModules.md): Linking `.so` modules at runtime.
- [**Working With Git**](developer/WorkingWithGit.md): Branching and commit workflow.
- [**Documenting Code**](developer/DocumentingCode.md): In-code documentation standards.
- [**Module Files Layout**](developer/ModuleFiles.md): Directory structure conventions.

## 📖 CLI & Tools Reference

- [**CLI Core Syntax**](cli/CLIcore.md): Argument parsing and command invocation rules.
- [**CLI User Input**](cli/UserInput.md): Writing interactive prompts.
- [**Readline Keys**](cli/helpreadline.md): Keyboard shortcuts inside the `milk` shell.
- [**Help Text**](cli/help.txt): Built-in help text reference.
- [**Automatically Generated Index**](Markdown_Index.md): Complete list of all Markdown files in the repository.

***
*Can't find what you're looking for? Check the [Automatically Generated Document Index](Markdown_Index.md) for all plugin READMEs.*
