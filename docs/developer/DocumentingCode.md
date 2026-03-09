# Documenting Code

This document outlines the conventions for documenting the `milk` project. The documentation is split into two halves:
1. **Manuals and Guides:** Written in Markdown (like the file you are reading) and hosted directly on GitHub.
2. **C API Reference:** Written directly in the `.c` and `.h` source code using Doxygen syntax.

## 1. Writing Manuals and Guides (Markdown)
General instructional documentation, architectural overviews, and tutorials should be placed in the `docs/` folder in standard GitHub Flavored Markdown (`.md`).
- Use `> [!NOTE]`, `> [!WARNING]`, and `> [!TIP]` to call out important information.
- Use Markdown tables for data and parameter lists.
- When referencing other files, use standard relative markdown links (e.g., `[Coding Standards](coding_standards.md)`).

## 2. Documenting Source Code (Doxygen)
The `milk` C/C++ source code uses Doxygen tags to generate the API reference. 

Doxygen comments should be placed directly above the function implementations in the `.c` files, or above the struct definitions in the `.h` files.

### 2.1 Standard Function Documentation
Use the standard `@param`, `@return`, and `@brief` tags to describe the function behavior.

```c
/**
 * @ingroup processing_module
 * @brief Computes the sum of two integers.
 *
 * Detailed description of the computation can go here. 
 * You can also include formulas or extended methodology.
 *
 * @param[in]  a first number
 * @param[in]  b second number to be added to first one
 * @param[out] output string buffer where the result is written
 *
 * @return error code, #RETURN_SUCCESS if OK
 */
inline static errno_t compute_sum(int a, int b, char *output) {
    int sum = a + b;
    sprintf(output, "%d", sum);
    return RETURN_SUCCESS;
}
```

### 2.2 Grouping Modules
Use `@defgroup` and `@ingroup` to group related functions into unified modules in the generated HTML reference.

```c
/**
 * @defgroup FPSconf Configuration function for Function Parameter Structure (FPS)
 * @defgroup FPSrun  Run function using Function Parameter Structure (FPS)
 */
```

## 3. Building the Doxygen HTML Reference
If you wish to view the generated C API documentation locally, you can build it using Doxygen.

### 3.1 Initial Requirements
Ensure you have Doxygen installed on your system:
```bash
# Ubuntu/Debian
sudo apt-get install doxygen graphviz

# CentOS/RHEL
sudo yum install doxygen graphviz
```

### 3.2 Generating the Documentation
There is currently no tracked `Doxyfile` in the repository by default, as development relies heavily on the GitHub markdown manuals. To generate the API reference locally:

1. Navigate to the repository root.
2. Generate a default Doxygen configuration tailored to the repository:
```bash
doxygen -g
```
3. Edit the generated `Doxyfile` to configure the source directories (e.g., set `INPUT = src/ plugins/` and `RECURSIVE = YES`).
4. Run doxygen to generate the `html/` output directory:
```bash
doxygen Doxyfile
```

You can then open `html/index.html` in your web browser to browse the C API reference.
