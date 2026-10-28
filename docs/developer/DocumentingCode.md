# Working with the documentation

This document outlines the conventions for documenting the `milk` project. The documentation is
split into two halves:

1. **Manuals and Guides:** Written in Markdown (like the file you are reading) and hosted directly
   on GitHub.
2. **C API Reference:** Written directly in the `.c` and `.h` source code using Doxygen syntax.

See also: [Programmer's Guide](../arch/programmers_guide.md) · [Working with Git](WorkingWithGit.md) ·
[Coding Standards](coding_standards.md)

---

## 1. Building and serving the documentation

The documentation website is generated using the `mkdocs` package; `mkdocs` is installed in a
Python environment along with the Material theme and extensions.

With `mkdocs` installed, start a local doc server with `mkdocs serve`.

Refer to [Developer tooling](tooling.md) for more information.

---

## 2. Writing Manuals and Guides (Markdown)

General instructional documentation, architectural overviews, and tutorials should be placed in the
`docs/` folder in standard-ish GitHub Flavored Markdown (`.md`).

We use the following conventions with `mkdocs`:

- Use MkDocs admonitions (`!!! note`, `!!! warning`, `!!! tip`) to call out important information.
  Indent the body by four spaces. Due to an unresolved interaction with the `prettier` hook for markdown linting, prefix the admonition with `<!-- prettier-ignore -->`.
- Use Markdown tables for data and parameter lists.
- When referencing other files within the documentation folder, use standard relative markdown links (e.g.,
  `[Coding Standards](coding_standards.md)`).

### Links & macros

A link is simply done [like that](./DocumentingCode.md/#links-macros).

```md
[This links to the page](./DocumentingCode.md)
[This links to the section with a permalink](./DocumentingCode.md/#links-macros)
```

For internal linkage, use relative links to `.md` files within the `docs/` folder.
To link to repository files _outside_ of the docs folder, use the `{{ github_blob_url }}` macro (add `render_macros: true` to the page front matter), see [Project management operations](../maintenance/pm_operations.md).


### Admonitions

=== "Admonition: rendered"

    !!! info "With a title"
        This is an admonition box !
        It can have one of 12 types: note, abstract, tip, success, question, warning, failure, danger, bug, example, quote.

        The title is optional.

        In python style, the _indentation_ controls the end of the admonition.

    ??? tip "Open me"
        And it can also be collapsible !

=== "Verbatim code"

    ```
    !!! info "With a title"
        This is an admonition box !
        It can have one of 12 types: note, abstract, tip, success, question, warning, failure, danger, bug, example, quote.

        The title is optional.

        In python style, the _indentation_ controls the end of the admonition.

    ??? tip "Open me"
        And it can also be collapsible !
    ```

More info [at this link](https://squidfunk.github.io/mkdocs-material/reference/admonitions/#inline-blocks-inline-end).


---

## 3. Documenting Source Code (Kernel-Doc / Doxygen)

The `milk` C/C++ source code uses Kernel-Doc style comments (also compatible with Doxygen) to
document functions and structures.

Place a brief API-facing description above declarations in `.h` files, and the full Kernel-Doc
comment directly above function definitions in `.c` files.

### 3.1 Standard Function Documentation

Use Kernel-Doc format (`function_name() - short description`, `@arg:` parameter descriptions, and
`Return:` section), or Doxygen `@param`/`@return` tags where already used in a module:

```c
/**
 * compute_sum() - Compute the sum of two integers and format as a string.
 * @a:      First integer operand.
 * @b:      Second integer operand.
 * @output: Output string buffer where the decimal sum is written.
 *
 * Detailed description of the computation can go here.
 *
 * Return: RETURN_SUCCESS on success, or an error code on failure.
 */
static inline errno_t compute_sum(
    int   a,
    int   b,
    char *output)
{
    int sum = a + b;
    sprintf(output, "%d", sum);
    return RETURN_SUCCESS;
}
```

### 3.2 Grouping Modules

Use `@defgroup` and `@ingroup` when grouping related functions into modules for Doxygen HTML
reference generation:

```c
/**
 * @defgroup FPSconf Configuration function for Function Parameter Structure (FPS)
 * @defgroup FPSrun  Run function using Function Parameter Structure (FPS)
 */
```

---

## 4. Building a Doxygen HTML Reference (Optional)

If you wish to generate a local C API reference using Doxygen:

### 4.1 Initial Requirements

Ensure you have Doxygen and Graphviz installed on your system:

```bash
# Ubuntu/Debian
sudo apt-get install doxygen graphviz

# CentOS/RHEL
sudo yum install doxygen graphviz
```

### 4.2 Generating the Documentation

Generate a default `Doxyfile` (or use `plugins/cacao-src/doc/Doxyfile` for `cacao`) and run
`doxygen`:

```bash
doxygen -g Doxyfile
doxygen Doxyfile
```

> [!NOTE]
> The MkDocs documentation site (`mkdocs.yml`) is built and deployed to GitHub Pages by
> `.github/workflows/docs-deploy.yaml` on pushes to `framework-dev`.

---

← [Documentation Index](../index.md)
