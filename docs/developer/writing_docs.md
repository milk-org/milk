---
render_macros: true
---

# Writing Docs

House rules for the pages under `docs/`: page layout, voice, and the Markdown features we use.

**Audience:** Developer · See also: [Developer tooling](tooling.md) ·
[Coding Standards](coding_standards.md) · [Project management operations](../maintenance/pm_operations.md)

## 1. Building the site

The site is built by MkDocs Material from `docs/` and `mkdocs.yml`. To install `mkdocs` and its
plugins, and to preview locally with `mkdocs serve`, see
[Developer tooling: MILK documentation](tooling.md#1-milk-documentation). Check every change
with `mkdocs build --strict`.

## 2. Page layout

Every page starts with:

1. an H1;
2. one sentence saying what the page is for;
3. one header line: audience, then 3-5 related pages.

```markdown
# Pipeline configuration files

How a confdir is laid out and what `conf.toml` can say.

**Audience:** Integrator · See also: [moka](moka.md) · [Glossary](glossary.md)
```

### 2.1. Audiences

| Audience    | Reads                                 | Example content                                 |
| ----------- | ------------------------------------- | ----------------------------------------------- |
| Operator    | how to run a system                   | commands, states, failures                      |
| Integrator  | how to write and tune a configuration | file formats, keys, custom plans                |
| Internals   | how the code works                    | package layout, contracts, C expansions of milk |
| Developer   | how to work on milk itself            | build, coding standards, tooling, these rules   |
| Maintainers | governance and maintenance            | TBC                                             |

A page has one audience. Put it in the nav group that matches: user pages under User Guide, or
under their package's own top-level group and `docs/` folder (`moka/`), with Internals as a
subgroup of the topic; developer pages under Developer Guide, maintainer pages under Maintenance.

### 2.2. Headings

- Numbered: `## 1. Title`, `### 1.1. Title`.
- ATX (`#`) headings, no skipped levels.

## 3. Voice

- Terse and factual. No marketing, no "simply" or "just".
- Present tense. Second person only where an instruction needs it.
- One idea per paragraph. Prose wrapped at ~100 characters; tables and code may be longer.
- One topic, one page. Other pages link to it instead of repeating it.

## 4. Code blocks

Every fenced block has a language (`bash`, `python`, `toml`, `c`, `text`...).

### 4.1. Commands

Shell commands go in `bash` blocks without `$` prompts, so the copy button gives runnable text.
Explain a line with a code annotation, `# (1)!`, and the numbered list right after the block:

````markdown
```bash
moka state -p basic          # (1)!
mokainit drift basic-rootdir  # (2)!
```

1. Prints the state, then a one-line drift summary.
2. Lists what changed in the confdir since init.
````

### 4.2. Output

Show output only where its format is the subject (a report, a status line, an error message),
in its own `text` block titled `Output`, after the command:

````markdown
```bash
mokainit drift basic-rootdir
```

```text title="Output"
conf.toml: source modified, local unchanged
STRUCTURAL: compute unit delay2: added
```
````

Walkthroughs show commands only, except Tutorial pages ([§10](#10-tutorial-pages)).

### 4.3. milk-cli scripts

Start milk-cli blocks with `#!/usr/bin/env milk-script`, without `milk-cli >` prompts.

## 5. CLI and Python tabs

When an operation exists both as a command and as a Python call, show both in content tabs.
Use the labels `CLI` and `Python` everywhere: linked tabs then keep the reader's choice across
pages.

````markdown
=== "CLI"

    ```bash
    moka deploy -p basic
    ```

=== "Python"

    ```python
    from moka.operate.pipeline import Pipeline
    from moka.operate.plans import run_plan

    run_plan(Pipeline("basic"), "deploy")
    ```
````

## 6. Admonitions

Use `note`, `warning` and `tip` only. A feature's implementation status goes in a single
admonition at the top of the page. The body is indented by four spaces:

```markdown
!!! warning "Not yet implemented"
    Calibration store/load is planned.
```

Syntax reference:
[Material admonitions](https://squidfunk.github.io/mkdocs-material/reference/admonitions/).

??? note "Markdown formatting: `mdformat`"

    The `mdformat` pre-commit hook (with the `mdformat-mkdocs` plugin, options in
    `.mdformat.toml`) understands admonitions, content tabs and front matter, and its output
    passes `.markdownlint.yml`. Nested lists use 4-space indents, as Python-Markdown requires.
    Older pages still carry `<!-- prettier-ignore -->` comments: they are obsolete.

## 7. Diagrams

Diagrams are Mermaid, in a `mermaid` block.

- One idea per diagram: a normal flow and its failure modes are two diagrams.
- Place the diagram right after the section's first sentence. The prose explains it and does not
  repeat it.
- At most ~12 nodes. Split a larger diagram.

## 8. Links

- Between pages: relative links to the `.md` file, `[Drift](conf.md#drift)`. Anchors
  are the heading slugs, as shown by the heading permalinks.
- To repository files outside `docs/`: the `github_blob_url` macro, for example
  [scripts/mkdocs_macros.py](<{{ github_blob_url }}/scripts/mkdocs_macros.py>). Relative `../`
  links out of `docs/` break on the published site. Macros are rendered only on pages with
  `render_macros: true` in their front matter:

{% raw %}

```markdown
---
render_macros: true
---

[scripts/mkdocs_macros.py]({{ github_blob_url }}/scripts/mkdocs_macros.py)
```

{% endraw %}

## 9. C API reference

Doxygen generation is currently removed; it may return.

## 10. Tutorial pages

A Tutorial is one page with one runnable path from nothing to a result, for a newcomer. It is the
only page type that departs from §3 and §4.2; every other rule applies.

- **Second person.** The reader is "you"; a Tutorial is the one place where second person is the
  default (§3).
- **Output after every stage.** Show the output of each command in its `Output` block (§4.2), as
  text or as an SVG terminal capture. Figures are PNG.
- **Images.** Stored under `img/<page name>/` next to the page. Every image has alt text and a
  one-line caption.
- **Exploration** goes in `tip` boxes titled "Look around"; **explanations** go in `note` boxes
  (§6).
- **Generated, not pasted.** The commands on the page are the source. A script runs them and
  writes the outputs and images; none is pasted by hand. The page footer names the script.

```markdown
![A rendered DM map](img/moka_tutorial/dm_map.png)
_Figure 1: The DM map after the first iteration._

!!! tip "Look around"
    Run `moka state -p basic` again and compare the compute units.

!!! note
    The state is read from shared memory, not from the confdir.

_Outputs and images on this page are generated by `scripts/docs/tutorial_capture.py`._
```
