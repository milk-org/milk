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

**Audience:** Integrator · See also: [Pipelines](pipelines.md) · [Glossary](pipelines_glossary.md)
```

### 2.1. Audiences

| Audience | Reads | Example content |
|---|---|---|
| Operator | how to run a system | commands, states, failures |
| Integrator | how to write and tune a configuration | file formats, keys, custom plans |
| Internals | how the code works | package layout, contracts, C expansions of milk |
| Developer | how to work on milk itself | build, coding standards, tooling, these rules |
| Maintainers | governance and maintenance | TBC |

A page has one audience. Put it in the nav group that matches: user pages under User Guide
(Internals as a subgroup of the topic), developer pages under Developer Guide, maintainer pages under Maintenance.

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
milk-pipes state -p basic          # (1)!
milk-pipeinit drift basic-rootdir  # (2)!
```

1. Prints the state, then a one-line drift summary.
2. Lists what changed in the confdir since init.
````

### 4.2. Output

Show output only where its format is the subject (a report, a status line, an error message),
in its own `text` block titled `Output`, after the command:

````markdown
```bash
milk-pipeinit drift basic-rootdir
```

```text title="Output"
conf.toml: source modified, local unchanged
STRUCTURAL: compute unit delay2: added
```
````

Walkthroughs show commands only.

### 4.3. milk-cli scripts

Start milk-cli blocks with `#!/usr/bin/env milk-script`, without `milk-cli >` prompts.

## 5. CLI and Python tabs

When an operation exists both as a command and as a Python call, show both in content tabs.
Use the labels `CLI` and `Python` everywhere: linked tabs then keep the reader's choice across
pages.

````markdown
=== "CLI"

    ```bash
    milk-pipes deploy -p basic
    ```

=== "Python"

    ```python
    from milk.operate.pipeline import Pipeline
    from milk.operate.plans import run_plan

    run_plan(Pipeline("basic"), "deploy")
    ```
````

## 6. Admonitions

Use `note`, `warning` and `tip` only. A feature's implementation status goes in a single
admonition at the top of the page. The body is indented by four spaces:

```markdown
!!! warning "Not yet implemented"
    Pipeline kinds other than `generic` are planned.
```

Syntax reference:
[Material admonitions](https://squidfunk.github.io/mkdocs-material/reference/admonitions/).

??? note "`<!-- prettier-ignore -->` comments"
    Prettier reflows indentation-based Markdown: grid cards, admonitions, content tabs, and any
    block whose body is indented. Older pages therefore put `<!-- prettier-ignore -->` (or a
    `prettier-ignore-start`/`-end` pair) before such blocks. The prettier hook is currently
    disabled in `.pre-commit-config.yaml`, so the comment is not needed until we settle on a
    better Markdown formatting hook.

## 7. Diagrams

Diagrams are Mermaid, in a `mermaid` block.

- One idea per diagram: a normal flow and its failure modes are two diagrams.
- Place the diagram right after the section's first sentence. The prose explains it and does not
  repeat it.
- At most ~12 nodes. Split a larger diagram.

## 8. Links

- Between pages: relative links to the `.md` file, `[Drift](pipelines_conf.md#drift)`. Anchors
  are the heading slugs, as shown by the heading permalinks.
- To repository files outside `docs/`: the `github_blob_url` macro, for example
  [scripts/mkdocs_macros.py]({{ github_blob_url }}/scripts/mkdocs_macros.py). Relative `../`
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
