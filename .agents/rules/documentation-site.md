---
description: Documentation site structure, MkDocs config,
  and deployment pipeline awareness.
---

# Documentation Site Structure

The milk documentation is published at
`https://milk-org.github.io/milk/` using two tools:

| Tool            | URL path     | Source            |
| --------------- | ------------ | ----------------- |
| MkDocs Material | `/` (root)   | `docs/` directory |
| Doxygen         | `/api/html/` | C headers (`*.h`) |

## Key Files

| File                         | Purpose                                       |
| ---------------------------- | --------------------------------------------- |
| `mkdocs.yml`                 | MkDocs configuration (nav, theme, extensions) |
| `docs/`                      | Markdown source files for MkDocs              |
| `docs/stylesheets/extra.css` | Custom CSS overrides                          |
| `docs/index.md`              | Homepage content                              |
| `Doxyfile`                   | Doxygen configuration                         |
| `.github/workflows/docs.yml` | CI: builds + deploys both                     |

## Navigation Tabs

The site uses top-level navigation tabs defined in the
`nav:` section of `mkdocs.yml`:

1. **Home** — landing page
2. **Getting Started** — install, build tiers, CLI overview, FAQ
3. **User Guide** — streams, FPS, procinfo, CLI, scripts, Python
4. **Developer Guide** — tutorial, coding standards, plugins
5. **Architecture** — programmer's guide, dependency graph
6. **Operations** — performance, PGO/LTO, debugging
7. **API Reference** — links to Doxygen at `/api/html/`

## When Adding a New Documentation Page

1. Create the `.md` file under `docs/` in the appropriate
   subdirectory.
2. Add the page to the correct tab section in `mkdocs.yml`
   under `nav:`.
3. Also add the page to `docs/index.md` in the appropriate
   section.
4. Add YAML frontmatter at the very top of the `.md` file
   with `tags: [topic1, topic2]` categorizing the content
   so that it appears in `docs/tags.md`.
5. Use `<details markdown="1">` (not bare `<details>`) for
   any collapsible sections — the `md_in_html` extension
   requires the `markdown="1"` attribute.
6. Run the `/update-docs-site` workflow to test locally.

## Deployment Pipeline

- **Workflow**: `.github/workflows/docs.yml`
- **Trigger**: Push to `framework-dev` when files in
  `docs/**`, `mkdocs.yml`, `Doxyfile`, `src/**/*.h`,
  `plugins/**/*.h`, or the workflow itself change.
- **Deploy**: Only from `framework-dev` branch pushes
  (not PRs).
- **Build**: MkDocs builds to `_site/`, Doxygen builds
  to `_site/api/`, both deployed to `gh-pages` branch.

## Markdown Extensions Available

These extensions are enabled in `mkdocs.yml` and can be
used in any docs page:

- **Admonitions**: `!!! note`, `!!! warning`, etc.
- **Details**: `??? note "Title"` or `<details markdown="1">`
- **Tags**: `tags: [topic]` YAML frontmatter
- **Mermaid diagrams**: ` ```mermaid ` code blocks
- **Code highlighting**: with line numbers and copy button
- **Content tabs**: `=== "Tab 1"` / `=== "Tab 2"` syntax
- **Tables**: standard markdown tables
- **Attribute lists**: `{ .class #id }` syntax
