---
description: Add or update pages on the MkDocs
  documentation site
---

# Update Documentation Site

Use this workflow when adding, removing, or modifying
pages on the MkDocs documentation site.

## Prerequisites

Install MkDocs Material in a Python virtual environment
(one-time setup):

```bash
$ python3 -m venv /tmp/mkdocs-venv
$ /tmp/mkdocs-venv/bin/pip install mkdocs-material
```

## Steps

### 1. Create or edit the markdown file

Place new files under `docs/` in the appropriate
subdirectory:

| Content type | Directory |
|-------------|-----------|
| Installation / setup | `docs/install/` |
| CLI documentation | `docs/cli/` |
| Developer guides | `docs/developer/` |
| Top-level concepts | `docs/` (root) |

Add YAML frontmatter to the very top of the new markdown
file to categorize it in the Tag Index (`docs/tags.md`):

```yaml
---
tags:
  - topic1
  - topic2
---
```

Use `<details markdown="1">` (not bare `<details>`) for
collapsible sections containing markdown tables or code.

### 2. Update `mkdocs.yml` navigation

Add the new page to the correct tab section in the
`nav:` key of `mkdocs.yml`:

```yaml
nav:
  - Home: index.md
  - Getting Started:    # install, build, CLI overview, FAQ
  - User Guide:         # streams, FPS, procinfo, CLI, scripts
  - Developer Guide:    # tutorial, coding standards, plugins
  - Architecture:       # programmer's guide, dep graph
  - Operations:         # performance, PGO/LTO, debugging
  - API Reference: api/html/index.html
```

### 3. Update `docs/index.md`

Add the new page link to the appropriate section on
the homepage so it appears in the landing page index.

// turbo
### 4. Test the build locally

```bash
$ /tmp/mkdocs-venv/bin/mkdocs build --strict -d /tmp/mkdocs-test
```

Fix any warnings about broken links or missing files.
Warnings about `../` links to files outside `docs/` are
expected and can be ignored.

// turbo
### 5. Preview locally (optional)

```bash
$ /tmp/mkdocs-venv/bin/mkdocs serve -a 127.0.0.1:8123
```

Open `http://127.0.0.1:8123/milk/` in a browser to
verify the page renders correctly.

### 6. Commit and push

Follow the standard git workflow: create a feature
branch from `framework-dev`, commit, push, and create
a PR targeting `framework-dev`.

The documentation CI workflow triggers automatically
when files matching these paths change:
- `docs/**`
- `mkdocs.yml`
- `Doxyfile`
- `src/**/*.h`
- `plugins/**/*.h`
- `.github/workflows/docs.yml`

Deployment to GitHub Pages only happens on push to
`framework-dev` (not from PRs).

## Updating the Doxygen API Reference

Doxygen is configured in `Doxyfile` at the repo root.
It automatically indexes all `*.h` files in `src/` and
`plugins/`. The CI workflow builds Doxygen output into
`_site/api/` and deploys alongside MkDocs.

To test Doxygen locally:

```bash
$ mkdir -p /tmp/doxygen-test
$ ( cat Doxyfile; echo "OUTPUT_DIRECTORY=/tmp/doxygen-test" ) \
    | doxygen -
```

## Updating Custom Styles

Custom CSS lives at `docs/stylesheets/extra.css` and is
loaded via the `extra_css` key in `mkdocs.yml`.
