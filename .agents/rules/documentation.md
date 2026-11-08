---
description: Documentation site structure and markdown standards for docs/ and READMEs.
---

# Documentation

Conventions and tooling: [`docs/developer/DocumentingCode.md`](../../docs/developer/DocumentingCode.md)
and [`docs/developer/tooling.md`](../../docs/developer/tooling.md). Site: MkDocs Material from `docs/`
(`mkdocs.yml`), deployed by `.github/workflows/docs.yml` from `framework-dev`; a Doxygen reference is
optional and built locally (DocumentingCode.md §4).

## Adding a page

1. Create it in the `docs/` subfolder matching its nav tab (`quickstart/`, `guide/`, `developer/`,
   `arch/`, `operations/`, `maintenance/`).
2. Add it to `nav:` in `mkdocs.yml` and to `docs/index.md`.
3. Optionally start the file with YAML frontmatter `tags: [topic1, topic2]` (the tags index is
   disabled in `mkdocs.yml`).
4. Put a "See also" bar of 3-5 related pages near the top.

## Formatting

- Prose wrapped at ~100 characters; tables and code blocks may exceed it.
- ATX headings, no skipped levels.
- Every fenced code block has a language (`bash`, `c`, `cmake`, `text`...).
- No `$` / `milk-cli >` prompts in code blocks: they prevent copy-paste. Start milk-cli blocks with
  `#!/usr/bin/env milk-script`.
- Collapsible sections use `<details markdown="1">` or the admonitions of DocumentingCode.md §2.
- Module READMEs: as [programmers_guide.md §7](../../docs/arch/programmers_guide.md#7-c-source-file-conventions)
  says.

## Before committing

```bash
npx markdown-link-check --config .markdown-link-check.json <file>
npx markdownlint-cli2 <file>
mkdocs build --strict -d /tmp/mkdocs-test
```
