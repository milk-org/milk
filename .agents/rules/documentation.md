---
description: Documentation site structure and markdown standards for docs/ and READMEs.
---

# Documentation

Conventions and tooling: [`docs/developer/DocumentingCode.md`](../../docs/developer/DocumentingCode.md)
and [`docs/developer/tooling.md`](../../docs/developer/tooling.md). Site: MkDocs Material from `docs/`
(`mkdocs.yml`), Doxygen from the headers; both deployed by `.github/workflows/docs.yml` from
`framework-dev`.

## Adding a page

1. Create it in the `docs/` subfolder matching its nav tab (`quickstart/`, `guide/`, `developer/`,
   `arch/`, `operations/`, `maintenance/`).
2. Add it to `nav:` in `mkdocs.yml` and to `docs/index.md`.
3. Start the file with YAML frontmatter `tags: [topic1, topic2]` (feeds `docs/tags.md`).
4. Put a "See also" bar of 3-5 related pages near the top.

## Formatting

- Prose wrapped at ~100 characters; tables and code blocks may exceed it.
- ATX headings, no skipped levels.
- Every fenced code block has a language (`bash`, `c`, `cmake`, `text`...).
- No `$` / `milk-cli >` prompts in code blocks: they prevent copy-paste. Start milk-cli blocks with
  `#!/usr/bin/env milk-script`.
- Collapsible sections use `<details markdown="1">`.
- Module READMEs: one-line description, source file table (`| File | Description |`), dependency list.

## Before committing

```bash
npx markdown-link-check --config .markdown-link-check.json <file>
npx markdownlint-cli2 <file>
mkdocs build --strict -d /tmp/mkdocs-test
```
