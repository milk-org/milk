---
description: Checklist for pages under docs/ and module READMEs; the rules live in docs/developer/writing_docs.md.
---

# Documentation

The rules are in [`docs/developer/writing_docs.md`](../../docs/developer/writing_docs.md) (human
source, takes precedence); building and previewing in
[`docs/developer/tooling.md`](../../docs/developer/tooling.md). Do not restate them here. To
rewrite or restructure a page, use the `docs-writer` skill.

## Adding or changing a page

1. Pick the page's one audience (writing_docs §2.1) and put it in the matching folder and nav group:
   `quickstart/`, `guide/` (Internals as a subgroup of the topic), `developer/`, `arch/`,
   `operations/`, `maintenance/`.
2. Add it to `nav:` in `mkdocs.yml`. Entry-point pages also get a card in `docs/index.md`.
3. Start it with the header line (writing_docs §2).
4. Check that no other page already covers the topic; link to it instead of repeating it.

## Module READMEs

One-line description, source file table (`| File | Description |`), dependency list.

## Before committing

```bash
npx markdown-link-check --config .markdown-link-check.json <file>
npx markdownlint-cli2 <file>
mkdocs build --strict -d /tmp/mkdocs-test
```
