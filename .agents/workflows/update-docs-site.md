---
description: Add or update pages on the MkDocs documentation site
---

# Update Documentation Site

Rules: [`docs/developer/writing_docs.md`](../../docs/developer/writing_docs.md). Checklist:
`.agents/rules/documentation.md`. Restructuring a page: the `docs-writer` skill.

1. Create or edit the page in the `docs/` folder and nav group of its audience.
2. Add it to `nav:` in `mkdocs.yml` (and a card in `docs/index.md` for an entry-point page); start
   it with the header line.
3. Build with link checking and fix every warning (warnings about `../` links to files outside
   `docs/` are expected):

   ```bash
   mkdocs build --strict -d /tmp/mkdocs-test
   ```

4. Optionally preview with `mkdocs serve -a 127.0.0.1:8123`.
5. Follow the git workflow (`.agents/rules/git-workflow.md`); the docs CI deploys from `framework-dev`.
