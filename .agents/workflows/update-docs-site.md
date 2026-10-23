---
description: Add or update pages on the MkDocs documentation site
---

# Update Documentation Site

Conventions, page checklist and tooling: `.agents/rules/documentation.md`,
[`docs/developer/DocumentingCode.md`](../../docs/developer/DocumentingCode.md) and
[`docs/developer/tooling.md`](../../docs/developer/tooling.md).

1. Create or edit the page under the `docs/` subfolder matching its nav tab.
2. Add it to `nav:` in `mkdocs.yml` and to `docs/index.md`; start it with `tags:` frontmatter.
3. Build with link checking, fix every warning (warnings about `../` links to files outside `docs/`
   are expected):

   ```bash
   mkdocs build --strict -d /tmp/mkdocs-test
   ```

4. Optionally preview with `mkdocs serve -a 127.0.0.1:8123`.
5. Follow the git workflow (`.agents/rules/git-workflow.md`); the docs CI deploys from `framework-dev`.
