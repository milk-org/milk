---
description: Ensure all new or modified documentation
  follows the project's adopted standards.
---

When creating or editing markdown documentation in the
`milk` repository, you MUST follow these standards:

## Formatting Rules

1. **Line length:** Wrap prose at ~72 characters. Tables
   and code blocks may exceed this limit.
2. **Headings:** Use ATX-style headings (`#`). Do not
   skip heading levels (e.g., `##` → `####`).
3. **Code blocks:** Every fenced code block MUST have a
   language specifier (CI enforces MD040). Common languages:
   `bash`, `c`, `python`, `cmake`, `text` (for plain output,
   compiler warnings, directory trees, etc.).
4. **Shell prompts:** Do NOT use `$` or `milk-cli >`
   prompts in code blocks — they prevent copy-paste.
   Instead, start milk-cli code blocks with
   `#!/usr/bin/env milk-script` (the shebang line).
   Use `#` for comments inside code blocks.
5. **No V1 macros:** Do not reference `FPS_MAIN_STANDALONE`
   (the V1 macro). Always use `FPS_MAIN_STANDALONE_V2` or
   `FPS_MAIN_STANDALONE_V2_CONFCHECK`.

## Structure Rules

1. **"See also" bar:** Every page under `docs/` MUST have
   a "See also" breadcrumb bar near the top, linking to
   3–5 related pages. Format:
   ```
   See also: [Page A](pageA.md) ·
   [Page B](pageB.md) ·
   [Page C](pageC.md)
   ```
2. **Plugin READMEs:** Must follow the standardized
   template with: one-line module description, source file
   table (`| File | Description |`), and dependency list.
3. **New topic pages:** Must be added to both
   `docs/index.md` and the `nav:` section of `mkdocs.yml`
   under the appropriate tab.
4. **MkDocs tags:** Every new `.md` page must have
   YAML frontmatter `tags: [topic1, topic2]` added to
   the very top to seed the `docs/tags.md` index.
5. **MkDocs details blocks:** Use `<details markdown="1">`
   (not bare `<details>`) so markdown content inside
   renders correctly. See `documentation-site.md` rule.

## Before Committing

1. Verify that all relative links resolve. Run:
   ```bash
   npx markdown-link-check \
     --config .markdown-link-check.json <file>
   ```
2. Run markdown linting:
   ```bash
   npx markdownlint-cli2 <file>
   ```
3. If you added a new `.md` file, check if
   `docs/Markdown_Index.md` needs regeneration.
