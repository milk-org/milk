---
name: docs-writer
description: Rewrite, split or restructure a page under docs/ to follow
  docs/developer/writing_docs.md
---

# Docs Writer

Procedure for reworking a documentation page. Every rule it applies is in
[`docs/developer/writing_docs.md`](../../../docs/developer/writing_docs.md); this skill only says
in which order to apply them. When the two disagree, writing_docs.md wins.

## When to Use

- A page mixes audiences, repeats another page, or predates writing_docs.md.
- A page must be split, merged or moved in the nav.

## Steps

1. **Read** the page and its nav neighbours in full. List its sections.
2. **Assign one audience** per page (writing_docs §2.1). If sections serve different audiences,
   plan a split: list each section with its target page.
3. **Deduplicate.** For each section, check whether another page already covers it. Keep one
   copy; replace the others with a link.
4. **Account for content.** Keep a table old section → new page/section. Nothing is dropped
   without the user's agreement; do not add behaviour the old text does not state.
5. **Header line and headings** per writing_docs §2.
6. **Code blocks** per §4: remove prompts, turn explanations into annotations, keep output only
   where its format is the subject. **CLI/Python tabs** per §5 where both exist.
7. **Admonitions** per §6: implementation status in one admonition at the top.
8. **Diagrams** per §7: check each against the rules; split or simplify.
9. **Links** per §8; update every inbound link and anchor to moved sections (grep `docs/`,
   `.agents/`, `AGENTS.md`).
10. **Nav and index:** update `mkdocs.yml` (and `docs/index.md` for entry-point pages).
11. **Build** with `mkdocs build --strict` and fix every warning caused by the change.
12. **Report** questions found in passing: contradictions between pages, unclear behaviour,
    audience judgment calls, each with a suggested default.
