---
name: docs-writer
description: Rewrite, split or restructure a page under docs/ to follow docs/developer/writing_docs.md; rebuild the generated outputs and images of the moka tutorial (scripts/docs/tutorial_capture.py)
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
   where its format is the subject (a Tutorial shows it after every stage, §10). **CLI/Python
   tabs** per §5 where both exist.
7. **Admonitions** per §6: implementation status in one admonition at the top.
8. **Diagrams** per §7: check each against the rules; split or simplify.
9. **Links** per §8; update every inbound link and anchor to moved sections (grep `docs/`,
   `.agents/`, `AGENTS.md`).
10. **Nav and index:** update `mkdocs.yml` (and `docs/index.md` for entry-point pages).
11. **Build** with `mkdocs build --strict` and fix every warning caused by the change.
12. **Report** questions found in passing: contradictions between pages, unclear behaviour,
   audience judgment calls, each with a suggested default.

## Rebuilding the moka tutorial

The outputs and images of [`docs/quickstart/moka_tutorial.md`](../../../docs/quickstart/moka_tutorial.md)
are generated from the page by
[`scripts/docs/tutorial_capture.py`](../../../scripts/docs/tutorial_capture.py). Never edit a
`tut:out` block or an image by hand. The script's docstring is the truth about the markers, the
modes and the run rules; read it, do not copy it here.

**When.** After any change to moka's output, to the example confdirs under `milk-examples/`, or to
a command of the page (a changed marker or command makes its block stale).

**Machine state `--run` needs.** A machine with milk and cacao installed and moka in the active
Python environment, and:

- `MILK_ROOT` set to the tree that holds `milk-examples/pipelines`.
- `~/AOloop2` and `/milk/shm/tutorial_capture` absent.
- Nothing else to prepare: the script gives everything it launches the private folder
  `/milk/shm/tutorial_capture` as `MILK_SHM_DIR` and `MILK_PROC_DIR`, and a private tmux socket
  folder (`tut_tmux_*` under the system temp dir), and removes both at the end. `/milk/shm` itself
  need not be clean.

It refuses to start, with the reason, when `MILK_ROOT`, `~/AOloop2` or the private folder is wrong;
a missing milk, cacao or moka shows up as the first failing block.

**Procedure.**

1. `python scripts/docs/tutorial_capture.py --check`: markers, ids and image files; no live system.
2. `python scripts/docs/tutorial_capture.py --run`: the full capture. It writes all outputs and
   images at the end, or none. `--from ID` resumes a hand run kept in place.
3. Run the formatting hooks and `mkdocs build --strict` as for any docs change.

**Manual: the desktop image.** `desktop.png` is the one image `--run` does not produce: a
screenshot of the arranged desktop (the page's "Set up your screen"), with the numbered discs added
by `python scripts/docs/tutorial_capture.py --callouts SCREENSHOT.png`, which saves `desktop.png`.
Retake it when the windows shown change (a TUI layout, a viewer, the stage the page ends on). The
discs sit at the fixed positions in the script's `CALLOUTS`: arrange the windows to match them, or
edit `CALLOUTS` when the layout changes, then rerun `--callouts`. Before publishing, look for what the
screenshot leaks (user and host in the prompts, branch names, local paths).

**What to look at in the diff.** Text outputs differ between runs in durations only (PIDs are
masked); anything else is a real change. Check every image against the text around it. The figures
need the eye in particular: the latency step response must show a step, the response matrix a clear
pattern (not noise), and the control matrix's singular values must drop where `-n` cuts. The
script refuses a calibration done without light or with a latency out of range (docstring), not
every bad figure. Images that vary from run to run, which is not a regression: the streamCTRL
captures (sort order, stale rows), PIDs and rates in the TUI captures, and the `milk-termview`
captures of streams, which show live data. Rasterize those (`rsvg-convert`) and look: pupil images
for the WFS, a compact PSF core with the loop closed, a spread one with it open.

**Failures.**

- The script exits 1 with a message: a refusal (precondition above), a failed page check, a
  `tut:run` exit code that is not the expected one, or a missing line to edit. Nothing is written;
  fix the cause (or the page, if the command changed) and run again. A `[FAIL]` from moka in a
  block means moka or the confdir changed; fix that, not the page.
- `bad calibration` after `dark-ref` or `measure-mlat`: every moka step said ok, but the light was
  off (a lost FPS write in milk, about 1 run in 10). Run again; nothing is written. A
  `RemoveLightSim ... [FAIL]` in `dark-ref` is the same race, caught by moka.
- Exit 0 with wrong figures (as above): do not commit them. Run again; if it recurs, it is a moka
  or confdir problem, to be reported, not worked around in the script.
- A killed or crashed run: the script cleans up after a failure, but after a kill, before the next
  run, stop the leftover tmux servers of each `tut_tmux_*` folder under the system temp dir
  (`TMUX_TMPDIR=<folder> tmux -L milk_tut kill-server` and `TMUX_TMPDIR=<folder> tmux kill-server`),
  then remove `~/AOloop2`, `/milk/shm/tutorial_capture` and the folder.
