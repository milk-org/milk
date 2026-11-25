# moka tutorial plan v1.2: one page, mlatsim + aoloop to a closed loop on turbulence

Status: plan v1.2, 2026-11-24, specified with Vincent in five question rounds (§2); v1.1 adds the dark and reference stage (D16). Number it when it
becomes a brief. Repo: `~/repos/milk-standalone` (the checkout with `python/moka`, `docs/moka/`, and
the UX commits of `claude/moka-cli-ux-plan.md` merged). Scope when built: `[docs]` plus one script
under `scripts/`, the `docs-writer` skill, and the removal of `sample.sh` / `sample.md`.

Read for this plan: `milk-examples/pipelines/{sample.sh,sample.md}`, `aoloop-conf/{conf.toml, fpsdetails.toml,scripts/dm_psf.py,scripts/moka_plans.py}`, `mlatsim-conf/conf.toml`,
`docs/moka/{tour,moka,operate,plans}.md` (plans: §1 only), `docs/developer/writing_docs.md`,
`docs/guide/fpsCTRL_reference.md`, `mkdocs.yml`, `.agents/skills/docs-writer/SKILL.md`,
`.agents/rules/documentation.md`, and `~/Documents/cacao/slides_onboarding.pptx` (5 slides, shapes
and images extracted). Not read: the moka code, `mokashow`'s code, `milk-streamCTRL`'s options, the
cacao executables. Nothing was run: this session cannot execute on spieluhr.

## 1. What is built

| Item                   | Path                                                                  | Note                                                         |
| ---------------------- | --------------------------------------------------------------------- | ------------------------------------------------------------ |
| The tutorial page      | `docs/quickstart/moka_tutorial.md`                                    | nav: Getting Started, after Installation. Audience: Operator |
| Its images             | `docs/quickstart/img/moka_tutorial/`                                  | SVG terminal captures, PNG figures, the annotated desktop    |
| The capture script     | `scripts/docs/tutorial_capture.py`                                    | runs the page's own blocks, rewrites its outputs and images  |
| A "Tutorial" page type | `docs/developer/writing_docs.md`                                      | new §, plus one line in `.agents/rules/documentation.md`     |
| Rebuild procedure      | `.agents/skills/docs-writer/SKILL.md`                                 | new section; the page footer links to it                     |
| Removed                | `docs/moka/tour.md`, `milk-examples/pipelines/sample.sh`, `sample.md` | links re-pointed                                             |

## 2. Decisions (Vincent, 2026-11-24)

| #   | Topic                    | Decision                                                                                                                                                                                                                     |
| --- | ------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| D1  | End point                | `sample.sh` to `spool`, then loop closed by hand, turbulence on (`dmturb`), PSF watched with `dm_psf.py`, then teardown.                                                                                                     |
| D2  | Captures                 | Taken by an agent on spieluhr, through a committed script. Rerun after any output change.                                                                                                                                    |
| D3  | House rules              | `writing_docs.md` gains a Tutorial page type: output after each stage, screenshots, exploration boxes. Other pages unchanged.                                                                                                |
| D4  | Location                 | Replaces the Tour. New file in `docs/quickstart/`, Getting Started nav. `tour.md` deleted, `pipelinebasic` walkthrough dropped (the confdir stays, as an example and test resource).                                         |
| D5  | Python                   | CLI everywhere. Python only where it adds something: reading streams and FPS values live.                                                                                                                                    |
| D6  | TUI captures             | `tmux capture-pane -e` converted to one SVG per capture. moka's plain output in `text` blocks.                                                                                                                               |
| D7  | Slide visuals            | Slide 1's desktop opens the page with numbered callouts and a legend table (slide 2's label text). Slides 3 to 5 at the mlat, measlinresp and mkcm stages. Slide 1's window arrangement is what "Set up your screen" builds. |
| D8  | UX output                | The UX commits are merged: capture now.                                                                                                                                                                                      |
| D9  | `sample.sh` state        | Runs end to end; the "Before it can work" list of `sample.md` is stale.                                                                                                                                                      |
| D10 | Start point              | milk + cacao installed; a check section proves the setup.                                                                                                                                                                    |
| D11 | Exploration              | All four: ask moka at every stage; look under the hood; play with the loop in fpsCTRL; change the configuration.                                                                                                             |
| D12 | Rebuild note             | A section of the existing `docs-writer` skill, referenced from the page footer.                                                                                                                                              |
| D13 | Source of commands       | The page. The script extracts and runs its blocks. `sample.sh` and `sample.md` are deleted.                                                                                                                                  |
| D14 | Loop steps in the script | The script sets `loopON`, `turbON` and the gains through pyMilk's FPS. The page shows the fpsCTRL keys first, and that Python as the tip of D5.                                                                              |
| D15 | Plan form                | This file: project plan with a commit table.                                                                                                                                                                                 |
| D16 | Dark and reference       | `moka takedark`, then `moka takeref`, immediately after the link, before `mlat`. The dark needs only the linkage to the data source.                                                                                         |

Defaults I took without asking (say if wrong):

- **E1. Outputs are written in place.** The script rewrites each `text title="Output"` block between
  markers. Alternative: `pymdownx.snippets` includes (commented out in `mkdocs.yml` today). In place
  keeps the page readable on GitHub and changes no site setting.
- **E2. Working folder `~/AOloop2`**, as `sample.sh` and the slides. Captured text has `$HOME`
  replaced by `~`.
- **E3. Title** "Tutorial: a simulated AO loop with moka"; nav label "moka Tutorial".
- **E4. mokashow figures are regenerated by the script**, headless, with
  `moka.show.cli.mlat_figure` and `moka.show.modes.modes_figure` (both named in `operate.md` §4).
  The slide images 3 to 5 are the model and the fallback.
- **E5. The desktop screenshot is the one manual image.** It is slide 1's picture (1865 x 1074), with
  the `aol0_psf` viewer (slide 1, Picture2) flattened onto it. It shows `vdeo@spieluhr`, a branch
  name and local paths: confirm that is fine to publish, or retake it.
- **E6. Slide label 9 is corrected.** Slide 2 says `shmImshow aol0_wfsim`; the window in the
  screenshot is `aol0_imWFS0`. The legend follows the window title.

## 3. The page

Header per writing_docs §2, `render_macros: true` front matter (the footer uses `github_blob_url`).
Numbered headings. One audience: Operator. Voice as the house rules; second person, since every
section is an instruction.

Each stage section has the same four parts, in this order:

1. one sentence: what the step does and why;
2. the command block;
3. what you should see: output block, or SVG capture, or figure;
4. a `tip` box "Look around" (D11), and where useful a `note` box "What moka did".

| §    | Section                          | Commands (the page's blocks)                                                                                                          | Shown                                                                                                          | Look around / remarks                                                                                                                                                                                                                                                                                   |
| ---- | -------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| top  | Where you will end up            | none                                                                                                                                  | desktop with 10 callouts + legend table; Mermaid: the two pipelines and the `delaysim` link (at most 12 nodes) | 30 to 45 minutes; nothing touches hardware                                                                                                                                                                                                                                                              |
| 1    | Before you start                 | `moka -h`, the three env variables, `which` on one cacao and one milk fpsexec, `tmux -V`, the Python imports                          | expected output of each                                                                                        | link to Installation §2 and §4; what to do when a check fails                                                                                                                                                                                                                                           |
| 2    | Set up your screen               | `milk-fpsCTRL`, `milk-streamCTRL`, two shells                                                                                         | fpsCTRL and streamCTRL empty                                                                                   | the arrangement of slide 1; both TUIs stay open for the whole page; `h` and `x`                                                                                                                                                                                                                         |
| 3    | Get the two confdirs             | `mkdir`, `mokainit clone --milk mlatsim .`, same for `aoloop`                                                                         | output; tree of a confdir                                                                                      | open `conf.toml`: compute units, `runstart`, `[systemlinkage.delaysim]`; confdir = what you keep in git                                                                                                                                                                                                 |
| 4    | Build the rootdirs               | `mokainit resolve`, `mokainit init`, for both                                                                                         | output; tree of `mlatsim-rootdir`                                                                              | hook products in `initdata/`; `moka state`, `moka plans -p mlatsim`: everything `cold`; the `-p` rule; working inside a rootdir without `-p`                                                                                                                                                            |
| 5    | Deploy and start the simulator   | `moka deploy -p mlatsim`, `moka runstart -p mlatsim`                                                                                  | step lines; fpsCTRL with the five mlatsim FPS; streamCTRL with `dm07disp*`, `mlatsim_wfs` at about 1 kHz       | `tmux ls`, attach to one CompU and detach; `shmImshow dm07disp`, `shmImshow mlatsim_wfs`; `moka state` now `running`; `dmturb` is not started (on demand)                                                                                                                                               |
| 6    | Deploy the loop and link it      | `moka deploy -p aoloop`, `moka link -p aoloop`, `moka link delaysim -p aoloop`                                                        | step lines; streamCTRL showing the three symlinks                                                              | Python tip: read `aol0_wfsim` with pyMilk `SHM`; why a linkage (swap simulator for hardware)                                                                                                                                                                                                            |
| 7    | Take a dark and a reference      | `moka takedark -p aoloop`, `moka takeref -p aoloop`                                                                                   | step lines of both: `RemoveLightSim`, `TakeDark`, `PutbackLightSim`, then `RunStart[acquWFS]`, `TakeRef`       | `takedark` here is the confdir-local plan (`aoloop-conf/scripts/moka_plans.py`) wrapping the built-in one: it turns the simulated light off and back on; `moka plans` marks it `(local)`; `calib/wfsdark.fits`, `calib/wfsref.fits`; `shmImshow aol0_imWFS2` before and after the reference; `-n`, `-t` |
| 8    | Measure the latency              | `moka mlat -p aoloop`, `mokashow mlat -p aoloop`                                                                                      | step lines with the wait; figure (slide 3)                                                                     | `moka tasks mlat -p aoloop` before and after; `acquWFS` stays running; where `hardwlatency.dat` is                                                                                                                                                                                                      |
| 9    | Measure the response             | first `moka mkcm -p aoloop` on purpose, then `moka mkpokes`, `moka measlinresp`, `mokashow rm`                                        | the refusal (`[ NO ] refused: ...`, as captured); step lines; figure (slide 4)                                 | reading a refusal: it names the fix; `calib/` files; `mkpokes zonal`, `-a AMPL`                                                                                                                                                                                                                         |
| 10   | Compute the control matrix       | `moka mkcm -p aoloop`, `mokashow cm`                                                                                                  | step line; figure (slide 5)                                                                                    | `-n`, `tikhonov -r`; singular values in the figure; `mkcm` also runs on a cold pipeline                                                                                                                                                                                                                 |
| 11   | Spool: the loop ready, open      | `moka spool -p aoloop`                                                                                                                | step lines; fpsCTRL on `aoloop_mfilt`, `loopON` OFF                                                            | `moka state`: `running`; streamCTRL: `aol0_modevalWFS` counting                                                                                                                                                                                                                                         |
| 12   | Close the loop                   | fpsCTRL keys (select `aoloop_mfilt`, `loopON`, Space)                                                                                 | fpsCTRL before and after; `shmImshow dm07disp03`                                                               | Python tab: the same through pyMilk `FPS`; `loopZERO`; nothing to correct yet, so the command stays near zero                                                                                                                                                                                           |
| 13   | Add turbulence and watch the PSF | `moka runstart -p mlatsim dmturb`, `turbON` in fpsCTRL, `python conf/scripts/dm_psf.py`, `shmImshow aol0_psf`, `shmImshow dm07disp10` | fpsCTRL on `mlatsim_dmturb`; desktop reference                                                                 | play: open and close the loop, `loopgain`, `loopmult`, `ampl`, `wspeed`; what each does to the PSF; a gain that diverges and `looplimit`                                                                                                                                                                |
| 14   | Change the configuration         | edit `loopgain` in `aoloop-conf/fpsdetails.toml`, `mokainit drift`, `mokainit resync`, teardown and deploy                            | drift report; `moka state` drift line                                                                          | fpsCTRL edits are live and lost at teardown, confdir edits are kept; pointer to Plans & Tasks for writing a local plan like the `takedark` of §7                                                                                                                                                        |
| 15   | Tear down                        | `moka teardown -p aoloop`, `-p mlatsim`; `moka nuke` named, not run                                                                   | step lines; `moka state`: `cold`                                                                               | what survives: rootdir, `work/`, `calib/`                                                                                                                                                                                                                                                               |
| 16   | Where next                       | none                                                                                                                                  | none                                                                                                           | Operating, Configuration, Plans & Tasks, fpsCTRL TUI, pyMilk, Glossary                                                                                                                                                                                                                                  |
| foot | How this page is built           | none                                                                                                                                  | none                                                                                                           | one line: outputs and images are generated by `scripts/docs/tutorial_capture.py`; link to the docs-writer skill section                                                                                                                                                                                 |

Remarks the page must carry, seen on slide 1's own terminal: the `INFO: ... extra keys ['_ln']` line
and the `WARNING: Local plan 'takedark' overrides another one.` line appear at every `moka` call on
`aoloop`; §4 says once that both are expected.

Not on the page: Python equivalents of the moka commands (the old tour's snippets move to
`operate.md`, C5), the AO theory behind mlat / RM / CM (one sentence each, links to `plans.md` §1),
`pipelinebasic`.

### 3.1. Callouts of the opening image

Positions are the top-left corners of slide 2's rectangles, as fractions of the image; the disc goes
there.

| #   | x    | y    | Window                  | Legend                                                             |
| --- | ---- | ---- | ----------------------- | ------------------------------------------------------------------ |
| 1   | 0.02 | 0.02 | `milk-fpsCTRL`          | Engineering control of all compute units. Arrow keys and keyboard. |
| 2   | 0.03 | 0.26 | `milk-streamCTRL`       | Framerate, counters and symlinks of every stream on the machine.   |
| 3   | 0.02 | 0.60 | shell                   | moka working shell.                                                |
| 4   | 0.07 | 0.92 | shell                   | second shell, for the viewers.                                     |
| 5   | 0.48 | 0.05 | `shmImshow dm07disp`    | Total DM output.                                                   |
| 6   | 0.50 | 0.42 | `shmImshow dm07disp03`  | AO loop output.                                                    |
| 7   | 0.49 | 0.70 | `shmImshow dm07disp10`  | Turbulence (FPS `mlatsim_dmturb`).                                 |
| 8   | 0.29 | 0.48 | `shmImshow aol0_psf`    | PSF of the DM phase (`dm_psf.py`).                                 |
| 9   | 0.68 | 0.03 | `shmImshow aol0_imWFS0` | PyWFS frames, 1 kHz (E6).                                          |
| 10  | 0.69 | 0.51 | `shmImshow aol0_imWFS2` | WFS frames: dark subtracted, normalized, reference subtracted.     |

## 4. The capture script

`scripts/docs/tutorial_capture.py`, stdlib plus what moka already needs (pyMilk, matplotlib), plus one
ANSI-to-SVG converter (§7 item 4). It reads the page, never a copy of its commands.

**Block markers.** An HTML comment on the line before a fenced block:

| Marker                                                                             | Block      | The script                                                                                                                        |
| ---------------------------------------------------------------------------------- | ---------- | --------------------------------------------------------------------------------------------------------------------------------- |
| `<!-- tut:run id=deploy-mlatsim -->`                                               | `bash`     | runs it in the working shell; its stdout+stderr go to the `text title="Output"` block marked `<!-- tut:out id=deploy-mlatsim -->` |
| `<!-- tut:run id=... expect=1 -->`                                                 | `bash`     | same; the exit code must be 1 (the refusal of §9)                                                                                 |
| `<!-- tut:py id=close-loop -->`                                                    | `python`   | runs it (D14); no output block needed                                                                                             |
| `<!-- tut:tui id=fpsctrl-mfilt cmd="milk-fpsCTRL -m aoloop_mfilt" size=150x16 -->` | image link | starts `cmd` in a detached tmux pane of that size, waits, `capture-pane -e`, writes `img/moka_tutorial/<id>.svg`, kills the pane  |
| `<!-- tut:fig id=mlat -->`                                                         | image link | writes `<id>.png` with the figure function of E4                                                                                  |
| `<!-- tut:manual -->`                                                              | any        | not run (viewers, `mokashow`, the fpsCTRL key steps, the confdir edit: the script does that edit itself)                          |

Every `bash` and `python` block of the page has exactly one marker.

**Modes.** `--check`: no live system; parses the page and fails on a block without marker, a
duplicate id, a `tut:out` without its `tut:run`, an image link whose file is missing. `--run`: the
full capture. `--from ID`: resume at a stage.

**Run rules.**

- Refuses to start if `~/AOloop2` exists, or if any FPS named `mlatsim_*` / `aoloop_*` or stream
  `dm07disp*` exists. It never deletes anything it did not create.
- Runs blocks in page order, in one shell, `--color=never` for text blocks; TUI captures keep color.
- Text is normalized: `$HOME` to `~`; nothing else. Durations (`done (0.8 s)`) change at each
  capture and are left as they are.
- A block that exits with an unexpected code stops the run; the page is left untouched (outputs are
  written at the end, all or none).
- Ends with teardown of both pipelines and removes `~/AOloop2`, also after a failure.

## 5. Commits

One commit per row, in this order. G0 is a gate, not a commit.

| #   | Commit                                                                                                   | Where    | Delegate |
| --- | -------------------------------------------------------------------------------------------------------- | -------- | -------- |
| G0  | Clean run of today's `sample.sh` in an empty folder, extended by hand to §7 and §12 to §15; report       | spieluhr | Opus     |
| C1  | `writing_docs.md`: Tutorial page type; `documentation.md`: one line                                      | anywhere | Sonnet   |
| C2  | `tutorial_capture.py` with `--check`, its tests                                                          | anywhere | Opus     |
| C3  | The page: text, markers, empty output blocks, placeholder images; nav entry                              | anywhere | Opus     |
| C4  | Captures: `--run`, outputs and images committed; desktop image with callouts                             | spieluhr | Opus     |
| C5  | Remove `tour.md`, `sample.sh`, `sample.md`; re-point links; tour's Python snippets to `operate.md`       | anywhere | Sonnet   |
| C6  | `docs-writer` skill: "Rebuilding the moka tutorial"; page footer; `AGENTS.md` if it lists skill contents | anywhere | Sonnet   |

### G0. Verification run

By hand, in an empty folder, with both TUIs open: every command of §3's table, in order. Report:
the exact output of each; anything refused or failed; the answers to §7. If D9 does not hold (a step
of `sample.sh` fails), stop and report: the page is not written around a workaround.

### C1. Tutorial page type

New `## 10. Tutorial pages` in `writing_docs.md` (or a subsection of §2, whichever reads better), and
§4.2's "Walkthroughs show commands only" gets "except Tutorial pages (§10)". Content:

- what a Tutorial is: one page, one runnable path from nothing to a result, for a newcomer;
- it may show output after every stage, SVG terminal captures and PNG figures, under
  `img/<page name>/` next to the page; every image has alt text and a one-line caption;
- its exploration goes in `tip` boxes titled "Look around"; explanations in `note` boxes;
- its commands are the source: outputs and images are generated from the page by a script, never
  pasted by hand; the page footer says which script;
- a Tutorial is the one place where second person is the default.

**Done when** `mkdocs build --strict` passes and the docs-writer skill's step 6 ("keep output only
where its format is the subject") names the exception.

### C2. The script

As §4. **Done when** py tests pin, on small fixture pages: each `--check` failure of §4; the marker
parser on all six kinds; the in-place rewrite of an output block (other text byte-identical, a second
run changes nothing); `$HOME` normalization; the start refusals, with a fake home and fake FPS list;
all-or-none writing when a block fails. An ANSI capture fixture converts to an SVG that an XML parser
accepts. No test needs milk.

### C3. The page

As §3. Images are placeholders until C4 (a 1 x 1 PNG per id, so `--check` and the strict build pass).
`mkdocs.yml`: `- moka Tutorial: quickstart/moka_tutorial.md` after Installation. `docs/index.md`: the
Pipelines card gains a link to the tutorial.

**Done when** `tutorial_capture.py --check` passes; `mkdocs build --strict` passes; `markdownlint` and
`mdformat` leave the markers and their blocks adjacent (if `mdformat` moves or reflows a marker, stop
and report); each of the four D11 trips appears at least twice; every stage has its four parts.

### C4. Captures

`--run` on spieluhr, then the manual image: slide 1's picture with Picture2 flattened on it, the ten
discs of §3.1 drawn by a `--callouts` mode of the script (Pillow), saved as `desktop.png`.

**Done when** the run ends with both pipelines `cold` and `~/AOloop2` gone; no output block is empty;
every figure and SVG is checked by eye against the page text around it (the fpsCTRL captures show the
FPS and the value the text names); a second `--run` changes only durations and counters.

### C5. Removals

Delete `docs/moka/tour.md` and its nav entry (the `Operate` subgroup keeps `Operating`; if a
one-page subgroup reads badly, flatten it and report). Re-point: `moka.md` (header line, the Operate
card), `operate.md` and `provision.md` (header lines). The old tour's Python snippets (`clone`,
`confresolve`, `init`, `Pipeline`, `run_plan`, `state`) go to `operate.md` as the Python tab of the
matching commands, where not already there. Delete `sample.sh` and `sample.md`; grep the repo for
both names and for `tour.md`.

Also correct `plans.md` §1, `moka takedark`: drop "acquWFS connects to it only when it starts, so a first
dark needs acquWFS restarted"; say that the dark needs only the linkage that gives `aol<loop_number>_wfsim`
its data source. Wording to be confirmed by Vincent in review.

**Done when** `mkdocs build --strict` and `markdown-link-check` pass; the grep is empty.

### C6. Skill and footer

A section "Rebuilding the moka tutorial" in `.agents/skills/docs-writer/SKILL.md`: when (any change
to moka output, to the example confdirs, or to the page's commands); the preconditions of §4's run
rules; `--check`, then `--run` on a machine with milk and cacao; what is manual (the desktop image,
and when to retake it); what to look at in the diff; the two failure cases and what to do. The page
footer links to that section through `github_blob_url`.

**Done when** someone following only the skill section can name the command to run, the machine
state it needs, and the one image it does not produce.

## 6. Verification

```bash
python scripts/docs/tutorial_capture.py --check    # (1)!
npx markdownlint-cli2 docs/quickstart/moka_tutorial.md docs/developer/writing_docs.md
mkdocs build --strict -d /tmp/mkdocs-test            # (2)!
```

1. Markers, ids, images.
2. No broken link after the tour's removal.

Plus, once: a person who did not write the page follows it on a machine other than spieluhr.

## 7. To check on the machine (G0)

Each is a stop-rule item: report, do not work around.

1. **Dark and reference (D16).** Vincent, 2026-11-24: a dark needs only the linkage to the data source
   (`aol0_wfsim`), no `acquWFS` restart; the sentence of `plans.md` that says otherwise is wrong (fixed in C5).
   G0 only records the outputs of both plans.
2. **`dmturb` on demand.** That `moka runstart -p mlatsim dmturb` plus `turbON` is the way, and how
   long the first start takes (it computes the seed screens).
3. **`milk-fpsCTRL` and `milk-streamCTRL` in a detached tmux pane**: that both draw at a fixed size
   without a client attached, that `capture-pane -e` returns their colors, and that
   `milk-fpsCTRL -m <name>` opens on that FPS with its parameters shown (else the capture needs
   navigation keys, which D14 wanted to avoid for setting values only).
4. **ANSI to SVG.** `rich` (`Text.from_ansi`, `Console.export_svg`) as a docs-only dependency, or a
   small converter in the script. `rich` is not in the moka venv (UX plan D2); it must not become a
   moka dependency.
5. **Headless figures.** That `mlat_figure` and `modes_figure` render under the `Agg` backend and
   look like slides 3 to 5.
6. **`mokainit clone --milk`** into `.` twice (two confdirs in one folder), as `sample.sh` does.
7. **`mdformat`** and the markers (C3).
8. **Closed-loop numbers.** With the conf's `loopgain 0.2`, `loopmult 0.99` and `dmturb.ampl 0.1`,
   that the PSF visibly sharpens when the loop closes; else the values §13 tells the reader to set.

## 8. Not in this plan

A second tutorial (hardware, custom plans), a video or GIF, translations, changes to moka or to the
example confdirs (if G0 shows one is needed, it is reported, not done here), the `Operate` nav
redesign beyond C5, capturing through keystrokes.
