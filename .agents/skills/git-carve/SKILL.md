---
name: git-carve
description: Carve one feature's commit out of a dirty git checkout that mixes several - list pending things, stage hunks chunk by chunk, commit on approval, optionally backport (cherry-pick) to an upstream branch or backporting-PR branch in a dedicated worktree, with safety snapshots. Sub-actions - pending, stage, unstage, commit, backport, snapshots. Use whenever the user is ready to commit, asks what is pending or uncommitted, wants only part of their changes staged or committed, or mentions a backport, cherry-pick to upstream, or upstream PR.
---

# git-carve

The user works on several features at once in one checkout. Carve one coherent commit out of the mix without disturbing the rest; optionally carry it to an upstream branch.

- **User's checkout**: staging and the commit happen here, on the current branch.
- **Your worktree** (`../<repo>-claude-backport`): backports only, after the commit exists.

## Sub-actions

Each does its step, then stops.

| Keyword                          | Does                                                | Stops after                            |
| -------------------------------- | --------------------------------------------------- | -------------------------------------- |
| `pending`                        | theme analysis                                      | the numbered list; index untouched     |
| `stage <n>`                      | snapshot, stage thing(s) `n`                        | reporting staged / left out; no commit |
| `unstage [<n>]`                  | snapshot, remove from index                         | reporting index state                  |
| `commit`                         | propose message for current index                   | snapshot + commit on approval + checks |
| `backport direct\|pr [<commit>]` | cherry-pick in your worktree (default: last commit) | local result; ask before push          |
| `snapshots`                      | list / restore / prune snapshots                    | the result                             |

- Plain words work too ("stage the dry-run stuff"). `<n>` = numbering of the latest pending list.
- `stage` with no list yet, or a tree that changed since: redo the analysis first.
- No keyword ("ready to commit"): pending → wait for pick → stage → propose message → commit on approval. Backport only when asked.

## Ground rule

Uncommitted work in the user's checkout exists nowhere else. There, change only the index and create commits. Never: `git stash`, `git checkout -- <path>`, `git restore` without `--staged`, `reset --hard`, `clean`, branch switch, rebase, `commit -a`. If a step seems to need one, ask. Sole exception: restoring damaged paths from a snapshot, after telling the user.

## Safety snapshot

Protects other themes from a bad patch or a pre-commit hook. Take one before `stage`/`unstage` and right before `git commit`. Plumbing only: touches no working tree, index, HEAD, stash list; runs no hooks.

```bash
carve_idx() {       # temp index = full working tree (tracked + untracked - ignored - $CARVE_EXCLUDE)
  local tmp; tmp=$(mktemp); cp "$(git rev-parse --git-path index)" "$tmp"
  ( cd "$(git rev-parse --show-toplevel)" && GIT_INDEX_FILE="$tmp" git add -A -- . $CARVE_EXCLUDE ) && echo "$tmp"
}
carve_snapshot() {  # $1 = stage | unstage | commit ; prints the ref
  local t i c ref
  t=$(carve_idx) || return 1
  i=$(git commit-tree "$(git write-tree)" -p HEAD -m "git-carve: index before $1") &&
  c=$(git commit-tree "$(GIT_INDEX_FILE="$t" git write-tree)" -p HEAD -p "$i" -m "git-carve: snapshot before $1") &&
  ref=refs/carve/$(date +%Y%m%d-%H%M%S)-$1-${c:0:7} && git update-ref "$ref" "$c" && echo "$ref"; rm -f "$t"
}
carve_check() {     # $1 = ref ; rc 0 and silent if working tree identical, else lists what moved
  local t rc; t=$(carve_idx) || return 1
  GIT_INDEX_FILE="$t" git diff --cached --stat --exit-code "$1"; rc=$?; rm -f "$t"; return $rc
}
```

- `$ref` = working tree; `$ref^2` = index as it was. Name the ref once, briefly.
- **Large untracked files get copied into `.git`.** First run `git ls-files --others --exclude-standard -z | xargs -0 -r du -ch | tail -1`. Above ~50 MB: tell the user, and exclude the big ones for the whole run with `export CARVE_EXCLUDE=':(exclude)path1 :(exclude)path2'` (those files are then unprotected; say so).
- Not captured: ignored files; uncommitted changes inside submodules. If a snapshot fails (no commits yet, unmerged index), stop and report; do not proceed unprotected.
- **Check**: `carve_check "$ref"` must be silent after stage/unstage. After a commit attempt, output means a hook touched files. Do not use plain `git diff "$ref"` for this: it shows untracked files as deleted.
- **On mismatch**: read `git diff "$ref" -- <path>` for the listed paths. Changes confined to the committed hunks (formatter) are expected: report. Anything else is damage.
- **Restore**: tell the user, then `git restore --source="$ref" --worktree -- <paths>` (index: `--source="$ref^2" --staged`). File mixing hook fix and damage: show the diff, let the user choose.
- **Housekeeping**: `git for-each-ref refs/carve/`. Keep after success; when taking a new one, `git update-ref -d` those older than 14 days.

## pending

```bash
git status --short --branch
git diff; git diff --cached --stat          # read the hunks, not just file names
git ls-files --others --exclude-standard
git log --oneline @{upstream}..HEAD 2>/dev/null; git stash list
```

Group changes into "things" (one feature/fix/chore = one sensible commit). State the source of each grouping:

- **AI work history** (strongest: records intent): what you or earlier AI turns, session/project notes and plans, edited and why.
- **Best guess**: identifiers, paths, comments, tests with their code, build-file changes with new sources, branch name, recent commit subjects.

Split at hunk level, not file level. Prefer an "Unattributed" bucket to a wrong guess.

```
Pending in <repo> on <branch>  (N files changed, M untracked)

1. <thing>   [source: AI work | best guess]  [confidence: high/medium/low]
   <one line: what it does>
   - path/file.c   (2 of 5 hunks: <what>)
   - path/new.h    (new)

Unattributed
   - path/other.py (1 hunk: <what>)

Already staged: <none | what>      Unpushed commits: <none | n>
```

## stage

Snapshot first. `git add -p` is interactive; apply patches to the index instead:

- **Whole file**: `git add -- <path>`.
- **Some hunks / part of a hunk**: start from `git diff -- <path>` (with context). Keep the header (`diff --git`, `---`, `+++`), drop unwanted hunks, and inside a mixed hunk delete unwanted `+` lines and turn unwanted `-` lines into context (leading space). Keep the remaining lines in the order they have in the working file. Apply with `git apply --cached --recount edited.patch` (context places the hunk; stale `@@` numbers are tolerated).
- **Do not apply a subset of `git diff -U0` hunks**: pure-insertion hunks are placed by line number, land at the wrong line once an earlier hunk is skipped, and git reports success. `-U0` is fine for *reading* fine-grained hunks.
- **Alternative for small files**: write the intended staged content to a temp file, then `git update-index --cacheinfo 100644,$(git hash-object -w tmp),<path>` (keep the file's real mode).
- **Part of an untracked file**: `git add -N -- <path>`, then as above.
- **Renames/deletions**: `git add -- <old> <new>`; confirm with `git status`.
- **Undo**: `git restore --staged -- <path>`, or `git apply --cached -R --recount` with the same patch.

Rules:

- Pre-existing staged content: report it, ask whether it belongs.
- Unsure about a hunk: leave it out and say so (a missed hunk costs a round; a wrong one costs a history fix).
- Flag staged code depending on unstaged hunks (commit would not build). To verify, build from `git checkout-index -a --prefix=/tmp/staged-check/`, never in the checkout.

Then verify, every time, per touched file:

- `git diff --cached -- <path>`: only the intended lines, at the right place.
- `git diff -- <path>`: exactly the other themes' changes remain, nothing shows as moved.
- `carve_check "$ref"` is silent.

Report: staged (per file, which hunks, plain words), left out, unsure. Apply adjustments the user asks for and re-report.

## commit

1. Re-read `git diff --cached` (the user may have changed the index). Nothing staged: say so.
2. Propose a message in the repo's style (`git log -n 15 --format='%s%n%b'`). Wait for approval or edits.
3. `ref=$(carve_snapshot commit)`, then `git commit -F <file>`. Do not push.
4. Hooks see a partially staged tree. No `--no-verify` unless the user asks. If the commit is rejected:
    - Auto-fixes left in the working tree: stage only the fixes inside what is being committed, retry.
    - Fixes rolled back by the hook framework ("Stashed changes conflicted with hook auto-fixes"): retrying cannot pass. Apply the fix to the staged content only (`--cacheinfo` technique) and retry; ask before touching the same lines in the working tree.
    - Real problem: report it.
5. After the attempt, two checks:
    - `carve_check "$ref"`: did a hook alter the working tree? Report or restore as above.
    - `[ "$(git rev-parse HEAD^{tree})" = "$(git rev-parse "$ref^2^{tree}")" ]`: is the commit exactly the approved index? If not, a hook added content (some `git add` whole files, sweeping in other themes): show `git diff "$ref^2" HEAD` and ask whether to redo it (`git reset --soft HEAD^` keeps the working tree) or accept.
6. Report: short hash, both check results in one line, updated pending list.

## backport

Cherry-pick of an existing commit; never from uncommitted or staged changes (run `commit` first). No snapshot needed: the user's checkout is not touched. Ask which form if unstated:

- **`direct`**: onto the upstream branch itself.
- **`pr`**: onto a new branch from upstream, to open a PR.

**Target**: ask each time, offering a best guess `<remote>/<branch>` (from `git remote -v`, `git branch -vv`, `git branch -r`, fork point).

**Worktree** (persistent: create once, reuse, never remove):

```bash
top=$(git rev-parse --show-toplevel); wt="$(dirname "$top")/$(basename "$top")-claude-backport"
git worktree list                                   # reuse if present
git worktree add --detach "$wt" <remote>/<branch>   # first time only
```

- Run everything as `git -C "$wt" ...`.
- Before use: `git -C "$wt" status`. Leftover changes or a cherry-pick in progress: report, do not wipe.

**Steps**:

```bash
git fetch <remote>; src=$(git rev-parse --short=12 <commit>)
# pr:
git -C "$wt" switch -c backport/<short-topic> <remote>/<branch>
# direct, local <branch> not checked out elsewhere (this advances the user's local <branch>):
git -C "$wt" switch <branch> && git -C "$wt" merge --ff-only <remote>/<branch>   # cannot ff: stop, ask
# direct, <branch> checked out in a user worktree:
git -C "$wt" switch --detach <remote>/<branch>      # report the hash; user decides how it lands

git -C "$wt" cherry-pick <commit>

kind="direct backport"                               # or: backporting PR
full=$(git -C "$wt" log -1 --format=%B); tr=$(git -C "$wt" log -1 --format='%(trailers)')
if [ -n "$tr" ]; then printf '%s(cherry-pick of %s, %s)\n\n%s\n' "${full%"$tr"}" "$src" "$kind" "$tr"
else printf '%s\n\n(cherry-pick of %s, %s)\n' "$full" "$src" "$kind"; fi > /tmp/carve-msg.txt
git -C "$wt" commit --amend -F /tmp/carve-msg.txt
```

- The marker goes after the body and before the trailer block (`Co-Authored-By:` etc.); appended after it, git no longer recognises the trailers. Check with `git -C "$wt" log -1 --format=%B`.
- Hooks are shared by all worktrees: `cherry-pick` does not run pre-commit, the `--amend` does. If it rejects or alters the backport, report; do not bypass without asking.
- Several commits: oldest first, one marker each.

**Conflicts** (usually a missing prerequisite upstream):

- Mechanical and obvious (context drift, renamed include): resolve, continue, say what changed.
- Otherwise: leave in progress, show the conflict, name prerequisites (`git log --oneline <remote>/<branch>..<commit>^ -- <paths>`), ask: backport those too, adapt, or abort.

**Finish**: report worktree path, branch or detached hash, new hash, final message, diff stat vs `<remote>/<branch>`. Ask before any push. If agreed: direct → push to the target branch; pr → push `backport/...` and offer `gh pr create --base <branch>` (title = original subject, body names the source commit). Never force-push.
