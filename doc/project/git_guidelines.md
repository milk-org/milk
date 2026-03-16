# Git usage guidelines

We follow the _gitflow_ model. More information [here](https://www.atlassian.com/en/git/tutorials/comparing-workflows/gitflow-workflow)

- `main` branch for releases
- `dev` branch for rolling updates
- secondary dev branches e.g. `scexao-dev`, which are equivalent to the `dev` branch of a forked repository.
- feature branches. Naming convention is loose, but prefer prefixing by your initials, `feature/`, `doc/`, `hotfix/`, or another thematic keyword.

The workflow is one of upstream merging through github pull requests.

## Rules that apply to your local machine.

- Configure your `git config pull.rebase true`. We do not want merge commits between the local and remote refs of the same branch. Ever.


## Commits

A commit should be focused on a single _intent_, which can be one of the following:
- Fixing a bug,
- Implementating a new function,
- or feature,
- or the skeleton thereof,
- typos,
- Writing or elaborating in documentation,
- etc.

Commit messages should include a `[<tag>] <message>`, e.g. `[doc] added X.md page`, `[fix] missing #include`.

Long sequences of debug / incremental commits should be squashed and/or fixup'd. Look up _git rebase -i_.

> [!NOTE]
> If going back and forth between your local machine and a testing server, it is convenient to
> - Make a temporary branch just for this purpose
> - Use ssh-copy-id to forward you github identity to the server
> - At the end, perform an _interactive rebase_ and _squash_ all the debug commits into one.

## Pull requests

A PR should as much as possible also follow the _single intent_ rule, although with broader scope:
- Moderate or large documentation updates.
- Documentation policy updates.
- CI/workflow change.
- Hotfixes directly into both `dev` and releases.
- Repository repackaging.
- Infrastructure (core engine) repackaging.
- Change of tooling
- Core engine feature development.
- Extension feature development.
- Systematic test, style, or documentation maintenance.

If using AI, see `ai_guidelines.md` on prompt requirements.

Increment the version number just prior to merge (FIXME: maybe we can make an action for that), make sure the submodules have the right references.

### PR Process

- Open pull requests into the appropriate upstream branch.
- See above for branch names
  - Collaborative branches should be `<intent>/<topic>` e.g. `feature/new_super_nice_hack`
  - Solo branches should be `<name>/<feature>` e.g. `vd/boring-repo-admin`
- Perform your development. You can open a draft PR to have a place to engage in conversation with other contributors, users, or maintainers.
- Run checks locally, preferably before submitting the PR / changing it to non-draft. You can definitily push to the remote so as to have backup and/or transit branch to a testing environment.
- We will implement __blocking__ workflows for build, test failures, and various code hygiene criteria. They are blocking in the sense that Github won't let you merge.

### PR merging

When merging a pull request on github, there are [three merging options](https://nausaf.hashnode.dev/types-of-merges-in-a-github-pull-request):
- Always keep the merge commit of a branch ("merge mode: __merge__") _except_
- if there are lots of small commits that could really be just one commit -- use __squash__ mode;
- if there is only one commit AND the upstream branch hasn't moved at all -- use __rebase__ mode and just add your commit on top of the upstream branch.


### Git workflows

Git workflows are automations that run upon certain actions occuring on github, typically pushes onto and pull requests into important branches. They are described by workflow YAML files located in `.github/workflows`.

`main`/`master` and `dev` are protected: one cannot push commit onto them, and one cannot merge PRs into them _unless_ all the automated checks pass.

Some workflows will just fail and block your PR (e.g. doesn't compile).
It's highly recommended you run the workflows locally to save time before PRing.

The pre-commit workflow runs the identical process as described in `.pre-commit-config.yaml` and that is executed locally upon each commit if you have dutyfully installed and activated pre-commit.


> [!NOTE] the pre-commit workflow, in PR context, will authoritatively fix non-compliant files and pre-commit on top of the PR branch (e.g. linting changes). This is identical to fixing files by running `pre-commit` locally.
> Unfortunately, if _the last commit of your PR_ is an automatic linting fix made by the github bot, the workflows won't run on it, and your PR won't pass, and merging will never become allowed. The fix:
> - `git pull` the updated branch to your local machine
> - `git rebase -i HEAD~1` and select `fixup` for the linting commit
> - `git push -f`
> And now, the last commit on the PR will have the check ran again.
