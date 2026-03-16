1)  Branch Naming (MagAOX)
  - Feature branches must be namespaced by the active username.
  - Format: `<username>/<feature-name>`
  - Example: `jrmales/gui-segfault-fixes`
  - For other users, substitute their username in place of `jrmales`.



# Git usage guidelines

## Misc

No pull-from-remote merges.

## Flow

Follow gitflow model.

- main branch for releases
- dev branch for rolling updates
- secondary dev branches OR use the dev branch of your fork for main instrument/team/collaborator lines.
    - Avoid excessive divergence. Good to release a local feature when the maintainers are otherwise too busy
- feature branches.
  - Naming.
- PRs into the upstream.

- hotfix branches into main or dev

- keep commits focused. One theme, one series of changes. A single commit should never ever mix two of the following:
  1) Repository tooling / administration
  2) Style changes
  3) Software implementation
  4)  Documentation (unless per 3)
  5)  ...

- commit messages. Please use a relevant prefix `[prefix] message` to your commit message.

## PR process

git workflows upon PR to key branches.

Some workflows will just fail and block your PR (e.g. doesn't compile)
- It's highly recommended you run the workflows locally to save time before PRing.
-
Some workflows will authoritatively fix and commit back to the PR branch (e.g. linting)

- keep PRs focused, increment the version number just prior to merge (maybe we can make an action for that), make sure the submodules have the right pointers.

- PR are locked until you pass all the github actions !
  - One of the actions is actually a pre-commit run.
  - Sometimes, you could have a problem because the actions bot may auto-commit on top of your work. This can only happen if you bypassed pre-commit locally.


- Always keep the merge commit of a branch ("merge mode: merge")-- EXCEPTs
  - If there are lots of small commits that could really be just one commit -- use SQUASH mode
  - If there is only one commit AND the upstream branch hasn't moved at all -- use REBASE mode and just add your commit on top of the upstream branch.
