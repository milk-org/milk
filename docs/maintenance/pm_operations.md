---
render_macros: true
---

# Project management operations

These are things maintainers should watch for during a release.

## Documentation branch

The published docs site is versioned with [mike](https://github.com/jimporter/mike):
each branch listed in `DOCS_DEPLOY_BRANCHES` in
[.github/workflows/docs.yml]({{ github_blob_url }}/.github/workflows/docs.yml)
is deployed to its own subfolder of `gh-pages` and appears in the site's version dropdown;
`DOCS_DEFAULT_VERSION` (`framework-dev`) is where the site root redirects.
`mkdocs.yml`'s `edit_uri: edit/framework-dev/docs/` records that branch, and
[scripts/mkdocs_macros.py]({{ github_blob_url }}/scripts/mkdocs_macros.py)
parses it from there to build the `github_blob_url` links to files outside `docs/`.

Any other doc that needs to link to a file outside `docs/` should use the same
`{{ '{{ github_blob_url }}' }}` macro variable (defined in
`scripts/mkdocs_macros.py`)
rather than a relative path — relative `../` links to files outside `docs_dir` are not
part of the mkdocs build and will 404 on the published site. Since macro rendering is
opt-in (`render_by_default: false` in `mkdocs.yml`), add `render_macros: true` to the
page's front matter to use it.

If the branch that carries the docs changes (e.g. at a release), update it in one
place — `edit_uri` in `mkdocs.yml`.
