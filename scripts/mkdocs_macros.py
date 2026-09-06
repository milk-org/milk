"""mkdocs-macros-plugin hook module.

Exposes variables usable in markdown pages as {{ variable_name }}.
"""

import re


def define_env(env):
    repo_url = (env.conf.get("repo_url") or "").rstrip("/")
    edit_uri = env.conf.get("edit_uri") or ""

    branch_match = re.match(r"edit/([^/]+)/", edit_uri)
    branch = branch_match.group(1) if branch_match else "framework-dev"

    env.variables["repo_branch"] = branch
    env.variables["github_blob_url"] = f"{repo_url}/blob/{branch}"
