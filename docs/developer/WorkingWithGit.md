# Working with git 

> [!NOTE]
> This file: `docs/developer/WorkingWithGit.md`



***

# 1. Source Code and Modules 

Most of milk's code is organized in directories under `src/` (core modules)
and `plugins/` (optional modules). Modules are compiled as shared objects
loaded by the main process at runtime.

Some plugin directories (e.g., `plugins/cacao-src`) may be symbolic links
to external source trees or git submodules.


## 1.1. Standard development workflow 

The primary branches are:
- **`main`**: Stable release branch.
- **`dev`**: Active development branch.

When developing, work in the `dev` branch:

	$ git checkout dev
	$ git pull
	# ... make changes ...
	$ git push


## 1.2. Updating main branch 

To synchronize `main` to latest `dev`:

	$ git checkout main
	$ git merge dev
	$ git push


## 1.3. Releasing a new version 

In `dev` branch:
- Update version number in `CMakeLists.txt`
- Update version information in `README.md`

Merge `dev` into `main`, then tag:

	$ git checkout main
	$ git merge dev
	$ git tag -a vX.YY.ZZ -m "milk version X.YY.ZZ"
	$ git push origin main --tags

> [!NOTE]
> Modules that are shared between packages (e.g., `milk` and `cacao`) can have
> parallel version number histories. Any new version, regardless of which
> package it is associated with, includes all previous changes.



***


# 2. Source Code Documentation (doxygen) 

For generating HTML source code documentation via Doxygen, refer to the
up-to-date guide in [DocumentingCode.md](DocumentingCode.md).
