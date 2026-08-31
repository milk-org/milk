# Developer tooling

## 1. Building documentation locally

The pure-markdown render of the docs can be seen with any usual preview tooling you may have locally.

It is sometimes quite useful to do a full website render of the documentation; here is how:

### 1.1 Install `mkdocs`

The preferred way is to use your daily python environment, or a dedicated one.

!!! Note
`mkdocs` can also be install with `apt`; in which case the `pip` commands for the extensions below must be run with `sudo pip3`.

```bash
# environment creation
mamba env create -n mkdocs
mamba activate mkdocs

# install mkdocs
pip install mkdocs

# install extensions used herein
pip install pymdown-extensions \
    mkdocs-material[plugins] \
    mkdocs-glightbox \
    mkdocs-git-revision-date-localized-plugin \
    mkdocs-minify-plugin
```

### 1.2 Build and serve the documentation

```bash
# Build and serve the website
mkdocs serve

# Access -- follow the link or
<your browser> http://127.0.0.1:8000/
```

You can leave `mkdocs serve` running and the broswer open while modifying documentation source file.
The render will update continuously.

## 2. (Future) nox, pytest, coverage.

TODO.

## 3. (Future) License checks with REUSE
