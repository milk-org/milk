# Local Install and Test

**Never** attempt to install to the system prefix
(`/usr/local/`, `/usr/`, etc.). The agent does not
have root privileges and `sudo` is not available.

## How to install and test locally

Use a temporary staging directory inside `_build`:

```bash
cd _build
cmake --install . --prefix _install
```

This populates `_build/_install/` with `bin/`,
`lib/`, `include/`, etc.

## Running binaries from the local install

Set `PATH` and `LD_LIBRARY_PATH` to pick up the
locally installed binaries and libraries:

```bash
export PATH="$(pwd)/_install/bin:$PATH"
export LD_LIBRARY_PATH="$(pwd)/_install/lib:$LD_LIBRARY_PATH"
```

Then run the binary under test directly.

## Cleanup

Remove the staging directory when done:

```bash
rm -rf _build/_install
```

## Key points

- Do **not** run `make install` or
  `cmake --install` without `--prefix`.
- Do **not** use `sudo`.
- Do **not** run `milk-setup-caps` when testing without `sudo`, as it requires root privileges
  to set capabilities.
- Disable capability setup in CMake when building/testing locally by configuring with
  `-DSETUP_CAPS=OFF`.
- Do **not** copy binaries into system directories.
- Always use a local `--prefix` under `_build/`.
