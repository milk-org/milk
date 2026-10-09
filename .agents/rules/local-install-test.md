# Local Install and Test

**Never** attempt to install to the system prefix
(`/usr/local/`, `/usr/`, etc.). The agent does not
have root privileges and `sudo` is not available.

## How to install and test locally

Follow [Local install without sudo](../../docs/developer/tooling.md#6-local-install-without-sudo)
(install, environment variables, cleanup).

## Key points

- Do **not** run `make install` or
  `cmake --install` without `--prefix`.
- Do **not** use `sudo`.
- Do **not** run `milk-setup-caps` when testing without `sudo`, as it requires root privileges
  to set capabilities.
- Do **not** copy binaries into system directories.
- Always use a local `--prefix` under `_build/`.
