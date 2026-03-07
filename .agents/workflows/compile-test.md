---
description: Compile and test milk after source code changes
---

# Compile and Test milk

Run this workflow after **any** source code change in the milk (or cacao) tree.

## Steps

// turbo-all

1. Run the incremental build from the existing `_build` directory:

```bash
cd /home/oguyon/src/milk/_build && cmake --build . -- -j$(nproc)
```

2. If the build **fails**, read the compiler errors, fix them, and re-run
   step 1 until the build succeeds.

3. After a successful build, report the result to the user (number of
   warnings, if any).
