# Always compile after editing source code

After modifying **any** C/C++/CMake source file in the milk or cacao
tree, you **must** run the [`/compile-test`](../workflows/compile-test.md) workflow to verify the build
still succeeds before considering the task complete.

If the build fails, fix the errors and rebuild until it passes.
