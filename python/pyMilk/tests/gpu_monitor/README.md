# tests/gpu_monitor

Compile the CUPTI spy library before running GPU integration tests:

```sh
make -C tests/gpu_monitor/
# Optional: explicit CUDA path
make -C tests/gpu_monitor/ CUDA_ROOT=/usr/local/cuda
```

Output: `tests/gpu_monitor/libcupti_spy.so`

Tests that use `no_gpu_host_transfers` or `assert_no_gpu_host_transfers`
are automatically skipped (not failed) when the library is absent.
