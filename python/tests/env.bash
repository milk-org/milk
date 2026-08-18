# Dev-only: source this to match the env the pytest suite spoofs.
# Keep in sync with ctfixt_change_MILK_SHM_DIR / ctfixt_change_tmux_server
# in conftestaux/milk.py.

export MILK_SHM_DIR=/tmp/milk_shm_dir_pytest
export MILK_PROC_DIR=/tmp/milk_shm_dir_pytest
export TMUX_TMPDIR=/tmp/milk_tmux_tmpdir_pytest
unset TMUX
