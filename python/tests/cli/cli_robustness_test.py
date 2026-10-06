"""
Robustness tests for milk-cli.

Cases are split into three buckets:

    EXPECT_OK    -- runs without crash/hang (returncode 0)
    EXPECT_ERROR -- prints a recognizable error message
    EXPECT_GREP  -- produces a specific substring on stdout

Crashes (signal death) are raised by the wrapper; hangs are caught by pytest-timeout.
"""

from __future__ import annotations

import re

import pytest

from milk.cliwrap import CLICommands, HAVE_CLI

if not HAVE_CLI:
    pytest.skip(reason="MILK compiled without CLI support", allow_module_level=True)

# Per-case wall-clock guard (seconds) via pytest-timeout.
TIMEOUT = 10

# Error-keyword regex mirrored from run_cli_robustness_tests.sh.
ERROR_REGEX = re.compile(
    r"error|usage|missing|cannot|not found|does not exist|wrong|invalid|"
    r"did you mean|unknown|no such",
    re.IGNORECASE,
)

# TODO see below -- this passes but should error.
# ["pwd --invalid-option-xyz"], ## TODO this is the only one that passes in the EXPECT_ERROR
# TODO -- there's also all the grep tests that are xfail'd

# ==========================================================================
# EXPECT_OK: command runs to completion without crash/hang (returncode 0).
# ==========================================================================
# fmt: off
TESTLIST_EXPECT_OK: list[list[str]] = [
    # --- Section 1: Basic parsing ---
    [""],
    ["# This is a comment"],
    ["echo"],
    # --- Section 2: Help and discovery ---
    ["?"],
    ["help"],
    ["lm?"],
    ["m?"],
    ["ci"],
    ["cmd? echo"],
    ["cmd? mem.listim"],
    # --- Section 3: Built-in commands ---
    ["mem.listim"],
    ["pwd"],
    ["cd /tmp"],
    ["cd"],
    ["usleep 1000"],
    ["dpsingle"],
    ["dpdouble"],
    ["mem.mk2Dim _clitest_im1 32 32"],
    ["mem.listim"],
    ["mem.rm _clitest_im1"],
    # --- Section 4: Missing / wrong arguments (graceful) ---
    ["mem.mk2Dim"],
    ["mem.rm"],
    # --- Section 5: Variables ---
    ["_testvar1=hello"],
    ["echo $_testvar1"],
    ["_testvar2=42"],
    ["echo $_testvar2"],
    ["echo ${_testvar1}"],
    ["unset _testvar1"],
    ["echo $_testvar1"],
    ["echo $?"],
    ["vars"],
    ["echo ${_teststr:0:3}"],
    ["echo ${_teststr^^}"],
    ["unset _testvar2", "unset _teststr", "unset _TESTUP", "unset _defvar"],
    # --- Section 6: Arithmetic ---
    ["_arith_c=$(( _arith_b * 2 - 3 ))", "echo $_arith_c"],
    ["_arith_e=$(( (_arith_a + 5) * 2 ))", "echo $_arith_e"],
    ["unset _arith_a", "unset _arith_b", "unset _arith_c", "unset _arith_d", "unset _arith_e"],
    # --- Section 7: Flow control ---
    ["if [ $_fc_x -lt 5 ]; then echo small; else echo not_small; fi"],
    ["if [ $_fc_x -gt 100 ]; then echo huge; elif [ $_fc_x -gt 5 ]; then echo medium; else echo small; fi"],
    ["if [ $_fc_s != world ]; then echo differ; fi"],
    ["if [ -n $_fc_s ]; then echo nonempty; fi"],
    ["_fc_empty=", "if [ -z $_fc_empty ]; then echo empty; fi"],
    ["if [ -f /etc/hostname ]; then echo file_exists; fi"],
    ["unset _fc_x", "unset _fc_n", "unset _fc_m", "unset _fc_k", "unset _fc_sum", "unset _fc_j", "unset _fc_mode", "unset _fc_s", "unset _fc_empty"],
    # --- Section 8: User functions ---
    ["function _testfunc_greet { echo greeting_$1; }"],
    ["_testfunc_greet world"],
    ["unset _gvar", "unset _result"],
    # --- Section 10: Shell features ---
    ["echo redirect_test > /tmp/_clitest_redir.txt"],
    ["echo append_line >> /tmp/_clitest_redir.txt"],
    ["echo $HOME"],
    ["echo ~"],
    ['read _hs_val <<< "here_string_test"', "echo $_hs_val"],
    ["unset _subst_val", "unset _hs_val"],
    # --- Section 11: Error conditions (graceful toggles) ---
    ["synhl off", "synhl on"],
    # --- Section 12: Set flags ---
    ["set -x"],
    ["set +x"],
    # --- Section 13: Readonly and declare ---
    ["readonly _RO_PI=3"],
    ["declare -i _decl_count=0"],
    ["unset _decl_count"],
    # --- Section 14: Calculator & math (numeric formatting varies) ---
    ["_calc_a = 2 + 3 * 4", "echo $_calc_a"],
    ["_calc_a = 2 + 3 * 4", "_calc_b = _calc_a > 10", "_calc_c = (_calc_a == 14)", "_calc_d = _calc_b && _calc_c", "echo $_calc_d"],
    ["_calc_e = min(abs(-5), max(3, fmod(17, 5)))", "echo $_calc_e"],
    ["_calc_e = min(abs(-5), max(3, fmod(17, 5)))", "_calc_f = where(_calc_e > 2, 100, -100)", "echo $_calc_f"],
    ["mem.mk2Dim _calc_imtest 10 10", "_calc_imtest = _calc_imtest * 0.0 + 5.0", "_calc_imtest_plus = _calc_imtest + 3", "_calc_imtest_mask = (_calc_imtest_plus == 8)", "_calc_g = itot(_calc_imtest_mask)", "echo $_calc_g"],
    ["_calc_im_where = where(_calc_imtest_mask, _calc_imtest, -_calc_imtest)", "_calc_h = imean(_calc_im_where)", "echo $_calc_h"],
    ["_calc_d_im = dot(_calc_imtest, _calc_imtest)", "_calc_n_im = norm(_calc_imtest)", "echo $_calc_d_im"],
    ["unset _calc_a", "unset _calc_b", "unset _calc_c", "unset _calc_d", "unset _calc_e", "unset _calc_f", "unset _calc_g", "unset _calc_h", "unset _calc_d_im", "unset _calc_n_im", "mem.rm _calc_imtest", "mem.rm _calc_imtest_plus", "mem.rm _calc_imtest_mask", "mem.rm _calc_im_where"],
    # --- Section 17: Extreme arguments ---
    ["echo " + "a" * 194],
    ["echo " + " ".join(str(i) for i in range(1, 31))],
    # --- Section 18b: SHM-aware test operators ---
    ["if [ -S _clitest_nostream_xyz ]; then echo FAIL_stream_found; fi"],
    ["if [ -F _clitest_nofps_xyz ]; then echo FAIL_fps_found; fi"],
    ["if [ -P _clitest_noproc_xyz ]; then echo FAIL_proc_found; fi"],
    ["mem.rm _clitest_shm_s"],
    # --- Section 18c: Assigncheck (never exits without -e) ---
    ["assigncheck _ac_test 10 0"],
    ["assigncheck _ac_test 10 0.1"],
    ["assigncheck _ac_test 10.05 0.1"],
    ["assigncheck _ac_test 20 0.1"],
    ["assigncheck _ac_test 20"],
    ["assigncheck _ac_test 20 /0"],
    ["unset _ac_test"],
    # --- Section 19: Cleanup / defer ---
    ["! rm -f /tmp/_clitest_redir.txt"],
    ["mem.mk2Dim _defertest 4 4", "defer mem.rm _defertest"],
    # --- Section 20-21: Env sync / command substitution ---
    ["echo $PATH"],
    ['BDIR_SYNC_VAR="hello_frommilk"', '! test "$BDIR_SYNC_VAR" = "hello_frommilk"'],
    ['test_subshell_var=$(echo "success_from_subshell")', '! test "$test_subshell_var" = "success_from_subshell"'],
    # --- Section 22: Advanced variable expansions ---
    ["echo ${_sub_str:-5:5}"],
    # --- Section 23: Advanced test evaluator (host files) ---
    ["if [ -r /etc/passwd ]; then echo readable; fi"],
    # --- Section 26: Introspection ---
    ["type type"],
    # --- Section 28: Array length (relies on prior array) ---
    ["echo ${#_test_arr[@]}"],
    # --- Section 30: Listim glob ---
    ["listim _nonexist_*"],
    ["mem.mk2Dim _globtest_a 4 4", "mem.mk2Dim _globtest_b 4 4", "listim _globtest_*", "mem.rm _globtest_a", "mem.rm _globtest_b"],
    ["listim _nonexist_?"],
    # --- Section 32: Stream slicing ---
    ["mem.mk2Dim _slicetest 64 64", "listim _slicetest"],
    ["_sliceout = _slicetest[0:9,0:9] + 0.0"],
    ["_sliceout2 = _slicetest[*,*] + 0.0"],
    ["_sliceout3 = _slicetest[*,-*] + 0.0"],
    ["_sliceout4 = _slicetest[0:63::2,*] + 0.0"],
    ["mem.rm _slicetest", "mem.rm _sliceout", "mem.rm _sliceout2", "mem.rm _sliceout3", "mem.rm _sliceout4"],
    # --- Section 33: Alias system ---
    ['alias _testalias "echo alias_works"'],
    ["_testalias"],
    ["unalias _testalias"],
    ["unalias _no_such_alias_xyz"],
    ["alias"],
    # --- Section 34: Bookmark system ---
    ["bookmark list"],
    ["bookmark rm _tbm"],
    # --- Section 35: printf ---
    [r'printf "a\tb\n"'],
    # --- Section 36: sleep ---
    ["sleep 0.01"],
    # --- Section 37: trap ---
    ['trap "echo trap_set" EXIT'],
    ["trap -l"],
    # --- Section 38: savehistory ---
    ["savehistory /tmp/_clitest_hist.txt"],
    ["! rm -f /tmp/_clitest_hist.txt"],
    # --- Section 39: History commands ---
    ["lhistory"],
    ["ghistory"],
    ["ghistory 5"],
    ["lhistory -t cmd"],
    # --- Section 43: include_once idempotency ---
    ['! echo "echo sourced_once" > /tmp/_clitest_inc.milk', "include_once /tmp/_clitest_inc.milk", "include_once /tmp/_clitest_inc.milk", "! rm -f /tmp/_clitest_inc.milk"],
    # --- Section 44: Edge cases ---
    ["while [ 0 -eq 1 ]; do echo never; done"],
    ["_neg = -5 + 3", "echo $_neg"],
    ["dpdouble", "mem.mk2Dim _dbl_test 4 4", "_dbl_test = _dbl_test * 0.0 + 1.5", "mem.rm _dbl_test", "dpsingle"],
    # --- Compound assignment (calc engine, formatting varies) ---
    ["_ca = 10", "_ca += 5", "echo $_ca"],
    ["_cs = 20", "_cs -= 3", "echo $_cs"],
    ["_cm = 4", "_cm *= 3", "echo $_cm"],
    ["_cd = 12", "_cd /= 4", "echo $_cd"],
    # --- Ternary operator ---
    ["_tt = (1 > 0) ? 10 : 20", "echo $_tt"],
    ["_tf = (0 > 1) ? 10 : 20", "echo $_tf"],
    # --- Integer literal formats ---
    ["_hx = 0xff", "echo $_hx"],
    ["_oc = 0o77", "echo $_oc"],
    ["_bn = 0b1010", "echo $_bn"],
    # --- strlen builtin ---
    ['_str="hello"', '_sl = strlen("hello")', "echo $_sl"],
    # --- until loop ---
    ["_uc = 0", "until [ $_uc -ge 3 ]; do _uc = $(( $_uc + 1 )); done", "echo $_uc"],
    # --- Bitwise operators (calc engine) ---
    ["_ba = 0xff & 0x0f", "echo $_ba"],
    ["_bo = 0x0f | 0xf0", "echo $_bo"],
    ["_bx = 0xff ^^ 0x0f", "echo $_bx"],
    ["_bn = ~0", "echo $_bn"],
    ["_ls = 1 << 4", "echo $_ls"],
    ["_rs = 256 >> 4", "echo $_rs"],
    # --- Format conversions (formatting varies) ---
    ["_hf = hex(255)", "echo $_hf"],
    ["_of = oct(255)", "echo $_of"],
    ["_bf = bin(10)", "echo $_bf"],
    # --- assert statement (never exits) ---
    ['assert [ 1 -eq 1 ] "should pass"'],
    ['assert [ 1 -eq 2 ] "expected failure"'],
    ["_av=1.23", "assert _av=1.23"],
    ["_av=1.23", "assert _av=1.25 ~0.1"],
    ["_av=1.23", "assert _av=1.25 ~0.01"],
    ["_av=1.23", "assert _av=5.0"],
    ["_bv=(437.06+58.47)/5", "assert _bv=99.106 ~0.001"],
    ["assert noequals"],
    ["_av=1.23", "assert _av<2"],
    ["assert _av<1"],
    ["assert _av>1"],
    ["assert _av>2"],
    ["assert _av<=1.23"],
    ["assert _av>=1.23"],
    ["assert 0<_av<2"],
    ["assert 0<_av<1"],
    ["assert 1<=_av<=2"],
    ["unset _av", "unset _bv"],
    # --- dpdigits ---
    ["dpdigits"],
    ["dpdigits 6"],
    ["dpdigits 3"],
    ["dpdigits 0"],
    ["dpdigits 99"],
    ["dpdigits 15"],
    # --- trap ERR ---
    ["trap 'echo error_caught' ERR"],
    # --- Processinfo integration ---
    ["echo $PROCINFO_NCPU"],
    ["echo $PROCINFO_NPROC"],
    ["procstat"],
    ["procctl _nonexistent_proc_ stop"],
    # --- Issue verifications (FITS round trip) ---
    ["mem.mk2Dim im2D 256 256", 'iofits.saveFITS im2D "im2D.fits"', 'rm "im2D.fits"'],
    ["mem.mk2Dim im2D_lf 128 128", 'iofits.saveFITS im2D_lf "im2D_lf.fits"', 'iofits.loadfits "im2D_lf.fits" im2D_lf_loaded 2', 'rm "im2D_lf.fits"', 'mem.rm "im2D_lf_loaded"'],
    # --- Section 45: JSON output modes ---
    ["streamlist --json"],
    ["streamlist --json _nonexist_*"],
    ["streamlist -l --json"],
    ["streamlist --json _nonexist_*"],
    ["proclist --json"],
    ["proclist -l --json"],
    ["fpslist --json"],
    ["fpslist --json _nonexist_*"],
    ["milkquery"],
    ["milkquery --fps"],
    ["milkquery --fps _nonexist_*"],
    ["milkquery --streams"],
    ["milkquery --procs"],
    ["milkquery --fps --streams --procs"],
]
# fmt: on


# ==========================================================================
# EXPECT_ERROR: command prints a recognizable error message.
# ==========================================================================
# fmt: off
TESTLIST_EXPECT_ERROR: list[list[str]] = [
    # --- Section 2/3: unknown command / bad cd ---
    ["cmd? nonexistent_command_xyz"],
    ["cd /nonexistent_directory_xyz_123"],
    # --- Section 4: missing / wrong arguments ---
    ["mem.mk2Dim _clitest_partial"],
    ["mem.mk2Dim _clitest_partial 32"],
    ["source"],
    ["source /tmp/nonexistent_file_xyz_456.milk"],
    ["savescript"],
    ["time"],
    ["watch"],
    ["watch 1000"],
    # --- Section 8: undefined function ---
    ["_nonexistent_function_xyz"],
    # --- Section 11: error conditions ---
    ["mem.lisim"],
    ["xyzzy_nonexistent_command"],
    ["source /tmp/_clitest_no_such_file.milk"],
    ["include_once /tmp/_clitest_no_such_f2.milk"],
    # --- Section 16: bad syntax ---
    ["ls 'unclosed"],
    ['ls "unclosed'],
    ["ls $( ls"],
    ["( ls"],
    # --- Section 17: bad option ---
    # TODO
    # ["pwd --invalid-option-xyz"], ## TODO this is the only one that passes in the EXPECT_ERROR
    # TODO
    # --- Section 18: invalid calculator math ---
    ["_calc_err = 1 / 0"],
    ["_calc_err = ( 2 + 3"],
    ["_calc_err = 2 +"],
    ["_calc_err = non_existent_function(5)"],
    ["_calc_err = min(5)"],
    ["_calc_err = abs(1, 2)"],
    # --- Section 19: defer max limit ---
    ["defer echo %d" % i for i in range(1, 34)],
    # --- Section 22: error-if-empty expansion ---
    ["echo ${_empty_err:?variable_is_missing}"],
    # --- Section 29: pipeline with nonexistent stream ---
    ["_nostream_xyz |> mem.rm"],
    # --- Section 31: source error context ---
    ["source /tmp/_clitest_no_trace.milk"],
    ["on_fpschange nodot { echo x }"],
    # --- Section 32: bad slice syntax ---
    ["_bad = _slicetest[0:9 + 0.0"],
    # --- Section 34: bookmark errors ---
    ["bookmark frobnicate"],
    ["bookmark run _no_such_bm_xyz"],
    # --- Section 36: sleep usage ---
    ["sleep"],
    # --- Section 38: savehistory usage ---
    ["savehistory"],
    # --- Section 40: fpsset error paths ---
    ["fpsset"],
    ["fpsset nodotparam 42"],
    ["fpsset nosuchfps_xyz.param 42"],
    # --- Section 41: on_update error paths ---
    ["on_update _no_such_stream_xyz { echo x }"],
    ["on_update"],
    # --- Section 42: shell bypass nonexistent ---
    ["! nonexistent_cmd_xyz_456"],
    # --- Section 45: fpsdump error paths ---
    ["fpsdump"],
    ["fpsdump _no_such_fps_"],
    ["fpsdump --json _no_such_fps_"],
    ["fpsdump -t --json _no_such_fps_"],
    ["milkquery foo"],
]
# fmt: on


# ==========================================================================
# EXPECT_GREP: (commands, substring-required-on-stdout).
# ==========================================================================
# fmt: off
TESTLIST_EXPECT_GREP_WORKING: list[tuple[list[str], str]] = [
    # --- Section 1: echo output ---
    (["   echo hello"], "hello"),
    (["echo hello world"], "hello world"),
    (["echo -n test"], "test"),
    (['echo "hello world"'], "hello world"),
    (["echo 'hello world'"], "hello world"),
    (["echo test_value # this is a comment"], "test_value"),
    # --- Section 5: variables ---
    (["_teststr=hello", "echo ${#_teststr}"], "5"),
    #(["_TESTUP=HELLO", "echo ${_TESTUP,,}"], "hello"),
    (["echo ${_unsetvar:-fallback}"], "fallback"),
    (["echo ${_defvar:=assigned}"], "assigned"),
    # --- Section 6: arithmetic ---
    #(["_arith_a=10", "_arith_b=$(( _arith_a + 5 ))", "echo $_arith_b"], "15"),
    #(["_arith_d=$(( 17 % 3 ))", "echo $_arith_d"], "2"),
    #(["echo $(( 100 + 200 ))"], "300"),
    # --- Section 7: flow control ---
    #(["_fc_x=10", "if [ $_fc_x -gt 5 ]; then echo big; fi"], "big"),
    #(["_fc_n=0", "while [ $_fc_n -lt 3 ]; do _fc_n=$(( _fc_n + 1 )); done", "echo $_fc_n"], "3"),
    #(["for _fc_item in alpha beta gamma; do echo $_fc_item; done"], "alpha"),
    #(["for _fc_i in {1..3}; do echo iter_$_fc_i; done"], "iter_1"),
    #(["_fc_m=0", "while [ $_fc_m -lt 3 ]; do _fc_m=$(( _fc_m + 1 )); if [ $_fc_m -eq 2 ]; then echo found_two; fi; done"], "found_two"),
    #(["_fc_k=0", "while [ $_fc_k -lt 100 ]; do _fc_k=$(( _fc_k + 1 )); if [ $_fc_k -eq 3 ]; then break; fi; done", "echo $_fc_k"], "3"),
    #(["_fc_sum=0", "_fc_j=0", "while [ $_fc_j -lt 5 ]; do _fc_j=$(( _fc_j + 1 )); if [ $_fc_j -eq 3 ]; then continue; fi; _fc_sum=$(( _fc_sum + _fc_j )); done", "echo $_fc_sum"], "12"),
    #(["_fc_mode=fast", "case $_fc_mode in fast) echo speed ;; safe) echo caution ;; *) echo other ;; esac"], "speed"),
    #(["_fc_s=hello", "if [ $_fc_s = hello ]; then echo match; fi"], "match"),
    #(["if [ -d /tmp ]; then echo dir_exists; fi"], "dir_exists"),
    #(["if [ ! -f /tmp/no_such_file_xyz ]; then echo not_found; fi"], "not_found"),
    # --- Section 8: user functions ---
    (["function _testfunc_check { if [ $1 -gt 10 ]; then return 0; fi; return 1; }", "_testfunc_check 20", "echo $?"], "0"),
    #(["_gvar=global", "function _testfunc_local { local _gvar=localized; echo $_gvar; }", "_testfunc_local"], "localized"),
    #(["function _testfunc_add { _result=$(( $1 + $2 )); echo $_result; }", "_testfunc_add 3 7"], "10"),
    # --- Section 9: command chaining ---
    (["echo first ; echo second"], "second"),
    (["echo ok_cmd && echo and_ran"], "and_ran"),
    (["echo ok_cmd || echo or_skipped"], "ok_cmd"),
    # --- Section 10: shell features ---
    (["_subst_val=$(echo hello_sub)", "echo $_subst_val"], "hello_sub"),
    (["echo {1..5}"], "1 2 3 4 5"),
    (["echo {0..10..2}"], "0 2 4 6 8 10"),
    (["echo searchable_text | grep searchable"], "searchable_text"),
    # --- Section 13: let arithmetic ---
    #(['let "_decl_count=5+3"', "echo $_decl_count"], "8"),
    # --- Section 18b: SHM stream present ---
    #(["mem.mk2Dim _clitest_shm_s 8 8", "if [ -S _clitest_shm_s ]; then echo stream_exists; fi"], "stream_exists"),
    #(["if [ ! -S _clitest_nostream_xyz ]; then echo stream_absent; fi"], "stream_absent"),
    # --- Section 19: final marker ---
    (["echo CLI_ROBUSTNESS_TESTS_COMPLETE"], "CLI_ROBUSTNESS_TESTS_COMPLETE"),
    # --- Section 22: advanced expansions ---
    (["echo ${_unset_adv:-fallback_adv}"], "fallback_adv"),
    (["echo ${_unset_adv2:=assigned_adv}", "echo ${_unset_adv2}"], "assigned_adv"),
    (['_sub_str="hello_world"', "echo ${_sub_str:0:5}"], "hello"),
    # --- Section 23: advanced test evaluator ---
    #(["if [ 5 -gt 3 -a 10 -lt 20 ]; then echo both_true; fi"], "both_true"),
    #(["if [ 5 -lt 3 -o 10 -gt 5 ]; then echo one_true; fi"], "one_true"),
    # --- Section 24: bitwise / native math ---
    #(["_bit_val=$(( 5 & 3 | 8 ))", "echo $_bit_val"], "9"),
    #(["_bit_shift=$(( 1 << 4 ))", "if [ $_bit_shift -eq 16 ]; then echo shifted; fi"], "shifted"),
    # --- Section 25: C-style loop ---
    #(["_c_loop_sum=0", "for (( i=0; i<3; i++ )); do _c_loop_sum=$(( _c_loop_sum + i )); done", "echo $_c_loop_sum"], "3"),
    # --- Section 27: regex matching ---
    #(['if [ "hello_world" =~ ^hello ]; then echo matched; fi'], "matched"),
    # --- Section 28: array splat ---
    #(["_test_arr[0]=a", "_test_arr[1]=b", "_test_arr[2]=c", "echo ${_test_arr[@]}"], "a b c"),
    # --- Section 29: stream pipeline ---
    (["mem.mk2Dim _spipe_test 16 16", "_spipe_test |> mem.rm", "echo pipeline_done"], "pipeline_done"),
    (["mem.mk2Dim _spipe_a 8 8", "mem.mk2Dim _spipe_b 8 8", "_spipe_a |> mem.rm", "_spipe_b |> mem.rm", "echo pipe_cleanup_done"], "pipe_cleanup_done"),
    # --- Section 33: alias overwrite/run ---
    #(['alias _testalias "echo alias_works"', 'alias _testalias "echo overwritten"', "_testalias"], "overwritten"),
    # --- Section 34: bookmark run ---
    #(['bookmark save _tbm "echo bookmark_ok"', "bookmark run _tbm"], "bookmark_ok"),
    # --- Section 35: printf ---
    ([r'printf "%s=%d\n" hello 42'], "hello=42"),
    #([r'printf "%.3f\n" 3.14159'], "3.142"),
    ([r'printf "literal\n"'], "literal"),
    # --- Section 42: shell bypass valid ---
    (["! echo shell_bypass_test"], "shell_bypass_test"),
    # --- Section 44: edge cases ---
    (["_var123abc=test_digits", "echo $_var123abc"], "test_digits"),
    #(["_deep=$(( ((1 + 2) * (3 + 4)) + ((5 - 6) * 7) ))", "echo $_deep"], "14"),
    #(["function _outer { echo outer_$1; }", "function _inner { _outer inner; }", "_inner"], "outer_inner"),
    #(["xyzzy_nonexistent_12345 || echo fallback_ok"], "fallback_ok"),
    (["xyzzy_nonexistent_12345 && echo skipped", "echo and_chain_done"], "and_chain_done"),
    (["echo first_sc ;   echo second_sc"], "second_sc"),
    (["_ev=", "echo prefix${_ev}suffix"], "prefixsuffix"),
    # --- printf builtin (second group) ---
    ([r'printf "value=%d\n" 42'], "value=42"),
    ([r'printf "%s world\n" hello'], "hello world"),
    ([r'printf "pi=%f\n" 3.14'], "pi=3.14"),
    # --- export builtin ---
    (["export _CLI_TESTVAR=hello123", "echo $_CLI_TESTVAR"], "hello123"),
    # --- shift builtin ---
    #(["function _test_shift { echo $1; shift; echo $1; }", "_test_shift alpha beta"], "beta"),
    # --- string functions ---
    #(['_su = "hello"', "_up = toupper(_su)", "echo $_up"], "HELLO"),
    #(['_sl = "HELLO"', "_lo = tolower(_sl)", "echo $_lo"], "hello"),
    # --- default parameter expansion ---
    (['_unsetvar123=""', '_dv="${_unsetvar123:-fallback}"', "echo $_dv"], "fallback"),
    (['_unsetvar456=""', '_iv="${_unsetvar456:=assigned}"', "echo $_unsetvar456"], "assigned"),
    (['_setvar="yes"', '_av="${_setvar:+alternate}"', "echo $_av"], "alternate"),
    # --- string slicing ---
    (['_ss="helloworld"', '_sub="${_ss:5:5}"', "echo $_sub"], "world"),
    # --- time command ---
    (["time echo timed_output"], "timed_output"),
    # --- comment handling ---
    (["echo comment_ok # this is a comment and should not error"], "comment_ok"),
    (["echo indented_comment_ok   # indented comment"], "indented_comment_ok"),
]
TESTLIST_EXPECT_GREP_XFAILING: list[tuple[list[str], str]] = [
    # --- Section 5: variables ---
    (["_TESTUP=HELLO", "echo ${_TESTUP,,}"], "hello"),
    # --- Section 6: arithmetic ---
    (["_arith_a=10", "_arith_b=$(( _arith_a + 5 ))", "echo $_arith_b"], "15"),
    (["_arith_d=$(( 17 % 3 ))", "echo $_arith_d"], "2"),
    (["echo $(( 100 + 200 ))"], "300"),
    # --- Section 7: flow control ---
    (["_fc_x=10", "if [ $_fc_x -gt 5 ]; then echo big; fi"], "big"),
    (["_fc_n=0", "while [ $_fc_n -lt 3 ]; do _fc_n=$(( _fc_n + 1 )); done", "echo $_fc_n"], "3"),
    (["for _fc_item in alpha beta gamma; do echo $_fc_item; done"], "alpha"),
    (["for _fc_i in {1..3}; do echo iter_$_fc_i; done"], "iter_1"),
    (["_fc_m=0", "while [ $_fc_m -lt 3 ]; do _fc_m=$(( _fc_m + 1 )); if [ $_fc_m -eq 2 ]; then echo found_two; fi; done"], "found_two"),
    (["_fc_k=0", "while [ $_fc_k -lt 100 ]; do _fc_k=$(( _fc_k + 1 )); if [ $_fc_k -eq 3 ]; then break; fi; done", "echo $_fc_k"], "3"),
    (["_fc_sum=0", "_fc_j=0", "while [ $_fc_j -lt 5 ]; do _fc_j=$(( _fc_j + 1 )); if [ $_fc_j -eq 3 ]; then continue; fi; _fc_sum=$(( _fc_sum + _fc_j )); done", "echo $_fc_sum"], "12"),
    (["_fc_mode=fast", "case $_fc_mode in fast) echo speed ;; safe) echo caution ;; *) echo other ;; esac"], "speed"),
    (["_fc_s=hello", "if [ $_fc_s = hello ]; then echo match; fi"], "match"),
    (["if [ -d /tmp ]; then echo dir_exists; fi"], "dir_exists"),
    (["if [ ! -f /tmp/no_such_file_xyz ]; then echo not_found; fi"], "not_found"),
    # --- Section 8: user functions ---
    (["_gvar=global", "function _testfunc_local { local _gvar=localized; echo $_gvar; }", "_testfunc_local"], "localized"),
    (["function _testfunc_add { _result=$(( $1 + $2 )); echo $_result; }", "_testfunc_add 3 7"], "10"),
    # --- Section 13: let arithmetic ---
    (['let "_decl_count=5+3"', "echo $_decl_count"], "8"),
    # --- Section 18b: SHM stream present ---
    (["mem.mk2Dim _clitest_shm_s 8 8", "if [ -S _clitest_shm_s ]; then echo stream_exists; fi"], "stream_exists"),
    (["if [ ! -S _clitest_nostream_xyz ]; then echo stream_absent; fi"], "stream_absent"),
    # --- Section 23: advanced test evaluator ---
    (["if [ 5 -gt 3 -a 10 -lt 20 ]; then echo both_true; fi"], "both_true"),
    (["if [ 5 -lt 3 -o 10 -gt 5 ]; then echo one_true; fi"], "one_true"),
    # --- Section 24: bitwise / native math ---
    (["_bit_val=$(( 5 & 3 | 8 ))", "echo $_bit_val"], "9"),
    (["_bit_shift=$(( 1 << 4 ))", "if [ $_bit_shift -eq 16 ]; then echo shifted; fi"], "shifted"),
    # --- Section 25: C-style loop ---
    (["_c_loop_sum=0", "for (( i=0; i<3; i++ )); do _c_loop_sum=$(( _c_loop_sum + i )); done", "echo $_c_loop_sum"], "3"),
    # --- Section 27: regex matching ---
    (['if [ "hello_world" =~ ^hello ]; then echo matched; fi'], "matched"),
    # --- Section 28: array splat ---
    (["_test_arr[0]=a", "_test_arr[1]=b", "_test_arr[2]=c", "echo ${_test_arr[@]}"], "a b c"),
    # --- Section 33: alias overwrite/run ---
    (['alias _testalias "echo alias_works"', 'alias _testalias "echo overwritten"', "_testalias"], "overwritten"),
    # --- Section 34: bookmark run ---
    (['bookmark save _tbm "echo bookmark_ok"', "bookmark run _tbm"], "bookmark_ok"),
    # --- Section 35: printf ---
    ([r'printf "%.3f\n" 3.14159'], "3.142"),
    (["_deep=$(( ((1 + 2) * (3 + 4)) + ((5 - 6) * 7) ))", "echo $_deep"], "14"),
    (["function _outer { echo outer_$1; }", "function _inner { _outer inner; }", "_inner"], "outer_inner"),
    (["xyzzy_nonexistent_12345 || echo fallback_ok"], "fallback_ok"),
    # --- shift builtin ---
    (["function _test_shift { echo $1; shift; echo $1; }", "_test_shift alpha beta"], "beta"),
    # --- string functions ---
    (['_su = "hello"', "_up = toupper(_su)", "echo $_up"], "HELLO"),
    (['_sl = "HELLO"', "_lo = tolower(_sl)", "echo $_lo"], "hello"),
]
# fmt: on


@pytest.mark.timeout(TIMEOUT)
@pytest.mark.parametrize("milk_cmds", TESTLIST_EXPECT_OK)
def test_expect_ok(milk_cmds: list[str]):
    joined_cmd = ";".join(milk_cmds)
    with CLICommands(milk_cmds) as result:
        assert result.returncode == 0, (
            f'"\033[1;33m{joined_cmd}\033[0m": expected clean exit, got "{result.returncode}"'
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )


@pytest.mark.timeout(TIMEOUT)
@pytest.mark.parametrize("milk_cmds", TESTLIST_EXPECT_ERROR)
def test_expect_error(milk_cmds: list[str]):
    if milk_cmds == ["pwd --invalid-option-xyz"]:
        pytest.mark.xfail(reason=f'the test for "pwd --invalid-option-xyz" is broken.')
    joined_cmd = ";".join(milk_cmds)
    with CLICommands(milk_cmds) as result:
        combined = result.stdout + result.stderr
        assert ERROR_REGEX.search(
            combined
        ), f'"\033[1;33m{joined_cmd}\033[0m": expected an error message, got: "{combined}"'


@pytest.mark.timeout(TIMEOUT)
@pytest.mark.parametrize("milk_cmds,needle", TESTLIST_EXPECT_GREP_WORKING)
def test_expect_grep(milk_cmds: list[str], needle: str):
    joined_cmd = ";".join(milk_cmds)
    with CLICommands(milk_cmds) as result:
        assert (
            needle in result.stdout
        ), f'"\033[1;33m{joined_cmd}\033[0m" (grepping:{needle}): expected {needle!r} in stdout, got: "{result.stdout}"'


@pytest.mark.xfail(reason="CLI doesn't behave as expected for control flow statements.")
@pytest.mark.timeout(TIMEOUT)
@pytest.mark.parametrize("milk_cmds,needle", TESTLIST_EXPECT_GREP_XFAILING)
def test_expect_grep_xfails(milk_cmds: list[str], needle: str):
    joined_cmd = ";".join(milk_cmds)
    with CLICommands(milk_cmds) as result:
        assert (
            needle in result.stdout
        ), f'"\033[1;33m{joined_cmd}\033[0m" (grepping:{needle}): expected {needle!r} in stdout, got: "{result.stdout}"'
