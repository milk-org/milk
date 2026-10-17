; SPDX-FileCopyrightText: 2026 Max Brunsfeld
;
; SPDX-License-Identifier: MIT

; highlights.scm — Neovim highlight queries
; for the milk-cli scripting language
;
; Maps tree-sitter node types to Neovim
; @capture groups. The active colorscheme
; then maps captures to terminal colors.


; ---- Comments ----
(comment) @comment


; ---- Strings ----
(double_quoted_string) @string
(single_quoted_string) @string


; ---- Numbers ----
(number) @number


; ---- Variables ----
(variable_assignment
  name: (variable_name) @variable.parameter)


; ---- Variable expansions ----
(simple_expansion) @variable.builtin
(expansion) @variable.builtin


; ---- milk: entity paths ----
; @s.cam.xsize, @f.${LOOP}.gain  → property color
(entity_path) @property


; ---- milk: calculator ----
; @{expr} inline, @ expr statement
(inline_calc
  ["@{" "}"] @punctuation.special)
(calc_statement
  "@" @function.macro)


; ---- Operators ----
["=" "==" "!=" "=~" "!" "|" "|&" "|>" "&&" "||"
 ">" ">>" "<" "<<<" "2>" "2>>" "&>" "&>>"
 "2>&1" ">&2" "1>&2"
 ";" ";;" ";&" ";;&" "&"] @operator


; ---- Punctuation ----
["(" ")" "{" "}" "[" "]"
 "[[" "]]" "$((" "))"
 "$(" "${"] @punctuation.bracket


; ---- Flow control keywords ----
; (only tokens that exist as literal strings
;  in the grammar)
["if" "elif" "else" "fi"
 "then"
 "for" "while" "until"
 "do" "done" "in"
 "case" "esac"
 "function"] @keyword


; ---- Control flow identifiers ----
; break/continue/return/exit are parsed as
; identifiers — match by text predicate
((identifier) @keyword.return
 (#any-of? @keyword.return
  "break" "continue" "return" "exit"))


; ---- Boolean literals ----
((identifier) @boolean
 (#any-of? @boolean "true" "false"))


; ---- Shell builtins ----
; Recognized by identifier text predicate
((identifier) @function.builtin
 (#any-of? @function.builtin
  "echo" "printf" "export" "source"
  "set" "readonly" "local" "declare"
  "let" "eval" "type" "command"
  "trap" "shift" "alias" "unalias"
  "pushd" "popd" "dirs" "getopts"
  "mapfile" "basename" "dirname"
  "seq" "time" "timeout" "wait"
  "select" "read" "unset" "cd"
  "test" "kill" "sleep"))


; ---- milk commands ----
; @calc, @waitfor, include_once, milksleep:
; a distinct color so users immediately see
; milk extensions.
(sigil_command) @function.macro
(milk_command) @function.macro


; ---- Module-qualified commands ----
; mem.listim, iofits.saveFITS; known ones are
; upgraded to @function.builtin by modules.scm
; (scripts/gen-module-highlights.sh).
(module_command) @function.call


; ---- Generic command names ----
; First word of a command that isn't matched
; by a more specific pattern above.
(command
  name: (identifier) @function.call)


; ---- Function definitions ----
(function_definition
  name: (identifier) @function)


; ---- File paths ----
(file_path) @string.special.path


; ---- Arithmetic ----
(arithmetic_expansion) @number


; ---- Command substitution ----
(command_substitution) @embedded


; ---- I/O redirections ----
(io_redirect) @operator


; ---- Test brackets ----
(test_bracket) @keyword.operator
(double_bracket) @keyword.operator
