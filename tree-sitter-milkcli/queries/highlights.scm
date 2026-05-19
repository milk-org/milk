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


; ---- milk: FPS variable references ----
; @fps.loopname.gain  → special property color
(fps_variable) @property


; ---- milk: Sequencer variable references ----
; @seq.myseq.status  → special property color
(seq_variable) @property


; ---- milk: Stream metadata ----
; ${s.dmcomb.cnt0}  → type color
(stream_metadata) @type


; ---- Operators ----
["=" "|" "|>" "&&" "||"
 ">" ">>" "<" "<<<" "2>&1" "2>"
 ";" ";;"] @operator


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


; ---- milk-specific commands ----
; These get a distinct color so users
; immediately see milk extensions.
(milk_command) @function.macro


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
