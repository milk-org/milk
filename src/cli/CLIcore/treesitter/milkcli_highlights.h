/**
 * @file milkcli_highlights.h
 *
 * @brief Embedded tree-sitter highlight queries
 *
 * Auto-generated from tree-sitter-milkcli/queries/
 * highlights.scm.  To regenerate, re-run the build
 * script in tree-sitter-milkcli/ and copy this file.
 *
 * This string is compiled into the tree-sitter
 * query via ts_query_new() at CLI startup.
 */

#ifndef MILKCLI_HIGHLIGHTS_H
#define MILKCLI_HIGHLIGHTS_H

static const char milkcli_highlights_scm[] = "(comment) @comment\n"
                                             "(double_quoted_string) @string\n"
                                             "(single_quoted_string) @string\n"
                                             "(number) @number\n"
                                             "(variable_assignment"
                                             " name: (variable_name) @variable.parameter)\n"
                                             "(simple_expansion) @variable.builtin\n"
                                             "(expansion) @variable.builtin\n"
                                             "(fps_variable) @property\n"
                                             "(seq_variable) @property\n"
                                             "(stream_metadata) @type\n"
                                             "[\"=\" \"|\" \"|>\" \"&&\" \"||\""
                                             " \">\" \">>\" \"<\" \"<<<\" \"2>&1\" \"2>\""
                                             " \";\" \";;\"] @operator\n"
                                             "[\"(\" \")\" \"{\" \"}\" \"[\" \"]\""
                                             " \"[[\" \"]]\" \"$((\""
                                             " \"))\" \"$(\" \"${\"] @punctuation.bracket\n"
                                             "[\"if\" \"elif\" \"else\" \"fi\""
                                             " \"then\""
                                             " \"for\" \"while\" \"until\""
                                             " \"do\" \"done\" \"in\""
                                             " \"case\" \"esac\""
                                             " \"function\"] @keyword\n"
                                             "((identifier) @keyword.return"
                                             " (#any-of? @keyword.return"
                                             " \"break\" \"continue\" \"return\""
                                             " \"exit\"))\n"
                                             "((identifier) @boolean"
                                             " (#any-of? @boolean"
                                             " \"true\" \"false\"))\n"
                                             "((identifier) @function.builtin"
                                             " (#any-of? @function.builtin"
                                             " \"echo\" \"printf\" \"export\" \"source\""
                                             " \"set\" \"readonly\" \"local\" \"declare\""
                                             " \"let\" \"eval\" \"type\" \"command\""
                                             " \"trap\" \"shift\" \"alias\" \"unalias\""
                                             " \"pushd\" \"popd\" \"dirs\" \"getopts\""
                                             " \"mapfile\" \"basename\" \"dirname\""
                                             " \"seq\" \"time\" \"timeout\" \"wait\""
                                             " \"select\" \"read\" \"unset\" \"cd\""
                                             " \"test\" \"kill\" \"sleep\"))\n"
                                             "(milk_command) @function.macro\n"
                                             "(command"
                                             " name: (identifier) @function.call)\n"
                                             "(function_definition"
                                             " name: (identifier) @function)\n"
                                             "(file_path) @string.special.path\n"
                                             "(arithmetic_expansion) @number\n"
                                             "(command_substitution) @embedded\n"
                                             "(io_redirect) @operator\n"
                                             "(test_bracket) @keyword.operator\n"
                                             "(double_bracket) @keyword.operator\n";

#endif /* MILKCLI_HIGHLIGHTS_H */
