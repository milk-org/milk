// SPDX-FileCopyrightText: 2026 Olivier Guyon et al
//
// SPDX-License-Identifier: LGPL-3.0-or-later

/**
 * @file milkcli_highlights.h
 *
 * @brief Embedded tree-sitter highlight queries
 *
 * Copy of tree-sitter-milkcli/queries/highlights.scm
 * without comments: one string per query line. Keep in
 * sync when the grammar or the queries change.
 *
 * This string is compiled into the tree-sitter
 * query via ts_query_new() at CLI startup.
 */

#ifndef MILKCLI_HIGHLIGHTS_H
#define MILKCLI_HIGHLIGHTS_H

static const char milkcli_highlights_scm[] =
    "(comment) @comment\n"
    "(double_quoted_string) @string\n"
    "(single_quoted_string) @string\n"
    "(number) @number\n"
    "(variable_assignment\n"
    "  name: (variable_name) @variable.parameter)\n"
    "(simple_expansion) @variable.builtin\n"
    "(expansion) @variable.builtin\n"
    "(entity_path) @property\n"
    "(inline_calc\n"
    "  [\"@{\" \"}\"] @punctuation.special)\n"
    "(calc_statement\n"
    "  \"@\" @function.macro)\n"
    "[\"=\" \"==\" \"!=\" \"=~\" \"!\" \"|\" \"|&\" \"|>\" \"&&\" \"||\"\n"
    " \">\" \">>\" \"<\" \"<<<\" \"2>\" \"2>>\" \"&>\" \"&>>\"\n"
    " \"2>&1\" \">&2\" \"1>&2\"\n"
    " \";\" \";;\" \";&\" \";;&\" \"&\"] @operator\n"
    "[\"(\" \")\" \"{\" \"}\" \"[\" \"]\"\n"
    " \"[[\" \"]]\" \"$((\" \"))\"\n"
    " \"$(\" \"${\"] @punctuation.bracket\n"
    "[\"if\" \"elif\" \"else\" \"fi\"\n"
    " \"then\"\n"
    " \"for\" \"while\" \"until\"\n"
    " \"do\" \"done\" \"in\"\n"
    " \"case\" \"esac\"\n"
    " \"function\"] @keyword\n"
    "((identifier) @keyword.return\n"
    " (#any-of? @keyword.return\n"
    "  \"break\" \"continue\" \"return\" \"exit\"))\n"
    "((identifier) @boolean\n"
    " (#any-of? @boolean \"true\" \"false\"))\n"
    "((identifier) @function.builtin\n"
    " (#any-of? @function.builtin\n"
    "  \"echo\" \"printf\" \"export\" \"source\"\n"
    "  \"set\" \"readonly\" \"local\" \"declare\"\n"
    "  \"let\" \"eval\" \"type\" \"command\"\n"
    "  \"trap\" \"shift\" \"alias\" \"unalias\"\n"
    "  \"pushd\" \"popd\" \"dirs\" \"getopts\"\n"
    "  \"mapfile\" \"basename\" \"dirname\"\n"
    "  \"seq\" \"time\" \"timeout\" \"wait\"\n"
    "  \"select\" \"read\" \"unset\" \"cd\"\n"
    "  \"test\" \"kill\" \"sleep\"))\n"
    "(sigil_command) @function.macro\n"
    "(milk_command) @function.macro\n"
    "(module_command) @function.call\n"
    "(command\n"
    "  name: (identifier) @function.call)\n"
    "(function_definition\n"
    "  name: (identifier) @function)\n"
    "(file_path) @string.special.path\n"
    "(arithmetic_expansion) @number\n"
    "(command_substitution) @embedded\n"
    "(io_redirect) @operator\n"
    "(test_bracket) @keyword.operator\n"
    "(double_bracket) @keyword.operator\n";

#endif /* MILKCLI_HIGHLIGHTS_H */
