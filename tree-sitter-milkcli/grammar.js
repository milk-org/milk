// SPDX-FileCopyrightText: 2026 Max Brunsfeld
//
// SPDX-License-Identifier: MIT

/**
 * @file grammar.js
 * @brief Tree-sitter grammar for milk-cli scripting
 *
 * Milk-cli is a bash-like scripting language with
 * extensions for shared-memory streams, FPS
 * parameters, and real-time process control.
 *
 * Key extensions over bash:
 *  - @fps.name.param    FPS variable access
 *  - @seq.name.field    Sequencer variable
 *  - ${s.stream.field}  Stream metadata
 *  - |>                 Stream pipe operator
 *  - waitfor_stream / waitfor_fps
 *  - on_update / on_fpschange
 *  - assigncheck / assert
 */

/// <reference types="tree-sitter-cli/dsl" />
// @ts-check

module.exports = grammar({
    name: "milkcli",

    extras: $ => [/\s/, $.line_continuation],

    conflicts: $ => [],

    word: $ => $.identifier,

    rules: {
        /* ---- Top-level ---- */

        program: $ => repeat($._statement),

        _statement: $ => choice(
            $.comment,
            $.variable_assignment,
            $.if_statement,
            $.while_statement,
            $.until_statement,
            $.for_statement,
            $.case_statement,
            $.function_definition,
            $.pipeline,
            $.command_list,
            $.command,
            $.subshell,
        ),

        /* ---- Comments ---- */
        comment: $ => token(seq("#", /.*/)),

        /* ---- Line continuation ---- */
        line_continuation: $ =>
            token(seq("\\", /\r?\n/)),

        /* ---- Identifiers ---- */
        identifier: $ =>
            /[a-zA-Z_][a-zA-Z0-9_]*/,

        /* ---- Variables ---- */

        variable_assignment: $ => prec.left(1, seq(
            field("name", alias(
                $.identifier, $.variable_name)),
            "=",
            optional(
                field("value", $._literal)),
        )),

        /* Variable expansion */
        simple_expansion: $ => seq(
            "$", $.identifier,
        ),

        expansion: $ => prec.left(1, seq(
            "${",
            repeat1(choice(
                $.expansion,
                $.simple_expansion,
                /[^}$]+/,
            )),
            "}",
        )),

        /* ---- milk-specific extensions ---- */

        /** @fps.name.param  or  @fps.${VAR}.param */
        fps_variable: $ => prec.left(2, seq(
            "@fps.",
            repeat1(choice(
                $.expansion,
                $.simple_expansion,
                /[a-zA-Z0-9_.-]+/,
            )),
        )),

        /** @seq.name.field  or  @seq.${VAR}.field */
        seq_variable: $ => prec.left(2, seq(
            "@seq.",
            repeat1(choice(
                $.expansion,
                $.simple_expansion,
                /[a-zA-Z0-9_.-]+/,
            )),
        )),

        /** ${s.stream.field} */
        stream_metadata: $ =>
            /\$\{s\.[a-zA-Z0-9_-]+\.[a-zA-Z0-9_]+\}/,

        /* ---- Arithmetic ---- */
        arithmetic_expansion: $ => seq(
            "$((",
            /[^)]+\)?[^)]*/,
            "))",
        ),

        /* ---- Command substitution ---- */
        command_substitution: $ => seq(
            "$(",
            repeat($._statement),
            ")",
        ),

        /* ---- Strings ---- */

        double_quoted_string: $ => seq(
            '"',
            repeat(choice(
                $.simple_expansion,
                $.expansion,
                $.fps_variable,
                $.seq_variable,
                $.stream_metadata,
                $.arithmetic_expansion,
                /\\./,
                /[^"\\$@]+/,
            )),
            '"',
        ),

        single_quoted_string: $ => seq(
            "'", optional(/[^']*/), "'",
        ),

        /* ---- Literals ---- */

        _literal: $ => choice(
            $.identifier,
            $.number,
            $.double_quoted_string,
            $.single_quoted_string,
            $.simple_expansion,
            $.expansion,
            $.fps_variable,
            $.seq_variable,
            $.stream_metadata,
            $.arithmetic_expansion,
            $.command_substitution,
            $.file_path,
            $.raw_string,
        ),

        number: $ =>
            /[0-9]+(\.[0-9]+)?([eE][+-]?[0-9]+)?/,

        file_path: $ =>
            /\.?\.?\/[a-zA-Z0-9_.\/-]+/,

        raw_string: $ =>
            /[a-zA-Z0-9_.\/~*?@\[\]{}-]+/,

        /* ---- Flow control ---- */

        if_statement: $ => seq(
            "if",
            field("condition", $._condition),
            optional(";"), "then",
            repeat($._statement),
            repeat($.elif_clause),
            optional($.else_clause),
            "fi",
        ),

        elif_clause: $ => seq(
            "elif",
            field("condition", $._condition),
            optional(";"), "then",
            repeat($._statement),
        ),

        else_clause: $ => seq(
            "else",
            repeat($._statement),
        ),

        while_statement: $ => seq(
            "while",
            field("condition", $._condition),
            optional(";"), "do",
            repeat($._statement),
            "done",
        ),

        until_statement: $ => seq(
            "until",
            field("condition", $._condition),
            optional(";"), "do",
            repeat($._statement),
            "done",
        ),

        for_statement: $ => seq(
            "for",
            field("variable", alias(
                $.identifier, $.variable_name)),
            "in",
            repeat1($._literal),
            optional(";"), "do",
            repeat($._statement),
            "done",
        ),

        case_statement: $ => seq(
            "case",
            field("value", $._literal),
            "in",
            repeat($.case_item),
            "esac",
        ),

        case_item: $ => seq(
            $._literal, ")",
            repeat($._statement),
            ";;",
        ),

        /* ---- Conditions ---- */

        _condition: $ => choice(
            $.test_bracket,
            $.double_bracket,
            $.command,
        ),

        test_bracket: $ => seq(
            "[", repeat($._literal), "]",
        ),

        double_bracket: $ => seq(
            "[[", repeat($._literal), "]]",
        ),

        /* ---- Functions ---- */

        function_definition: $ => seq(
            "function",
            field("name", $.identifier),
            optional(seq("(", ")")),
            "{",
            repeat($._statement),
            "}",
        ),

        /* ---- Commands ---- */

        command: $ => prec.left(-1, seq(
            field("name", $._command_name),
            repeat(field("argument", $._literal)),
            repeat($.io_redirect),
        )),

        _command_name: $ => choice(
            $.milk_command,
            $.identifier,
            $.file_path,
        ),

        /**
         * milk-specific commands — highlighted
         * distinctly from generic commands.
         * These are the extensions that make
         * milk-cli different from bash.
         */
        milk_command: $ => choice(
            "assert",
            "assigncheck",
            "dpdigits",
            "procctl",
            "procwait",
            "procstat",
            "waitfor_stream",
            "waitfor_fps",
            "wait_any",
            "on_update",
            "on_fpschange",
            "include_once",
            "savescript",
            "savehistory",
        ),

        /* ---- Pipelines and lists ---- */

        pipeline: $ => prec.left(1, seq(
            $._statement,
            choice("|", "|>"),
            $._statement,
        )),

        command_list: $ => prec.left(0, seq(
            $._statement,
            choice("&&", "||", ";"),
            $._statement,
        )),

        /* ---- Subshell ---- */

        subshell: $ => seq(
            "(", repeat($._statement), ")",
        ),

        /* ---- I/O Redirection ---- */

        io_redirect: $ => seq(
            choice(">", ">>", "<", "<<<",
                   "2>&1", "2>"),
            $._literal,
        ),
    },
});
