// SPDX-FileCopyrightText: 2026 Max Brunsfeld
//
// SPDX-License-Identifier: MIT

/**
 * @file grammar.js
 * @brief Tree-sitter grammar for milk-cli scripting
 *
 * A milk script is a bash script with two additions
 * (see docs/guide/scripting.md):
 *  - @keyword args      language keyword (@calc, @get, @waitfor)
 *  - @{expr}            inline expansion, also inside "..."
 * plus module commands (mem.listim) and entity paths
 * (s.cam.xsize, namespaces: ENTITY_NS) as arguments.
 *
 * Statements end at a newline, ';' or '&', as in bash.
 * Not covered: heredocs (need an external scanner),
 * word concatenation, arrays, C-style for.
 */

/// <reference types="tree-sitter-cli/dsl" />
// @ts-check

/* Entity namespaces, as in src/cli/libmilkscript/shm_prop/milk_entity_ns.h */
const ENTITY_NS = [
    "s", "stream", "l", "local", "f", "fps",
    "p", "proc",
];

module.exports = grammar({
    name: "milkcli",

    extras: $ => [/\s/, $.line_continuation, $.comment],

    word: $ => $.identifier,

    rules: {
        /* ---- Top-level ---- */

        program: $ => optional($._statements),

        _statements: $ => prec(1, seq(
            repeat(seq($._statement, $._terminator)),
            $._statement,
            optional($._terminator),
        )),

        _terminated_statement: $ => seq(
            $._statement, $._terminator,
        ),

        /* '\n' outranks the prec-2 '@' tokens, or "\n@x" would lex as one word */
        _terminator: $ => choice(";", "&", token(prec(3, "\n"))),

        _statement: $ => choice(
            $.variable_assignment,
            $.if_statement,
            $.while_statement,
            $.until_statement,
            $.for_statement,
            $.case_statement,
            $.function_definition,
            $.compound_statement,
            $.test_bracket,
            $.double_bracket,
            $.pipeline,
            $.command_list,
            $.negated_command,
            $.command,
            $.subshell,
        ),

        /* ---- Comments ---- */
        comment: $ => token(prec(-1, /#.*/)),

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

        /* $VAR, $1, $@, $?, $# ... */
        simple_expansion: $ => choice(
            seq("$", $.identifier),
            token(/\$[0-9@*#?$!-]/),
        ),

        expansion: $ => seq(
            "${",
            repeat1(choice(
                $.expansion,
                $.simple_expansion,
                /[^}$]+/,
            )),
            "}",
        ),

        /* ---- milk-specific extensions ---- */

        /*
         * Path parts glued to the previous token (no
         * whitespace), so `@s.cam.xsize next` ends at
         * xsize. Aliased to the ordinary nodes.
         */
        _immediate_expansion: $ => seq(
            token.immediate("${"),
            repeat1(choice(
                $.expansion,
                $.simple_expansion,
                /[^}$]+/,
            )),
            "}",
        ),

        _immediate_simple_expansion: $ => seq(
            token.immediate("$"), $.identifier,
        ),

        /** s.cam.xsize, f.${LOOP}.gain, p.loop.status (an argument) */
        entity_path: $ => prec.right(seq(
            token(prec(2, new RegExp(
                "(" + ENTITY_NS.join("|") + ")\\."))),
            repeat1(choice(
                alias($._immediate_expansion,
                      $.expansion),
                alias($._immediate_simple_expansion,
                      $.simple_expansion),
                token.immediate(prec(2,
                    /[a-zA-Z0-9_.*-]+/)),
            )),
        )),

        /** @{imean(cam) * 2} */
        inline_calc: $ => seq(
            token(prec(2, "@{")),
            repeat(choice(
                $.expansion,
                $.simple_expansion,
                /[^}$]+/,
            )),
            "}",
        ),

        /** @calc, @waitfor: the language keywords */
        sigil_command: $ => token(prec(1,
            /@[a-zA-Z_][a-zA-Z0-9_]*/)),

        /** mem.listim, iofits.saveFITS */
        module_command: $ =>
            /[a-zA-Z_][a-zA-Z0-9_]*\.[a-zA-Z_][a-zA-Z0-9_.]*/,

        /* ---- Arithmetic ---- */
        arithmetic_expansion: $ => seq(
            "$((",
            repeat(choice(/[^()]+/, $._arithmetic_group)),
            "))",
        ),

        _arithmetic_group: $ => seq(
            "(",
            repeat(choice(/[^()]+/, $._arithmetic_group)),
            ")",
        ),

        /* ---- Command substitution ---- */
        command_substitution: $ => seq(
            "$(",
            optional($._statements),
            ")",
        ),

        /* ---- Strings ---- */

        double_quoted_string: $ => seq(
            '"',
            repeat(choice(
                $.simple_expansion,
                $.expansion,
                $.inline_calc,
                $.arithmetic_expansion,
                $.command_substitution,
                token.immediate(/\\./),
                token.immediate(prec(1, /[^"\\$@]+/)),
                "@",
            )),
            '"',
        ),

        single_quoted_string: $ =>
            token(seq("'", /[^']*/, "'")),

        /* ---- Literals ---- */

        _literal: $ => choice(
            $.identifier,
            $.number,
            $.double_quoted_string,
            $.single_quoted_string,
            $.simple_expansion,
            $.expansion,
            $.entity_path,
            $.inline_calc,
            $.arithmetic_expansion,
            $.command_substitution,
            $.file_path,
            $.raw_string,
        ),

        number: $ =>
            /[0-9]+(\.[0-9]+)?([eE][+-]?[0-9]+)?/,

        file_path: $ =>
            /\.?\.?\/[a-zA-Z0-9_.\/-]+/,

        /* key=value, @L:name, a,b: '=' ':' ',' '+' '#' after the first char */
        raw_string: $ =>
            /[a-zA-Z0-9_.\/~*?@\[\]{}%^+-][a-zA-Z0-9_.\/~*?@\[\]{}=:,+%^#-]*/,

        /* ---- Flow control ---- */

        if_statement: $ => seq(
            "if",
            field("condition", $._terminated_statement),
            "then",
            optional($._statements),
            repeat($.elif_clause),
            optional($.else_clause),
            "fi",
        ),

        elif_clause: $ => seq(
            "elif",
            field("condition", $._terminated_statement),
            "then",
            optional($._statements),
        ),

        else_clause: $ => seq(
            "else",
            optional($._statements),
        ),

        while_statement: $ => seq(
            "while",
            field("condition", $._terminated_statement),
            "do",
            optional($._statements),
            "done",
        ),

        until_statement: $ => seq(
            "until",
            field("condition", $._terminated_statement),
            "do",
            optional($._statements),
            "done",
        ),

        for_statement: $ => seq(
            "for",
            field("variable", alias(
                $.identifier, $.variable_name)),
            optional(seq("in", repeat1($._literal))),
            $._terminator,
            "do",
            optional($._statements),
            "done",
        ),

        case_statement: $ => seq(
            "case",
            field("value", $._literal),
            optional($._terminator),
            "in",
            optional($._terminator),
            repeat($.case_item),
            "esac",
        ),

        case_item: $ => seq(
            optional("("),
            field("value", $._literal),
            repeat(seq("|", field("value", $._literal))),
            ")",
            optional($._statements),
            choice(";;", ";&", ";;&"),
        ),

        /* ---- Tests ---- */

        _test_operator: $ => choice(
            "=", "==", "!=", "!", "(", ")",
        ),

        test_bracket: $ => seq(
            "[",
            repeat(choice($._literal, $._test_operator)),
            "]",
        ),

        double_bracket: $ => seq(
            "[[",
            repeat(choice(
                $._literal, $._test_operator,
                "=~", "&&", "||", "<", ">",
            )),
            "]]",
        ),

        /* ---- Functions and groups ---- */

        function_definition: $ => choice(
            seq(
                "function",
                field("name", $.identifier),
                optional(seq("(", ")")),
                "{",
                optional($._statements),
                "}",
            ),
            seq(
                field("name", $.identifier),
                "(", ")",
                "{",
                optional($._statements),
                "}",
            ),
        ),

        compound_statement: $ => seq(
            "{", optional($._statements), "}",
        ),

        subshell: $ => seq(
            "(", optional($._statements), ")",
        ),

        /* ---- Commands ---- */

        command: $ => prec.left(seq(
            field("name", $._command_name),
            repeat(choice(
                field("argument", $._literal),
                field("argument", alias("=", $.raw_string)),
                $.io_redirect,
            )),
        )),

        _command_name: $ => choice(
            $.sigil_command,
            $.module_command,
            $.identifier,
            $.file_path,
            $.simple_expansion,
            $.expansion,
        ),

        negated_command: $ => prec.right(seq(
            "!", $._statement,
        )),

        /* ---- Pipelines and lists ---- */

        pipeline: $ => prec.left(1, seq(
            $._statement,
            choice("|", "|&"),
            $._statement,
        )),

        command_list: $ => prec.left(0, seq(
            $._statement,
            choice("&&", "||"),
            $._statement,
        )),

        /* ---- I/O Redirection ---- */

        io_redirect: $ => choice(
            seq(
                choice(">", ">>", "<", "<<<", "2>",
                       "2>>", "&>", "&>>"),
                $._literal,
            ),
            "2>&1",
            ">&2",
            "1>&2",
        ),
    },
});
