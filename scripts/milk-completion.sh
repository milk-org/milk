#!/bin/bash

# Early-exit help flags — only execute when run standalone, not when sourced
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    case "${1-}" in
        -h1|--help-oneline)
            echo "bash tab-completion script for milk-cli" \
                 "and cacao fpsexec commands"
            exit 0 ;;
        -h2|--help-description)
            echo "Provides bash tab-completion for" \
                 "milk-cli, milk-fpsexec-*, and" \
                 "cacao-fpsexec-* commands."
            exit 0 ;;
        -h|--help|-hm|--help-mono)
            echo "Source this file in your shell to" \
                 "enable tab-completion."
            exit 0 ;;
    esac
fi
# milk-completion.sh
# Bash completion script for milk-cli and cacao fpsexec commands

_milk_fpsexec_complete()
{
    local cur prev words cword cmd
    COMPREPLY=()

    # Simple fallback parsing
    cur="${COMP_WORDS[COMP_CWORD]}"
    prev="${COMP_WORDS[COMP_CWORD-1]}"
    words=("${COMP_WORDS[@]}")
    cword=$COMP_CWORD
    cmd="${words[0]}"

    # If the user is typing an option starting with '-'
    if [[ "$cur" == -* ]]; then
        # Dynamically extract options from the command's -h output
        # It looks for lines starting with "  -" or "    -", extracts the flag (e.g. -T)
        local opts
        opts=$("$cmd" -h 2>/dev/null | grep -o -E '^\s*-[a-zA-Z0-9-]+' | tr -d ' ' | sort -u)
        COMPREPLY=( $(compgen -W "${opts}" -- "$cur") )
        return 0
    fi
}

_milk_complete()
{
    local cur prev words cword
    COMPREPLY=()

    cur="${COMP_WORDS[COMP_CWORD]}"
    prev="${COMP_WORDS[COMP_CWORD-1]}"
    words=("${COMP_WORDS[@]}")
    cword=$COMP_CWORD

    if [[ "$cur" == -* ]]; then
        # Standard options for `milk-cli` main executable
        local opts="-h --help -v --version -i --info --verbose -d -o --overwrite -e --errorexit -Z --idle -A --autocomplete --no-autocomplete --no-history-suggest --no-arg-hints --no-fuzzy -f --fifoflag -F -s -n -p"
        COMPREPLY=( $(compgen -W "${opts}" -- "$cur") )
        return 0
    fi
}

# Register completion functions
# Bash 4+ and Bash 3 support
complete -o default -F _milk_complete milk-cli

# Dynamically bind all installed milk-fpsexec and cacao-fpsexec commands
# Redirect stderr to avoid errors if compgen finds nothing
for exe in $(compgen -c milk-fpsexec- 2>/dev/null); do
    complete -o default -F _milk_fpsexec_complete "$exe"
done

for exe in $(compgen -c cacao-fpsexec- 2>/dev/null); do
    complete -o default -F _milk_fpsexec_complete "$exe"
done
