# milk-fps-set completion
# Only provide help when run directly, not when sourced
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    case "${1-}" in
        -h1|--help-oneline)
            echo "bash completion for milk-fps-set"
            exit 0 ;;
        -h2|--help-description)
            echo "Completion script; source via" \
                 "bash_completion.d."
            exit 0 ;;
        -h|--help|-hm|--help-mono)
            echo "Source this file; do not run directly."
            exit 0 ;;
    esac
fi
# Using complete -C calls the command itself with [cmd] [cur] [prev]
# and expects matches on stdout, one per line.
complete -C milk-fps-set milk-fps-set