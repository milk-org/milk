#!/usr/bin/env bash
set -ex
echo "Module CACAO"
if [ -d "plugins/cacao-src" ]; then
    echo "	Already installed -> pulling"
    pushd plugins/cacao-src
    git pull
    popd
else
    # If a local mirror exists in $HOME/githubalt/cacao, use it
    if [ -d "$HOME/githubalt/cacao/.git" ]; then
        repository="file://$HOME/githubalt/cacao"
        branchopt=""
    else
        # Allow overriding the upstream repository and branch via environment variables.
        # CACAO_REPOSITORY - full git URL to clone (defaults to official cacao repo)
        # CACAO_BRANCH     - branch to checkout (defaults to 'dev')
        repository="${CACAO_REPOSITORY:-https://github.com/cacao-org/cacao.git}"
        branch="${CACAO_BRANCH:-dev}"
        branchopt="-b ${CACAO_BRANCH:-dev}"
    fi
    git clone $branchopt "$repository" plugins/cacao-src
fi
