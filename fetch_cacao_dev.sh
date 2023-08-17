#!/usr/bin/env bash

echo "Module CACAO"
if [ -d "plugins/cacao-src" ]; then
    echo "	Already installed -> pulling"
    (cd plugins/cacao-src; git pull)
else
    [ -d "$HOME/githubalt/cacao/.git" ] \
    && repository="file://$HOME/githubalt/cacao" branchopt= \
    || repository="https://github.com/cacao-org/cacao.git" branchopt="-b dev"
    echo git clone $branchopt "$repository" plugins/cacao-src
    git clone $branchopt "$repository" plugins/cacao-src
    echo ""
fi

PATCHFN=patch_cacao_lapacke_optional.txt
( [ -r  "./$PATCHFN" ] \
  && cd plugins/cacao-src/computeCalib/ \
  && PATCHPATH="../../../$PATCHFN" \
  && [ -r  "$PATCHPATH" ] \
  && patch -s --reject-file=- -f -p 2 < "$PATCHPATH" \
  && echo "Successfully patched CACAO computeCalib/CMakeList.txt" \
  || echo "Failed to patch CACAO computeCalib/CMakeList.txt" \
  || true
)
