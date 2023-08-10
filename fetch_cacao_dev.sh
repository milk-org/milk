#!/usr/bin/env bash

echo "Module CACAO"
if [ -d "plugins/cacao-src" ]; then
    echo "	Already installed -> pulling"
    (cd plugins/cacao-src; git pull)
else
    git clone -b dev https://github.com/cacao-org/cacao plugins/cacao-src
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
