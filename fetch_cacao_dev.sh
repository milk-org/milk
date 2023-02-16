#!/usr/bin/env bash

echo "Module CACAO"
if [ -d "plugins/cacao-src" ]; then
    echo "	Already installed -> pulling"
    (cd plugins/cacao-src; git pull)
else
    git clone -b dev https://github.com/cacao-org/cacao plugins/cacao-src
    echo ""
fi
