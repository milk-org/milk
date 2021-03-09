#!/usr/bin/env bash

./fetch_milk_dev.sh

echo "COFFEE modules"
if [ -d "plugins/coffee-src" ]; then
  echo "	Already installed -> pulling"
  (cd plugins/coffee-src; git pull)
else
  git clone -b dev https://github.com/coffee-org/coffee plugins/coffee-src
  echo ""
fi
