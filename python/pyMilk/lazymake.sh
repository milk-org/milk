#!/bin/bash

mkdir -p _build
cd _build
cmake ../bindings
make
find . -name *.so | xargs -I {} cp {} ../pyMilk/
