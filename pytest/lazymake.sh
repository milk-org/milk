#!/bin/bash

cd _build
cmake ../milkengine -DUSE_CUDA=ON
make
find . -name *.so | xargs -I {} cp {} ../pyMilk/
