#!/bin/bash

cd _build
cmake ../milkengine -DNOCUDA=ON
make
find . -name *.so | xargs -I {} cp {} ../pyMilk/
