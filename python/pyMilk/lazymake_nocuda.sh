#!/bin/bash

cd _build
cmake ../bindings
make
find . -name *.so | xargs -I {} cp {} ../pyMilk/
