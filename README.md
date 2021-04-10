[![License: GPL v3](https://img.shields.io/badge/License-GPL%20v3-blue.svg)](http://www.gnu.org/licenses/gpl-3.0)



**main**

[![CMake badge](https://github.com/milk-org/milk/actions/workflows/cmake.yml/badge.svg)](https://github.com/milk-org/milk/actions/workflows/cmake.yml)
[![Build Status](https://www.travis-ci.com/milk-org/milk.svg?branch=main)](https://www.travis-ci.com/milk-org/milk)
![lastcommit](https://img.shields.io/github/last-commit/milk-org/milk/main.svg)


**dev**

[![CMake badge](https://github.com/milk-org/milk/actions/workflows/cmake.yml/badge.svg)](https://github.com/milk-org/milk/actions/workflows/cmake.yml)
[![Build Status dev](https://www.travis-ci.com/milk-org/milk.svg?branch=dev)](https://www.travis-ci.com/milk-org/milk)
![lastcommit](https://img.shields.io/github/last-commit/milk-org/milk/dev.svg)




# Milk

milk-core for **milk** package.


Module inclues key frameworks :

- **image streams** : low-latency shared memory streams
- **processinfo** : process management and control
- **function parameter structure (FPS)** : reading/writing function parameters

## Compilation

    ./compile.sh $PWD/local

## Adding plugins

Compile with cacao plugins:

    ./fetch_cacao_dev.sh
    ./compile.sh $PWD/local

Compile with coffee plugins:

    ./fetch_coffee_dev.sh
    ./compile.sh $PWD/local


