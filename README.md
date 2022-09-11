[![License: GPL v3](https://img.shields.io/badge/License-GPL%20v3-blue.svg)](http://www.gnu.org/licenses/gpl-3.0)


Latest Version: [![latesttag](https://img.shields.io/github/tag/milk-org/milk.svg)](https://github.com/milk-org/milk/tree/master)

| Branch    | Build   | Docker Deployment    | Travis-CI    | Activity   |
|-------------|-------------|-------------|-------------|-------------|
**main**|[![CMake badge](https://github.com/milk-org/milk/actions/workflows/cmake.yml/badge.svg?branch=main)](https://github.com/milk-org/milk/actions/workflows/cmake.yml)|[![CMake badge](https://github.com/milk-org/milk/actions/workflows/docker-image.yml/badge.svg?branch=main)](https://github.com/milk-org/milk/actions/workflows/docker-image.yml)|[![Build Status](https://www.travis-ci.com/milk-org/milk.svg?branch=main)](https://www.travis-ci.com/milk-org/milk)|![lastcommit](https://img.shields.io/github/last-commit/milk-org/milk/main.svg)|
**dev**|[![CMake badge](https://github.com/milk-org/milk/actions/workflows/cmake.yml/badge.svg?branch=dev)](https://github.com/milk-org/milk/actions/workflows/cmake.yml)|[![CMake badge](https://github.com/milk-org/milk/actions/workflows/docker-image.yml/badge.svg?branch=dev)](https://github.com/milk-org/milk/actions/workflows/docker-image.yml)|[![Build Status dev](https://www.travis-ci.com/milk-org/milk.svg?branch=dev)](https://www.travis-ci.com/milk-org/milk)|![lastcommit](https://img.shields.io/github/last-commit/milk-org/milk/dev.svg)|


Code metrics (dev branch) :
[![CodeScene Code Health](https://codescene.io/projects/14777/status-badges/code-health)](https://codescene.io/projects/14777)
[![CodeScene System Mastery](https://codescene.io/projects/14777/status-badges/system-mastery)](https://codescene.io/projects/14777)
[![Codacy Badge](https://app.codacy.com/project/badge/Grade/1c9a67a8529340359a2047eba5c971bf)](https://www.codacy.com/gh/milk-org/milk/dashboard?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=milk-org/milk&amp;utm_campaign=Badge_Grade)
[![CodeFactor](https://www.codefactor.io/repository/github/milk-org/milk/badge)](https://www.codefactor.io/repository/github/milk-org/milk)



***

# Milk

milk-core for **milk** package.


Module inclues key frameworks :

- **image streams** : low-latency shared memory streams
- **processinfo** : process management and control
- **function parameter structure (FPS)** : reading/writing function parameters

## Download

	git clone https://github.com/milk-org/milk.git
	cd milk
	./fetch_milk_dev.sh


## Compile

Standard compile:

	mkdir _build
	cd _build
	cmake ..
	make
	sudo make install

Compile with Python module (check script help with -h option for details):

    ./compile.sh $PWD/local


## Interactive tutorial

Pre-requisites: tmux, nnn

	milk-tutorial

## Adding plugins

Compile with cacao plugins:

    ./fetch_cacao_dev.sh
    ./compile.sh $PWD/local

Compile with coffee plugins:

    ./fetch_coffee_dev.sh
    ./compile.sh $PWD/local
