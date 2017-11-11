# Installation {#page_installation}


# Downloading source code
You can clone this repository, or download the latest .tar.gz distribution.


# 1 Libraries 

## 1.1 Pre-requisites

Libraries required :

- **gcc**
- **openMP**
- **fitsio**
- **fftw** (single and double precision), for performing Fourier Transforms
- **gsl**
- **readline**, for reading the command line input
- **tmux**
- **bash dialog**, version 1.2 minimum
- **flex**, for parsing the command line input
- **bison**, to interpret the command line input
- **gsl**, for math functions and tools

Install above libraries (centOS):

		sudo yum install readline-devel flex bison-devel fftw3-devel gsl-devel

Install above libraries (Ubuntu):

		sudo apt-get install libcfitsio3 libcfitsio3-dev libreadline6-dev libncurses5-dev libfftw3-dev libgsl0-dev flex bison


## 1.2 FITSIO install

For reading and writing FITS image files

- Visit https://heasarc.gsfc.nasa.gov/fitsio/fitsio.html and download the file Unix .tar file cfitsio3410.tar.gz
- Extract it , README , install it 
There is the fitsio.h in it. Move it to usr :

		./configure --prefix=/usr
		make 
		sudo make install 

## 1.3 GPU acceleration (optional, but highly recommended)

- install **NVIDIA driver**
- install **CUDA**
- install **MAGMA**, version 2.x


## 1.4 Shared Memory Image Stream Viewer

Two options:

- shared memory image viewer (`shmimview` or similar)
- qt-based `shmimviewqt`




# 2 Compilation from git clone (recommended for developers)

## 2.1 Additional libraries

### 2.1.1 CentOS

Install Development tools, use the command bellow. This will search the yum repositories, and install the tools from the closest repo.

		sudo yum groupinstall "Development tools"

### 2.1.2 Ubuntu

		sudo apt-get install autoconf libtool git


## 2.2 Compilation

The source code follows the standard GNU build process:

		autoreconf -vif
		./configure
		make
		make install


# 3 Compilation from tarball


Unpack

		gunzip <package>.tar.gz
		tar -xvf <package>.tar

The source code follows the standard GNU build process:

		./configure
		make
		sudo make install

