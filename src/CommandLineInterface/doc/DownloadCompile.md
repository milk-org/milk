[TOC]


# Install {#page_installation}

@note This file: ./src/CommandLineInterface/doc/DownloadCompile.md


To compile :

	cd cacao
	mkdir _build
	cd _build
	cmake ..
	make
	sudo make install




---



# Details {#page_installation_details}


## Libraries {#page_installation_details_libraries}

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


### FITSIO install {#page_installation_details_libraries_fitsio}

For reading and writing FITS image files

- Visit https://heasarc.gsfc.nasa.gov/fitsio/fitsio.html and download the file Unix .tar file cfitsio3410.tar.gz
- Extract it , README , install it 
There is the fitsio.h in it. Move it to usr :

		./configure --prefix=/usr
		make 
		sudo make install 


---


## GPU acceleration (optional, but highly recommended) {#page_installation_details_gpuacceleration}

Required libraries:

- install **NVIDIA driver**
- install **CUDA**
- install **MAGMA**


---




## Shared Memory Image Stream Viewer {#page_installation_details_sharedmemviewer}

Two options:

- shared memory image viewer (`shmimview` or similar)
- qt-based `shmimviewqt`



---


## Compilation  {#page_installation_details_compilation}

### Installing cmake {#page_installation_details_compilation_installingcmake}

Use cmake version 3.xx.

To install cmake on centOS system (cmake executable will be cmake3):

	sudo yum install cmake3
	

### Compile source code {#page_installation_details_compilation_compilesourcecode}

To compile using cmake

	cd cacao
	mkdir _build
	cd _build
	cmake ..
	make
	sudo make install



### Post-installation configuration {#page_installation_details_compilation_postinstall}

You may need to add /usr/local/lib to LD_LIBRARY_PATH environment variable:

	echo "/usr/local/lib" > usrlocal.conf
	sudo mv usrlocal.conf /etc/ld.so.conf.d/
	sudo ldconfig -v


Add milk executable scripts to PATH environment variable. Add this line to the .bashrc file (change source code location as needed):

	export PATH=$PATH:/home/myname/src/cacao/src/CommandLineInterface/scripts

	



---

## Troubleshooting and FAQs {#page_installation_details_troubleshooting}


### No package 'magma' found

configure script uses pkg-config to find the package. You need to add in .bashrc :

	export PKG_CONFIG_PATH=/usr/local/magma/lib/pkgconfig


