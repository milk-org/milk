# Installation 

> [!NOTE]
> This file: ./src/CommandLineInterface/doc/DownloadCompile.md



---


# 1. Download and install milk {#milkinstall}

> [!WARNING]
> This page describes installation of the core package milk. If you install application package (cacao or coffee), replace "milk" with "cacao" in these instructions.


## 1.1. Download and compile {#milkinstall_downloadcompile}

```text
git clone --recursive https://github.com/cacao-org/milk milk
cd milk
mkdir _build
cd _build
cmake ..
# If you use NVIDIA GPUs, install cuda and magma libraries, and use "cmake .. -DUSE_MAGMA=ON"
make
sudo make install
```


## 1.2. Post-installation {#milkinstall_postinstall}

You may need to add /usr/local/lib to LD_LIBRARY_PATH environment variable:

```text
echo "/usr/local/lib" > usrlocal.conf
sudo mv usrlocal.conf /etc/ld.so.conf.d/
sudo ldconfig -v
```


## 1.3. tmpfs (optional) {#milkinstall_tmpfs}

OPTIONAL: Create tmpfs disk for high performance I/O:

```text
echo "tmpfs /milk/shm tmpfs rw,nosuid,nodev" | sudo tee -a /etc/fstab
sudo mkdir -p /milk/shm
sudo mount /milk/shm
```


---

# 2. Dependancies 


## 2.1. Libraries 

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


## 2.2. FITSIO install 

For reading and writing FITS image files

- Visit https://heasarc.gsfc.nasa.gov/fitsio/fitsio.html and download the file Unix .tar file cfitsio3410.tar.gz
- Extract it , README , install it
There is the fitsio.h in it. Move it to usr :

		./configure --prefix=/usr
		make
		sudo make install


## 2.3. GPU acceleration (optional, but highly recommended) 

Required libraries:

- install **NVIDIA driver**
- install **CUDA**
- install **MAGMA**

### No package 'magma' found

configure script uses pkg-config to find the package. You need to add in .bashrc :

	export PKG_CONFIG_PATH=/usr/local/magma/lib/pkgconfig

---


# 3. Running multiple versions {#milk_multipleversions}


> [!WARNING]
> Untested, may require tweaking

To install independant versions on the same system, download source code in separate source directories:

```text
cd $HOME/src
git clone --recursive https://github.com/milk-org/milk milk-1
git clone --recursive https://github.com/milk-org/milk milk-2
```



Compile each copy with a different target directory :

```text
cd $HOME/src/milk-1
mkdir _build
cd _build
cmake -DCMAKE_INSTALL_PREFIX=/usr/local/milk-1 ..
sudo make install
```

```text
cd $HOME/src/milk-2
mkdir _build
cd _build
cmake -DCMAKE_INSTALL_PREFIX=/usr/local/milk-2 ..
sudo make install
```


To make version 1 the default on the system :

```text
sudo ln -s /usr/local/milk-1 /usr/local/milk
```


To run an instance of version 2 :

```text
LD_LIBRARY_PATH=/usr/local/milk-2/lib PATH=/usr/local/milk-2/bin milk
```


Additionally, each version may have its own independent shared memory space for streams :
```text
MILK_SHM_DIR=/milk-2/shm LD_LIBRARY_PATH=/usr/local/milk-2/lib PATH=/usr/local/milk-2/bin milk
```
