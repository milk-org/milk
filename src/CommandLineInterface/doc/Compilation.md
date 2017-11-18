# Compilation {#page_Compilation}

Source code is written in C.
The package follows the GNU build process, and uses autotools (autmake, autoconf, libtool).

To compile and install:

	autoreconf -i
	./configure
	make
	make install
	
## Compilation options

By default, libraries are dynamically linked.

Full list of compilation options is obtained by :

	./configure --help

For high performance (fast execution speed), use :

	./configure CFLAGS='-Ofast -march=native' --enable-cuda --enable-magma
	
## MAGMA library

The configure script uses pkg-config to configure magma for linking. The MAGMA-provided pkg-config file (.pc file) must be made visible to pkg-config by running (ideally in .bashrc):

	export PKG_CONFIG_PATH=/usr/local/magma/lib/pkgconfig

