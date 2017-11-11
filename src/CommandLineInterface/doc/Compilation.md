# Compilation {#page_Compilation}

Source code is written in C.
The package follows the GNU build process, and uses autotools (autmake, autoconf, libtool).

By default, libraries are dynamically linked.

Full list of compilation options is obtained by :

	./configure --help

For high performance (fast execution speed), use :

	./configure_highperf
	
