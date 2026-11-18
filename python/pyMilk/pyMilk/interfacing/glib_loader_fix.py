# This forces the resolution of libstdc++ from
# $CONDA_PREFIX/lib instead of the system one IF IT EXIST
# by the time we get to the ImageStreamIOWrap import
# This matters because the conda one will be the more recent GLIBC
# and linking an old one will cause ruckus.

# vdeo: stumbled upon this
# E     * The Python version is: Python 3.13 from "/home/pllab-vis/miniforge3/bin/python3.13"
# E     * The NumPy version is: "2.4.3"
# On a ubuntu 20.04 in April 2026.
# Importing everything separately works completely fine.

from ctypes import CDLL

try:
    lib = CDLL("libstdc++.so")
except:
    # print('No libstdc++ in $CONDA_PREFIX/lib. Proceeding.')
    pass
