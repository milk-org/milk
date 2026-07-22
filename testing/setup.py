"""
pyMilk setup.py
"""

import os
import sys
import platform
import subprocess
from distutils.version import LooseVersion

from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext

import shlex
import pathlib


class CMakeExtension(Extension):
    """
    This is made to build a sub-project
    <setup.py's directory>/<folder>
    with CMake

    Each <folder> becomes its own CMakeExtension and is built with
    the CMakeBuildExt class
    """

    def __init__(self, name, package, sourcedir=""):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)  # + '/' + self.name
        self.package = package


class CMakeSelfExtension(Extension):
    """
    This extension is a singleton approach that calls
    the parent CMakeLists.txt that is in the same folder than setup.py
    """

    def __init__(self, package, sourcedir=""):
        Extension.__init__(
            self, "SELF", sources=[]
        )  # Must avoid name collision with the root package.
        self.sourcedir = os.path.abspath(sourcedir)
        self.package = package


class CMakeBuildExt(build_ext):

    def run(self):
        self.inplace = True
        try:
            out = subprocess.check_output(
                ["cmake", "--version"]
            )  # Will raise FileNotFoundError
        except:
            raise RuntimeError(
                "CMake and nanobind must be installed to build the following extensions: "
                + ", ".join(e.name for e in self.extensions)
                + "\n (pip install nanobind)"
            )

        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        self.announce("Preparing the build environment", level=3)

        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))

        build_temp_subdir = self.build_temp + "/" + ext.name

        if self.editable_mode:
            # When running `pip install -e .`
            # extdir = $HOME/src/pyMilk/
            # drop lib in $HOME/src/pyMilk/pyMilk
            lib_drop_in_directory = extdir + "/" + ext.package
        else:
            # self.build_temp is:    build/temp.linux-x86_64-cpython-310
            # build_temp_subdir is:  build/temp.linux-x86_64-cpython-310/ImageStreamIO
            # drop lib in:
            #       build/lib.linux-x86_64-cpython/pyMilk
            build_temp_path = pathlib.Path(os.path.abspath(self.build_temp))
            lib_drop_in_directory = str(
                build_temp_path.parent
                / build_temp_path.name.replace("temp", "lib", 1)
                / ext.package
            )

        os.makedirs(build_temp_subdir, exist_ok=True)

        cmake_args = [
            "-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=" + lib_drop_in_directory,
        ]

        if os.environ.get("COVERAGE", None) == "ON":
            cfg = "Coverage"
        else:
            cfg = "Debug" if self.debug else "Release"
        build_args = ["--config", cfg]

        cmake_args += ["-DCMAKE_BUILD_TYPE=" + cfg]
        build_args += ["--", "-j%d" % os.cpu_count()]  # , 'VERBOSE=1']

        if "CUDA_ROOT" in os.environ:
            if os.path.isfile("{}/bin/gcc".format(os.environ["CUDA_ROOT"])):
                cmake_args += [
                    "-DCMAKE_C_COMPILER={}/bin/gcc".format(os.environ["CUDA_ROOT"])
                ]
            if os.path.isfile("{}/bin/g++".format(os.environ["CUDA_ROOT"])):
                cmake_args += [
                    "-DCMAKE_CXX_COMPILER={}/bin/g++".format(os.environ["CUDA_ROOT"])
                ]
            cmake_args += ["-DUSE_CUDA=ON"]
        else:
            cmake_args += ["-DUSE_CUDA=OFF"]

        self.announce("Configuring cmake project", level=3)
        command_a = f"cmake {ext.sourcedir} " + " ".join(cmake_args)
        command_b = f"cmake --build . " + " ".join(build_args)

        # Great way to locate a print-debug quickly when running pip -v
        #'''
        print(
            f'{ext=} \n\t {ext.name=} \n\t {ext.sourcedir} \n\t command_a = "{command_a}"'
            f' \n\t command _b = "{command_b}" \n\t cmake cwd = {build_temp_subdir}'
        )
        #'''

        subprocess.check_call(shlex.split(command_a), cwd=build_temp_subdir)
        subprocess.check_call(shlex.split(command_b), cwd=build_temp_subdir)


with open("README.md", "r") as f:
    long_description = f.read()

# import sys
# raise ValueError(sys.argv)

PACKAGE_PARENT = "noop_milktest"
PACKAGE_LIBS = "milk"
setup(
    packages=[PACKAGE_LIBS],  # same as name
    ext_modules=[
        CMakeExtension(PACKAGE_LIBS, PACKAGE_PARENT, sourcedir=PACKAGE_LIBS),
    ],
    cmdclass=dict(build_ext=CMakeBuildExt),
    long_description=long_description,
)
