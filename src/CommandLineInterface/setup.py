import os
import platform
import re
import subprocess
import sys
from distutils.version import LooseVersion

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext


class CMakeExtension(Extension):

    def __init__(self, name, sourcedir=''):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuildExt(build_ext):

    def run(self):
        try:
            out = subprocess.check_output(['cmake', '--version'])
        except OSError:
            raise RuntimeError(
                    "CMake must be installed to build the following extensions: "
                    + ", ".join(e.name for e in self.extensions))

        if platform.system() == "Windows":
            cmake_version = LooseVersion(
                    re.search(r'version\s*([\d.]+)', out.decode()).group(1))
            if cmake_version < '3.1.0':
                raise RuntimeError("CMake >= 3.1.0 is required on Windows")

        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        extdir = os.path.abspath(
                os.path.dirname(self.get_ext_fullpath(ext.name)))

        cmake_args = [
                '-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' + extdir,
                '-DPYTHON_EXECUTABLE=' + sys.executable
        ]

        cfg = 'Debug' if self.debug else 'Release'
        build_args = ['--config', cfg]

        if platform.system() == "Windows":
            cmake_args += [
                    '-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{}={}'.format(
                            cfg.upper(), extdir)
            ]
            if sys.maxsize > 2**32:
                cmake_args += ['-A', 'x64']
            build_args += ['--', '/m']
        else:
            cmake_args += ['-DCMAKE_BUILD_TYPE=' + cfg]
            build_args += ['--', '-j4']

        if "CUDA_ROOT" in os.environ:
            if os.path.isfile('{}/bin/gcc'.format(os.environ["CUDA_ROOT"])):
                cmake_args += [
                        '-DCMAKE_C_COMPILER={}/bin/gcc'.format(
                                os.environ["CUDA_ROOT"])
                ]
            if os.path.isfile('{}/bin/g++'.format(os.environ["CUDA_ROOT"])):
                cmake_args += [
                        '-DCMAKE_CXX_COMPILER={}/bin/g++'.format(
                                os.environ["CUDA_ROOT"])
                ]
            cmake_args += ['-DUSE_CUDA=ON']

        cmake_args += ['-Dpython_build=ON']

        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)

        subprocess.check_call(['cmake', ext.sourcedir] + cmake_args,
                              cwd=self.build_temp)
        subprocess.check_call(['cmake', '--build', '.'] + build_args,
                              cwd=self.build_temp)


setup(
        name='CacaoProcessTools',
        version='1.0.0',
        author=['Olivier Guyon', 'Arnaud Sevin'],
        author_email=['oliv.guyon@gmail.com', 'arnaud.sevin@obspm.fr'],
        description='',
        long_description='',
        scripts=['bin/tmuxsessionname'],
        ext_modules=[CMakeExtension('CacaoProcessTools')],
        cmdclass={'build_ext': CMakeBuildExt},
        zip_safe=False,
)
