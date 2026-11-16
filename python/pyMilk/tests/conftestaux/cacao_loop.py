import os, shutil, pathlib

import pytest

from importlib.resources import files
import subprocess as sproc

import tests  # should be the pytest file tree root

# TODO: port cacao_loop_deploy to pyMilk
# from aorts.cacao_stuff.loop_manager import CacaoConfigReader, CacaoLoopManager, cacao_loop_deploy
"""
# ConfTest.py FIXTture == ctfixt_
@pytest.fixture(scope='session')
def ctfixt_parse_config(request, tmpdir_factory):
    ...
"""
