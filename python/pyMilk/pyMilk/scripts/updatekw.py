#!/usr/bin/env python
"""
updatekw

Usage:
    updatekw <shm_name> <kw_name> <kw_value>

Options:
    <shm_name>       Stream name
    <kw_name>        Keyword name
    <kw_value>       Keyword value to set
"""

from docopt import docopt

from pyMilk.interfacing.shm import SHM


def main():
    args = docopt(__doc__)
    shm_name = args["<shm_name>"]
    kw_name = args["<kw_name>"]
    kw_value = args["<kw_value>"]
    try:
        kw_value = float(kw_value)
    except:
        pass

    shm = SHM(shm_name)
    try:
        shm.update_keyword(kw_name, kw_value)
    except:
        shm.set_keywords({kw_name: kw_value})
