#!/usr/bin/env python
"""
creahmim (pyMilk)

Replaces bash creashmim. Adding versatily.

Usage:
    creashmim <name> [<size>...] [--type=<type>] [-z] [--kw=<kw>] [--force] [--del]

Options:
    <size>...          1, 2 or 3 successive integers.
    -z                 zero out the SHM
    --type=<type>      datatype (see below) [default: f32].
    --kw=<kw>          number of keywords to allocate [default: 50].
    --force            disable reuse_existing
    --del              enable delete_existing (does not exclude reusing existing!)

Types: f32, f64, c64, c128, u8, u16, u32, u64, i8, i16, i32, i64
"""

from docopt import docopt

from pyMilk.interfacing.shm_functions import creashmim

import numpy as np

TYPE_DICT = {
    "f32": np.float32,
    "f64": np.float64,
    "c64": np.csingle,
    "c128": np.cdouble,
    "i8": np.int8,
    "i16": np.int16,
    "i32": np.int32,
    "i64": np.int64,
    "u8": np.uint8,
    "u16": np.uint16,
    "u32": np.uint32,
    "u64": np.uint64,
}


def main():
    args = docopt(__doc__)

    name = args["<name>"]
    shape = tuple([int(k) for k in args["<size>"]])
    data_type = TYPE_DICT[args["--type"]]
    nb_kw = int(args["--kw"])

    creashmim(
        name,
        shape,
        data_type,
        nb_kw,
        attempt_reuse=not args["--force"],
        delete_existing=args["--del"],
        do_zero=args["-z"],
    )
