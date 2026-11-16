#!/bin/env python
"""
zmq_recv.py

Usage:
    zmq_recv.py <host:port> <shm_name> [options]

Options:
    <host:port>     IP adress : port to receive from.
    <shm_name>    SHM to listen for - carried over from source server
    -s=<name_OR>  name override - reassign the SHM with a different name
"""

from __future__ import annotations

"""bash
IP=""
for CH in 00 01 02 03 04 05 06 07 08 09 10 11; do
    zmq_recv.py ${IP}:321${CH} dm00disp${CH}
done
"""

import zmq
import pickle
import numpy as np

from docopt import docopt
from pyMilk.interfacing.shm import SHM

import logging

logg = logging.getLogger(__name__)


def zmq_recv_loop(host_port: tuple[str, int], topic: str, out_name: str):

    # Get the ZMQ side ready
    context = zmq.Context()
    socket = context.socket(zmq.SUB)
    socket.setsockopt(
        zmq.SUBSCRIBE, topic.encode("ascii")
    )  # Subscribe to everything on this port.

    socket.connect(f"tcp://{host_port[0]}:{host_port[1]}")

    # We will get the pyMilk side ready in the loop upon the first reception
    pymilk_ready = False
    out_shm: SHM | None = None

    while True:
        message = socket.recv()
        sp_idx = message.index(b" ")

        keywords, data = pickle.loads(message[sp_idx + 1 :])

        if pymilk_ready:
            assert out_shm is not None
            out_shm.set_data(data)
        else:
            # Needs a little kick on dtype parsing o_O
            try:  # Try reuse
                out_shm = SHM(out_name, nbkw=0)
                out_shm.set_data(
                    data.astype(np.dtype(data.dtype.name))
                )  # Make it crash here if dtype/dshape noncompliant
                out_shm.set_keywords(
                    keywords
                )  # Make it crash here if we need keyword space
            except:
                out_shm = SHM(
                    out_name,
                    data=data.astype(np.dtype(data.dtype.name)),
                    nbkw=len(keywords) * 2,
                )
            pymilk_ready = True

        out_shm.set_keywords(keywords)


def main():
    # Parse
    from docopt import docopt

    doc = docopt(__doc__)

    hp = doc["<host:port>"].split(":")
    host = hp[0]
    port = int(hp[1])
    topic = doc["<shm_name>"]

    out_name = doc["-s"]
    if out_name is None:
        out_name = topic

    zmq_recv_loop((host, port), topic, out_name)


if __name__ == "__main__":
    main()
