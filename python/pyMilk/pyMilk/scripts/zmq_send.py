#!/bin/env python
"""
zmq_send.py

Usage:
    zmq_send.py <host:port> <shm_name> [options]

Options:
    <host:port>     IP address : port to bind to.
    <shm_name>      SHMs to send
    -t <timeout>    Timeout to re-send data [default: -1]
    -f <frate>      Throttling down to <frate> FPS
    -s <skip>       Send every n-th frame [default: 1]
"""

from __future__ import annotations

"""bash
IP=""
for CH in 00 01 02 03 04 05 06 07 08 09 10 11; do
    zmq_send.py ${IP}:321${CH} dm00disp${CH} &
done
"""

import time
import zmq
import pickle
from docopt import docopt
from pyMilk.interfacing.shm import SHM


def zmq_send_loop(
    host_port: tuple[str, int],
    shm_name: str,
    timeout: float,
    skip: int = 1,
    throttle_sec: float = 0.0,
):

    # Open shared memories
    shm_obj = SHM(shm_name)

    # Get the ZMQ side ready
    context = zmq.Context()
    socket = context.socket(zmq.PUB)

    socket.bind(f"tcp://{host_port[0]}:{host_port[1]}")

    init = True
    counter = 0
    t1 = time.time()

    while True:
        t2 = time.time()
        if t2 - t1 < throttle_sec:
            time.sleep(throttle_sec - t2 + t1)
        t1 = time.time()

        data = shm_obj.get_data(check=True, checkSemAndFlush=init, timeout=timeout)
        init = False

        if counter % skip == 0:
            kw = shm_obj.get_keywords()

            message = (shm_name + " ").encode("ascii")
            message += pickle.dumps((kw, data))
            socket.send(message)

        counter += 1


def main():
    # Parse
    from docopt import docopt

    doc = docopt(__doc__)

    hp = doc["<host:port>"].split(":")
    host = hp[0]
    port = int(hp[1])
    shm_name = doc["<shm_name>"]
    timeout = float(doc["-t"])
    if doc["-f"] is None:
        throttle_sec = 0.0
    else:
        throttle_sec = 1 / float(doc["-f"])

    skip = int(doc["-s"])

    zmq_send_loop((host, port), shm_name, timeout, skip=skip, throttle_sec=throttle_sec)


if __name__ == "__main__":
    main()
