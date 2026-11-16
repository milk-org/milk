from pyMilk.interfacing.fps import FPS
from pyMilk.interfacing.shm import SHM

import numpy as np

from milk.operate.compute_unit import ComputeUnit

f1 = ComputeUnit("milk-fpsexec-gpusynctest-addone", "fa")
f2 = ComputeUnit("milk-fpsexec-gpusynctest-addone", "fb")

f1.fpsinit(procinfo=True)
f2.fpsinit(procinfo=True)

assert f1.fps
f1.fps["procinfo.enabled"] = True
f1.fps["procinfo.loopcntMax"] = True
f1.fps["procinfo.triggermode"] = 3
f1.fps["procinfo.triggersname"] = "a"
f1.fps["in_name"] = "a"
f1.fps["out_name"] = "b"
f1.fps["gpusync"] = False

assert f2.fps
f2.fps["procinfo.enabled"] = True
f2.fps["procinfo.loopcntMax"] = True
f2.fps["procinfo.triggermode"] = 3
f2.fps["procinfo.triggersname"] = "b"
f2.fps["in_name"] = "b"
f2.fps["out_name"] = "c"
f2.fps["gpusync"] = True

a = SHM("a", np.zeros((256, 256), np.uint16), location=0)
b = SHM("b", np.zeros((256, 256), np.uint16), location=0)
c = SHM("c", np.zeros((256, 256), np.uint16), location=0)

f1.confstart()
f2.confstart()
f1.runstart()
f2.runstart()

input("Hanging on...")

# Caracterisations to do: with/without nanosleep, with/without MPS.

# Teardown
f1.fps.destroy()
f2.fps.destroy()
a.destroy()
b.destroy()
c.destroy()
