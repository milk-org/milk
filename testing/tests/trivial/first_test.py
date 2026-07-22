"""
My first test
Trying to get nox and gcovr happy

Will just run any piece of milk code
"""

import subprocess


def test_a_trivial_thing():
    subprocess.run(["milk-fps-list"])

    assert True
