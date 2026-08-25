import pytest

try:
    from pyMilk.ImageStreamIOWrap import _gcov_dump
except ImportError:

    def _gcov_dump():
        print(
            "pyMilk.ImageStreamIOWrap._gcov_dump not available"
            " -- compiled without -DCMAKE_BUILD_TYPE=Coverage ?"
        )


@pytest.fixture(scope="session", autouse=True)
def ctfixt_coverage_gcov_dump_at_session_end():
    yield
    _gcov_dump()
