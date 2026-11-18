import pytest
import os

from pyMilk.interfacing import fps


def test_fps_errno_raiser():
    f = fps.FPS.create("some_name")

    f._errno_raiser(0, "info")

    with pytest.raises(fps.FPSErrnoError):
        f._errno_raiser(1, "error 1")

    f.destroy()


def test_fps_keyerrors():
    f = fps.FPS.create("some_name")

    # Successful get on trivial FPS
    assert f["Name"] == "some_name"

    with pytest.raises(ValueError):
        f["no_param"]

    with pytest.raises(ValueError):
        f["no_param_2"] = 1.0

    f.destroy()


def test_fps_overwrite():
    with pytest.raises(fps.FPSDoesntExistError):
        f = fps.FPS("some_name")

    f = fps.FPS.create("some_name")
    with pytest.raises(fps.FPSOverrideError):
        f = fps.FPS.create("some_name")

    f2 = fps.FPS.create("some_name", True)
    f2.destroy()


def test_smart_fps_instantiation():

    class A(fps.SmartAttributesFPS):
        pass

    with pytest.raises(fps.SmartFPSInitError):
        A("uniq_name_A")  # no _DICT_METADATA

    class B(fps.SmartAttributesFPS):
        gain: float

    with pytest.raises(fps.SmartFPSInitError):
        B("uniq_name_B")  # No _DICT_METADATA

    class C(fps.SmartAttributesFPS):
        gain: float
        _DICT_METADATA = {}

    with pytest.raises(fps.SmartFPSInitError):
        C("uniq_name_C")  # 'gain' not in _DICT_METADATA

    class D(fps.SmartAttributesFPS):
        _DICT_METADATA = {}

    assert isinstance(D.create("uniq_name_D"), D)

    class E(fps.SmartAttributesFPS):
        gain: float
        _DICT_METADATA = {"yolo": ()}

    with pytest.raises(fps.SmartFPSInitError):
        E("uniq_name_E")  # Unrelated key in _DICT_METADATA
    with pytest.raises(fps.SmartFPSInitError):
        E.create("uniq_name_E")  # _DICT_METADATA tuple-value noncompliant

    class F(fps.SmartAttributesFPS):
        gain: float
        _DICT_METADATA = {"gain": (1, 2, 3)}

    with pytest.raises(fps.SmartFPSInitError):
        F("uniq_name_F")  # _DICT_METADATA tuple-value noncompliant
    with pytest.raises(fps.SmartFPSInitError):
        F.create("uniq_name_F")  # _DICT_METADATA tuple-value noncompliant

    class G(fps.SmartAttributesFPS):
        gain: float
        _DICT_METADATA = {
            "gain": ("toto", fps.FpsType.FLOAT32, fps.FpsFlag.DEFAULT_INPUT)
        }

    with pytest.raises(fps.FPSDoesntExistError):
        G("uniq_name_G")  # _DICT_METADATA tuple-value noncompliant
    g = G.create("uniq_name_G")
    assert isinstance(g, G)
    assert isinstance(g.gain, float)

    g.gain = 3.1415
    assert g.gain == pytest.approx(3.1415)

    class H(fps.SmartAttributesFPS):
        gain: float
        _DICT_METADATA = {
            "gain": (
                "toto",
                fps.FpsType.FLOAT32,
                fps.FpsFlag.DEFAULT_INPUT,
                "an_extra_argument",
            )
        }

    with pytest.raises(fps.SmartFPSInitError):
        h = H.create("another_uniquename")


def test_smart_fps_find_annotations_cannot_be_overridden():

    class H(fps.SmartAttributesFPS):
        gain: float
        _DICT_METADATA = {
            "gain": ("gain", fps.FpsType.FLOAT32, fps.FpsFlag.DEFAULT_INPUT)
        }

        @classmethod
        def _cls_find_annotations(cls):
            return {"gain": float}

    with pytest.raises(fps.SmartFPSInitError, match="overrides _cls_find_annotations"):
        H.create("some_name")


@pytest.fixture
def fixt_dumb_fps():
    with pytest.raises(fps.FPSDoesntExistError):
        a = fps.FPS("uniq_fps")

    a = fps.FPS.create("uniq_fps")

    yield a

    a.destroy()


def test_pass_data(fixt_dumb_fps):
    a = fixt_dumb_fps
    a.add_param("abc", "abcComment", fps.FpsType.INT32, fps.FpsFlag.DEFAULT_INPUT)
    a["abc"] = 1

    b = fps.FPS(a.name)
    assert b["abc"] == 1


def test_flags(fixt_dumb_fps):
    a = fixt_dumb_fps
    # Added first: a substring lookup of "trig" would hit "trigx"
    a.add_param("trigx", "c", fps.FpsType.FLOAT32)
    a.add_param("trig", "c", fps.FpsType.STREAMNAME, fps.FpsFlag.DEFAULT_TRIGGER_STREAM)
    a.add_param("gain", "c", fps.FpsType.FLOAT32)

    # Reconnect: index cache built from shm, not from add_param
    b = fps.FPS(a.name)
    assert fps.FpsFlag.TRIGGER_STREAM in b.get_flags("trig")
    assert fps.FpsFlag.TRIGGER_STREAM not in b.get_flags("gain")
    assert b.get_flags("gain") == fps.FpsFlag.DEFAULT_INPUT

    # Same object that added the params
    assert fps.FpsFlag.TRIGGER_STREAM in a.get_flags("trig")
    assert a.get_flags("gain") == fps.FpsFlag.DEFAULT_INPUT

    with pytest.raises(ValueError):
        a.get_flags("nonexistent")


@pytest.fixture
def fixt_smart_fps_properties():

    class A(fps.SmartAttributesFPS):
        i4: int
        i8: int
        u4: int
        u8: int
        f4: float
        f8: float
        s: str

        _DICT_METADATA = {
            "i4": ("i4", fps.FpsType.INT32, fps.FpsFlag.DEFAULT_INPUT),
            "i8": ("i8", fps.FpsType.INT64, fps.FpsFlag.DEFAULT_INPUT),
            "u4": ("u4", fps.FpsType.UINT32, fps.FpsFlag.DEFAULT_INPUT),
            "u8": ("u8", fps.FpsType.UINT64, fps.FpsFlag.DEFAULT_INPUT),
            "f4": ("f4", fps.FpsType.FLOAT32, fps.FpsFlag.DEFAULT_INPUT),
            "f8": ("f8", fps.FpsType.FLOAT64, fps.FpsFlag.DEFAULT_INPUT),
            "s": ("s", fps.FpsType.STRING, fps.FpsFlag.DEFAULT_INPUT),
        }

    with pytest.raises(fps.FPSDoesntExistError):
        a = A("uniq_a")

    a = A.create("uniq_a")

    yield a

    try:
        a.destroy()
    except FileNotFoundError:
        print(f"Could not destroy {a.name} - already gone.")


def test_smart_to_dumb(fixt_smart_fps_properties):
    a = fixt_smart_fps_properties
    a.i4 = 12345

    f = fps.FPS(a.name)
    with pytest.raises(AttributeError):
        _ = f.i4

    assert f["i4"] == 12345


def test_downcasting(fixt_smart_fps_properties):
    a = fixt_smart_fps_properties
    a.i4 = 12345

    f = fps.FPS(a.name)

    f_as_a = type(a).smartfps_downcast(f)
    assert f_as_a.i4 == 12345
    f_as_a.s = "yolo"
    assert f["s"] == "yolo"


def test_md(fixt_dumb_fps):
    fps = fixt_dumb_fps
    md = fps.fps.md()
    for tag in md.__dir__():
        if not tag.startswith("_"):
            print(tag, getattr(md, tag))

    assert True


def test_fixture_works(fixt_smart_fps_properties):
    fps = fixt_smart_fps_properties
    assert True


def test_fixture_works_twice(fixt_smart_fps_properties):
    assert True


def test_smart_fps_property_types(fixt_smart_fps_properties):
    fps = fixt_smart_fps_properties

    # Test integer properties
    fps.i4 = 42
    assert fps.i4 == 42
    fps.i8 = -9223372036854775807  # Test large int64
    assert fps.i8 == -9223372036854775807

    # Test unsigned integers
    fps.u4 = 4294967295  # Max uint32
    assert fps.u4 == 4294967295
    fps.u8 = 18446744073709551615  # Max uint64
    assert fps.u8 == 18446744073709551615

    # Test floats
    fps.f4 = 3.14159
    assert fps.f4 == pytest.approx(3.14159)
    fps.f8 = 2.718281828459045
    assert fps.f8 == pytest.approx(2.718281828459045)

    # Test string
    fps.s = "Hello FPS!"
    assert fps.s == "Hello FPS!"


def test_smart_fps_type_validation(fixt_smart_fps_properties):
    fps = fixt_smart_fps_properties

    # Test invalid type assignments
    with pytest.raises(RuntimeError):
        fps.i4 = "not integer-able"
    with pytest.raises(RuntimeError):
        fps.f4 = "not float-able"
    """ That worked with pybind, it had more
    autocasting...
    fps.s = 42
    assert fps.s == '42'  # 'Dis JS or what?
    """
    # Nanobind is less friendly:
    with pytest.raises(RuntimeError):
        fps.s = 42


def test_smart_fps_concurrent_access():
    """Test that multiple SmartAttributesFPS instances can coexist"""

    class TestFPS(fps.SmartAttributesFPS):
        value: float
        _DICT_METADATA = {
            "value": ("val", fps.FpsType.FLOAT32, fps.FpsFlag.DEFAULT_INPUT)
        }

    fps1 = TestFPS.create("fps_test_1")
    fps2 = TestFPS.create("fps_test_2")

    fps1.value = 1.0
    fps2.value = 2.0

    assert fps1.value == pytest.approx(1.0)
    assert fps2.value == pytest.approx(2.0)

    fps1.destroy()
    fps2.destroy()

    # Verify FPS was properly destroyed
    with pytest.raises(fps.FPSDoesntExistError):
        TestFPS("fps_test_1")
    with pytest.raises(fps.FPSDoesntExistError):
        TestFPS("fps_test_2")


def test_fps_filesystem_operations():
    """Test that FPS create/destroy operations are reflected in the filesystem"""

    fps_name = "test_fps_filesystem_ops"
    fps_path = (
        os.environ["MILK_SHM_DIR"] + f"/{fps_name}.fps.shm"
    )  # Standard FPS file location

    # Ensure FPS doesn't exist before test
    if os.path.exists(fps_path):
        raise RuntimeError(f"FPS file {fps_path} already exists")

    # Test creation
    test_fps = fps.FPS.create(fps_name)
    assert os.path.exists(fps_path), "FPS file was not created"

    # Test destruction
    test_fps.destroy()
    assert not os.path.exists(fps_path), "FPS file was not removed"

    # Verify FPS was properly destroyed
    with pytest.raises(fps.FPSDoesntExistError):
        fps.FPS(fps_name)


def test_smart_fps_auto():

    class NewFPS(fps.SmartAttributesFPSAutoMetadata):
        gain: float
        counter: int
        name_of_something: str

    f = NewFPS.create("a_name")

    f2 = NewFPS("a_name")
    f2.gain = 3.1415
    f2.counter = 100
    f2.name_of_something = "YOLO"

    assert f.gain == 3.1415
    assert f.counter == 100
    assert f.name_of_something == "YOLO"

    # TODO breakpoint() here and figure out the interaction with milk-fpsCTRL.

    f.destroy()


def test_metadata_mot_mappable():

    class NewFPS(fps.SmartAttributesFPSAutoMetadata):
        gain: None

    with pytest.raises(fps.SmartFPSInitError):
        f = NewFPS.create("xxxx")


def test_fps_isvalid(fixt_smart_fps_properties):
    f: fps.FPS = fixt_smart_fps_properties
    assert f.is_valid()

    # drop bytes of garbage in f
    fps_path = os.environ["MILK_SHM_DIR"] + f"/{f.name}.fps.shm"
    # TODO rather than False, this is most likely to return a segfault
    # size_bytes = os.stat(fps_path).st_size
    # import random
    # with open(fps_path, 'rb+') as fps_file:
    #    fps_file.write(random.randbytes(16))

    os.remove(fps_path)
    assert not f.is_valid()


def test_non_leak_maps_with_isvalid(fixt_smart_fps_properties):

    def _count_mmaps_of(fps_filepath: str) -> int:
        with open(f"/proc/{os.getpid()}/maps") as fh:
            return sum(1 for line in fh if fps_filepath in line)

    f: fps.FPS = fixt_smart_fps_properties

    fps_path = os.environ["MILK_SHM_DIR"] + f"/{f.name}.fps.shm"
    baseline = _count_mmaps_of(fps_path)
    assert baseline > 0

    for _ in range(50):
        assert f.is_valid()

    assert _count_mmaps_of(fps_path) == baseline
