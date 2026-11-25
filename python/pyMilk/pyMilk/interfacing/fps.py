"""
Pure python class FPS

Binds over the CacaoProcessTools pybind binding

Main purpose is to allow pyroification.

CacaoProcessTools needs to be compiled and installed:
- Have all of milk installed
- cd ${MILK_ROOT}
- pip install . (with your userland pip)
- Check the import works

Alternatively:
- Compile milk with -dbuild_python_module=ON
- Add the installdir ${MILK_INSTALLDIR} to your LD_LIBRARY_PATH.

First version:
    Requires milk more recent than Sep 27, 2023.
"""

from __future__ import annotations

import typing as typ

import os
import time
import glob
import weakref

_CORES = os.sched_getaffinity(0)
from . import glib_loader_fix
from pyMilk.FpsWrap import fps as CPTFPS, FpsType, FpsFlag

from pyMilk.interfacing.pinfo import ProcessInfo
from pyMilk.interfacing.shm_names import build_FPS_path, check_FPS_name

os.sched_setaffinity(0, _CORES)

if typ.TYPE_CHECKING:
    FPVal: typ.TypeAlias = str | bool | int | float


class FPSBaseError(Exception):
    pass


class FPSErrnoError(FPSBaseError):
    pass


class FPSOverrideError(FPSBaseError):
    pass


class FPSDoesntExistError(FPSBaseError):
    pass


class SmartFPSInitError(FPSBaseError):
    pass


if typ.TYPE_CHECKING:
    # TODO pack this in a stub file -- same for PINFO
    from dataclasses import dataclass

    @dataclass
    class FPS_METADATA:
        """Metadata exposed by the FPS_md nanobind class."""

        name: str
        description: str
        kwarray: str
        workdir: str
        datadir: str
        confdir: str
        sourcefname: str
        sourceline: int
        pname: str
        callprogname: str
        NBnameindex: int
        confpid: int
        confpidstarttime: object
        runpid: int
        runpidstarttime: object
        signal: int
        confwaitus: int
        status: int
        NBparamMAX: int
        msgpindex: typ.Sequence[int]
        msgcode: typ.Sequence[int]
        msgcnt: int
        conferrcnt: int


class FPS:

    _T_Subclass = typ.TypeVar("_T_Subclass", bound="FPS")  # Any subclass of this class

    @classmethod
    def create(
        cls: type[_T_Subclass], name: str, force_recreate: bool = False
    ) -> _T_Subclass:
        """
        Create an empty FPS.
        The only attribute set is the name.
        """
        name = check_FPS_name(name)
        fps_filepath = build_FPS_path(name)
        if os.path.exists(fps_filepath):
            if force_recreate:
                os.remove(fps_filepath)
            else:
                raise FPSOverrideError(f"FPS {name} exist and {force_recreate=}.")

        _fps = CPTFPS(name, True)  # thrown away, opened as "read" in constructor

        fps = cls(name)
        fps.add_param(
            "Name", "NameComment", FpsType.STRING, FpsFlag.DEFAULT_STATUS
        )  # 0x5 = VISIBLE | ACTIVE
        fps["Name"] = name

        return fps

    def __init__(self, name: str) -> None:
        self.name = check_FPS_name(name)
        self.filepath = build_FPS_path(self.name)
        if not os.path.isfile(self.filepath):
            raise FPSDoesntExistError(self.filepath)
        try:
            self.fps = CPTFPS(self.name)
        except RuntimeError as exc:
            raise FPSDoesntExistError from exc
        self.key_types: typ.Dict[str, int] = self.fps.keys

        self.procinfo: ProcessInfoFps | None = None
        if "procinfo.enabled" in self.key_types:
            self.procinfo = ProcessInfoFps(self)
            self._pinfo_struct: ProcessInfo | None = None

    def __str__(self) -> str:
        # FIXME append tmux status
        return f'{self.name} | CONF: {("N", "Y")[self.conf_isrunning()]} | RUN: {("N", "Y")[self.run_isrunning()]}'

    def _errno_raiser(self, retcode: int, info: str) -> None:
        if retcode != 0:
            raise FPSErrnoError(
                f"FPS {self.name}: errno raise code {retcode} with info {info}."
            )

    def is_valid(self) -> bool:
        if not os.path.isfile(self.filepath):
            return False
        try:
            _fps = CPTFPS(self.name, True)  # test that no error
        except:
            return False

        return True

    def add_param(
        self,
        key: str,
        comment: str,
        datatype: int,
        flags: int = FpsFlag.DEFAULT_INPUT,
    ) -> None:
        # TODO: _autorelink implementation !
        self.fps.add_entry(key, comment, datatype, flags)
        self.key_types[key] = datatype

    def get_flags(self, key: str) -> FpsFlag:
        """Current flags of parameter `key`, read live from shared memory."""
        if key in self.key_types:
            return FpsFlag(self.fps.get_flags(key))
        else:
            raise ValueError(f"get_flags: key {key} not in FPS {self.name}.")

    def set_param(self, key: str, value: FPVal) -> None:
        # TODO: _autorelink implementation !
        if key in self.key_types:
            self.fps[key] = value
        else:
            raise ValueError(f"set_param: key {key} not in FPS {self.name}.")

    def get_param(self, key: str) -> FPVal:
        # TODO: _autorelink implementation !
        if key in self.key_types:
            return self.fps[key]
        else:
            raise ValueError(f"get_param: key {key} not in FPS {self.name}.")

    __setitem__ = set_param
    __getitem__ = get_param

    def __contains__(self, key: str) -> bool:
        return key in self.key_types

    """
    We don't have that in pyFPS!
    def tmux_isrunning(self) -> bool:
        return self.fps.TMUXrunning
    """

    def conf_isrunning(self) -> bool:
        return self.fps.CONFrunning == 1

    def run_isrunning(self) -> bool:
        return self.fps.RUNrunning == 1

    def conf_start(self, timeoutsync: float | None = None) -> bool:
        self._errno_raiser(self.fps.CONFstart(), "conf_start")
        if timeoutsync:
            start = time.time()
            while (not self.conf_isrunning()) and (time.time() - start < timeoutsync):
                time.sleep(1e-4)

        return self.conf_isrunning()

    def conf_stop(self, timeoutsync: float | None = None) -> bool:
        self._errno_raiser(self.fps.CONFstop(), "conf_stop")
        if timeoutsync:
            start = time.time()
            while self.conf_isrunning() and (time.time() - start < timeoutsync):
                time.sleep(1e-4)
        return not self.conf_isrunning()

    def run_start(self, timeoutsync: float | None = None) -> bool:
        self._errno_raiser(self.fps.RUNstart(), "run_start")
        if timeoutsync:
            start = time.time()
            while (not self.run_isrunning()) and (time.time() - start < timeoutsync):
                time.sleep(1e-4)
        return self.run_isrunning()

    def run_stop(self, timeoutsync: float | None = None) -> bool:
        self._errno_raiser(self.fps.RUNstop(), "run_stop")
        if timeoutsync:
            start = time.time()
            while (self.run_isrunning()) and (time.time() - start < timeoutsync):
                time.sleep(1e-4)
        return not self.run_isrunning()

    def tmux_start(self) -> None:
        self._errno_raiser(self.fps.TMUXstart(), "tmux_start")

    def tmux_stop(self) -> None:
        self._errno_raiser(self.fps.TMUXstop(), "tmux_stop")

    def disconnect(self) -> None:
        self.fps.disconnect()

    def destroy(self) -> None:
        self.disconnect()
        os.remove(self.filepath)

    @property
    def md(self) -> FPS_METADATA:
        return self.fps.md()

    @property
    def pinfo_struct(self) -> ProcessInfo:
        run_pid = self.md.runpid

        if self._pinfo_struct is not None and run_pid != self._pinfo_struct._pinfo.PID:
            self._pinfo_struct._pinfo.close()  # TODO wrap function.
            self._pinfo_struct = None

        if self._pinfo_struct is None:
            self._pinfo_struct = ProcessInfo(self.name, run_pid)

        return self._pinfo_struct


class FPSCollection:
    # TODO Non-regex FPSManager base class
    ...


class FPSManager(FPSCollection):

    def __init__(self, fps_name_glob: str = "*", fps_keyword_glob: str = "*") -> None:
        self.fps_cache: typ.Dict[str, FPS] = {}
        self.fps_name_glob = fps_name_glob
        self.fps_keyword_glob = fps_keyword_glob
        self.rescan_all()

    def find_fps(self, name: str) -> FPS:
        if not name in self.fps_cache or self.fps_cache[name] is None:
            self.fps_cache[name] = FPS(name)  # throws a runtime error

        return self.fps_cache[name]

    def rescan_all(
        self, fps_name_glob: str | None = None, fps_keyword_glob: str | None = None
    ):

        if fps_name_glob is None:
            fps_name_glob = self.fps_name_glob
        if fps_keyword_glob is None:
            fps_keyword_glob = self.fps_keyword_glob
        import fnmatch

        self.purge_cache()
        system_fps_files = glob.glob(build_FPS_path(fps_name_glob))
        for fps_file in system_fps_files:
            self.find_fps(check_FPS_name(fps_file))

        non_matching_keys = set()
        for name, fpsobj in self.fps_cache.items():
            kw_str = fpsobj.md.kwarray.strip(":")
            if len(kw_str) == 0 or not any(
                (fnmatch.fnmatch(kw, fps_keyword_glob) for kw in kw_str.split(":"))
            ):
                non_matching_keys.add(name)

        for name in non_matching_keys:
            fps = self.fps_cache.pop(name)
            fps.disconnect()

    def purge_cache(self) -> None:
        for k, v in self.fps_cache.items():
            v.fps.disconnect()
        self.fps_cache = {}

    def set_param(self, fps: str, key: str, value: FPVal) -> None:
        self.find_fps(fps).set_param(key, value)

    def get_param(self, fps: str, key: str) -> FPVal:
        return self.find_fps(fps).get_param(key)

    def conf_start(self, fps: str, timeout: float | None = None) -> None:
        self.find_fps(fps).conf_start(timeout)

    def conf_stop(self, fps: str, timeout: float | None = None) -> None:
        self.find_fps(fps).conf_stop(timeout)

    def run_start(self, fps: str, timeout: float | None = None) -> None:
        self.find_fps(fps).run_start(timeout)

    def run_stop(self, fps: str, timeout: float | None = None) -> None:
        self.find_fps(fps).run_stop(timeout)

    def tmux_start(self, fps: str) -> None:
        self.find_fps(fps).tmux_start()

    def tmux_stop(self, fps: str) -> None:
        self.find_fps(fps).tmux_stop()


class SmartAttributesFPS(FPS):
    """
    This is meant for FPS that get reused a lot, so that we get static typing

    E.g. define

    class SFPS(SmartAttributesFPS):
        gain: float

    in the class preamble

    and now you may do:

    f = SPFS()

    f.gain # return f['gain'] but correctly hinted to float
    f.gain = 1.0 # pyright no screaming
    f.gain = 'yolo' # pyright screaming,
    although this would completely work at runtime since 'yolo' is a valid FPVal


    Remains the question of the initializer...

    e.g. we would love to pass something like:
    gain: ('Averaging gain', FpsType.FLOAT64, FpsFlag.DEFAULT_INPUT)

    Am I reinventing named tuples / named dicts?

    # TODO: make a subclass wherein _DICT_METADATA just maps all typical types to FPS types.
    # TODO: and just puts FpsFlag to RW at runtime at all time.
    """

    _DICT_METADATA: dict[str, tuple[str, FpsType, FpsFlag]]

    _T_Subclass = typ.TypeVar(
        "_T_Subclass", bound="SmartAttributesFPS"
    )  # Any subclass of this class

    def __new__(cls: type[_T_Subclass], name: str) -> _T_Subclass:
        """
        We're doing all this work in __new__ so that subclasses can inherit the behavior of
        create once the runtime checks are a go.
        Hol' up.
        Probs we could just work it out in the init...
        but we'd have to delete the FPS created by __new__ in case of non-compliance and aborting
        the __init__... so better in __new__
        """
        cls._cls_assert_find_annotations_not_overridden()
        cls._cls_metadata_checks()
        return super(SmartAttributesFPS, cls).__new__(cls)

    @classmethod
    def smartfps_downcast(cls: type[_T_Subclass], fps: FPS) -> _T_Subclass:
        cls._cls_metadata_checks()
        # This actually casts...
        # fps.__class__ = cls
        # Let's prefer making a copy.
        return cls(fps.name)

    @classmethod
    def _cls_metadata_checks(cls) -> None:
        if not hasattr(cls, "_DICT_METADATA"):
            raise SmartFPSInitError(
                f"Missing _DICT_METADATA to instantiate SmartAttributesFPS subclass {cls.__name__}"
            )
        if not cls._DICT_METADATA.keys() == cls._cls_find_annotations().keys():
            raise SmartFPSInitError(
                f"__annotations__ and _DICT_METADATA differ to instantiate "
                f"SmartAttributesFPS subclass {cls.__name__}"
            )
        for param, value_tuple in cls._DICT_METADATA.items():
            if not len(value_tuple) == 3:
                raise SmartFPSInitError(
                    f"_DICT_METADATA tuple not len 3 for {param} to "
                    f"instantiate SmartAttributesFPS subclass {cls.__name__}"
                )
            comment, tipe, flag = value_tuple
            if not (
                isinstance(comment, str)
                and isinstance(tipe, FpsType)
                and isinstance(flag, FpsFlag)
            ):
                raise SmartFPSInitError(
                    f"_DICT_METADATA tuple type not OK for {param} -"
                    f"- {value_tuple} to instantiate SmartAttributesFPS subclass {cls.__name__}"
                )

    @classmethod
    def _cls_assert_find_annotations_not_overridden(cls) -> None:
        for cls_in_hierarchy in cls.__mro__:
            if cls_in_hierarchy is SmartAttributesFPS:
                break
            if "_cls_find_annotations" in cls_in_hierarchy.__dict__:
                raise SmartFPSInitError(
                    f"{cls_in_hierarchy.__name__} overrides "
                    "_cls_find_annotations, which is forbidden. "
                    "Use _cls_annotation_technical_keys_to_remove instead."
                )

    @classmethod
    def _cls_find_annotations(
        cls: type[_T_Subclass],
    ) -> dict[str, typ.Any]:  # Really a dict_keys

        # Since this explores the MRO, it must NOT be subclassed !!!
        annotations = {}
        all_to_remove = set()
        for cls_in_hierarchy in cls.__mro__[::-1]:
            if not hasattr(cls_in_hierarchy, "__annotations__"):
                continue

            annotations.update(cls_in_hierarchy.__annotations__.copy())
            if not hasattr(
                cls_in_hierarchy, "_cls_annotation_technical_keys_to_remove"
            ):
                continue

            all_to_remove.update(
                cls_in_hierarchy._cls_annotation_technical_keys_to_remove()
            )

        for key in all_to_remove:
            del annotations[key]

        return annotations

    @classmethod
    def _cls_annotation_technical_keys_to_remove(cls: type[_T_Subclass]) -> list[str]:
        return ["_DICT_METADATA"]

    def _prop_fget(self, prop_name: str):

        def fget(self):
            return FPS.__getitem__(self, prop_name)

        return fget

    def _prop_fset(self, prop_name: str):

        def fset(self, value):
            FPS.__setitem__(self, prop_name, value)

        return fset

    def __init__(self, name: str) -> None:
        super().__init__(name)

        for key, (comment, tipe, flags) in self._DICT_METADATA.items():
            self.add_param(key, comment, tipe, flags)
        """
        This is for Pyro4 exposes and preventing a __getattribute / __setattribute
        metaprogramming pain

        We wrap DICT_METADATA in properties.
        The underlying data storage is not in hidden attributes, but directly in the FPS
        SHM tree.

        There will come a problem with this whole thing for attributes with dots
        in their names.
        """
        for key in self._DICT_METADATA:
            # setattr(self, '___' + key, getattr(self, key))
            setattr(
                self.__class__,
                key,
                property(self._prop_fget(key), self._prop_fset(key)),
            )


class ProcessInfoFps:

    RTprio: int
    cset: str
    taskset: str
    NBthread: int
    enabled: bool
    loopcntMax: int
    triggermode: int
    triggersname: str
    MeasureTiming: bool
    semindexrequested: int
    triggerdelay: float
    triggertimeout: float

    def __init__(self, parent_fps: FPS) -> None:
        self._fps = weakref.proxy(
            parent_fps
        )  # no cycle: FPS is freed (and disconnected) by refcount

        # create the properties
        for key in self.__annotations__:
            setattr(
                self.__class__,
                key,
                property(self._prop_fget(key), self._prop_fset(key)),
            )

    def _prop_fget(self, prop_name: str):

        def fget(self: ProcessInfoFps):
            return FPS.__getitem__(self._fps, "procinfo." + prop_name)

        return fget

    def _prop_fset(self, prop_name: str):

        def fset(self: ProcessInfoFps, value):
            FPS.__setitem__(self._fps, "procinfo." + prop_name, value)

        return fset


class SmartAttributesFPSAutoMetadata(SmartAttributesFPS):
    """
    Subclass of SmartAttributeFPS intended for mostly python use.

    Question: what happens if the underlying FPS exists and the flags don't match ???
    """

    _cls_metadata_is_init: typ.ClassVar[bool] = False
    _ANNOTATION_TYPE_MAPPER: typ.ClassVar = {
        bool: FpsType.ONOFF,
        int: FpsType.INT64,
        float: FpsType.FLOAT64,
        str: FpsType.STRING,
    }

    _T_Subclass = typ.TypeVar(
        "_T_Subclass", bound="SmartAttributesFPSAutoMetadata"
    )  # Any subclass of this class

    def __new__(cls: type[_T_Subclass], name: str) -> _T_Subclass:
        if not cls._cls_metadata_is_init:
            cls._cls_metadata_fill_dict()
            cls._cls_metadata_is_init = True
        return super(SmartAttributesFPSAutoMetadata, cls).__new__(cls, name)

    @classmethod
    def _cls_metadata_fill_dict(cls) -> None:
        if cls._cls_metadata_is_init:
            raise SmartFPSInitError("Already initialized, should not be here.")
        annotations = cls._cls_find_annotations()

        cls._DICT_METADATA = {}

        for attr_name, annotation_type in annotations.items():
            if not annotation_type in cls._ANNOTATION_TYPE_MAPPER:
                raise SmartFPSInitError(
                    "Cannot SmartAttributesFPSAutoMetadata with "
                    f"attribute {attr_name} annotation type {annotation_type}."
                )

            cls._DICT_METADATA[attr_name] = (
                f"{attr_name}:{annotation_type}",
                cls._ANNOTATION_TYPE_MAPPER[annotation_type],
                FpsFlag.DEFAULT_INPUT,
            )

    @classmethod
    def _cls_annotation_technical_keys_to_remove(cls: type[_T_Subclass]) -> list[str]:
        return ["_ANNOTATION_TYPE_MAPPER", "_cls_metadata_is_init"]
