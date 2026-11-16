"""
shm.py

Author: V. Deo
Date: April 13, 2020

Offers a python binding to MILK streams.

Is based on the ImageStreamIOWrap (ISIOW) C wrapper to python
which must be compiled and accessible in the PYTHONPATH
(default /usr/local/python is included automatically in this file)

    Your code
        ^
        |
isio_shmlib.py SHM class     (python) <- You are here
        ^
        |
ImageStreamIOWrap.cpp     (nanobind)
        ^
        |
ImagestreamIO.c           (C)
        ^
        |
Shared memory


Credit for the ideas, templates, docstrings, and xaosim.shmlib:
    F. Martinache (www.github.com/fmartinache/xaosim)

Credit for ImageStreamIO C library: O. Guyon

Credit for ImageStreamIOWrap pybind interface: A. Sevin
"""

from __future__ import annotations

import typing as typ
import numpy as np

if typ.TYPE_CHECKING:
    IMAGESTREAMIO_HAVE_CUDA: int
    KWType = str | int | float
    KWDict = dict[str, KWType]
    KWCommentDict = dict[str, tuple[KWType, str]]
    KWOptCommentDict = dict[str, KWType | tuple[KWType] | tuple[KWType, str]]
    # Mapping is essentially a read-only dict?
    KWOptCommentMapping = typ.Mapping[str, KWType | tuple[KWType] | tuple[KWType, str]]

    from types import TracebackType

    ExcTpl = typ.TypeVar("ExcTpl", bound=BaseException)

    import numpy.typing as npt
    import cupy as cp

    xp_ndarray = typ.TypeVar("xp_ndarray", np.ndarray, cp.ndarray)

from . import glib_loader_fix
from pyMilk.ImageStreamIOWrap import Image, Image_kw, Image_md, IMAGESTREAMIO_HAVE_CUDA

# FIXME all np calls... in case the data is a cp array ! Review img_shapes as well.

import datetime

import time

from ..util import img_shapes
from .. import errors

import os


def MILK_SHM_DIR() -> str:
    """
    When testing, we may have to spoof the environment AFTER importing this module.
    It's important to regenerate MILK_SHM_DIR dynamically for this reason.
    """
    return os.environ["MILK_SHM_DIR"]


class SHM:
    """
    Main interfacing class
    """

    #############################################################
    # CORE COMPATIBILITY WITH xaosim.shmlib
    #############################################################

    def __init__(
        self,
        fname: str,
        data: None | xp_ndarray | tuple[tuple[int, ...], npt.DTypeLike] = None,
        nbkw: int = 50,
        shared: bool | typ.Literal[0, 1] = True,
        location: int = -1,
        verbose: bool = False,
        packed=False,
        autoSqueeze: bool = True,
        symcode: int = 4,
        triDim: int = img_shapes.Which3DState.LAST2LAST,
    ) -> None:
        """
        Constructor for a SHM (shared memory) object.

        Parameters:
        ----------
        fname: name of the shm file
                 the resulting name will be $MILK_SHM_DIR/<fname>.im.shm
        data: a numpy array (1, 2 or 3D of data)
                alternatively, a tuple ((int, ...), dtype) is accepted
                    which provides shape and datatype (string or np type)

            WARNING: if data is None, we're reading
                     the userside shape will be found from the C-side shape, the symcode, and the triDim
                     BUT
                     if data isnt None and we're creating - data.shape should be the python shape -- not the c-shape
                     in that case, the C-shape must be found from data.shape

        nbkw: # of keywords to be appended to the data structure (optional)
        shared: True if the memory is shared among users
        location: -1 for CPU RAM, >= 0 provides the # of the GPU.

        autoSqueeze: Remove singleton dimensions between C-side and user side
                     Otherwise, we take the necessary steps to squeeze / pad the singleton dimensions.
                 Warning: [data not None] assumes [autoSqueeze=False].
                     If you're creating a SHM, it's assumed you know what you're passing.
        symcode: symmetry to apply to the data - see pyMilk.util.symcode
        triDim: what to do with tri-dimensional arrays. See util.symcode.


        NOTE:
            The order of operations on read is
                squeeze -> 3D swap -> symcode
            And on write
                symcode -> 3D swap -> unsqueeze

        Depending on whether the file already exists, and/or some new
        data is provided, the file will be created or overwritten.


        CHANGES from xaosim:
            arguments "packed" and "verbose" are unused.
            (this is handled as a compilation flag in ImageStreamIO)

            arguments shared and location added.
        """

        self.IMAGE = Image()
        self.FNAME = check_SHM_name(fname)
        self.FILEPATH = MILK_SHM_DIR() + "/" + self.FNAME + ".im.shm"

        self.semID: int | None = None

        self.symcode = (
            symcode  # Handle image symetries; 0-7, see pyMilk.util.img_shapes
        )
        self.triDimState = triDim

        self.location: int  # assigned in _finalize_init
        self.shared: bool

        if data is None:
            self._finalize_init_openmode(autoSqueeze)
        else:
            self._finalize_init_creationmode(data, location, shared, nbkw)

    def _finalize_init_openmode(self, autoSqueeze: bool):

        if not self._open_from_isio():
            raise FileNotFoundError(f"Requested SHM {self.FNAME} does not exist")
        # _checkExists already performed the self.IMAGE.open()

        self._checkGrabSemaphore()
        data_arr: np.ndarray = self.IMAGE.copy()  # Data is C-side shaped

        self._init_internals_read(data_arr, autoSqueeze)

        self.shared = True  # can't be other with OpenIm...
        self.location = self.IMAGE.md.location

    def _finalize_init_creationmode(
        self,
        data: xp_ndarray | tuple[tuple[int, ...], npt.DTypeLike],
        location: int,
        shared: bool | typ.Literal[0, 1],
        nbkw: int,
    ):
        # Check if numpy or cupy array but without importing cupy ;) Adding mro() to work on subclasses of ndarray.
        if "py.ndarray" in repr(type(data).mro()):
            data_arr = data
        else:
            # data is (Shape, type) tuple
            # making np array (shape as python-side)
            data_arr = np.zeros(data[0], dtype=data[1])

        data_c = self._init_internals_creation(data_arr)  # type: ignore

        if not self._open_from_isio():
            print(f"{self.FNAME}.im.shm will be created")
        else:
            print(f"{self.FNAME}.im.shm will be overwritten")
            # _checkExist opened the image, we can destroy.
            # But we don't need to destroy
            # And that creates a brief lapse where the file, old nor new, doesn't exits
            # So better just close, not destroy, then overwrite.
            self.IMAGE.close()

        self.location = location
        self.shared = bool(shared)

        self.IMAGE.create(
            self.FNAME, data_c, location=location, shared=shared, NBkw=nbkw
        )

    def _init_internals_read(self, data: np.ndarray, autoSqueeze: bool) -> None:
        """
        Aux function for initializing
        data is C-side shape

        Must assign
            self.nptype
            self.nDim
            self.shape_c
            self.shape
        """

        self.nptype: npt.DTypeLike = data.dtype
        self.shape_c: tuple[int, ...] = data.shape

        # Autosqueeze
        if autoSqueeze:
            self.shape: tuple[int, ...] = np.squeeze(data).shape
        else:
            self.shape = self.shape_c

        self.readSlice, self.writeSlice = img_shapes.gen_squeeze_unsqueeze_slices(
            self.shape_c, autoSqueeze
        )

        # Quick ref
        self.nDim = len(self.shape)
        self.nDim_c = len(self.shape_c)

        # 3D ordering
        if self.nDim == 2:
            self.shape = img_shapes.image_decode(
                data[self.readSlice], self.symcode
            ).shape
        elif self.nDim == 3:
            self.shape = img_shapes.full_cube_decode(
                data[self.readSlice], self.symcode, self.triDimState
            ).shape

    def _attempt_autorelink_if_needed(self) -> None:
        if not self.shared:
            return

        if self.IMAGE.md.inode == os.stat(self.FILEPATH).st_ino:
            return

        self.IMAGE.close()
        if not self._open_from_isio():  # Perform open
            # unreachable? Since we'd have raise a FileNotFoundError upon os.stat
            raise errors.AutoRelinkError(f"pyMilk @ SHM {self.FNAME}: checkExist")

        self._checkGrabSemaphore()

        if self.shape_c != tuple(self.IMAGE.md.size):
            raise errors.AutoRelinkSizeError(
                f"pyMilk @ SHM {self.FNAME}: Error during autorelink --"
                f" size mismatch {self.shape_c} vs. {tuple(self.IMAGE.md.size)}"
            )

        new_dtype = self.IMAGE.view().dtype
        if new_dtype != self.nptype:
            raise errors.AutoRelinkTypeError(
                f"pyMilk @ SHM {self.FNAME}: Error during autorelink --"
                f" type mismatch {new_dtype} vs. {self.nptype}"
            )

    def _init_internals_creation(self, data: np.ndarray) -> np.ndarray:
        """
        Aux function for initializing
        data is python-side shape

        Must assign
            self.nptype
            self.nDim
            self.shape_c
            self.shape
        """

        data = _ensure_native_byteorder(data)

        self.nptype = data.dtype
        self.shape = data.shape
        self.nDim = len(self.shape)
        self.nDim_c = self.nDim  # No autosqueeze in creation mode

        # Autosqueeze
        if 1 in self.shape:
            print(
                "Warning: ignoring autoSqueeze parameter when creating SHM."
                "Remove the singleton dimensions yourself."
            )

        self.readSlice, self.writeSlice = img_shapes.gen_squeeze_unsqueeze_slices(
            self.shape, False
        )  # Trivial slices - no squeezing anyway

        if self.nDim <= 1:
            data_c = data  # Starting point
        elif self.nDim == 2:
            data_c = img_shapes.image_encode(data, self.symcode)
        elif self.nDim == 3:
            data_c = img_shapes.full_cube_encode(data, self.symcode, self.triDimState)
        else:
            raise ValueError("number of dimensions 1 <= d <= 3")

        self.shape_c = data_c.shape
        return data_c

    def __enter__(self) -> SHM:
        """
        Use this class as a context manager.
        __enter__ does nothing
        __exit__ calls close()

        This allows a variation of the pattern of (repeatedly) calling:
        data = SHM('x').get_data() # BAD
        using
        with SHM('x') as _shm:
            data = _shm.get_data() # Good
        """
        return self

    def __exit__(
        self,
        type_: type[ExcTpl] | None,
        value: ExcTpl | None,
        traceback: TracebackType | None,
    ) -> bool | None:
        self.close()

    def close(self) -> None:
        """
        Clean close of a SHM data structure link
        """
        self.IMAGE.close()

    def destroy(self) -> None:
        # Do not destroy if the inode is not owned anymore.
        if self.IMAGE.used and (
            self.shared is False or self.IMAGE.md.inode == os.stat(self.FILEPATH).st_ino
        ):
            self.IMAGE.destroy()

    """
    Metadata as property
    """

    @property
    def md(self) -> Image_md:
        return self.IMAGE.md

    """
    __getitem__, __setitem__ used for keywords
    __setitem__ allows keyword creation, but with no comment.
    """

    def __getitem__(self, key: str) -> KWType:
        return self.get_keyword(key)

    def __setitem__(self, key: str, value: KWType) -> None:
        self.set_keywords({key: value})

    def update_keyword(
        self, name: str, value: KWType, comment: str | None = None
    ) -> None:
        """
        Mind the signature change from xaosim: ii (kw index) is not used.
        """

        kws = self.IMAGE.get_kws_list()
        kws_names = [kw.name for kw in kws]
        if name not in kws_names:
            raise ValueError("Updating a keyword that does not exist yet.")
        idx = kws_names.index(name)
        if comment is None:  # Keeping prev comment, updating value only
            kws[idx] = Image_kw(name, value, kws[idx].comment)
        else:
            kws[idx] = Image_kw(name, value, comment)
        self.IMAGE.set_kws_list(kws)

    def _kw_opt_comment_unpacker(
        self, val: KWType | tuple[KWType] | tuple[KWType, str]
    ) -> tuple[KWType, str]:
        if not isinstance(val, tuple):
            v, c = val, ""
        elif len(val) == 1:
            v, c = val[0], ""
        else:
            v, c = val
        return v, c

    def set_keywords(self, kw_dict: KWOptCommentMapping) -> None:
        """
        Sets a keyword dictionnary into the SHM
        This is a tentatively non-destructive write

        kw_dict: {key: value}, {key: (value,)} and {key: (value, comment)} are all accepted
        """
        kws = self.IMAGE.get_kws_list()
        kws_names = [kw.name for kw in kws]

        for name in kw_dict:
            if name in kws_names:
                idx = kws_names.index(name)  # Current location
            else:
                idx = -1  # Append

            v, c = self._kw_opt_comment_unpacker(kw_dict[name])
            if idx >= 0:
                kws[idx] = Image_kw(name, v, c)
            else:
                kws.append(Image_kw(name, v, c))

        self.IMAGE.set_kws_list(kws)

    def reset_keywords(self, kw_dict: KWOptCommentMapping) -> None:
        """
        Sets a keyword dictionnary into the SHM
        This is a complete write - it erases pre-existing keywords

        kw_dict: {key: value}, {key: (value,)} and {key: (value, comment)} are all accepted
        """
        kws = {}
        for name in kw_dict:
            kws[name] = Image_kw(name, *self._kw_opt_comment_unpacker(kw_dict[name]))

        self.IMAGE.set_kws(kws)

    @typ.overload
    def get_keyword(
        self, key: str, comment: typ.Literal[False] = False, n_tries: int = 4
    ) -> KWType: ...

    @typ.overload
    def get_keyword(
        self, key: str, comment: typ.Literal[True], n_tries: int = 4
    ) -> tuple[KWType, str]: ...

    def get_keyword(
        self, key: str, comment: bool = False, n_tries: int = 4
    ) -> KWType | tuple[KWType, str]:
        return self.get_keywords(comment, n_tries)[key]

    @typ.overload
    def get_keywords(
        self, comments: typ.Literal[False] = False, n_tries: int = 4
    ) -> KWDict: ...

    @typ.overload
    def get_keywords(
        self, comments: typ.Literal[True], n_tries: int = 4
    ) -> KWCommentDict: ...

    def get_keywords(
        self, comments: bool = False, n_tries: int = 4
    ) -> KWDict | KWCommentDict:
        """
        Return the keyword dictionary from the SHM

        Will return {name: value} if comments is False
        Will return {name: (value, comments)} if comments is True
        """

        # We'll give ourselves several tries - for race conditions with camera servers
        for i in range(n_tries):
            try:
                return self._get_keywords_nofail(comments=comments)
            except:
                time.sleep(0.002)
        # If we havent's succeeded: succeed or fail, last chance
        return self._get_keywords_nofail(comments=comments)

    def _get_keywords_nofail(self, comments=False) -> KWDict | KWCommentDict:
        self._attempt_autorelink_if_needed()
        kws = self.IMAGE.get_kws()
        kws_ret = {}
        for name in kws:
            if comments:
                kws_ret[name] = (kws[name].value, kws[name].comment)
            else:
                kws_ret[name] = kws[name].value

        return kws_ret

    def print_meta_data(self) -> None:
        print(self.IMAGE.md)

    def get_counter(self) -> int:
        return self.IMAGE.md.cnt0

    def non_block_wait_semaphore(self, sleep_time=0.1) -> int:
        """
        This will spin on SHM semaphore even in case of re-creation to a new inode
        It will raise AutorelinkErrors if the size or dtype has changed.

        To wait without erroring until a SHM is re-alive or recreated for sure,
        see non_block_wait_until_recreation
        """
        self._attempt_autorelink_if_needed()  # can raise!
        self.IMAGE.semflush(self.semID)
        ret = -1
        while ret < 0:
            self._attempt_autorelink_if_needed()  # can raise!
            # ret is -1 is semaphore is alive and not posted
            self.IMAGE.getsemwaitindex(self.semID)
            ret = self.IMAGE.semtimedwait(self.semID, sleep_time)

        # ret will be 1 (sem destroyed) or 0 (sem posted)
        return ret

    def non_block_wait_until_recreation(self, sleep_time=0.1) -> SHM:
        """
        Except the shared memory to come back to life at about any cost:
        - Semaphore becomes posted again
        - New shared memory with the same name appears and posts semaphore

        Returns a pointer to whatever shared memory is alive under self.FNAME.
        """
        try:
            self.non_block_wait_semaphore(sleep_time=sleep_time)
            return self
        except errors.AutoRelinkError:
            return SHM(self.FNAME)

    def check_sem_trywait(self) -> bool:
        """
        Performs a trywait on the currently held semaphore
        Returns True if the semaphore was posted (and therefore acquired and decremented)
        Returns False if the semaphore was 0 and the acquisition failed.
        """
        self._checkGrabSemaphore()
        return self.IMAGE.semtrywait(self.semID) == 0

    def check_sem_timedwait(self, timeout: float) -> bool:
        self._checkGrabSemaphore()
        return self.IMAGE.semtimedwait(self.semID, timeout) == 0

    def check_sem_value(self) -> int:
        """
        Check the semaphore value of the currently held read semaphore.
        Returns the value, or -1 if error.
        """
        self._checkGrabSemaphore()
        return self.IMAGE.semvalue(self.semID)

    @typ.overload
    def get_data(
        self,
        check: bool = False,
        timeout: float | None = 5.0,
        copy: bool = True,
        checkSemAndFlush: bool = True,
        autorelink_if_need: bool = True,
        return_none_on_timeout: typ.Literal[False] = False,
    ) -> xp_ndarray: ...

    @typ.overload
    def get_data(
        self,
        check: bool = False,
        timeout: float | None = 5.0,
        copy: bool = True,
        checkSemAndFlush: bool = True,
        autorelink_if_need: bool = True,
        return_none_on_timeout: typ.Literal[True] = True,
    ) -> xp_ndarray | None: ...

    def get_data(
        self,
        check: bool = False,
        timeout: float | None = 5.0,
        copy: bool = True,
        checkSemAndFlush: bool = True,
        autorelink_if_need: bool = True,
        return_none_on_timeout: bool = False,
    ) -> xp_ndarray | None:
        """
        Reads and returns the data part of the SHM file
        Parameters:
        ----------
        - check: boolean (integer supported); if not False, waits image update
        - copy: boolean, if False returns a np.array pointing to the shm, not a copy.
        - checkSemAndFlush: flush the semaphore to 0, to guarantee SHM
                            was written after this function was called

        CHANGES from xaosim:
            reform, sleepT not implemented
            - reform: use the symcode, autoSqueeze, and triDim constructor arguments (see pyMilk.util.img_shapes for more info)
            - sleepT not useful as we use semaphore waiting now.
        """
        if autorelink_if_need:
            self._attempt_autorelink_if_needed()

        if check:
            self._checkGrabSemaphore()
            if checkSemAndFlush:
                self.IMAGE.semflush(self.semID)
            if timeout is None or timeout <= 0:
                self.IMAGE.semwait(self.semID)
            else:
                err = self.IMAGE.semtimedwait(self.semID, timeout)
                if err != 0:  # Timeout
                    if return_none_on_timeout:
                        return None
                    else:  # Warn, and proceed to return stale data.
                        print(
                            f"Warning SHM {self.FNAME} - isio_shmlib.SHM.get_data has timed out and returned old data."
                        )

        # FIXME ! image_decode, full_cube_decode don't have the same meaning
        # in case of autoSqueeze collapsing dimensions.
        # Should they apply before or after reducing dimensions??
        arr_sliced = (self.IMAGE.copy() if copy else self.IMAGE.view())[self.readSlice]
        if self.nDim == 2:
            return img_shapes.image_decode(arr_sliced, self.symcode)
        elif self.nDim == 3:
            return img_shapes.full_cube_decode(
                arr_sliced,
                self.symcode,
                self.triDimState,
            )
        else:
            return arr_sliced

    def set_data(
        self, data: xp_ndarray, check_dt: bool = False, autorelink_if_need: bool = False
    ) -> None:
        """
        Upload new data to the SHM file.
        Parameters:
        ----------
        - data: the array to upload to SHM
        - check_dt: boolean (default: false) recasts data
        """
        # SHM is actually a scalar, autosqueezed to 0 dimensions.
        if self.nDim == 0:
            # A scalar array with () shape
            data = np.array(data)  # type: ignore

        if autorelink_if_need:
            self._attempt_autorelink_if_needed()

        data = _ensure_native_byteorder(data)

        if check_dt:
            data = data.astype(self.nptype)  # type: ignore

        if self.nDim == 2:
            data_towrite = img_shapes.image_encode(data, self.symcode)[self.writeSlice]
        elif self.nDim == 3:
            data_towrite = img_shapes.full_cube_encode(
                data, self.symcode, self.triDimState
            )[self.writeSlice]
        else:
            data_towrite = data[self.writeSlice]

        if not data_towrite.flags["F_CONTIGUOUS"]:
            data_towrite = data_towrite.copy("F")
        self.IMAGE.write(data_towrite)

    def repost(self, atime: float | datetime.datetime | None = None) -> None:
        """
        Repost semaphore and writetimes || hang on it actually repost acqtimes as well.
        But does not handle data.
        """
        if atime is None:
            self.IMAGE.update()
        elif isinstance(atime, float) or isinstance(atime, int):
            self.IMAGE.update_atime(atime)
        else:
            self.IMAGE.update_atime(atime.timestamp())

    def save_as_fits(self, fitsname: str, **kwargs) -> int:
        """
        Convenient sometimes, to be able to export the data as a fits file.

        Parameters:
        ----------
        - fitsname: a filename (overwrite=True)
        """
        from pyMilk.interfacing import fits_lib

        fits_lib.multi_write(
            fitsname,
            self.get_data(**kwargs),  # type: ignore
            symcode=self.symcode,
            tri_dim=self.triDimState,
        )
        return 0

    #############################################################
    # EXTENSION proposed by xaosim.scexao_shmlib
    #############################################################

    def get_expt(self) -> float:
        """
        Returns exposure time (sec)
        """
        kws = self.get_keywords()
        if "EXPTIME" in kws:
            retval = kws["EXPTIME"]
        else:
            retval = kws.get("tint", 0.0)

        assert isinstance(retval, float)
        return retval

    def get_fps(self) -> float:
        """
        Returns framerate (Hz)
        """
        kws = self.get_keywords()
        if "FRATE" in kws:
            retval = kws["FRATE"]
        else:
            retval = kws.get("fps", 0.0)

        assert isinstance(retval, float)
        return retval

    def get_ndr(self) -> int:
        """
        Returns NDR status
        """
        kws = self.get_keywords()
        if "DET-NSMP" in kws:
            retval = kws["DET-NSMP"]
        else:
            retval = kws.get("NDR", 1)

        assert isinstance(retval, int)
        return retval

    def get_crop(self) -> tuple[int, int, int, int]:
        """
        Return image crop boundaries
        FIXME: old specs for what the keywords mean are lost. Is this function correct?
        """

        kws = self.get_keywords()
        new_new_key = ["PRD-MIN1", "PRD-RNG1", "PRD-MIN2", "PRD-RNG2"]
        if new_new_key[0] in kws:
            x0x1y0y1 = [int(kws[key]) for key in new_new_key]
            # We switched from (start, end) to (start, range)
            # Also it's gonna be very broken in case we're binning
            x0x1y0y1[1] = x0x1y0y1[0] + x0x1y0y1[1] - 1
            x0x1y0y1[3] = x0x1y0y1[2] + x0x1y0y1[3] - 1
            tup = tuple(x0x1y0y1)
            assert len(tup) == 4
            return tup

        new_key = ["CROP_OR1", "CROP_EN1", "CROP_OR2", "CROP_EN2"]
        if new_key[0] in kws:
            tup = tuple([int(kws[key]) for key in new_key])
            assert len(tup) == 4
            return tup

        old_key = ["x0", "x1", "y0", "y1"]
        if old_key[0] in kws:
            x0x1y0y1 = [int(kws[key]) for key in old_key]
            x0x1y0y1[1] = x0x1y0y1[0] + x0x1y0y1[1] - 1
            x0x1y0y1[3] = x0x1y0y1[2] + x0x1y0y1[3] - 1
            tup = tuple(x0x1y0y1)
            assert len(tup) == 4
            return tup

        return (0, self.shape[0], 0, self.shape[1])

    #############################################################
    # ADDITIONAL FEATURES - NOT SUPPORTED BY xaosim shm STRUCTURE
    #############################################################

    # A handy property to access the self.IMAGE keywords from top-level
    keywords = property(lambda x: x.IMAGE.kw, None)

    def _open_from_isio(self) -> bool:
        """
        Check if the requeste SHM file is already allocated
        This is called by the constructor when built for writing.

        Returns True if file exists.
        """
        retcode = self.IMAGE.open(self.FNAME)
        return retcode == 0

    def _checkGrabSemaphore(self) -> None:
        """
        Called by constructor in read mode
        Called by synchronous receive functions

        Sets, and returns, semaphore ID
        """
        if self.semID is None:
            self.semID = self.IMAGE.getsemwaitindex(0)

    @typ.overload
    def multi_recv_data(
        self,
        n: int,
        output_as_cube: typ.Literal[False] = False,
        monitorCount: bool = False,
        timeout: float = 5.0,
    ) -> list[np.ndarray]: ...

    @typ.overload
    def multi_recv_data(
        self,
        n: int,
        output_as_cube: typ.Literal[True],
        monitorCount: bool = False,
        timeout: float = 5.0,
    ) -> np.ndarray: ...

    def multi_recv_data(
        self,
        n: int,
        output_as_cube: bool = False,
        monitorCount: bool = False,
        timeout: float = 5.0,
    ) -> list[np.ndarray] | np.ndarray:
        """
        Synchronous read of n successive images in a stream.

        Parameters:
        ----------
        - n: number of frames to read and return
        - outputFormat: flag to indicate what is desired
                        0 for List[np.ndarray]
                        1 for aggregated np.ndarray
        - monitorSem: Monitor and report the counter states when ready to receive - WIP
        """
        # Prep output

        output: list[np.ndarray] | np.ndarray
        if output_as_cube is False:
            output = [None for _ in range(n)]  # type: ignore
        else:
            output = np.zeros((n, *self.shape), dtype=self.nptype)

        if monitorCount:
            countValues = np.zeros((2, n), dtype=np.uint64)
        else:
            countValues = None

        # Check and flush the semaphore
        self._checkGrabSemaphore()
        self.IMAGE.semflush(self.semID)

        for k in range(n):
            if countValues is not None:
                countValues[0, k] = self.IMAGE.md.cnt0

            # if output[k] is a list slot, we must copy or we're gonna get a pointer
            # if output[k] is a ndarray slice, we avoid a doublecopy with copy=False
            output[k] = self.get_data(
                check=True,
                copy=(self.location >= 0) or (not output_as_cube),
                checkSemAndFlush=False,
                timeout=timeout,
            )
            if countValues is not None:
                countValues[1, k] = self.IMAGE.md.cnt0

        if countValues is not None:

            x = countValues[:, 1:] - countValues[:, :-1]
            p1, p2 = np.sum(x < 1, axis=1), np.sum(x > 1, axis=1)
            y = countValues[1] - countValues[0]
            p3, p4 = np.sum(y < 1), np.sum(y > 1)
            print(f"Irregularities:")
            print(f"{p1[0]} < 1, {p1[1]} > 1 deltas in pre.")
            print(f"{p2[0]} < 1, {p2[1]} > 1 deltas in post.")
            print(f"{p3} < 1, {p4} > 1 pre/post diff.")

        return output


def _ensure_native_byteorder(data: xp_ndarray) -> xp_ndarray:
    if not data.dtype.isnative:
        data = data.astype(data.dtype.newbyteorder("="), copy=True)  # type: ignore
    return data


def check_SHM_name(fname: str) -> str:
    """
    Check if the name provided in the constructor is
    - a string in ISIO style
    - a full path + filename + in.shm in xaosim style

    Parameters:
    - fname: the string to check

    Returns:
    The cleaned, ISIO formatted file name (no extension)

    Raises:
    Errors if it's a path-like name that
    1 / does not fit in $MILK_SHM_DIR
    2 / does not end with .im.shm
    """
    if not "/" in fname:  # filename only
        if fname.endswith(".im.shm"):
            return fname[:-7]
        else:
            return fname

    # It has slashes
    pre, end = os.path.split(fname)

    # We use it a lot in here, no need to call the dynamic env func every time.
    SHM_DIR = MILK_SHM_DIR()

    # Third condition is for future use if we allow subdirs
    if not pre.startswith(SHM_DIR) or (
        len(pre) > len(SHM_DIR) and pre[len(SHM_DIR)] != "/"
    ):
        raise ValueError(f"Only files in MILK_SHR_DIR ({SHM_DIR}) are supported")

    # Do we need to make folders ? We can't... maybe some day
    if "/" in pre[len(SHM_DIR) :]:
        raise ValueError(
            f"Only files at the root of MILK_SHR_DIR ({SHM_DIR}) are supported"
        )

    # OK we're good, just filter the extension
    if end.endswith(".im.shm"):
        return end[:-7]
    else:
        return end
