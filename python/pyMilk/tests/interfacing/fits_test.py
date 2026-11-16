"""
Comprehensive tests for pyMilk.interfacing.fits module

Tests cover:
- multi_read: reading single HDU (2D and 3D), and multiple HDUs
- multi_write: writing single array (2D and 3D), and multiple arrays
- read_fits_load_shm: loading FITS into shared memory
- shm_to_fits: saving shared memory to FITS file
"""

import pytest
import tempfile
import os
import numpy as np

from pyMilk.interfacing import fits
from pyMilk.interfacing.isio_shmlib import SHM
from pyMilk.util import img_shapes


@pytest.fixture
def fixt_tmpdir_fits():
    """Fixture providing a temporary directory for FITS file tests"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


class TestMultiRead:
    """Test multi_read function"""

    def test_multi_read_single_2d_image_no_symcode(self, fixt_tmpdir_fits):
        """Test reading a single 2D image with no symcode"""
        file_path = os.path.join(fixt_tmpdir_fits, "test_2d.fits")

        # Create test data
        test_data = np.random.rand(10, 20)
        fits.multi_write(file_path, test_data, symcode=0)

        # Read it back
        result = fits.multi_read(file_path, symcode=0)

        assert isinstance(result, np.ndarray)
        assert result.shape == (10, 20)
        np.testing.assert_array_almost_equal(result, test_data)

    @pytest.mark.parametrize("symcode", range(8))
    def test_multi_read_single_2d_image_with_symcode(self, symcode, fixt_tmpdir_fits):
        """Test reading a single 2D image with various symcodes"""
        test_data = np.arange(20).reshape(4, 5)

        file_path = os.path.join(fixt_tmpdir_fits, f"test_2d_symcode_{symcode}.fits")
        fits.multi_write(file_path, test_data, symcode=symcode)
        result = fits.multi_read(file_path, symcode=symcode)

        assert isinstance(result, np.ndarray)
        assert result.shape == test_data.shape
        np.testing.assert_array_almost_equal(result, test_data)

    def test_multi_read_single_3d_cube_last2last(self, fixt_tmpdir_fits):
        """Test reading a single 3D cube with LAST2LAST tri_dim"""
        file_path = os.path.join(fixt_tmpdir_fits, "test_3d.fits")

        # Create 3D test data
        test_data = np.random.rand(5, 10, 20)
        fits.multi_write(
            file_path, test_data, symcode=0, tri_dim=img_shapes.Which3DState.LAST2LAST
        )

        result = fits.multi_read(
            file_path, symcode=0, tri_dim=img_shapes.Which3DState.LAST2LAST
        )

        assert isinstance(result, np.ndarray)
        assert result.ndim == 3
        np.testing.assert_array_almost_equal(result, test_data)

    def test_multi_read_single_3d_cube_FRONT2FRONT(self, fixt_tmpdir_fits):
        """Test reading a single 3D cube with FRONT2FRONT tri_dim"""
        file_path = os.path.join(fixt_tmpdir_fits, "test_3d_FRONT2FRONT.fits")

        test_data = np.random.rand(5, 10, 20)
        fits.multi_write(
            file_path, test_data, symcode=0, tri_dim=img_shapes.Which3DState.FRONT2FRONT
        )

        result = fits.multi_read(
            file_path, symcode=0, tri_dim=img_shapes.Which3DState.FRONT2FRONT
        )

        assert isinstance(result, np.ndarray)
        assert result.ndim == 3
        np.testing.assert_array_almost_equal(result, test_data)

    def test_multi_read_multiple_hdus(self, fixt_tmpdir_fits):
        """Test reading multiple HDUs returns tuple"""
        file_path = os.path.join(fixt_tmpdir_fits, "test_multi_hdu.fits")

        # Create multiple arrays
        data_list = [np.random.rand(5, 10), np.random.rand(8, 12), np.random.rand(6, 7)]

        fits.multi_write(file_path, data_list)

        result = fits.multi_read(file_path)

        assert isinstance(result, tuple)
        assert len(result) == 3
        for i, data in enumerate(data_list):
            np.testing.assert_array_almost_equal(result[i], data)

    def test_multi_read_multiple_3d_hdus(self, fixt_tmpdir_fits):
        """Test reading multiple 3D HDUs"""
        file_path = os.path.join(fixt_tmpdir_fits, "test_multi_3d_hdu.fits")

        # Create multiple 3D arrays
        data_list = [np.random.rand(3, 5, 10), np.random.rand(4, 6, 8)]

        fits.multi_write(file_path, data_list)

        result = fits.multi_read(file_path)

        assert isinstance(result, tuple)
        assert len(result) == 2
        for i, data in enumerate(data_list):
            np.testing.assert_array_almost_equal(result[i], data)


class TestMultiWrite:
    """Test multi_write function"""

    def test_multi_write_single_2d_array(self, fixt_tmpdir_fits):
        """Test writing a single 2D array"""
        file_path = os.path.join(fixt_tmpdir_fits, "test_write_2d.fits")
        test_data = np.random.rand(10, 20)

        fits.multi_write(file_path, test_data)

        assert os.path.exists(file_path)
        result = fits.multi_read(file_path)
        np.testing.assert_array_almost_equal(result, test_data)

    def test_multi_write_single_3d_array(self, fixt_tmpdir_fits):
        """Test writing a single 3D array"""
        file_path = os.path.join(fixt_tmpdir_fits, "test_write_3d.fits")
        test_data = np.random.rand(5, 10, 20)

        fits.multi_write(file_path, test_data)

        assert os.path.exists(file_path)
        result = fits.multi_read(file_path)
        assert result.ndim == 3

    def test_multi_write_multiple_arrays_as_list(self, fixt_tmpdir_fits):
        """Test writing multiple arrays as list"""
        file_path = os.path.join(fixt_tmpdir_fits, "test_write_multi.fits")

        data_list = [np.random.rand(5, 10), np.random.rand(8, 12), np.random.rand(6, 7)]

        fits.multi_write(file_path, data_list)

        assert os.path.exists(file_path)
        result = fits.multi_read(file_path)
        assert isinstance(result, tuple)
        assert len(result) == 3
        for i, data in enumerate(data_list):
            np.testing.assert_array_almost_equal(result[i], data)

    def test_multi_write_multiple_arrays_as_tuple(self, fixt_tmpdir_fits):
        """Test writing multiple arrays as tuple"""
        file_path = os.path.join(fixt_tmpdir_fits, "test_write_multi_tuple.fits")

        data_tuple = (
            np.random.rand(5, 10),
            np.random.rand(8, 12),
        )

        fits.multi_write(file_path, data_tuple)

        assert os.path.exists(file_path)
        result = fits.multi_read(file_path)
        assert isinstance(result, tuple)
        for i, data in enumerate(data_tuple):
            np.testing.assert_array_almost_equal(result[i], data)

    @pytest.mark.parametrize("symcode", range(8))
    def test_multi_write_with_symcode(self, symcode, fixt_tmpdir_fits):
        """Test writing with various symcodes"""
        test_data = np.arange(20).reshape(4, 5)

        file_path = os.path.join(fixt_tmpdir_fits, f"test_write_symcode_{symcode}.fits")
        fits.multi_write(file_path, test_data, symcode=symcode)
        result = fits.multi_read(file_path, symcode=symcode)

        np.testing.assert_array_almost_equal(result, test_data)

    def test_multi_write_overwrite_existing(self, fixt_tmpdir_fits):
        """Test that overwrite=True allows rewriting files"""
        file_path = os.path.join(fixt_tmpdir_fits, "test_overwrite.fits")

        data1 = np.ones((5, 10))
        data2 = np.zeros((6, 12))

        fits.multi_write(file_path, data1)
        fits.multi_write(file_path, data2)  # Should overwrite

        result = fits.multi_read(file_path)
        np.testing.assert_array_equal(result, data2)


class TestReadFitsLoadShm:
    """Test read_fits_load_shm function"""

    def test_read_fits_load_shm_basic(self, fixt_tmpdir_fits):
        """Test basic loading of FITS into shared memory"""
        file_path = os.path.join(fixt_tmpdir_fits, "test_shm_load.fits")
        test_data = np.random.rand(10, 20)

        fits.multi_write(file_path, test_data, symcode=3)

        shm = fits.read_fits_load_shm(
            file_path, "test_shm_basic", create_shm=True, symcode=3
        )

        assert isinstance(shm, SHM)
        np.testing.assert_array_almost_equal(shm.get_data(), test_data)

        # Cleanup
        shm.close()

    @pytest.mark.parametrize(
        "symcode_in,symcode_out", [(a, b) for a in range(8) for b in range(8)]
    )
    def test_read_fits_load_shm_existing_shm(
        self, fixt_tmpdir_fits, symcode_in, symcode_out
    ):
        """Test loading into existing shared memory"""
        file_path = os.path.join(fixt_tmpdir_fits, "test_shm_load_existing.fits")
        test_data = np.arange(6).reshape(2, 3).astype(np.float32)

        fits.multi_write(file_path, test_data, symcode=symcode_in)  # default symcode 0

        # Create SHM first
        shm = SHM(
            "test_shm_existing", test_data * 0, symcode=symcode_in
        )  # default symcode 4

        # Load into existing
        result_shm = fits.read_fits_load_shm(
            file_path, "test_shm_existing", create_shm=False, symcode=symcode_out
        )

        assert isinstance(result_shm, SHM)
        np.testing.assert_array_almost_equal(shm.get_data(), test_data)

    def test_read_fits_load_shm_3d_cube(self, fixt_tmpdir_fits):
        """Test loading 3D cube into shared memory"""
        file_path = os.path.join(fixt_tmpdir_fits, "test_shm_load_3d.fits")
        test_data = np.random.rand(5, 10, 20)

        fits.multi_write(
            file_path, test_data, tri_dim=img_shapes.Which3DState.LAST2LAST
        )

        shm = fits.read_fits_load_shm(
            file_path,
            "test_shm_3d",
            create_shm=True,
            tri_dim=img_shapes.Which3DState.LAST2LAST,
        )

        assert isinstance(shm, SHM)
        assert shm.get_data().ndim == 3

        # Cleanup
        shm.close()

    def test_read_fits_load_shm_with_symcode(self, fixt_tmpdir_fits):
        """Test loading FITS with symcode into shared memory"""
        file_path = os.path.join(fixt_tmpdir_fits, "test_shm_load_symcode.fits")
        test_data = np.arange(20).reshape(4, 5)
        symcode = 2

        fits.multi_write(file_path, test_data, symcode=symcode)

        shm = fits.read_fits_load_shm(
            file_path, "test_shm_symcode", symcode=symcode, create_shm=True
        )

        assert isinstance(shm, SHM)
        np.testing.assert_array_almost_equal(shm.get_data(), test_data)

        # Cleanup
        shm.close()

    def test_read_fits_load_shm_fails_on_multiple_hdus(self, fixt_tmpdir_fits):
        """Test that multi-HDU FITS raises assertion error"""
        file_path = os.path.join(fixt_tmpdir_fits, "test_multi_hdu.fits")

        data_list = [np.random.rand(5, 10), np.random.rand(8, 12)]

        fits.multi_write(file_path, data_list)

        with pytest.raises(AssertionError):
            fits.read_fits_load_shm(file_path, "test_shm_fail", create_shm=True)


class TestShmToFits:
    """Test shm_to_fits function"""

    def test_shm_to_fits_with_shm_object(self, fixt_tmpdir_fits):
        """Test writing SHM to FITS using SHM object"""
        file_path = os.path.join(fixt_tmpdir_fits, "test_shm_to_fits.fits")
        test_data = np.random.rand(10, 20)

        # Create SHM
        shm = SHM("test_shm_to_fits_obj", test_data, shared=1)

        # Write to FITS
        fits.shm_to_fits(shm, file_path)

        assert os.path.exists(file_path)
        result = fits.multi_read(file_path)

        # Note: shm data may have symcode applied, so we check shape
        assert result.shape == test_data.shape
        np.testing.assert_array_almost_equal(result, test_data)

        # Cleanup
        shm.close()

    def test_shm_to_fits_with_shm_name(self, fixt_tmpdir_fits):
        """Test writing SHM to FITS using SHM name"""
        file_path = os.path.join(fixt_tmpdir_fits, "test_shm_to_fits_name.fits")
        test_data = np.random.rand(10, 20)

        # Create SHM
        shm = SHM("test_shm_to_fits_name", test_data, shared=1)

        # Write to FITS using name
        fits.shm_to_fits("test_shm_to_fits_name", file_path)

        assert os.path.exists(file_path)
        result = fits.multi_read(file_path)
        assert result.shape == test_data.shape
        np.testing.assert_array_almost_equal(result, test_data)

        # Cleanup
        shm.close()

    def test_shm_to_fits_3d_cube(self, fixt_tmpdir_fits):
        """Test writing 3D SHM cube to FITS"""
        file_path = os.path.join(fixt_tmpdir_fits, "test_shm_to_fits_3d.fits")
        test_data = np.random.rand(5, 10, 20)

        # Create 3D SHM
        shm = SHM(
            "test_shm_to_fits_3d",
            test_data,
            shared=1,
            triDim=img_shapes.Which3DState.LAST2LAST,
        )

        # Write to FITS
        fits.shm_to_fits(shm, file_path)

        assert os.path.exists(file_path)
        result = fits.multi_read(file_path)
        assert result.ndim == 3

        # Cleanup
        shm.close()


class TestRoundTripConsistency:
    """Test round-trip consistency across various data types"""

    def test_roundtrip_2d_uint8(self, fixt_tmpdir_fits):
        """Test round-trip with uint8 data"""
        file_path = os.path.join(fixt_tmpdir_fits, "test_roundtrip_uint8.fits")
        test_data = (np.random.rand(10, 20) * 255).astype(np.uint8)

        fits.multi_write(file_path, test_data)
        result = fits.multi_read(file_path)

        np.testing.assert_array_equal(result, test_data)

    def test_roundtrip_2d_float32(self, fixt_tmpdir_fits):
        """Test round-trip with float32 data"""
        file_path = os.path.join(fixt_tmpdir_fits, "test_roundtrip_float32.fits")
        test_data = np.random.rand(10, 20).astype(np.float32)

        fits.multi_write(file_path, test_data)
        result = fits.multi_read(file_path)

        np.testing.assert_array_almost_equal(result, test_data)

    def test_roundtrip_2d_int32(self, fixt_tmpdir_fits):
        """Test round-trip with int32 data"""
        file_path = os.path.join(fixt_tmpdir_fits, "test_roundtrip_int32.fits")
        test_data = (np.random.rand(10, 20) * 1000).astype(np.int32)

        fits.multi_write(file_path, test_data)
        result = fits.multi_read(file_path)

        np.testing.assert_array_equal(result, test_data)

    @pytest.mark.parametrize("shape", [(1, 1), (10, 20), (100, 1), (1, 100), (50, 50)])
    def test_roundtrip_various_shapes(self, fixt_tmpdir_fits, shape):
        """Test round-trip with various array shapes"""

        file_path = os.path.join(fixt_tmpdir_fits, f"test_shape.fits")
        test_data = np.random.rand(*shape)

        fits.multi_write(file_path, test_data)
        result = fits.multi_read(file_path)

        assert result.shape == shape
        np.testing.assert_array_almost_equal(result, test_data)


'''
class TestEdgeCases:
    """Test edge cases and special scenarios"""

    def test_write_single_element_array(self, fixt_tmpdir_fits):
        """Test writing single element array"""
        file_path = os.path.join(fixt_tmpdir_fits, 'test_single_element.fits')
        test_data = np.array([[42.0]])

        fits.multi_write(file_path, test_data)
        result = fits.multi_read(file_path)

        np.testing.assert_array_almost_equal(result, test_data)

    def test_write_large_array(self, fixt_tmpdir_fits):
        """Test writing reasonably large array"""
        file_path = os.path.join(fixt_tmpdir_fits, 'test_large_array.fits')
        test_data = np.random.rand(500, 600)

        fits.multi_write(file_path, test_data)
        result = fits.multi_read(file_path)

        assert result.shape == test_data.shape
        np.testing.assert_array_almost_equal(result, test_data)

    def test_write_zero_array(self, fixt_tmpdir_fits):
        """Test writing array of zeros"""
        file_path = os.path.join(fixt_tmpdir_fits, 'test_zero_array.fits')
        test_data = np.zeros((10, 20))

        fits.multi_write(file_path, test_data)
        result = fits.multi_read(file_path)

        np.testing.assert_array_equal(result, test_data)

    def test_write_negative_values(self, fixt_tmpdir_fits):
        """Test writing arrays with negative values"""
        file_path = os.path.join(fixt_tmpdir_fits, 'test_negative.fits')
        test_data = np.random.randn(
                10, 20)  # Normal distribution includes negatives

        fits.multi_write(file_path, test_data)
        result = fits.multi_read(file_path)

        np.testing.assert_array_almost_equal(result, test_data)

    def test_write_empty_list(self, fixt_tmpdir_fits):
        """Test writing single array preserves single HDU structure"""
        file_path = os.path.join(fixt_tmpdir_fits, 'test_single_as_iterable.fits')
        test_data = np.random.rand(10, 20)

        # Pass as single element in list
        fits.multi_write(file_path, [test_data])
        result = fits.multi_read(file_path)

        # Should return tuple since we wrote multiple HDUs
        assert isinstance(result, np.ndarray)
'''
