from __future__ import annotations

import numpy as np
import pytest
from pyMilk.util.img_shapes import (
    image_encode,
    image_decode,
    cube_front_image_encode,
    cube_front_image_decode,
    cube_back_image_encode,
    cube_back_image_decode,
    gen_squeeze_unsqueeze_slices,
    cube_roll_forw,
    cube_roll_back,
    full_cube_encode,
    full_cube_decode,
    Which3DState,
    Which3DRoll,
)


def make_random_array(shape, dtype=np.float32, seed: int | None = None):
    """Generate random array of given shape and dtype with optional seed"""
    rng = np.random.default_rng(seed)

    if dtype in [np.complex64, np.complex128]:  # Complex types
        return (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)).astype(
            dtype
        )
    elif dtype in [np.float32, np.float64]:
        return rng.standard_normal(shape).astype(dtype)
    else:
        return rng.integers(0, 100, size=shape, dtype=dtype)


# Test shapes covering 1D, 2D, and 3D with various axis sizes
TEST_SHAPES_2D = [
    (1, 1),  # Minimal 2D
    (1, 5),  # 2D with one singleton
    (5, 1),  # 2D with other singleton
    (3, 4),  # Regular 2D
]

TEST_SHAPES_3D = [
    (1, 1, 1),  # Minimal 3D
    (1, 3, 4),  # 3D with first singleton
    (3, 1, 4),  # 3D with middle singleton
    (3, 4, 1),  # 3D with last singleton
    (2, 3, 4),  # Regular 3D
]

TEST_DTYPES = [
    np.float32,
    np.float64,  # Floating point
    np.int32,
    np.uint16,  # Integer
    np.complex64,
    np.complex128,  # Complex
]


@pytest.mark.parametrize("shape", TEST_SHAPES_2D)
@pytest.mark.parametrize("dtype", TEST_DTYPES)
def test_image_encode_decode(shape, dtype):
    test_data = make_random_array(shape, dtype)
    for s in range(8):
        encoded = image_encode(test_data, s)
        decoded = image_decode(encoded, s)
        np.testing.assert_array_equal(decoded, test_data)


def test_image_encode_invalid_symcode():
    test_data_2d = make_random_array((2, 3))
    test_data_3d = make_random_array((2, 3, 4))

    for s in [-1, 8]:
        with pytest.raises(ValueError):
            image_encode(test_data_2d, s)
        with pytest.raises(ValueError):
            image_decode(test_data_2d, s)
        with pytest.raises(ValueError):
            cube_front_image_encode(test_data_3d, s)
        with pytest.raises(ValueError):
            cube_front_image_decode(test_data_3d, s)
        with pytest.raises(ValueError):
            cube_back_image_encode(test_data_3d, s)
        with pytest.raises(ValueError):
            cube_back_image_decode(test_data_3d, s)


@pytest.mark.parametrize("shape", TEST_SHAPES_3D)
@pytest.mark.parametrize("dtype", TEST_DTYPES)
def test_cube_front_encode_decode(shape, dtype):
    test_data = make_random_array(shape, dtype)
    for s in range(8):
        encoded = cube_front_image_encode(test_data, s)
        decoded = cube_front_image_decode(encoded, s)
        np.testing.assert_array_equal(decoded, test_data)


@pytest.mark.parametrize("shape", TEST_SHAPES_3D)
@pytest.mark.parametrize("dtype", TEST_DTYPES)
def test_cube_back_encode_decode(shape, dtype):
    test_data = make_random_array(shape, dtype)
    for s in range(8):
        encoded = cube_back_image_encode(test_data, s)
        decoded = cube_back_image_decode(encoded, s)
        np.testing.assert_array_equal(decoded, test_data)


@pytest.mark.parametrize(
    "shape",
    [
        (1,),
        (5,),  # 1D
        (1, 1),
        (1, 5),
        (5, 5),  # 2D
        (1, 1, 1),
        (1, 5, 5),
        (5, 5, 5),  # 3D
    ],
)
@pytest.mark.parametrize("auto_squeeze", [True, False])
def test_squeeze_unsqueeze_slices(shape, auto_squeeze):
    read_slice, write_slice = gen_squeeze_unsqueeze_slices(shape, auto_squeeze)

    # Verify slices are valid
    test_data = make_random_array(shape)
    squeezed = test_data[read_slice]
    if auto_squeeze:
        assert squeezed.shape == tuple(s for s in shape if s != 1)
    else:
        assert squeezed.shape == shape

    # Verify reconstruction
    reconstructed = np.zeros_like(test_data)
    reconstructed[write_slice] = squeezed
    np.testing.assert_array_equal(reconstructed, test_data)


@pytest.mark.parametrize("shape", TEST_SHAPES_3D)
@pytest.mark.parametrize("dtype", TEST_DTYPES)
def test_cube_roll_operations(shape, dtype):
    test_data = make_random_array(shape, dtype)
    rolled_forward = cube_roll_forw(test_data)
    # Shape should be rotated: (a,b,c) -> (c,a,b)
    expected_shape = (shape[2], shape[0], shape[1])
    assert rolled_forward.shape == expected_shape

    rolled_back = cube_roll_back(rolled_forward)
    np.testing.assert_array_equal(rolled_back, test_data)


@pytest.mark.parametrize("shape", TEST_SHAPES_3D)
@pytest.mark.parametrize("dtype", TEST_DTYPES)
def test_full_cube_encode_decode(shape, dtype):
    test_data = make_random_array(shape, dtype)
    for s in range(8):
        for tri_dim in [
            Which3DState.LAST2LAST,
            Which3DState.LAST2FRONT,
            Which3DState.FRONT2FRONT,
            Which3DState.FRONT2LAST,
        ]:
            encoded = full_cube_encode(test_data, s, tri_dim)
            decoded = full_cube_decode(encoded, s, tri_dim)
            np.testing.assert_array_equal(decoded, test_data)


@pytest.mark.parametrize("shape", TEST_SHAPES_2D)
@pytest.mark.parametrize("dtype", TEST_DTYPES)
def test_specific_transformations(shape, dtype):
    test_data = make_random_array(shape, dtype)

    # Symcode 1: Flip vertically
    encoded = image_encode(test_data, 1)
    expected = np.flip(test_data, axis=0)
    np.testing.assert_array_equal(encoded, expected)

    # Symcode 2: Flip horizontally
    encoded = image_encode(test_data, 2)
    expected = np.flip(test_data, axis=1)
    np.testing.assert_array_equal(encoded, expected)

    # Symcode 4: Transpose
    encoded = image_encode(test_data, 4)
    expected = test_data.T
    np.testing.assert_array_equal(encoded, expected)


def test_cube_front_and_back_symmetry():
    """Test that cube_front and cube_back operations maintain symmetry"""
    test_data = make_random_array((3, 4, 5))

    # Test all symcodes for front
    for s in range(8):
        encoded = cube_front_image_encode(test_data, s)
        decoded = cube_front_image_decode(encoded, s)
        np.testing.assert_array_equal(decoded, test_data)

    # Test all symcodes for back
    for s in range(8):
        encoded = cube_back_image_encode(test_data, s)
        decoded = cube_back_image_decode(encoded, s)
        np.testing.assert_array_equal(decoded, test_data)
