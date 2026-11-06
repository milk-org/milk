"""Dataset catalog (§6.2) for calculator expression testing."""

from __future__ import annotations

from dataclasses import dataclass, field

BASE_IMAGES = ("A", "B", "C", "D", "G", "O", "U", "V", "M")
BASE_SCALARS = (
    "v",
    "m",
    "s",
    "lo",
    "hi",
    "cx",
    "cy",
    "flux",
    "H",
    "x",
    "y",
    "n",
    "k",
    "t",
    "alpha",
)


@dataclass(frozen=True)
class DatasetSpec:
    """Descriptor for one calculator test dataset."""

    id: str
    description: str
    shape: tuple[int, ...] = (64, 64)
    images: tuple[str, ...] = BASE_IMAGES
    int_images: tuple[str, ...] = ()
    complex_images: tuple[str, ...] = ()
    double_images: tuple[str, ...] = ()
    mask_images: tuple[str, ...] = ("M",)
    cube_images: tuple[str, ...] = ()
    broadcast_images: tuple[str, ...] = ()
    scalars: tuple[str, ...] = BASE_SCALARS
    setup: tuple[str, ...] = field(default_factory=tuple)


DATASETS: dict[str, DatasetSpec] = {
    "default": DatasetSpec(
        id="default",
        description="Standard 64x64 float32 images A B C D G O U V M and scalar variables",
        shape=(64, 64),
    ),
    "random": DatasetSpec(
        id="random",
        description="Random floats in [0, 1) across 64x64 images A B C D G O and random mask M",
        shape=(64, 64),
    ),
    "offset_0": DatasetSpec(
        id="offset_0",
        description="Offset frame: offset 0 + sin(0.37*x) + cos(0.21*y) (256x256 float32)",
        shape=(256, 256),
    ),
    "offset_1e3": DatasetSpec(
        id="offset_1e3",
        description="Offset frame: offset 1e3 + sin(0.37*x) + cos(0.21*y) (256x256 float32)",
        shape=(256, 256),
    ),
    "offset_1e4": DatasetSpec(
        id="offset_1e4",
        description="Offset frame: offset 1e4 + sin(0.37*x) + cos(0.21*y) (256x256 float32)",
        shape=(256, 256),
    ),
    "offset_3e4": DatasetSpec(
        id="offset_3e4",
        description="Offset frame: offset 3e4 + sin(0.37*x) + cos(0.21*y) (256x256 float32)",
        shape=(256, 256),
    ),
    "nan_special": DatasetSpec(
        id="nan_special",
        description="NaN, +-inf, +-0 and denormals scattered at random pixels (64x64 float32)",
        shape=(64, 64),
    ),
    "int_limits": DatasetSpec(
        id="int_limits",
        description="Every integer type (u8 i8 u16 i16 u32 i32 u64 i64) at its limits (16x16)",
        shape=(16, 16),
        int_images=(
            "im_u8",
            "im_i8",
            "im_u16",
            "im_i16",
            "im_u32",
            "im_i32",
            "im_u64",
            "im_i64",
        ),
        scalars=BASE_SCALARS + ("imin64", "imax64"),
    ),
    "complex": DatasetSpec(
        id="complex",
        description="Complex float (zA zB zC), complex double (zAd zBd zCd), and double (Ad Bd Cd)",
        shape=(32, 32),
        complex_images=("zA", "zB", "zC", "zAd", "zBd", "zCd"),
        double_images=("Ad", "Bd", "Cd"),
    ),
    "masks": DatasetSpec(
        id="masks",
        description="Masks all zero (M0), all one (M1), random float (M, M_rand), and uint8 (Mu8)",
        shape=(64, 64),
        mask_images=("M", "M0", "M1", "M_rand", "Mu8"),
    ),
    "cubes": DatasetSpec(
        id="cubes",
        description="3-D float32 cubes (cube, cubeB, cubeC, 16x12x4) and 2-D images (16x12)",
        shape=(16, 12),
        cube_images=("cube", "cubeB", "cubeC"),
    ),
    "size_1": DatasetSpec(
        id="size_1",
        description="Size 1x1 float32 images",
        shape=(1, 1),
    ),
    "size_7": DatasetSpec(
        id="size_7",
        description="Size 7x7 float32 images (odd, below SIMD width)",
        shape=(7, 7),
    ),
    "size_255": DatasetSpec(
        id="size_255",
        description="Size 255x255 float32 images (one below block boundary)",
        shape=(255, 255),
    ),
    "size_256": DatasetSpec(
        id="size_256",
        description="Size 256x256 float32 images",
        shape=(256, 256),
    ),
    "size_257": DatasetSpec(
        id="size_257",
        description="Size 257x257 float32 images (one above block boundary)",
        shape=(257, 257),
    ),
    "size_1024": DatasetSpec(
        id="size_1024",
        description="Size 1024x1024 float32 images",
        shape=(1024, 1024),
    ),
    "size_2048": DatasetSpec(
        id="size_2048",
        description="Size 2048x2048 float32 images",
        shape=(2048, 2048),
    ),
    "nonsquare": DatasetSpec(
        id="nonsquare",
        description="Non-square 23x11 float32 images",
        shape=(23, 11),
    ),
    "broadcast": DatasetSpec(
        id="broadcast",
        description="Broadcast shapes: 32x16 2-D, 32x1 row, 1x16 col, 1x1 sc1, 32x16x4 cube",
        shape=(32, 16),
        cube_images=("cube", "cubeC"),
        broadcast_images=("row", "col", "sc1"),
    ),
}


def get_dataset(dataset_id: str) -> DatasetSpec:
    """Return the DatasetSpec for @dataset_id."""
    return DATASETS[dataset_id]


def all_dataset_ids(include_large: bool = False) -> list[str]:
    """Return dataset IDs (excluding 1024/2048 by default for fast per-statement sweeps)."""
    skip = set() if include_large else {"size_1024", "size_2048"}
    return [k for k in DATASETS if k not in skip]


def offset_dataset_ids() -> list[str]:
    """Return the four offset dataset IDs for reduction numerical stability checks."""
    return ["offset_0", "offset_1e3", "offset_1e4", "offset_3e4"]
