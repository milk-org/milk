"""016-1: the CM numerics of pycacao.calib.mkcm (see specs/016-*/brief.md §3, §4)."""

from __future__ import annotations

from typing import Callable

import numpy as np
import pytest

from pycacao.calib.mkcm import CM, Array, cm_pinv, cm_tikhonov, cm_tsvd

N = 6  # poke modes; DM 3 x 2 (x by y), WFS 4 x 3
RNG = np.random.default_rng(16)
D = RNG.standard_normal((N, 2, 3))
W = RNG.standard_normal((N, 3, 4))
P = RNG.standard_normal(N)
P /= np.linalg.norm(P)
W5 = np.einsum("ij,jkl->ikl", np.eye(N) - np.outer(P, P), W)  # rank 5: W5ᵀ p = 0
C = RNG.standard_normal(N)


def flat(cube: Array) -> Array:
    return cube.reshape(cube.shape[0], -1)


def loop(cm_wfs: Array, cm_dm: Array, w: Array, c: Array = C) -> Array:
    """§1: wfs = Wᵀc, a = CMmodesWFS · wfs, dm = CMmodesDMᵀ · a."""
    a = flat(cm_wfs) @ (flat(w).T @ c)
    return flat(cm_dm).T @ a


def svd(w: Array) -> tuple[Array, Array, Array]:
    return np.linalg.svd(flat(w), full_matrices=False)


def test_tsvd_full_rank_is_identity() -> None:
    cm_wfs, cm_dm, _ = cm_tsvd(D, W)
    assert cm_wfs.shape[0] == N
    np.testing.assert_allclose(loop(cm_wfs, cm_dm, W), flat(D).T @ C)


def test_tsvd_rank5_default_drops_the_null_mode() -> None:
    cm_wfs, cm_dm, _ = cm_tsvd(D, W5)
    assert cm_wfs.shape[0] == 5
    np.testing.assert_allclose(
        loop(cm_wfs, cm_dm, W5), flat(D).T @ (C - P * (P @ C)), atol=1e-12
    )


@pytest.mark.parametrize("fn", [cm_tsvd, cm_pinv])
@pytest.mark.parametrize("nmodes, rcond", [(3, 1e-6), (None, 0.5), (2, 0.5), (5, 0.5)])
def test_n_and_r_combine(
    fn: Callable[..., CM], nmodes: int | None, rcond: float
) -> None:
    v, sv, _ = svd(W)
    k = [
        i for i in range(N) if (nmodes is None or i < nmodes) and sv[i] > rcond * sv[0]
    ]
    assert k and k == list(range(len(k)))  # the kept modes are the largest ones
    cm_wfs, cm_dm, _ = fn(D, W, nmodes=nmodes, rcond=rcond)
    if fn is cm_tsvd:
        assert cm_wfs.shape[0] == len(k)
        np.testing.assert_allclose(
            flat(cm_wfs) @ flat(W).T @ v[:, k], np.eye(len(k)), atol=1e-12
        )
    else:
        assert cm_wfs.shape[0] == N
        vk = v[:, k]
        np.testing.assert_allclose(
            loop(cm_wfs, cm_dm, W), flat(D).T @ vk @ vk.T @ C, atol=1e-12
        )


def test_tikhonov_gains_and_cut() -> None:
    v, sv, _ = svd(W)
    reg = 0.3
    lam2 = (reg * sv[0]) ** 2
    cm_wfs, _, _ = cm_tikhonov(D, W, reg)
    np.testing.assert_allclose(
        flat(cm_wfs) @ flat(W).T @ v, np.diag(sv**2 / (sv**2 + lam2)), atol=1e-12
    )
    cm_wfs, _, _ = cm_tikhonov(D, W, reg, nmodes=4)
    np.testing.assert_allclose(
        flat(cm_wfs) @ flat(W).T @ v[:, :4],
        np.diag(sv[:4] ** 2 / (sv[:4] ** 2 + lam2)),
        atol=1e-12,
    )


def test_tikhonov_reg0_equals_tsvd() -> None:
    for a, b in zip(cm_tikhonov(D, W, 0.0), cm_tsvd(D, W)):
        np.testing.assert_allclose(a, b)


def test_pinv_defaults_and_cut() -> None:
    cm_wfs, cm_dm, _ = cm_pinv(D, W)
    np.testing.assert_allclose(flat(cm_wfs), np.linalg.pinv(flat(W).T, rcond=1e-6))
    np.testing.assert_array_equal(cm_dm, D)
    cm_wfs, cm_dm, _ = cm_pinv(D, W, nmodes=2)
    assert cm_wfs.shape[0] == cm_dm.shape[0] == N
    v2 = svd(W)[0][:, :2]
    np.testing.assert_allclose(
        loop(cm_wfs, cm_dm, W), flat(D).T @ v2 @ v2.T @ C, atol=1e-12
    )


@pytest.mark.parametrize(
    "call",
    [
        lambda: cm_tsvd(D, W, nmodes=4),
        lambda: cm_tikhonov(D, W, 0.1, nmodes=4),
        lambda: cm_pinv(D, W, nmodes=4),
    ],
)
def test_shapes_and_sv(call: Callable[[], CM]) -> None:
    cm_wfs, cm_dm, sv = call()
    assert cm_wfs.shape[1:] == (3, 4) and cm_dm.shape[1:] == (2, 3)
    assert sv.shape == (min(N, 12),) and np.all(np.diff(sv) <= 0)


def test_dtype_follows_input() -> None:
    cm_wfs, cm_dm, _ = cm_tsvd(D.astype(np.float32), W.astype(np.float32))
    assert cm_wfs.dtype == cm_dm.dtype == np.float32


@pytest.mark.parametrize(
    "call",
    [
        lambda: cm_tsvd(D, W, nmodes=0),
        lambda: cm_tikhonov(D, W, 0.1, nmodes=0),
        lambda: cm_pinv(D, W, nmodes=0),
        lambda: cm_tsvd(D, 0 * W),
        lambda: cm_pinv(D, 0 * W),
    ],
)
def test_empty_k_raises(call: Callable[[], CM]) -> None:
    with pytest.raises(ValueError):
        call()
