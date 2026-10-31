"""Periodic cell tests: wrapping, exact nearest images, reciprocal space."""

from itertools import product

import numpy as np
import pytest

from qmc.cells import BCCCell, CubicCell, OrthorhombicCell, PeriodicCell


def test_cubic_minimum_image_and_periodicity():
    cell = CubicCell(10.0)
    delta = np.array([[7., -6., 1.], [25., 5., -5.]])
    np.testing.assert_allclose(
        cell.minimum_image(delta),
        [[-3., 4., 1.], [5., 5., -5.]],
    )
    np.testing.assert_allclose(cell.displacement([9., 1., 1.], [1., 2., 1.]), [2., 1., 0.])
    assert cell.volume == pytest.approx(1000.)
    assert cell.wrap([11., -1., 0.]).shape == (3,)
    np.testing.assert_allclose(cell.wrap([11., -1., 0.]), [1., 9., 0.])


def test_orthorhombic_box_and_batch_shapes():
    cell = OrthorhombicCell([4., 6., 8.])
    xyz = np.array([[[3., 4., 5.], [-7., 10., -13.]]])
    result = cell.minimum_image(xyz)
    assert result.shape == (1, 2, 3)
    np.testing.assert_allclose(result[0, 0], [-1., -2., -3.])
    np.testing.assert_allclose(result[0, 1], [1., -2., 3.])
    assert cell.volume == pytest.approx(192.)
    np.testing.assert_allclose(cell.wrap(xyz), xyz % np.array([4., 6., 8.]))


def test_reciprocal_and_fractional_round_trip():
    cell = PeriodicCell([[2.0, 0.0, 0.0], [0.6, 2.2, 0.0], [0.4, 0.7, 3.0]])
    np.testing.assert_allclose(cell.vectors @ cell.reciprocal_vectors.T, 2*np.pi*np.eye(3), atol=1e-12)
    r = np.random.default_rng(2026).normal(size=(4, 6, 3))
    np.testing.assert_allclose(cell.to_cartesian(cell.to_fractional(r)), r, atol=1e-12)
    k = cell.wavevectors([[1, 0, -1], [2, -3, 4]])
    np.testing.assert_allclose(k, np.array([[1, 0, -1], [2, -3, 4]]) @ cell.reciprocal_vectors)
    with pytest.raises(ValueError, match="integer"):
        cell.wavevectors([0.5, 1, 0])


def test_triclinic_minimum_image_is_global_minimum():
    # A strongly non-orthogonal lattice: naive rounding of each fractional
    # coordinate can give a non-minimal image.
    cell = PeriodicCell([[2.0, 0.0, 0.0], [1.6, 1.1, 0.0], [0.9, 0.7, 1.4]])
    rng = np.random.default_rng(123)
    fractional = rng.uniform(-1.5, 1.5, size=(60, 3))
    delta = cell.to_cartesian(fractional)
    actual = cell.minimum_image(delta)
    shifts = np.array(list(product(range(-4, 5), repeat=3))) @ cell.vectors
    expected2 = np.min(np.sum((delta[:, None, :] - shifts[None, :, :])**2, axis=-1), axis=1)
    np.testing.assert_allclose(np.sum(actual * actual, axis=-1), expected2, atol=1e-12)
    np.testing.assert_allclose(cell.minimum_image(delta + cell.vectors[0]), actual, atol=1e-12)


def test_bcc_volume_and_truncated_octahedral_nearest_image():
    cell = BCCCell(4.0)
    assert cell.volume == pytest.approx(32.0)
    # The nearest image of (1.96,1.96,1.96) is the half-cube lattice
    # translation (2,2,2), not the origin or a simple-cubic translation.
    np.testing.assert_allclose(cell.minimum_image([1.96, 1.96, 1.96]), [-0.04]*3, atol=1e-12)
    np.testing.assert_allclose(cell.wrap([1.96]*3, centered=True), [-0.04]*3, atol=1e-12)
    np.testing.assert_allclose(cell.wrap([2., 2., 2.]), [0., 0., 0.], atol=1e-12)


def test_bcc_random_minimum_image_matches_brute_force():
    cell = BCCCell(4.0)
    rng = np.random.default_rng(24680)
    dr = rng.uniform(-13., 13., size=(3, 18, 3))
    nearest = cell.minimum_image(dr)
    ints = np.array(list(product(range(-5, 6), repeat=3)), dtype=float)
    reference = np.concatenate([4*ints, 4*(ints + 0.5)], axis=0)
    brute2 = np.min(np.sum((dr.reshape(-1, 1, 3)-reference[None, :, :])**2, axis=-1), axis=1)
    np.testing.assert_allclose(np.sum(nearest**2, axis=-1).reshape(-1), brute2, atol=1e-12)
    np.testing.assert_allclose(cell.displacement(dr*0., dr), nearest, atol=1e-12)


def test_cell_rejects_invalid_vectors_and_coordinates():
    with pytest.raises(ValueError):
        PeriodicCell(np.zeros((3, 3)))
    with pytest.raises(ValueError):
        PeriodicCell(np.eye(2))
    with pytest.raises(ValueError):
        OrthorhombicCell([3., -2., 5.])
    with pytest.raises(ValueError):
        BCCCell(0.)
    with pytest.raises(ValueError):
        CubicCell(1).minimum_image([1., 2.])
    with pytest.raises(ValueError):
        CubicCell(1).wrap([np.nan, 0, 1])
    assert CubicCell(2.).minimum_image(np.empty((0, 3))).shape == (0, 3)
    assert CubicCell(1e-6).volume == pytest.approx(1e-18)
