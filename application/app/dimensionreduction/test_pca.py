# Copyright (c) 2025 TOYOTA MOTOR CORPORATION. ALL RIGHTS RESERVED.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for `test_pca.py`."""

from collections.abc import (
    Iterable,
    Sequence,
)
from sys import float_info
from typing import Any, cast

import pytest

import numpy as np
from numpy.testing import (
    assert_allclose,
    assert_array_max_ulp,
)
from numpy.typing import ArrayLike
from scipy.stats import ortho_group

from sklearn import decomposition
from sklearn.preprocessing import normalize

import pylog
from pyqoolloop.testutils import combine_lists

from ..mathutils import NPArray, OptionalVectorPair
from .linear import (
    LinearOrientedDimensionReducer,
    _calculate_vector_from_3d,
    _find_3_orthonormal,
    _find_matching_axes,
    _map,
)
from .pca import (
    OrientedPCA,
    OrientedPCALess,
    OrientedWeightedPCA,
    _OrientedPCA,
)

_logger = pylog.getLogger(__name__)


_epsilon = float_info.epsilon


def _check_array_max_ulp(x: NPArray, y: NPArray, maxulp: int = 1) -> bool:
    """
    Check for equality and return `bool` instead of raising `AssertionError`.

    So that `py.test` can show values of the arguments.
    """
    try:
        assert_array_max_ulp(x, y, maxulp)

    except AssertionError:
        return False

    return True


# === _map


@pytest.mark.parametrize(
    'vector, axes, expected',
    [
        ((1, 0, 0, 0), ((1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0)), (1, 0, 0)),
        ((0, 1, 0, 0), ((1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0)), (0, 1, 0)),
        ((0, 0, 1, 0), ((1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0)), (0, 0, 1)),
        ((0, 0, 0, 1), ((1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0)), (0, 0, 0)),
        (
            ((1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1)),
            ((1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0)),
            ((1, 0, 0), (0, 1, 0), (0, 0, 1), (0, 0, 0)),
        ),
    ],
)
def test__map(
    vector: Iterable[float], axes: Iterable[Iterable[float]], expected: Iterable[float]
) -> None:
    """Test `_map()`."""
    mapped_vector = _map(np.array(vector), np.array(axes))
    assert np.all(np.array(expected) == mapped_vector)


# === _find_3_orthonormal


@pytest.mark.parametrize(
    'original_vectors, axes, expected',
    [
        (
            ((1, 0, 0), (0, 1, 0), (0, 0, 1)),
            ((1, 0, 0), (0, 1, 0), (0, 0, 1)),
            ((1, 0, 0), (0, 1, 0), (0, 0, 1)),
        ),
        (
            ((0, 1, 0), (0, 0, 1), (1, 0, 0)),
            ((1, 0, 0), (0, 1, 0), (0, 0, 1)),
            ((0, 1, 0), (0, 0, 1), (1, 0, 0)),
        ),
        (
            ((0, 0, 1), (1, 0, 0), (0, 1, 0)),
            ((1, 0, 0), (0, 1, 0), (0, 0, 1)),
            ((0, 0, 1), (1, 0, 0), (0, 1, 0)),
        ),
        (
            ((1, 0, 0), (0, 1, 0), (0, 0, 1)),
            ((1, 0, 0), (0, -1, 0), (0, 0, -1)),
            ((1, 0, 0), (0, 1, 0), (0, 0, 1)),
        ),
        (
            ((-1, 0, 0), (0, 0, -1), (0, -1, 0)),
            ((0, -1, 0), (-1, 0, 0), (0, 0, -1)),
            ((-1, 0, 0), (0, 0, -1), (0, -1, 0)),
        ),
    ],
)
def test__find_3_orthonormal(
    original_vectors: Iterable[Iterable[float]],
    axes: Iterable[Iterable[float]],
    expected: Iterable[Iterable[float]],
) -> None:
    """Test `_find_3_orthonormal()` with simple vectors."""
    for prioritize in [0, 1, 2]:
        result = _find_3_orthonormal(
            np.array(original_vectors), prioritize, np.array(axes)
        )
        assert np.all(expected == result)

        result = _find_3_orthonormal(
            -np.array(original_vectors), prioritize, np.array(axes)
        )
        assert np.all(expected == -result)


@pytest.mark.parametrize(
    'original_vectors, axes, expected',
    [
        (
            ((1, 0.1, 0), (0, 0.1, 0), (0, 0.1, 1)),
            ((1, 0, 0), (0, 1, 0), (0, 0, 1)),
            ((1, 0, 0), (0, 1, 0), (0, 0, 1)),
        ),
    ],
)
def test__find_3_orthonormal__close(
    original_vectors: Iterable[Iterable[float]],
    axes: Iterable[Iterable[float]],
    expected: Iterable[Iterable[float]],
) -> None:
    """Test `_find_3_orthonormal()` with tolerance."""
    prioritize = 1

    result = _find_3_orthonormal(np.array(original_vectors), prioritize, np.array(axes))
    assert_allclose(np.array(expected), result, atol=1e-16)


@pytest.mark.parametrize(
    'original_vectors, prioritize, axes, expected',
    [
        (
            ((1, 0, 0), (0, 0.5, 0), (0, 0, 0.5)),
            0,
            ((1, 0, 0), (0, 1, 0), (0, 0, 1)),
            (1, 0, 0),
        ),
        (
            ((0.2, 0, 0), (0, 1, 0), (0, 0, 0.2)),
            1,
            ((1, 0, 0), (0, 1, 0), (0, 0, 1)),
            (0, 1, 0),
        ),
        (
            ((0.7, 0, 0), (0, 0.7, 0), (0, 0, 1)),
            2,
            ((1, 0, 0), (0, 1, 0), (0, 0, 1)),
            (0, 0, 1),
        ),
    ],
)
def test__find_3_orthonormal__prioritize(
    original_vectors: Iterable[Iterable[float]],
    prioritize: int,
    axes: Iterable[Iterable[float]],
    expected: Iterable[float],
) -> None:
    """Test `_find_3_orthonormal()` with priority."""
    result = _find_3_orthonormal(np.array(original_vectors), prioritize, np.array(axes))
    assert np.all(np.array(expected) == result[prioritize, :])


def _are_orthonormal(vectors: NPArray) -> bool:
    return np.allclose(np.identity(3), vectors @ vectors.transpose())


@pytest.mark.parametrize('dimensions', [3, 10, 200])
def test__find_3_orthonormal__orthonormal(dimensions: int) -> None:
    """Test that `_find_3_orthonormal()` returns orthonormal vectors."""
    generator = np.random.default_rng(seed=1234)

    while True:
        original_vectors = generator.uniform(low=-1.0, high=1.0, size=(3, dimensions))

        for each_row in original_vectors:
            if np.allclose(np.zeros((1, dimensions)), each_row):
                continue
        break

    orthogonal_matrix = ortho_group.rvs(dim=dimensions)

    for prioritize in range(3):
        result = _find_3_orthonormal(
            original_vectors, prioritize, orthogonal_matrix[:3, :]
        )

        assert _are_orthonormal(result)


# === _find_matching_axes


@pytest.mark.parametrize(
    'weights, axes, desired, expected',
    combine_lists(
        [(0.1, 1, 0.1), (1, 1, 1)],
        [
            [
                ([1, 0, 0], [0, 1, 0], [0, 0, 1]),
                ([1, 0, 0], [0, 1, 0], [0, 0, 1]),
                (0, 1, 2),
            ],
            [
                ([0, 0, 0, 1], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]),
                ([1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]),
                (1, 2, 3),
            ],
            [
                (
                    [1, 0, 0, 0, 0],
                    [0, 0, 0, 0, 1],
                    [0, 1, 0, 0, 0],
                    [0, 0, 1, 0, 0],
                    [0, 0, 0, 1, 0],
                ),
                ([0, 1, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 1, 0]),
                (2, 3, 4),
            ],
            [
                (
                    [1, 0, 0, 0, 0],
                    [0, 0, 0, 0, 1],
                    [0, 1, 0, 0, 0],
                    [0, 0, 1, 0, 0],
                    [0, 0, 0, 1, 0],
                ),
                ([0, 1, 1, 0, 0], [0, 0, 1, 1, 0], [0, 1, 0, 1, 0]),
                (2, 3, 4),
            ],
            [
                (
                    [0, 0, 0, 1, 0, 0],
                    [0, 1, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0],
                    [0, 0, 0, 0, 0, 1],
                    [1, 0, 0, 0, 0, 0],
                ),
                ([1, -1, -1, 0, 0, 0], [6, 5, 1, 1, 4, 0], [0, 1, -1, 0, -1, 0]),
                (1, 2, 4),
            ],
        ],
    )
    + combine_lists(
        [(1, 1, 1)],
        [
            [
                (
                    [0, 0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 1, 0],
                    [0, 1, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0],
                    [0, 0, 0, 0, 0, 1],
                    [1, 0, 0, 0, 0, 0],
                ),
                ([1, -1, 0, 0, -1, 0], [2, 1, 1, 1, 0, 0], [0, -1, 0, 0, 1, 0]),
                (1, 2, 5),
            ],
            [
                (
                    [0, 0, 0, 1, 0, 0],
                    [0, 1, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0],
                    [0, 0, 0, 0, 0, 1],
                    [1, 0, 0, 0, 0, 0],
                ),
                ([1, -1, -1, 0, 0, 0], [6, 5, 1, 1, 4, 5], [0, 1, -1, 0, -1, 0]),
                (1, 2, 4),
            ],
        ],
    )
    + combine_lists(
        [(0.1, 1, 0.1)],
        [
            [
                (
                    [0, 0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 1, 0],
                    [0, 1, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0],
                    [0, 0, 0, 0, 0, 1],
                    [1, 0, 0, 0, 0, 0],
                ),
                ([1, -1, 0, 0, -1, 0], [2, 1, 0.5, 1, 0, 0], [0, -1, 0, 0, 1, 0]),
                (0, 2, 5),
            ],
            [
                (
                    [0, 0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 1, 0],
                    [0, 1, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0],
                    [0, 0, 0, 0, 0, 1],
                    [1, 0, 0, 0, 0, 0],
                ),
                ([1, -1, 0, 0, -1, 0], [2, 1, 1, 0.5, 0, 0], [0, -1, 0, 0, 1, 0]),
                (2, 3, 5),
            ],
            [
                (
                    [0, 0, 0, 1, 0, 0],
                    [0, 1, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0],
                    [0, 0, 0, 0, 0, 1],
                    [1, 0, 0, 0, 0, 0],
                ),
                ([1, -1, -1, 0, 0, 0], [6, 5, 1, 1, 4, 5], [0, 1, -1, 0, -1, 0]),
                (1, 3, 4),
            ],
        ],
    ),
)
def test__find_matching_axes(
    weights: Iterable[float],
    axes: Iterable[Iterable[float]],
    desired: Iterable[Iterable[float]],
    expected: tuple[int, ...],
) -> None:
    """Test `_find_matching_axes()`."""
    result = _find_matching_axes(
        np.array(axes), normalize(np.array(desired), axis=1), np.array(weights)
    )
    assert set(expected) == set(result)


# === _calculateDirection


_CALCULATEDIRECTION_PARAMETERS = (
    'new_position_3d, old_position_3d, directions_3d',
    combine_lists(
        [
            [(0, 0, 0), (0, 0, 0)],
            [(1, 0, 0), (0, 0, 0)],
            [(1, 0, 0), (0, 1, 0)],
        ],
        [
            [((1, 0, 0), (0, 1, 0), (0, 0, 1))],
            [((0, -1, 0), (1, 0, 0), (0, 0, 1))],
        ],
    ),
)


@pytest.mark.parametrize(*_CALCULATEDIRECTION_PARAMETERS)
def test__calculate_vector_from_3d__3d(
    new_position_3d: Iterable[float],
    old_position_3d: Iterable[float],
    directions_3d: Iterable[Iterable[float]],
) -> None:
    """Test `_calculate_direction()` in 3D."""
    new_position_3d_array: NPArray = np.array(new_position_3d)
    old_position_3d_array: NPArray = np.array(old_position_3d)

    expected_movement = new_position_3d_array - old_position_3d_array

    movement = _calculate_vector_from_3d(
        new_position_3d_array - old_position_3d_array,
        np.array(directions_3d),
        np.array(directions_3d),
    )

    assert (expected_movement == movement).all()


@pytest.mark.parametrize(*_CALCULATEDIRECTION_PARAMETERS)
def test__calculate_vector_from_3d__4d(
    new_position_3d: Iterable[float],
    old_position_3d: Iterable[float],
    directions_3d: Sequence[Iterable[float]],
) -> None:
    """Test `_calculate_direction()` in 4D."""
    new_position_3d_array: NPArray = np.array(new_position_3d)
    old_position_3d_array: NPArray = np.array(old_position_3d)

    movement_3d = new_position_3d_array - old_position_3d_array

    expected_movement: NPArray = np.hstack((movement_3d, 0))

    movement = _calculate_vector_from_3d(
        movement_3d,
        np.array(directions_3d),
        np.hstack((np.array(directions_3d), np.zeros([len(directions_3d), 1]))),
    )

    assert (expected_movement == movement).all()


# === PCA


def _fit_oriented_pca(
    pca: LinearOrientedDimensionReducer,
    data_point_array: NPArray,
    *,
    previous_pca: LinearOrientedDimensionReducer | None = None,
    centers: OptionalVectorPair | None = None,
    camera_directions: tuple[ArrayLike, ...] | None = None,
) -> LinearOrientedDimensionReducer:
    _logger.info("Checking: %s", type(pca))

    pca.set_previous(previous_pca)
    pca.set_global(is_global=previous_pca is None)

    if camera_directions is None:
        camera_directions = ([1, 0, 0], [0, 1, 0], [0, 0, 1])

    _logger.info("camera_directions: %r", camera_directions)
    # Not really necessary for Global mode.
    pca.set_camera_directions(*camera_directions)

    if centers is None:
        centers = OptionalVectorPair(
            visual_3d=np.zeros((3,)),
            virtual=np.zeros(
                data_point_array.shape[1],
            ),
        )

    _logger.info("centers: %r", centers)
    if centers.virtual is None:
        pca.move_center_3d(centers.visual_3d)

    else:
        pca.move_center(centers.visual_3d, centers.virtual)

    pca._fit(data_point_array)  # noqa: SLF001
    if hasattr(pca._pca, 'components_'):  # noqa: SLF001
        _logger.info(
            "OrientedPCA Components: %r",
            pca._pca.components_,  # noqa: SLF001
        )

    return pca


_DETERMINISTIC_PCA = [
    # pca_class, arguments
    [OrientedPCA, {"random_state": False}],
    [OrientedPCALess, {"random_state": False}],
    [OrientedPCA, {"random_state": 1234}],
    [OrientedPCALess, {"random_state": 1234}],
]

_STOCHASTIC_PCA = [
    # pca_class, arguments
    [_OrientedPCA, {}],
    [
        OrientedWeightedPCA,
        {},
    ],
    [
        OrientedWeightedPCA,
        {'weight': {'type': 'Equal', 'parameters': {}}},
    ],
]


@pytest.mark.parametrize(
    'pca_class, arguments, allow_sign_inversion',
    combine_lists(_DETERMINISTIC_PCA, [False]) + combine_lists(_STOCHASTIC_PCA, [True]),
)
def test__compare_with_regular_pca(
    pca_class: type[LinearOrientedDimensionReducer],
    arguments: dict[str, Any],
    allow_sign_inversion: bool,  # noqa: FBT001
) -> None:
    """
    Test that PCA classes produce the same results for regular PCA.

    :param pca_class: PCA class.
    :param arguments: Arguments for the constructor.
    :param allow_sign_inversion: Whether to allow components to have different sign.
    """

    def _regular_pca(data_points: NPArray) -> NPArray:
        pca = decomposition.PCA(svd_solver='full', n_components=3)
        pca.fit(data_points)

        _logger.info("PCA Components: %r", pca.components_)

        pca.mean_ = None

        return cast(NPArray, pca.transform(data_points))

    def _assert_compare(
        one: NPArray, another: NPArray, *, allow_sign_inversion: bool
    ) -> None:
        if allow_sign_inversion:
            one = np.abs(one)
            another = np.abs(another)

        assert_allclose(one, another)

    features = 6

    zero_point = np.zeros((1, features))

    generator = np.random.default_rng(seed=1234)
    data_point_array: NPArray = generator.random(size=(100, features))

    reducer = _fit_oriented_pca(pca_class(**arguments), data_point_array)
    assert isinstance(reducer, LinearOrientedDimensionReducer)

    pca_result = _regular_pca(data_point_array)

    result = reducer._transform(zero_point)  # noqa: SLF001
    assert_array_max_ulp(result, np.zeros((1, 3)))

    result = reducer._transform(data_point_array)  # noqa: SLF001
    _assert_compare(result, pca_result, allow_sign_inversion=allow_sign_inversion)


@pytest.mark.parametrize('pca_class, arguments', _DETERMINISTIC_PCA + _STOCHASTIC_PCA)
def test__pca_stability_with_perturbation(
    pca_class: type[LinearOrientedDimensionReducer],
    arguments: dict[str, Any],
) -> None:
    """Test that small perturbation in data doesn't cause large changes in transform."""
    samples = 100
    features = 5

    generator = np.random.default_rng(seed=1234)

    original_array: NPArray = generator.random(size=(samples, features))

    perturbed_array = (
        original_array + generator.random(size=(samples, features)) * _epsilon
    )

    original_reducer = _fit_oriented_pca(pca_class(**arguments), original_array)
    original_result = original_reducer._transform(original_array)  # noqa: SLF001

    perturbed_reducer = _fit_oriented_pca(
        pca_class(**arguments), perturbed_array, previous_pca=original_reducer
    )
    perturbed_result = perturbed_reducer._transform(original_array)  # noqa: SLF001

    assert_allclose(original_result, perturbed_result)


@pytest.mark.parametrize('pca_class, arguments', _DETERMINISTIC_PCA + _STOCHASTIC_PCA)
def test__pca__center_stability(
    pca_class: type[LinearOrientedDimensionReducer],
    arguments: dict[str, Any],
) -> None:
    """
    Test that the center doesn't change despite distribution of data.

    The "center" is the current position.
    """
    samples = 100
    features = 5

    generator = np.random.default_rng(seed=1234)

    previous_pca = None

    for _iteration in range(2):
        # First loop is Global Mode. Second loop is Local Mode

        data_points: NPArray = generator.random(size=(samples, features))

        center_3d = generator.random(size=(3,))
        center = generator.random(size=(1, features))

        centers = OptionalVectorPair(visual_3d=center_3d, virtual=center)

        reducer = _fit_oriented_pca(
            pca_class(**arguments),
            data_points,
            previous_pca=previous_pca,
            centers=centers,
        )
        result = reducer._transform(center)  # noqa: SLF001

        assert_allclose(result[0, :], center_3d)


@pytest.mark.parametrize('pca_class, arguments', _DETERMINISTIC_PCA + _STOCHASTIC_PCA)
def test__pca__stability_with_movement_and_rotation(
    pca_class: type[LinearOrientedDimensionReducer],
    arguments: dict[str, Any],
) -> None:
    """Test that the distribution of mapped data points doesn't change with movement."""
    samples = 100
    features = 7

    camera_direction_choices = (
        ([1, 0, 0], [0, 1, 0], [0, 0, 1]),
        ([0, 1, 0], [0, 0, 1], [1, 0, 0]),
        ([0, 0, 1], [1, 0, 0], [0, 1, 0]),
    )

    generator = np.random.default_rng(seed=1234)

    data_points: NPArray = generator.random(size=(samples, features))
    center: NPArray | None = generator.random(size=(1, features))

    previous_pca = None
    previous_result = None

    for _iteration in range(3):
        # First loop is Global Mode. Second and third loops are Local Mode

        center_3d = generator.random(size=(3,))

        centers = OptionalVectorPair(visual_3d=center_3d, virtual=center)

        direction_index = generator.integers(0, len(camera_direction_choices))

        reducer = _fit_oriented_pca(
            pca_class(**arguments),
            data_points,
            previous_pca=previous_pca,
            centers=centers,
            # FUTURE: false positive for pylint 2.15.5?
            # pylint: disable=invalid-sequence-index
            camera_directions=camera_direction_choices[direction_index],
        )
        result = reducer._transform(data_points)  # noqa: SLF001

        if previous_result is not None:
            assert_allclose(result, previous_result)

        previous_pca = reducer
        previous_result = result
        center = None


@pytest.mark.parametrize('pca_class, arguments', _DETERMINISTIC_PCA + _STOCHASTIC_PCA)
def test__pca__stability_with_no_movement(
    pca_class: type[LinearOrientedDimensionReducer],
    arguments: dict[str, Any],
) -> None:
    """Test that the distribution of mapped data points doesn't change with movement."""
    samples = 100
    features = 8

    generator = np.random.default_rng(seed=1234)

    data_points: NPArray = generator.random(size=(samples, features))
    center: NPArray | None = generator.random(size=(1, features))

    previous_pca = None
    previous_result = None

    center_3d = generator.random(size=(3,))

    for _iteration in range(3):
        # First loop is Global Mode. Second and third loops are Local Mode

        centers = OptionalVectorPair(visual_3d=center_3d, virtual=center)

        reducer = _fit_oriented_pca(
            pca_class(**arguments),
            data_points,
            previous_pca=previous_pca,
            centers=centers,
        )
        result = reducer._transform(data_points)  # noqa: SLF001

        if previous_result is not None:
            assert_allclose(result, previous_result)

        previous_pca = reducer
        previous_result = result
        center = None
