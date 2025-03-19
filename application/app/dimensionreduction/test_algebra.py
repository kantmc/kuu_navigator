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

"""Tests for `algebra.py`."""

from collections.abc import Iterable, Sequence
from math import cos, sin, sqrt
from typing import cast

import pytest

import numpy as np
from numpy.testing import assert_allclose, assert_array_max_ulp

from pyqoolloop.testutils import combine_lists

from ..mathutils import NPArray
from .algebra import find_orthonormal_2d, generate_orthonormal_3d
from .test_pca import _check_array_max_ulp, _epsilon

# === generate_orthonormal_3d()


@pytest.mark.parametrize(
    'first_vector',
    [
        (1, 0, 0),
        (0, 1, 0),
        (0, 0, 1),
        (0, 1, 1),
        (1, 0, 1),
        (1, 1, 0),
        (1, 1, 1),
        (_epsilon, 0, 0),
        (0, _epsilon, 0),
        (0, 0, _epsilon),
        (0, _epsilon, _epsilon),
        (_epsilon, 0, _epsilon),
        (_epsilon, _epsilon, 0),
        (_epsilon, _epsilon, _epsilon),
    ],
)
def test__generate_orthonormal_3d(first_vector: tuple[float, float, float]) -> None:
    """Test `_generate_orthonormal_3d()`."""
    v1, v2, v3 = generate_orthonormal_3d(np.array(first_vector))

    assert_allclose(np.cross(v1, first_vector), 0)  # comparing array and scalar

    assert_array_max_ulp(np.inner(v1, v1), 1, maxulp=2)
    assert_array_max_ulp(np.inner(v2, v2), 1, maxulp=2)
    assert_array_max_ulp(np.inner(v3, v3), 1, maxulp=2)

    assert_array_max_ulp(np.inner(v1, v2), 0)
    assert_array_max_ulp(np.inner(v2, v3), 0)
    assert_array_max_ulp(np.inner(v3, v1), 0)


# === _findOrthonormal2d()


@pytest.mark.parametrize(
    'first_vector, second_vector, expected',
    [
        ([1, 0], [0, 1], ([1, 0], [0, 1])),
        ([0, 1], [1, 0], ([0, 1], [1, 0])),
        ([-1, 0], [0, 1], ([-1, 0], [0, 1])),
        ([0, -1], [1, 0], ([0, -1], [1, 0])),
        ([1, 0], [0, -1], ([1, 0], [0, -1])),
        ([0, 1], [-1, 0], ([0, 1], [-1, 0])),
        ([0, -1], [-1, 0], ([0, -1], [-1, 0])),
    ],
)
def test__find_orthonormal_2d(
    first_vector: Sequence[float],
    second_vector: Sequence[float],
    expected: tuple[Iterable[float], Iterable[float]],
) -> None:
    """Test `_find_orthonormal_2d()` with simple vectors."""
    expected_array: NPArray = np.array(expected)

    for each in (0.1, 1.0, 2.0):
        scaled_first_vector = np.multiply(each, np.array(first_vector))
        scaled_second_vector = np.multiply(each, np.array(second_vector))

        result = find_orthonormal_2d(scaled_first_vector, np.array(second_vector))
        assert np.all(result == expected_array)

        result = find_orthonormal_2d(np.array(first_vector), scaled_second_vector)
        assert np.all(result == expected_array)

        result = find_orthonormal_2d(scaled_first_vector, scaled_second_vector)
        assert np.all(result == expected_array)


@pytest.mark.parametrize(
    'first_vector, second_vector, rotation',
    (
        combine_lists(
            [
                [
                    [1, 0],
                    [0, 1],
                ],
                [
                    [0, 1],
                    [1, 0],
                ],
            ],
            [value / 10 * 2 * np.pi for value in range(10)],
        )
    ),
)
def test__find_orthonormal_2d__orthogonal_rotate(
    first_vector: Sequence[float], second_vector: Sequence[float], rotation: float
) -> None:
    """Test `_find_orthonormal_2d()` by rotating vectors."""
    rotation_matrix: NPArray = np.array(
        [[cos(rotation), sin(rotation)], [-sin(rotation), cos(rotation)]]
    )

    rotated_first_vector = cast(NPArray, np.array(first_vector)) @ rotation_matrix
    rotated_second_vector = cast(NPArray, np.array(second_vector)) @ rotation_matrix

    expected_array: NPArray = np.vstack(
        (np.array(rotated_first_vector), np.array(rotated_second_vector))
    )

    for each in (0.1, 1.0, 2.0):
        result = find_orthonormal_2d(each * rotated_first_vector, rotated_second_vector)
        assert _check_array_max_ulp(result, expected_array)

        result = find_orthonormal_2d(rotated_first_vector, each * rotated_second_vector)
        assert _check_array_max_ulp(result, expected_array)

        result = find_orthonormal_2d(
            each * rotated_first_vector, each * rotated_second_vector
        )
        assert _check_array_max_ulp(result, expected_array)


@pytest.mark.parametrize(
    'first_vector, second_vector, expected',
    [
        ([1, 1], [1, -1], ([sqrt(0.5), sqrt(0.5)], [sqrt(0.5), -sqrt(0.5)])),
        # FUTURE: more
    ],
)
def test__find_orthonormal_2d__close(
    first_vector: Sequence[float],
    second_vector: Sequence[float],
    expected: tuple[Iterable[float], Iterable[float]],
) -> None:
    """Test `_find_orthonormal_2d()` with approximate values."""
    expected_array: NPArray = np.array(expected)

    for each in (0.1, 1.0, 2.0):
        scaled_first_vector = np.multiply(each, np.array(first_vector))
        scaled_second_vector = np.multiply(each, np.array(second_vector))

        result = find_orthonormal_2d(scaled_first_vector, np.array(second_vector))
        assert_array_max_ulp(result, expected_array)

        result = find_orthonormal_2d(np.array(first_vector), scaled_second_vector)
        assert_array_max_ulp(result, expected_array)

        result = find_orthonormal_2d(scaled_first_vector, scaled_second_vector)
        assert_array_max_ulp(result, expected_array)
