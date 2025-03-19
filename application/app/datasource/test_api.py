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

"""Tests for `api.py`."""

from collections.abc import (
    Sequence,
)

import pytest

import numpy as np

from ..mathutils import NPArray, np_all
from .api import (
    DataPointCollection,
    NPExtremes,
    equal_coordinates,
)
from .simulated import (
    _DataPointCard,
)

# == NPExtremes


@pytest.mark.parametrize(
    'array, dimensions, expected_mins, expected_maxs',
    [
        ([[1], [2], [3], [4]], None, [1], [4]),
        ([[1, 2], [3, 4], [5, 6]], None, [1, 2], [5, 6]),
        ([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]], 2, [1, 2], [11, 12]),
        (
            [[1.1, 2.2, 3.3, 4.4], [-1.1, 13.3, 14.4, -2.2], [9.9, 10.0, 11.1, 12.2]],
            2,
            [-1.1, -2.2],
            [14.4, 13.3],
        ),
    ],
)
def test__NPExtremes__from_array(
    array: list[float],
    dimensions: int | None,
    expected_mins: list[float],
    expected_maxs: list[float],
) -> None:
    """Test :meth:`NPExtremes.from_array()`."""
    nparray = np.array(array)
    extremes = NPExtremes.from_array(nparray, dimensions=dimensions)

    assert np_all(expected_mins == extremes.lower)
    assert np_all(expected_maxs == extremes.upper)


# === DataPoint


def test__DataPoint__immutable() -> None:
    """
    Test that `DataPoint`_ does not change with value used for initialization.

    .. _`DataPoint`: ../datasource/index.html#app.datasource.DataPoint
    """
    original_coordinates = (0, 1, 2)
    coordinates = list(original_coordinates)

    data_point = _DataPointCard(coordinates)

    coordinates[0] = 1

    assert equal_coordinates(data_point.get_coordinates(), original_coordinates)


# === DataPointCollection


@pytest.mark.parametrize(
    'coordinates',
    [
        [1.0, 2.0],
        [1.0, 2.0, 3.0],
    ],
)
def test__DataPointCollection__contains(coordinates: Sequence[float]) -> None:
    """Tests whether the `in` operator works for :class:`.DataPointCollection`."""
    data_point = _DataPointCard(coordinates)

    collection = DataPointCollection()

    assert data_point not in collection

    collection.add(data_point)

    assert data_point in collection


def test__DataPointCollection__iter_to_array_consistency() -> None:
    """
    Test that `__iter__()`_ and `to_array()`_ return values in the same order.

    .. _`__iter__()`:
      ../datasource/index.html#app.datasource.DataPointCollection.__iter__
    .. _`to_array()`:
      ../datasource/index.html#app.datasource.DataPointCollection.to_array
    """
    samples = 10
    features = 4

    generator = np.random.default_rng(seed=1234)

    def _make_data_point_collection() -> DataPointCollection:
        array: NPArray = generator.random(size=(samples, features))

        data_point_collection = DataPointCollection()
        for each_point in array:
            data_point_collection.add(_DataPointCard(each_point))

        return data_point_collection

    data_point_collection = _make_data_point_collection()
    data_point_array = data_point_collection.to_array()

    for each_in_array, each_in_collection in zip(
        data_point_array, data_point_collection, strict=True
    ):
        assert all(each_in_array == each_in_collection.get_coordinates())
