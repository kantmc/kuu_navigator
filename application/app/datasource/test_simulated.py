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

"""Tests for `simulated.py`."""

from collections.abc import (
    Callable,
    Iterable,
    Sequence,
)
from functools import partial, reduce
from math import (
    inf,
    sqrt,
)
import operator
from sys import float_info
from typing import Any, cast
from warnings import warn

import pytest

import numpy as np
from numpy.testing import assert_allclose

from pyqoolloop.testutils import combine_lists

from ..mathutils import NPArray
from ..testutils import TestMode
from .api import (
    Coordinates,
    DataPointCollection,
    DataSource,
    distance_between,
)
from .simulated import (
    Desk,
    Euclidean,
    Gaussian,
    GridData,
    RandomData,
    SwissRoll,
    _nested_loop,
)

# === _nestedLoop()


def test__nested_loop__1d() -> None:
    """Test :func:`~.datasource._nested_loop()` with 1D arguments."""
    for range_max in [1, 5]:
        for each_from_generator, each_from_range in zip(
            _nested_loop([0], [range_max]), range(range_max + 1), strict=True
        ):
            assert each_from_generator == [each_from_range]


def test__nested_loop__2d() -> None:
    """Test :func:`~.datasource._nested_loop()` with 2D arguments."""
    mins = [-1, -2]
    maxs = [4, 3]

    candidates = set()
    for i in range(mins[0], maxs[0] + 1):
        for j in range(mins[1], maxs[1] + 1):
            candidates.add((i, j))

    for each in _nested_loop(mins, maxs):
        each_tuple = cast(tuple[int, int], tuple(each))
        candidates.remove(each_tuple)

    assert len(candidates) == 0


def test__nested_loop__3d() -> None:
    """Test :func:`~.datasource._nested_loop()` with 3D arguments."""
    mins = [-1, -2, 0]
    maxs = [4, 3, 7]

    candidates = set()
    for i in range(mins[0], maxs[0] + 1):
        for j in range(mins[1], maxs[1] + 1):
            for k in range(mins[2], maxs[2] + 1):
                candidates.add((i, j, k))

    for each in _nested_loop(mins, maxs):
        each_tuple = cast(tuple[int, int, int], tuple(each))
        candidates.remove(each_tuple)

    assert len(candidates) == 0


# === DataSource

_MAX_DIMENSIONS = 6

_MINS_MAXS_1D_PARAMETERS = (
    # mins: Sequence[int], maxs: Sequence[int]
    ([0], [1]),
    ([0], [2]),
)

_MINS_MAXS_2D_PARAMETERS = (
    # mins: Sequence[int], maxs: Sequence[int]
    ([0] * 2, [1] * 2),
    ([-5] * 2, [5] * 2),
)

_MINS_MAXS_3D_PARAMETERS = (
    # mins: Sequence[int], maxs: Sequence[int]
    ([-5] * 3, [5] * 3),
)

_MINS_MAXS_PARAMETERS_PER_DIMENSION = [
    _MINS_MAXS_1D_PARAMETERS,
    _MINS_MAXS_2D_PARAMETERS,
    _MINS_MAXS_3D_PARAMETERS,
    (),
    (),
    (),
]

_MINS_MAXS_ALL_PARAMETERS = sum(_MINS_MAXS_PARAMETERS_PER_DIMENSION, ())


_CENTER_PARAMETERS_PER_DIMENSION = [
    # [ center: Sequence[float] ]
    [[[0] * (dimensions_1 + 1)], [[0.5] * (dimensions_1 + 1)]]
    for dimensions_1 in range(_MAX_DIMENSIONS)
]


_RADIUS_PARAMETERS = [
    # radius: float
    10,
]


_MAX_NUMBER_PARAMETERS = [
    # max_number: int
    1,
    2,
    3,
    100,
]


@pytest.mark.parametrize(
    'int_mins, int_maxs',
    _MINS_MAXS_ALL_PARAMETERS,
)
def test__GridData__get_total_number(
    int_mins: Sequence[int], int_maxs: Sequence[int]
) -> None:
    """Test :meth:`.GridData.get_total_number()` returns expected number."""

    def _count_total_number(int_mins: Sequence[int], int_maxs: Sequence[int]) -> int:
        count = 0
        for _ in _nested_loop(int_mins, int_maxs):
            count += 1

        return count

    grid_data = GridData(int_mins=int_mins, int_maxs=int_maxs)
    total_number = grid_data.get_total_number()
    assert total_number == _count_total_number(int_mins, int_maxs)


@pytest.mark.parametrize(
    'mins, maxs',
    _MINS_MAXS_ALL_PARAMETERS,
)
def test__GridData__get_total_number__matches_get_from_all(
    mins: Sequence[int], maxs: Sequence[int]
) -> None:
    """
    Test that `GridData.get_total_number()`_ is consistent with `get_from_all()`_.

    .. _`GridData.get_total_number()`:
      ../datasource/index.html#app.datasource.GridData.get_total_number
    .. _`get_from_all()`:
      ../datasource/index.html#app.datasource.GridData.get_from_all
    """
    grid_data = GridData(int_mins=mins, int_maxs=maxs)
    total_number = grid_data.get_total_number()

    all_points = grid_data.get_from_all(2 * total_number)
    assert total_number == len(all_points)

    all_points = grid_data.get_from_all(total_number)
    assert total_number == len(all_points)


@pytest.mark.parametrize(
    'mins, maxs',
    _MINS_MAXS_ALL_PARAMETERS,
)
def test__GridData__get_from_all__small_max_number(
    mins: Sequence[int], maxs: Sequence[int]
) -> None:
    """Test that `GridData.get_from_all()` returns expected number with small max."""
    grid_data = GridData(int_mins=mins, int_maxs=maxs)

    total_number = grid_data.get_total_number()

    for divisor in [2, 3, 5, 7]:
        max_number = total_number // divisor
        all_points = grid_data.get_from_all(max_number)

        assert len(all_points) <= max_number
        assert max_number - 1 <= len(all_points)


_SCALE_PARAMETERS_PER_DIMENSION = [
    # Power of 2 to avoid calculation error.
    [[pow(2, i - 2) for i in range(dimensions_1 + 1)]]
    for dimensions_1 in range(5)
] + [[]]


def _inject_shape(
    parameters: list[list[object]],
    shapes: Iterable[DataSource.Shape] = tuple(DataSource.Shape),
) -> list[list[object]]:
    return [
        [
            parameters[0],  # constructor
            cast(dict[str, Any], parameters[1])  # arguments
            | {"neighborhood_shape": shape.name.lower()},
            *parameters[2:],
        ]
        for parameters in parameters
        for shape in shapes
    ]


_GRID_DATA__CONSTRUCTORS_PER_DIMENSION = [
    # [ constructor: Callable[..., DataSource], arguments: dict[str, Any] ]
    combine_lists(
        GridData,
        [
            {'int_mins': mins, 'int_maxs': maxs}
            for mins, maxs in _MINS_MAXS_PARAMETERS_PER_DIMENSION[dimensions_1]
        ],
        raise_if_empty=False,
    )
    + combine_lists(
        GridData,
        [
            {'int_mins': mins, 'int_maxs': maxs, "scale": scale}
            for mins, maxs in _MINS_MAXS_PARAMETERS_PER_DIMENSION[dimensions_1]
            for scale in _SCALE_PARAMETERS_PER_DIMENSION[dimensions_1]
        ],
        raise_if_empty=False,
    )
    for dimensions_1 in range(_MAX_DIMENSIONS)
]


_RANDOM_DATA__CONSTRUCTORS_PER_DIMENSION = [
    # [ constructor: Callable[..., DataSource], arguments: dict[str, Any] ]
    combine_lists(
        RandomData,
        [
            {'count': 100, 'mins': mins, 'maxs': maxs}
            for mins, maxs in _MINS_MAXS_PARAMETERS_PER_DIMENSION[dimensions_1]
        ],
        raise_if_empty=False,
    )
    for dimensions_1 in range(_MAX_DIMENSIONS)
]

_GAUSSIAN__CONSTRUCTORS_PER_DIMENSION = [
    # [ constructor: Callable[..., DataSource], arguments: dict[str, Any] ]
    combine_lists(
        Gaussian,
        [
            {"scale": scale, "count": 100, "color_radius": 1}
            for scale in scale_parameters
        ],
        raise_if_empty=False,
    )
    for scale_parameters in _SCALE_PARAMETERS_PER_DIMENSION
]


_SWISS_ROLL__CONTRUCTORS_PER_DIMENSION = [
    (
        []
        if dimensions_1 < 3 - 1
        else combine_lists(
            SwissRoll,
            [
                {
                    "count": 100,
                    "start_radius": 5,
                    "fall_rate": 0.5,
                    "rotations": 4,
                    "widths": [10],
                    "virtual_dimensions": dimensions_1 + 1,
                }
            ],
        )
    )
    for dimensions_1 in range(_MAX_DIMENSIONS)
]


# Constructors for all `DataSource` classes.
_DETERMINISTIC_CONSTRUCTORS_LIST_PER_DATA_SOURCE: tuple[list[list[object]], ...] = (
    _GRID_DATA__CONSTRUCTORS_PER_DIMENSION,
)

_RANDOM_CONSTRUCTORS_LIST_PER_DATA_SOURCE: tuple[list[list[object]], ...] = (
    _RANDOM_DATA__CONSTRUCTORS_PER_DIMENSION,
    _GAUSSIAN__CONSTRUCTORS_PER_DIMENSION,
    _SWISS_ROLL__CONTRUCTORS_PER_DIMENSION,
)

_CONSTRUCTORS_LIST_PER_DATA_SOURCE: tuple[list[list[object]], ...] = tuple(
    list(_DETERMINISTIC_CONSTRUCTORS_LIST_PER_DATA_SOURCE)
    + list(_RANDOM_CONSTRUCTORS_LIST_PER_DATA_SOURCE)
)


# Constructors for `DataSource` classes with minimum/maximum boundaries
_MIN_MAX_CONSTRUCTORS_LIST_PER_DATA_SOURCE: tuple[list[list[object]], ...] = (
    _GRID_DATA__CONSTRUCTORS_PER_DIMENSION,
    _RANDOM_DATA__CONSTRUCTORS_PER_DIMENSION,
    # FUTURE: _SWISS_ROLL__CONTRUCTORS_PER_DIMENSION,
)


def _flatten_list_per_data_source(
    list_per_data_source: tuple[list[list[object]], ...],
) -> list[list[object]]:
    return [
        cast(list[object], constructor_and_argument)
        for per_data_source in list_per_data_source
        for constructor_list in per_data_source
        for constructor_and_argument in constructor_list
    ]


def _get_from_neighorhood_parameters(
    constructors_list_per_data_source: tuple[list[list[object]], ...],
) -> list[list[object]]:
    # constructor: Callable[..., DataSource], arguments: dict[str, Any],
    # center: Sequence[float], radius: float
    return reduce(
        operator.iadd,
        (
            cast(
                list[list[object]],
                combine_lists(
                    constructors_list_per_dimension[dimensions_1],
                    _CENTER_PARAMETERS_PER_DIMENSION[dimensions_1],
                    _RADIUS_PARAMETERS,
                    raise_if_empty=False,
                ),
            )
            for constructors_list_per_dimension in constructors_list_per_data_source
            for dimensions_1 in range(_MAX_DIMENSIONS)
        ),
        [],
    )


_DETERMINISTIC_DATA_SOURCE__GET_FROM_NEIGHBORHOOD_PARAMETERS = _inject_shape(
    _get_from_neighorhood_parameters(_DETERMINISTIC_CONSTRUCTORS_LIST_PER_DATA_SOURCE)
)

_RANDOM_DATA_SOURCE__GET_FROM_NEIGHBORHOOD_PARAMETERS = _inject_shape(
    _get_from_neighorhood_parameters(_RANDOM_CONSTRUCTORS_LIST_PER_DATA_SOURCE)
)

_DATA_SOURCE__GET_FROM_NEIGHBORHOOD_PARAMETERS = _inject_shape(
    _get_from_neighorhood_parameters(_CONSTRUCTORS_LIST_PER_DATA_SOURCE)
)


_MIN_MAX_DATA_SOURCE__GET_FROM_NEIGHBORHOOD_PARAMETERS = _inject_shape(
    _get_from_neighorhood_parameters(_MIN_MAX_CONSTRUCTORS_LIST_PER_DATA_SOURCE)
)


_DETERMINISTIC_DATA_SOURCE__GET_FROM_NEIGHBORHOOD_PARAMETERS_WITH_MAX_NUMBER = (
    combine_lists(
        # constructor: Callable[..., DataSource], arguments: dict[str, Any],
        # center: Sequence[float], radius: float, max_number: int
        _DETERMINISTIC_DATA_SOURCE__GET_FROM_NEIGHBORHOOD_PARAMETERS,
        _MAX_NUMBER_PARAMETERS,
    )
)

_RANDOM_DATA_SOURCE__GET_FROM_NEIGHBORHOOD_PARAMETERS_WITH_MAX_NUMBER = combine_lists(
    # constructor: Callable[..., DataSource], arguments: dict[str, Any],
    # center: Sequence[float], radius: float, max_number: int
    _RANDOM_DATA_SOURCE__GET_FROM_NEIGHBORHOOD_PARAMETERS,
    _MAX_NUMBER_PARAMETERS,
)

_DATA_SOURCE__GET_FROM_NEIGHBORHOOD_PARAMETERS_WITH_MAX_NUMBER = combine_lists(
    # constructor: Callable[..., DataSource], arguments: dict[str, Any],
    # center: Sequence[float], radius: float, max_number: int
    _DATA_SOURCE__GET_FROM_NEIGHBORHOOD_PARAMETERS,
    _MAX_NUMBER_PARAMETERS,
)


def _equal_collection(one: DataPointCollection, another: DataPointCollection) -> bool:
    for each_in_self, each_in_other in zip(one, another, strict=True):
        if each_in_self != each_in_other:
            return False

    return True


def _test__consistency(
    function: Callable[..., DataPointCollection],
    args: list[Any],
    *,
    test_mode: TestMode,
) -> None:
    data_points = function(*args)
    if test_mode == TestMode.TestTest:
        assert len(data_points) > 0, (
            "Don't choose arguments that consistently produce no Data Points."
        )

    new_data_points = function(*args)
    assert _equal_collection(data_points, new_data_points)


@pytest.mark.parametrize(
    'constructor, arguments, max_number',
    combine_lists(
        _inject_shape(
            _flatten_list_per_data_source(_MIN_MAX_CONSTRUCTORS_LIST_PER_DATA_SOURCE)
        ),
        _MAX_NUMBER_PARAMETERS,
    ),
)
def test__min_max_datasource__get_from_all(
    test_mode: TestMode,
    constructor: Callable[..., DataSource],
    arguments: dict[str, Any],
    max_number: int,
) -> None:
    """
    Test :meth:`~.DataSource.get_from_all()` respects mins and maxs.

    Tests with :mod:`DataSource <app.datasource>` subclasses with min/max initializers.
    """
    data_source = constructor(**arguments)

    mins = arguments.get('mins')
    if mins is None:
        mins = arguments['int_mins']

    maxs = arguments.get('maxs')
    if maxs is None:
        maxs = arguments['int_maxs']

    data_points = data_source.get_from_all(max_number)
    if test_mode == TestMode.TestTest:
        assert len(data_points) > 0, (
            "Don't choose arguments that consistently produce no Data Points."
        )

    for each in data_points:
        coordinates: NPArray = np.array(each.get_coordinates())

        each_mins = np.minimum(coordinates, mins)
        assert (each_mins == mins).all()

        each_maxs = np.maximum(coordinates, maxs)
        assert (each_maxs == maxs).all()


@pytest.mark.parametrize(
    'constructor, arguments, max_number',
    combine_lists(
        _inject_shape(
            _flatten_list_per_data_source(_CONSTRUCTORS_LIST_PER_DATA_SOURCE)
        ),
        _MAX_NUMBER_PARAMETERS,
    ),
)
def test__DataSource__get_from_all__consistent(
    test_mode: TestMode,
    constructor: Callable[..., DataSource],
    arguments: dict[str, Any],
    max_number: int,
) -> None:
    """Test that :meth:`~.DataSource.get_from_all()` returns consistent data points."""
    data_source = constructor(**arguments)
    _test__consistency(data_source.get_from_all, [max_number], test_mode=test_mode)


@pytest.mark.parametrize(
    'constructor, arguments, center, radius, max_number',
    _DATA_SOURCE__GET_FROM_NEIGHBORHOOD_PARAMETERS_WITH_MAX_NUMBER,
)
def test__DataSource__get_from_neighborhood__small_max_number(
    test_mode: TestMode,
    constructor: Callable[..., DataSource],
    arguments: dict[str, Any],
    center: Coordinates,
    radius: float,
    max_number: int,
) -> None:
    """Test number of data points from :meth:`~.DataSource.get_from_neighborhood()`."""

    def _total_in_range(
        data_source: DataSource,
        center: Coordinates,
        radius: float,
    ) -> int:
        some_large_number = 10000

        total_number = data_source.get_total_number()
        assert total_number is not None
        assert total_number <= some_large_number, (
            "Shouldn't test with too large `DataSource`."
        )

        all_points = data_source.get_from_all(total_number)
        count = 0
        for each in all_points:
            within_distance = True
            for each_in_coordinates, each_in_point in zip(
                each.get_coordinates(), center, strict=True
            ):
                if abs(each_in_coordinates - each_in_point) > radius:
                    within_distance = False

            if within_distance:
                count += 1

        assert count > 0
        return count

    data_source = constructor(**arguments)

    total_in_range = _total_in_range(data_source, center, radius)
    neighborhood = data_source.get_from_neighborhood(center, radius, max_number)

    if test_mode == TestMode.TestTest:
        assert len(neighborhood) > 0, (
            "Don't choose arguments that consistently produce no Data Points."
        )

    assert len(neighborhood) <= min(max_number, total_in_range)


@pytest.mark.parametrize(
    'constructor, arguments, center, radius, max_number',
    [
        (
            GridData,
            {
                'int_mins': [-1] * 10,
                'int_maxs': [1] * 10,
                'neighborhood_shape': 'sphere',
            },
            [0] * 10,
            2,
            10,
        ),
        (
            GridData,
            {
                'int_mins': [-1] * 10,
                'int_maxs': [1] * 10,
                'neighborhood_shape': 'sphere',
            },
            [1] * 10,
            2,
            10,
        ),
        (
            GridData,
            {
                'int_mins': [-1] * 200,
                'int_maxs': [1] * 200,
                'neighborhood_shape': 'cube',
            },
            [1] * 200,
            20,
            100,
        ),
    ],
)
def test__DataSource__get_from_neighborhood__max_number(
    test_mode: TestMode,
    constructor: Callable[..., DataSource],
    arguments: dict[str, Any],
    center: Coordinates,
    radius: float,
    max_number: int,
) -> None:
    """Test number of data points from :meth:`~.DataSource.get_from_neighborhood()`."""
    data_source = constructor(**arguments)

    neighborhood = data_source.get_from_neighborhood(center, radius, max_number)

    if test_mode == TestMode.TestTest:
        assert len(neighborhood) > 0, (
            "Don't choose arguments that consistently produce no Data Points."
        )

    assert len(neighborhood) <= max_number


@pytest.mark.parametrize(
    'constructor, arguments, center, radius, max_number',
    _DATA_SOURCE__GET_FROM_NEIGHBORHOOD_PARAMETERS_WITH_MAX_NUMBER,
)
def test__DataSource__get_from_neighborhood__consistent(
    constructor: Callable[..., DataSource],
    arguments: dict[str, Any],
    center: Coordinates,
    radius: float,
    max_number: int,
    test_mode: TestMode,
) -> None:
    """
    Test that `get_from_neighborhood()`_ returns consistent data points.

    .. _`get_from_neighborhood()`:
      ../datasource/index.html#app.datasource.DataSource.get_from_neighborhood
    """
    data_source = constructor(**arguments)
    _test__consistency(
        partial(data_source.get_from_neighborhood, center, radius),
        [max_number],
        test_mode=test_mode,
    )


@pytest.mark.parametrize(
    'constructor, arguments, center, radius',
    _DATA_SOURCE__GET_FROM_NEIGHBORHOOD_PARAMETERS,
)
def test__DataSource__get_from_neighborhood__include_boundary(
    constructor: Callable[..., DataSource],
    arguments: dict[str, Any],
    center: Coordinates,
    radius: float,
) -> None:
    """
    Test `get_from_neighborhood()`_ returns furthest point in range.

    .. _`get_from_neighborhood()`:
      ../datasource/index.html#app.datasource.DataSource.get_from_neighborhood
    """

    def _get_furthest_distance_within_range(
        data_source: DataSource, center: Coordinates, radius: float
    ) -> float:
        total_number = data_source.get_total_number()
        assert total_number is not None
        all_points = data_source.get_from_all(total_number)
        assert len(all_points) > 0

        count = 0
        maximum = -inf
        for each in all_points:
            distance = distance_between(each.get_coordinates(), center)
            if distance <= radius:
                maximum = max(maximum, distance)
                count += 1

        return maximum

    data_source = constructor(**arguments)

    furthest_distance_within_range = _get_furthest_distance_within_range(
        data_source, center, radius
    )

    total_number = data_source.get_total_number()
    assert total_number is not None
    neighborhood = data_source.get_from_neighborhood(center, radius, total_number)

    if furthest_distance_within_range == -inf:
        if arguments['neighborhood_shape'] == 'sphere':
            assert len(neighborhood) == 0

        for each in neighborhood:
            distance = distance_between(each.get_coordinates(), center)
            assert distance > radius

        return

    has_furthest = False
    for each in neighborhood:
        distance = distance_between(each.get_coordinates(), center)
        if distance == furthest_distance_within_range:
            has_furthest = True

    assert has_furthest


@pytest.mark.parametrize(
    'constructor, arguments, center, radius',
    [
        *_MIN_MAX_DATA_SOURCE__GET_FROM_NEIGHBORHOOD_PARAMETERS,
        (GridData, {'int_mins': [-1] * 200, 'int_maxs': [1] * 200}, [0] * 200, 2),
        (GridData, {'int_mins': [-1] * 200, 'int_maxs': [1] * 200}, [1] * 200, 2),
    ],
)
def test__DataSource__get_from_neighborhood__outside_range(
    constructor: Callable[..., DataSource],
    arguments: dict[str, Any],
    center: Coordinates,
    radius: float,
) -> None:
    """
    Test `get_from_neighborhood()`_ returns empty list, if out of range.

    .. _`get_from_neighborhood()`:
      ../datasource/index.html#app.datasource.DataSource.get_from_neighborhood
    """

    def _calc_points_outside_range(
        mins: Sequence[int], maxs: Sequence[int], center: Coordinates, radius: float
    ) -> list[Coordinates]:
        eps = 2 * float_info.epsilon
        point_list: list[Coordinates] = []
        for index, (minimum, maximum) in enumerate(zip(mins, maxs, strict=True)):
            new_point = list(center)
            new_point[index] = (minimum - radius) - abs(minimum - radius) * eps
            point_list.append(cast(Coordinates, new_point))

            new_point = list(center)
            new_point[index] = (maximum + radius) + abs(maximum + radius) * eps
            point_list.append(cast(Coordinates, new_point))

        return point_list

    data_source = constructor(**arguments)

    total_number = data_source.get_total_number()
    assert total_number is not None

    mins = arguments.get('mins')
    if mins is None:
        mins = arguments['int_mins']

    maxs = arguments.get('maxs')
    if maxs is None:
        maxs = arguments['int_maxs']

    for each in _calc_points_outside_range(mins, maxs, center, radius):
        neighborhood = data_source.get_from_neighborhood(each, radius, total_number)
        assert len(neighborhood) == 0


@pytest.mark.parametrize(
    'constructor, arguments, center, max_number, radius_before, radius_after, atol',
    [
        (
            GridData,
            {
                'int_mins': [-5] * 200,
                'int_maxs': [5] * 200,
                'random_seed': 1234,
                'neighborhood_shape': "sphere",
            },
            [0] * 200,
            3000,
            100.0,
            1000.0,
            1.0,
        ),
        (
            GridData,
            {
                'int_mins': [-5] * 200,
                'int_maxs': [5] * 200,
                'random_seed': 1234,
                'neighborhood_shape': "cube",
            },
            [0] * 200,
            3000,
            10.0,
            100.0,
            10.0,
        ),
        (
            SwissRoll,
            {
                'count': 20000,
                'start_radius': 100,
                'fall_rate': 0.3,
                'rotations': 2,
                'widths': [40],
                'virtual_dimensions': 3,
                'random_seed': 1234,
            },
            [0] * 3,
            1000,
            100.0 * 0.7,
            100.0,
            10.0,
        ),
        (
            Desk,
            {
                'connection_length': 10.0,
                'connection_count': 2000,
                'cube_lengths': [10, 9, 8, 7, 6],
                'cube_count': 10000,
                'neighborhood_shape': "sphere",
                'random_seed': 1234,
            },
            [5, 4.5, 4, 3.5, 3],
            1000,
            5.0,
            50.0,
            11.0,
        ),
        (
            Desk,
            {
                'connection_length': 10.0,
                'connection_count': 2000,
                'cube_lengths': [10, 9, 8, 7, 6],
                'cube_count': 10000,
                'neighborhood_shape': "sphere",
                'random_seed': 1234,
            },
            [10 + 10 + 5, 4.5, 4, 3.5, 3],
            1000,
            5.0,
            50.0,
            11.0,
        ),
    ],
)
def test__DataSource__get_from_neighborhood__unbiased(
    constructor: Callable[..., DataSource],
    arguments: dict[str, Any],
    center: Coordinates,
    max_number: int,
    radius_before: float,
    radius_after: float,
    atol: float,
) -> None:
    """Test that increasing neighborhood radius doesn't change distribution."""
    data_source = constructor(**arguments)

    results = data_source.get_from_neighborhood(center, radius_before, max_number)

    mean_before = results.to_array().mean(axis=0)

    results = data_source.get_from_neighborhood(center, radius_after, max_number)

    mean_after = results.to_array().mean(axis=0)

    assert_allclose(
        mean_before,
        mean_after,
        atol=atol,
    )


@pytest.mark.parametrize(
    'constructor, arguments, center, radius, max_number',
    _DETERMINISTIC_DATA_SOURCE__GET_FROM_NEIGHBORHOOD_PARAMETERS_WITH_MAX_NUMBER,
)
def test__deterministic_DataSource__get_from_neighborhood__within_range(
    test_mode: TestMode,
    constructor: Callable[..., DataSource],
    arguments: dict[str, Any],
    center: Coordinates,
    radius: float,
    max_number: int,
) -> None:
    """
    Test that `get_from_neighborhood()`_ returns points within distance.

    "Distance" being `sqrt(dimensions) * radius`, which is the distance from the
    center to a corner of the cube of width `2 * radius`.

    .. note:: for deterministic Data Sources

    .. _`get_from_neighborhood()`:
      ../datasource/index.html#app.datasource.DataSource.get_from_neighborhood
    """
    data_source = constructor(**arguments)
    dimensions = len(center)

    results = data_source.get_from_neighborhood(center, radius, max_number)
    if test_mode == TestMode.TestTest:
        assert len(results) > 0, (
            "Don't choose arguments that consistently produce no Data Points."
        )

    for each in results:
        distance = distance_between(each.get_coordinates(), center)
        assert distance <= sqrt(dimensions) * radius


@pytest.mark.parametrize(
    'constructor, arguments, center, radius, max_number',
    _RANDOM_DATA_SOURCE__GET_FROM_NEIGHBORHOOD_PARAMETERS_WITH_MAX_NUMBER,
)
def test__random_DataSource__get_from_neighborhood__within_range(
    constructor: Callable[..., DataSource],
    arguments: dict[str, Any],
    center: Coordinates,
    radius: float,
    max_number: int,
) -> None:
    """
    Test that `get_from_neighborhood()` returns points within distance.

    "Distance" being `sqrt(dimensions) * radius`, which is the distance from the
    center to a corner of the cube of width `2 * radius`.

    .. note:: for random Data Sources
    """
    data_source = constructor(**arguments)
    dimensions = len(center)

    results = data_source.get_from_neighborhood(center, radius, max_number)
    if len(results) == 0:
        warn(
            UserWarning("Shouldn't test with empty `result`. Ok if only sometimes."),
            stacklevel=2,
        )

    for each in results:
        distance = distance_between(each.get_coordinates(), center)
        assert distance <= sqrt(dimensions) * radius


# FUTURE: Remove when hash equality is abandoned
@pytest.mark.parametrize(
    'int_mins, int_maxs',
    _MINS_MAXS_ALL_PARAMETERS,
)
def test__GridData__data_point_identity(
    test_mode: TestMode, int_mins: Sequence[int], int_maxs: Sequence[int]
) -> None:
    """Test that each :class:`.DataPoint` is unique for :class:`.GridData`."""
    grid_data = GridData(int_mins=int_mins, int_maxs=int_maxs)

    total_number = grid_data.get_total_number()

    all_points = grid_data.get_from_all(total_number)
    if test_mode == TestMode.TestTest:
        assert len(all_points) > 0, (
            "Don't choose arguments that consistently produce no Data Points."
        )

    past_points = set()
    for each in all_points:
        assert each not in past_points
        past_points.add(each)


@pytest.mark.parametrize(
    'grid_mins, grid_maxs, grid_scale, center, radius, expected_mins, expected_maxs',
    [
        ((-1, -2, -3), (1, 2, 3), None, (0, 0, 0), 10, (-1, -2, -3), (1, 2, 3)),
        ((-1, -2, -3), (1, 2, 3), None, (0, 0, 0), 3, (-1, -2, -3), (1, 2, 3)),
        ((-1, -2, -3), (1, 2, 3), None, (0, 0, 0), 2, (-1, -2, -2), (1, 2, 2)),
        ((-1, -2, -3), (1, 2, 3), None, (0, 0, 0), 1, (-1, -1, -1), (1, 1, 1)),
        ((-1, -2, -3), (1, 2, 3), None, (0, 0, 0), 0.9, (0, 0, 0), (0, 0, 0)),
        ((-1, -1, -1), (1, 1, 1), (1, 2, 3), (0, 0, 0), 10, (-1, -2, -3), (1, 2, 3)),
        ((-1, -1, -1), (1, 1, 1), (1, 2, 3), (0, 0, 0), 3, (-1, -2, -3), (1, 2, 3)),
        ((-1, -1, -1), (1, 1, 1), (1, 2, 3), (0, 0, 0), 2, (-1, -2, 0), (1, 2, 0)),
        ((-1, -1, -1), (1, 1, 1), (1, 2, 3), (0, 0, 0), 1, (-1, 0, 0), (1, 0, 0)),
        ((-1, -1, -1), (1, 1, 1), (1, 2, 3), (0, 0, 0), 0.9, (0, 0, 0), (0, 0, 0)),
        ((-1, -1, -1), (1, 1, 1), (1, 2, 3), (1, 2, 3), 10, (-1, -2, -3), (1, 2, 3)),
        ((-1, -1, -1), (1, 1, 1), (1, 2, 3), (1, 2, 3), 3, (-1, 0, 0), (1, 2, 3)),
        ((-1, -1, -1), (1, 1, 1), (1, 2, 3), (1, 2, 3), 2, (-1, 0, 3), (1, 2, 3)),
        ((-1, -1, -1), (1, 1, 1), (1, 2, 3), (1, 2, 3), 1, (0, 2, 3), (1, 2, 3)),
        ((-1, -1, -1), (1, 1, 1), (1, 2, 3), (1, 2, 3), 0.9, (1, 2, 3), (1, 2, 3)),
        ((-1, -1, -1), (1, 1, 1), (1, 2, 3), (-1, -2, -3), 10, (-1, -2, -3), (1, 2, 3)),
        ((-1, -1, -1), (1, 1, 1), (1, 2, 3), (-1, -2, -3), 3, (-1, -2, -3), (1, 0, 0)),
        ((-1, -1, -1), (1, 1, 1), (1, 2, 3), (-1, -2, -3), 2, (-1, -2, -3), (1, 0, -3)),
        (
            (-1, -1, -1),
            (1, 1, 1),
            (1, 2, 3),
            (-1, -2, -3),
            1,
            (-1, -2, -3),
            (0, -2, -3),
        ),
        (
            (-1, -1, -1),
            (1, 1, 1),
            (1, 2, 3),
            (-1, -2, -3),
            0.9,
            (-1, -2, -3),
            (-1, -2, -3),
        ),
    ],
)
def test__GridData__get_grid_int_min_max(
    grid_mins: Sequence[int],
    grid_maxs: Sequence[int],
    grid_scale: Sequence[float] | float | None,
    center: Coordinates,
    radius: float,
    expected_mins: Sequence[int],
    expected_maxs: Sequence[int],
) -> None:
    """Test `GridData._get_grid_min_max()`."""
    grid_data = GridData(int_mins=grid_mins, int_maxs=grid_maxs, scale=grid_scale)

    result_mins, result_maxs = grid_data._get_grid_int_min_max(center, radius)  # noqa: SLF001

    if grid_scale is None:
        assert np.all(expected_mins == result_mins)
        assert np.all(expected_maxs == result_maxs)

    else:
        assert np.all(expected_mins == result_mins * grid_scale)
        assert np.all(expected_maxs == result_maxs * grid_scale)


@pytest.mark.parametrize(
    'first, second, expected',
    [
        ((0, 0, 0), (0, 0, 0), 0),
        ((0, 0, 0), (0, 0, 1), 1),
        ((0, 2, 0), (0, 0, 0), 2),
        ((3, 0, 0), (0, 0, 0), 3),
    ],
)
def test__Euclidean(first: Coordinates, second: Coordinates, expected: float) -> None:
    """Test :class:`.Euclidean`."""
    metric = Euclidean()
    distance = metric.get_distance(first, second)
    assert expected == distance
