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

"""Tests for `visual_node.py`."""

from collections.abc import Collection, Hashable, Iterable
from typing import (
    cast,
)

import pytest

import numpy as np

from direct.showbase.ShowBase import ShowBase

from pyqoolloop.testutils import combine_lists

from ..datasource.api import (
    ColoredDataPointRectangles,
    Coordinates,
    DataPoint,
    DataPointCollection,
    NPExtremes,
)
from ..datasource.simulated import _DataPointCard
from .visualnode import (
    VisualNode,
    VisualNodeCollection,
)

# === VisualNodeCollection


def test__VisualNodeCollection__iter(base: ShowBase) -> None:
    """Test consistent iteration with :class:`~.VisualNodeCollection`."""
    generator = np.random.default_rng(seed=1234)

    collection = VisualNodeCollection(base.render, base.loader)

    for index in range(10):
        data_point = _DataPointCard([0, 0, index])
        collection.add_data_points(
            [data_point],
            visual_size=1,
            detail=DataPoint.Detail.MEDIUM,
            number_of_new_detailed=0,
            point_to_camera=None,
            random_generator=generator,
        )

    result = [*collection]

    for each, expected in zip(collection, result, strict=True):
        assert each is expected


_UPDATE_DATA_POINTS_PARAMETERS = (
    'original_points, new_points, common_count, data_point_type',
    (
        combine_lists(
            [
                [(), (), 0],
                [([1, 2, 3],), ([1, 2, 3],), 1],
                [([1, 2, 3],), (), 0],
                [([1, 2, 3], [2, 3, 4]), ([2, 3, 4],), 1],
                [([1, 2, 3], [2, 3, 4]), (), 0],
                [([1, 2, 3],), ([2, 3, 4],), 0],
                [([1, 2, 3], [2, 3, 4]), ([2, 3, 4],), 1],
                [([1, 2, 3], [2, 3, 4]), ([2, 3, 4], [3, 4, 5]), 1],
                [([2, 3, 4],), ([2, 3, 4], [3, 4, 5]), 1],
            ],
            [_DataPointCard, ColoredDataPointRectangles],
        )
        + combine_lists(
            [
                [([1.1, 2.2, 3.3],), ([1.1, 2.2, 3.3],), 1],
                [([0.1, 0.2, 0.3],), (), 0],
                [([1, 2.1, 3.2], [2.1, 3.2, 4.3]), ([2.1, 3.2, 4.3],), 1],
                [([1.0, 2.0, 3.0], [2.0, 3.0, 4.0]), (), 0],
                [([-1, -2, -3],), ([-2, -3, -4],), 0],
                [([1, 2, 3], [-2, -3, -4]), ([-2, -3, -4],), 1],
                [([1, 2, 3], [2.11, 3.11, 4.11]), ([2.11, 3.11, 4.11], [3, 4, 5]), 1],
                [([2.11, 3.11, 4.11],), ([2.11, 3.11, 4.11], [3, 4, 5]), 1],
            ],
            [_DataPointCard, ColoredDataPointRectangles],
        )
    ),
)


def _points_to_data_point_collection(
    points: Iterable[Coordinates], data_point_type: type[DataPoint]
) -> DataPointCollection:
    extremes = NPExtremes(lower=np.array([0, 0, 0]), upper=np.array([1, 1, 1]))

    collection = DataPointCollection()
    for each in points:
        data_point = data_point_type(  # type: ignore[call-arg]
            each, attributes=each, extremes=extremes
        )
        collection.add(data_point)

    return collection


@pytest.mark.parametrize(*_UPDATE_DATA_POINTS_PARAMETERS)
def test__VisualNodeCollection__update_data_points(
    base: ShowBase,
    original_points: Iterable[Coordinates],
    new_points: Iterable[Coordinates],
    common_count: int,
    data_point_type: type[DataPoint],
) -> None:
    """Test that `update_data_points()` keeps nodes that it doesn't have to discard."""

    def _same_points(
        collection: VisualNodeCollection, points: Iterable[Coordinates]
    ) -> bool:
        points_set = {tuple(each) for each in points}

        collection_set = {
            tuple(each_visual_node.get_data_point().get_coordinates())
            for each_visual_node in collection
        }

        return points_set == collection_set

    def _check_kept_visual_nodes(
        old_visual_nodes: dict[Hashable, VisualNode],
        new_collection: VisualNodeCollection,
        common_count: int,
    ) -> None:
        count = 0
        for each in new_collection:
            coordinates = tuple(each.get_data_point().get_coordinates())
            if coordinates in old_visual_nodes:
                old = old_visual_nodes[coordinates]
                assert each is old
                count += 1

        assert count == common_count

    visual_size = 0.1

    generator = np.random.default_rng(seed=6789)

    collection = VisualNodeCollection(base.render, base.loader)

    original_data_points = _points_to_data_point_collection(
        original_points, data_point_type
    )
    collection.update_data_points(
        original_data_points,
        visual_size=visual_size,
        max_number_of_detailed=0,
        random_generator=generator,
    )
    assert _same_points(collection, original_points)

    original_visual_nodes: dict[Hashable, VisualNode] = {
        tuple(each.get_data_point().get_coordinates()): each for each in collection
    }
    # Note that `VisualNode` can't be displayed once deleted from `VisualNodeCollection`

    new_data_points = _points_to_data_point_collection(new_points, data_point_type)
    collection.update_data_points(
        new_data_points,
        visual_size=visual_size,
        max_number_of_detailed=0,
        random_generator=generator,
    )
    assert _same_points(collection, new_points)

    _check_kept_visual_nodes(original_visual_nodes, collection, common_count)


_UPDATE_DATA_POINTS_SELECTION_PARAMETERS = (
    'original_points, new_points, data_point_type',
    (
        combine_lists(
            [
                [([1, 2, 3],), ([1, 2, 3],)],
                [([1, 2, 3],), ()],
                [([1, 2, 3], [2, 3, 4]), ([2, 3, 4],)],
                [([1, 2, 3], [2, 3, 4]), ()],
                [([1, 2, 3],), ([2, 3, 4],)],
                [([1, 2, 3], [2, 3, 4]), ([2, 3, 4], [3, 4, 5])],
            ],
            [_DataPointCard, ColoredDataPointRectangles],
        )
    ),
)


def test__UPDATE_DATA_POINTS_SELECTION_PARAMETERS() -> None:
    """
    Check parameters.

    For `test__VisualNodeCollection__update_data_points__selection()`.
    """
    parameters = cast(list[list[object]], _UPDATE_DATA_POINTS_SELECTION_PARAMETERS[1])

    at_least_one_inclusion = False

    at_least_one_exclusion = False

    for each in parameters:
        print(f"{each=}")

        assert isinstance(each[0], tuple)
        original_points = cast(tuple[int, ...], each[0])

        assert isinstance(each[1], tuple)
        new_points = each[1]

        first = original_points[0]

        included = first in new_points

        at_least_one_inclusion = at_least_one_inclusion or included

        at_least_one_exclusion = at_least_one_exclusion or not included

    assert at_least_one_inclusion

    assert at_least_one_exclusion


@pytest.mark.parametrize(*_UPDATE_DATA_POINTS_SELECTION_PARAMETERS)
def test__VisualNodeCollection__update_data_points__selection(
    base: ShowBase,
    original_points: Iterable[Coordinates],
    new_points: Iterable[Coordinates],
    data_point_type: type[DataPoint],
) -> None:
    """Test that Visual Node selection in `update_data_points()` is handled properly."""

    def _get_first(data_points: DataPointCollection) -> DataPoint:
        for each in data_points:
            return each

        raise AssertionError("`DataPointCollection` is empty.")

    generator = np.random.default_rng(seed=2345)

    visual_nodes = VisualNodeCollection(base.render, base.loader)

    original_data_points = _points_to_data_point_collection(
        original_points, data_point_type
    )
    visual_nodes.update_data_points(
        original_data_points,
        visual_size=1,
        max_number_of_detailed=0,
        random_generator=generator,
    )

    visual_nodes.select_data_point(_get_first(original_data_points))
    node_selection = visual_nodes.get_selected()
    assert node_selection is not None

    new_data_points = _points_to_data_point_collection(new_points, data_point_type)
    visual_nodes.update_data_points(
        new_data_points,
        visual_size=1,
        max_number_of_detailed=0,
        random_generator=generator,
    )

    if node_selection in visual_nodes:
        assert visual_nodes.get_selected() is node_selection

    else:
        assert visual_nodes.get_selected() is None


@pytest.mark.parametrize(*_UPDATE_DATA_POINTS_PARAMETERS)
def test__VisualNodeCollection__update_data_points__number_of_detailed(
    base: ShowBase,
    original_points: Collection[Coordinates],
    new_points: Collection[Coordinates],
    common_count: int,
    data_point_type: type[DataPoint],
) -> None:
    """Test `get_number_of_detailed()` after `update_data_points()`."""
    visual_size = 0.1

    generator = np.random.default_rng(seed=123)

    for each in (
        0,
        1,
        len(original_points) - 1,
        len(original_points),
        len(original_points) + 1,
        len(original_points) + len(new_points) - 1,
        len(original_points) + len(new_points),
        len(original_points) + len(new_points) + 1,
    ):
        if each < 0:
            continue

        collection = VisualNodeCollection(base.render, base.loader)

        original_data_points = _points_to_data_point_collection(
            original_points, data_point_type
        )
        collection.update_data_points(
            original_data_points,
            visual_size=visual_size,
            max_number_of_detailed=each,
            random_generator=generator,
        )

        first_number_of_detailed = min(each, len(original_points))
        assert collection.get_number_of_detailed() == first_number_of_detailed

        new_data_points = _points_to_data_point_collection(new_points, data_point_type)
        collection.update_data_points(
            new_data_points,
            visual_size=visual_size,
            max_number_of_detailed=each,
            random_generator=generator,
        )
        second_number_of_detailed = collection.get_number_of_detailed()

        assert min(each, len(new_points) - common_count) <= second_number_of_detailed
        assert second_number_of_detailed <= min(each, len(new_points))
