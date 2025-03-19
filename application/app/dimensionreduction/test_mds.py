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

"""Tests for `mds` module."""

# pylint: disable=duplicate-code  # OK for tests.

from collections.abc import Collection
from typing import Any

import pytest

import numpy as np

from direct.showbase.ShowBase import ShowBase

import pylog

from ..datasource.api import ColoredDataPointRectangles, DataPoint
from ..dimensionreducer import DimensionReducerCoordinator
from ..dimensionreduction.api import DimensionReducer
from ..mathutils import NPArray, magnitude
from ..nodes.visualnode import VisualNodeCollection
from .mds import OrientedMDS, OrientedPCoA

_logger = pylog.getLogger(__name__)


def _generate_data_points(
    size: tuple[int, int], scale: float, generator: np.random.Generator
) -> Collection[DataPoint]:
    array = generator.uniform(low=-scale, high=scale, size=size)

    dummy_color = [0, 0, 0]

    data_points = [
        ColoredDataPointRectangles(each, attributes=dummy_color, raw_color=True)
        for each in array
    ]

    return data_points


def _to_visual_nodes(
    base: ShowBase,
    data_points: Collection[DataPoint],
    random_generator: np.random.Generator,
) -> VisualNodeCollection:
    collection = VisualNodeCollection(base.render, base.loader)
    collection.add_data_points(
        data_points,
        visual_size=1.0,
        detail=DataPoint.Detail.MEDIUM,
        number_of_new_detailed=0,
        point_to_camera=None,
        random_generator=random_generator,
    )

    return collection


def _generate_visual_nodes(
    base: ShowBase, size: tuple[int, int], scale: float, generator: np.random.Generator
) -> VisualNodeCollection:
    data_points = _generate_data_points(size, scale, generator)
    visual_nodes = _to_visual_nodes(base, data_points, random_generator=generator)
    return visual_nodes


def _print_distances(array: NPArray) -> None:
    count = array.shape[0]

    distances = np.zeros((count, count))

    for first in range(count):
        for second in range(first):
            each_distance = magnitude(array[first, :] - array[second, :])
            distances[first, second] = each_distance
            distances[second, first] = each_distance

    _logger.info("distances: %r", distances)


@pytest.mark.parametrize(
    'cls, first_arguments, second_arguments',
    [
        (
            OrientedMDS,
            {'eps': 1e-5, 'random_seed': 5679},
            {'eps': 1e-5, 'random_seed': 5680},
        ),
        (OrientedPCoA, {}, {}),
    ],
)
def test__stability(
    base: ShowBase,
    cls: type[DimensionReducer],
    first_arguments: dict[str, Any],
    second_arguments: dict[str, Any],
) -> None:
    """Test that MDS implementations produces stable results."""

    def _calculate_error(
        visual_nodes: VisualNodeCollection, new_positions_3d: NPArray
    ) -> float:
        def _calculate_error_from_statistics(
            statistics: DimensionReducerCoordinator._Statistics,
        ) -> float:
            # Taken from `DimensionReducerCoordinator.setup_local_visual_nodes()`
            assert statistics.sum_squared_dislocation is not None
            assert statistics.variance_dislocated is not None

            mean_squared_dislocation = (
                statistics.sum_squared_dislocation / statistics.dislocation_count
            )
            standardized_mean_squared_dislocation = (
                mean_squared_dislocation / statistics.variance_dislocated
            )
            return standardized_mean_squared_dislocation

        statistics = DimensionReducerCoordinator._calculate_statistics(  # noqa: SLF001
            visual_nodes, new_positions_3d
        )
        error = _calculate_error_from_statistics(statistics)
        return error

    data_size = (50, 3)

    generator = np.random.default_rng(seed=1234)
    visual_nodes = _generate_visual_nodes(
        base, size=data_size, scale=1.0, generator=generator
    )

    previous_mds = cls(**first_arguments)
    previous_result = previous_mds.fit_transform(visual_nodes, force_fit=True)
    _logger.info("previous_result: %r", previous_result)
    _logger.info("mean: %r", np.mean(previous_result, axis=0))
    _print_distances(previous_result)

    DimensionReducerCoordinator._set_positions_3d(  # noqa: SLF001
        visual_nodes, previous_result, set_new_position=False
    )

    new_mds = cls(**second_arguments)
    new_mds.set_previous(previous_mds)

    new_mds.move_center(np.zeros((3,)), np.zeros((data_size[1],)))

    new_result = new_mds.fit_transform(visual_nodes, force_fit=True)
    _logger.info("new_result: %r", new_result)
    _logger.info("mean: %r", np.mean(new_result, axis=0))
    _print_distances(new_result)

    error = _calculate_error(visual_nodes, new_result)

    assert error < 0.005  # noqa: PLR2004
    # Orthogonal Procrustes
    # data size: error
    # 5 : 0.006482135182590425
    # 10 : 0.0037255449217103664
    # 50 : 0.002192679683483986
