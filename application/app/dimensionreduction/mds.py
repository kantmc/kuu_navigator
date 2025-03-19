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

"""
Module for MDS (Multi-Dimensional Scaling).

Array type: numpy
"""

from types import NoneType
from typing import cast

from typing_extensions import override

import numpy as np
from numpy.typing import (
    ArrayLike,
)
from scipy.linalg import orthogonal_procrustes

from sklearn.manifold import MDS as SkMDS  # noqa: N811

import pylog

from ..mathutils import NPArray, np_all
from ..nodes.visualnode import VisualNodeCollection
from ._algorithm.api import MDSProtocol
from ._algorithm.mds import PCoA
from .api import DimensionReducer

_logger = pylog.getLogger(__name__)


class _OrientedMDS(DimensionReducer):
    """
    Common implementation of Oriented MDS.

    :param mds: The actual (not oriented) MDS algorithm to use.
    """

    def __init__(self, mds: MDSProtocol) -> None:
        super().__init__()

        self._mds = mds
        self._previous: _OrientedMDS | None = None

    @override
    def set_previous(self, previous: DimensionReducer | None) -> None:
        assert isinstance(previous, (type(self), NoneType)), (
            "Previous instance is not same type"
        )

        if previous is not None:
            # Release unnecessary instance. Can cause some methods to malfunction on
            # previous instance.
            previous._previous = None  # noqa: SLF001

        self._previous = previous

    @override
    def set_camera_directions(
        self,
        right_3d: ArrayLike,
        forward_3d: ArrayLike,
        up_3d: ArrayLike,
    ) -> None: ...

    def _fit_transform(self, nodes: VisualNodeCollection) -> NPArray:
        data_point_array = nodes.coordinates_to_array()
        return cast(NPArray, self._mds.fit_transform(data_point_array))

    class _NoSharedNodesError(Exception): ...

    def _fit_transform_with_orientation(self, nodes: VisualNodeCollection) -> NPArray:
        def _fit_transform(nodes: VisualNodeCollection) -> NPArray:
            data_point_array = nodes.coordinates_to_array()
            transformed_array_3d = self._mds.fit_transform(data_point_array)
            return cast(NPArray, transformed_array_3d)

        def _subtract_vector(matrix: NPArray, offset: NPArray) -> NPArray:
            return matrix - offset.reshape((1, offset.shape[0]))

        def _calculate_rotation_matrix_procrustes(
            new_coordinates_3d: NPArray, old_coordinates_3d: NPArray
        ) -> NPArray:
            # Kabsh algorithm doesn't handle reflection, only rotation.
            rotation_matrix, _ = orthogonal_procrustes(
                old_coordinates_3d, new_coordinates_3d
            )
            return cast(NPArray, rotation_matrix)

        def _calculate_affine(
            new_coordinates_3d: NPArray,
            old_coordinates_3d: NPArray,
            old_center_3d: NPArray,
        ) -> tuple[NPArray, NPArray]:
            assert self._center is not None

            center_index: int | None = None
            for index, each in enumerate(old_coordinates_3d):
                if np_all(each == old_center_3d):
                    center_index = index
                    break

            if center_index is not None:
                new_center_3d = new_coordinates_3d[center_index, :]

            else:
                new_center_3d = np.mean(new_coordinates_3d, axis=0)

            new_coordinates_centered_3d = _subtract_vector(
                new_coordinates_3d, new_center_3d
            )
            old_coordinates_centered_3d = _subtract_vector(
                old_coordinates_3d, self._center.visual_3d
            )

            rotation_matrix_3d = _calculate_rotation_matrix_procrustes(
                new_coordinates_centered_3d, old_coordinates_centered_3d
            )
            return rotation_matrix_3d, new_center_3d

        new_data_point_array_3d = _fit_transform(nodes)
        _logger.info("len(new_data_point_array)= %d", len(new_data_point_array_3d))
        old_coordinates_3d = nodes.coordinates_to_list_3d()

        shared_indexes = [each is not None for each in old_coordinates_3d]

        new_data_point_array_shared_3d = new_data_point_array_3d[shared_indexes]
        _logger.info(
            "len(new_data_point_array_filtered)= %d",
            len(new_data_point_array_shared_3d),
        )

        if len(new_data_point_array_shared_3d) == 0:
            raise _OrientedMDS._NoSharedNodesError

        # `np.array(old_coordinates)` is `numpy.array` of `LPoint3f | None`
        old_coordinates_array_shared_3d = np.array(
            list(np.array(old_coordinates_3d, dtype=object)[shared_indexes]),
            dtype=np.float64,
        )

        assert self._center is not None

        rotation_matrix_3d, translation_3d = _calculate_affine(
            new_data_point_array_shared_3d,
            old_coordinates_array_shared_3d,
            self._center.visual_3d,
        )

        centered_3d = _subtract_vector(new_data_point_array_3d, translation_3d)

        result_3d = _subtract_vector(
            centered_3d @ rotation_matrix_3d.transpose(),
            -self._center.visual_3d,
        )
        return result_3d

    @override
    def fit_transform(self, nodes: VisualNodeCollection, *, force_fit: bool) -> NPArray:
        if self._previous is not None:
            try:
                return self._fit_transform_with_orientation(nodes)

            except _OrientedMDS._NoSharedNodesError:
                pass

        return self._fit_transform(nodes)


@DimensionReducer.factory.register
class OrientedMDS(_OrientedMDS):
    """
    Implements Multi-Dimensional Scaling (MDS) with orientation alignment.

    :param eps: Relative tolerance with respect to stress at which to declare
      convergence. The default value specified by Scikit Learn (1e-3) is rather weak.
    :param random_seed: Seed for random number generator.

    .. note::
      - :meth:`move_center()` will be ignored.
      - Current implementation uses Euclidean metric. `--shared-neighborhood`
        needs to be enabled.
    """

    def __init__(
        self,
        eps: float,
        random_seed: int | None,
    ) -> None:
        self._my_mds = SkMDS(
            n_components=3,
            random_state=random_seed,
            normalized_stress="auto",
            eps=eps,
            n_jobs=-2,
        )
        super().__init__(self._my_mds)

    @override
    def fit_transform(self, nodes: VisualNodeCollection, *, force_fit: bool) -> NPArray:
        result_3d = super().fit_transform(nodes, force_fit=force_fit)

        _logger.info("MDS iterations: %d", self._my_mds.n_iter_)
        _logger.info("MDS stress: %f", self._my_mds.stress_)

        return result_3d


@DimensionReducer.factory.register
class OrientedPCoA(_OrientedMDS):
    """
    Implements Classical Multi-Dimensional Scaling (cMDS) with orientation alignment.

    .. note::
      - Classical Multi-Dimensional Scaling is also known as Principal Coordinate
        Analysis (PCoA).
      - :meth:`move_center()` will be ignored.
      - Current implementation uses Euclidean metric. `--shared-neighborhood`
        needs to be enabled.
    """

    def __init__(self) -> None:
        mds = PCoA(dimensions=3)
        super().__init__(mds)
