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
Define the common API for dimension reduction.

Array type: numpy
"""

from abc import (
    ABC,
    abstractmethod,
)

import numpy as np
from numpy.typing import ArrayLike

from pyqoolloop.factory import RegistryFactory

from ..mathutils import NPArray, OptionalVectorPair, VectorPair
from ..nodes.visualinformation import VisualInformation
from ..nodes.visualnode import VisualNodeCollection


class DimensionReducer(ABC):
    """
    API for dimension reduction.

    See :mod:`.pca` for usage.
    """

    def __init__(self) -> None:
        super().__init__()

        self._is_global: bool | None = None  # `None` means not set

        self._center: VectorPair | None = None

        self._camera_directions: OptionalVectorPair | None = None

    factory = RegistryFactory['DimensionReducer']()
    """:class:`pyqoolloop.factory.RegistryFactory` for :class:`DimensionReducer`"""

    def set_global(self, *, is_global: bool) -> None:
        """Set whether dimensionality reduction is done in Global mode."""
        self._is_global = is_global

    @abstractmethod
    def set_previous(self, previous: 'DimensionReducer | None') -> None:
        """
        Set previous instance of :class:`DimensionReducer`.

        :param previous: Previous instance. `None` for Global mode.
        """

    def get_previous(self) -> 'DimensionReducer | None':
        """
        Get the previous instance of :class:`DimensionReducer`, if available.

        :raise NotImplementedError: Doesn't keep record of the previous instance.
        """
        raise NotImplementedError

    def set_camera_directions(
        self,
        right_3d: ArrayLike,
        forward_3d: ArrayLike,
        up_3d: ArrayLike,
    ) -> None:
        """
        Set directions of the camera.

        :param right_3d: Right direction in visual space.
        :param forward_3d: Forward direction in visual space.
        :param up_3d: Up direction in visual space.
        """
        self._camera_directions = OptionalVectorPair(
            visual_3d=np.vstack([right_3d, forward_3d, up_3d]), virtual=None
        )

    def move_center(self, position_3d: NPArray, position: NPArray) -> NPArray:
        """
        Move center to specified position.

        To be called before fitting. If not called, the center will remain at the
        origin.

        The center is a position that won't change with dimensionality reduction.

        :param position_3d: Preferred new position in visual space for center. Some
          nonlinear dimension reducers may ignore this value.
        :param position: New position in virtual space for center.
          c.f. :meth:`LinearDimensionReducer.move_center_3d()`.

        :returns: New position in virtual space.
        """
        assert position_3d.shape == (3,), f"{position_3d.shape=}"

        self._center = VectorPair(visual_3d=position_3d, virtual=position)

        return position

    def get_depth(self) -> float | None:
        """
        Get depth after fitting.

        For PCA, depth is the minimum index for the components that were
        chosen divided by the number of total components.

        :returns: Depth between [0, 1], or `None` if depth not available.
        """
        return None

    def get_message(self) -> list[str]:
        """
        Get message to be shown on screen.

        :returns: Message as a list of strings to be shown on each line.
        """
        return []

    def get_visual_information(self) -> VisualInformation | None:
        """
        Get information to show on screen.

        :returns: `VisualInformation` that represents information about the reducer.
          `None`, if there is nothing to be shown.
        """
        return None

    @abstractmethod
    def fit_transform(self, nodes: VisualNodeCollection, *, force_fit: bool) -> NPArray:
        """
        Transform data points based on dimension reduction analysis.

        :param nodes: Visual Nodes to transform.
        :param force_fit: Force dimension reduction. May not perform dimension reduction
          if `False`.
        :returns: Transformed coordinates in each row.
        """
