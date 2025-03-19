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

"""Defines visual objects."""

from abc import ABC, abstractmethod

from direct.interval.Interval import Interval
from panda3d.core import (
    NodePath,
)

from ..mathutils import VectorPair


class VisualInformation(ABC):
    """Holds information to show on the screen."""

    def __init__(self) -> None:
        super().__init__()

        self._nodepath = NodePath('VisualInformation')
        self._nodepath.setPos(0, 0, 0)

    @abstractmethod
    def set_width(self, width: float) -> None:
        """Set the preferred width of this Visual Information."""

    @abstractmethod
    def get_width(self) -> float:
        """Get the width of this Visual Information."""

    @abstractmethod
    def update_directions(self, directions: VectorPair) -> None:
        """
        Notify `VisualInformation` of new camera directions.

        :param directions: Directions of the camera.
        """

    def get_nodepath(self) -> NodePath:
        """Get the :class:`panda3d.core.NodePath` to show on."""
        return self._nodepath

    @abstractmethod
    def make_intervals(self, duration: float, *, clear: bool) -> list[Interval] | None:
        """
        Make animation interval for visual information regarding the dimension reducer.

        :param duration: Length in time for movement.
        :param clear: Whether to clear new transformation.
        :returns: Animation intervals for this visual information. `None`, if no
          animation is necessary.
        """

    @abstractmethod
    def apply_new_transformation(self) -> None:
        """Finalize animation."""

    def set_visibility(self, *, show: bool) -> None:
        """Set whether to make this Visual Information visible or not."""
        if show:
            self._nodepath.show()

        else:
            self._nodepath.hide()
