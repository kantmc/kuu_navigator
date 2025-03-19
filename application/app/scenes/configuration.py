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

"""Defines dataclasses for configuration of the Navigation Scene."""

from dataclasses import dataclass
from typing import Any

import numpy as np

from panda3d.core import LPoint2i, LVector2i

from pyqoolloop.parallel import Guard

from ..window import Window


@dataclass(frozen=True)
class ReducerConfiguration:
    """Configuration parameters for dimension reduction."""

    global_mode: dict[str, Any]
    """
    Configuration of the dimension reduction algorithm for Global Mode.

    The `dict` contains the following keys:
        - `"type"`: Name of algorithm (registered to
            :class:`.dimensionreduction.api.DimensionReducer`).
        - `"parameters"`: Arguments passed to the initializer of the algorithm
            class.
    """
    local_mode: dict[str, Any] | None
    """
    Configuration of the dimension reduction algorithm for Local Mode.

    The `dict` has the same format as for `global_mode`.
    If `None`, no dimension reduction will be performed.
    """
    random_generator: np.random.Generator
    """A random number generator."""
    visual_information: bool
    """Whether to show Visual Information about dimension reduction."""


@dataclass
class AnimationConfiguration:
    """Configuration parameters for animation."""

    node_animation: bool
    """Enable animation for Visual Node."""

    camera_animation: bool
    """Enable animation for Camera."""

    animation_interval_secs: float
    """Interval in seconds for animation."""


class SharedNavigationSceneConfiguration:
    """Guarded configuration parameters shared between Navigation Scenes."""

    @dataclass
    class Data:
        """The actual data held for `SharedNavigationSceneConfiguration`."""

        animation: AnimationConfiguration

        max_number_of_points: int
        """Maximum number of Data Points to handle at a time."""

        number_of_detailed_nodes: int
        """Number of Visual Nodes to render with detail."""

        neighborhood_radius: float
        """Radius for local neighborhood."""

        shared_neighborhood: bool
        """
        If `True`, make sure to share points while moving around neighborhoods.

        This may cause bias in sampling. Will also increase number of data points.
        """

        def has_animation(self) -> bool:
            """Return whether animation is provided."""
            return self.animation.camera_animation or self.animation.node_animation

    def __init__(self, **kwargs: Any) -> None:
        data = SharedNavigationSceneConfiguration.Data(**kwargs)
        self._guarded = Guard(data)

    def with_lock(self) -> Guard[Data]:
        """
        Access data with lock.

        ..note:: To be used in `with` statement.
        """
        return self._guarded


@dataclass
class UniqueNavigationSceneConfiguration:
    """Configuration parameters unique to each instance of Navigation Scene."""

    visual_size: float
    """Size of Visual Nodes."""

    background_color: tuple[float, float, float]
    """
    Default color (RGB) of the background. Background color may change depending on
      depth (selection of components).
    """

    screenshot_prefix: str | None
    """
    Prefix of filenames for screenshots. Characters after the last '/' will be used as
      the prefix of the name of the file. Directories will be created as necessary.
    """


@dataclass
class WindowConfiguration:
    """Configuration for the window of a Navigation Scene."""

    title: str
    """The title for the window."""

    parent: Window | None = None
    """The parent window."""

    window_origin: LPoint2i | None = None
    """Coordinates of the origin of the window. `None` for Main Navigation Scene."""

    window_size: LVector2i | None = None
    """Size of the window. `None` for default."""


@dataclass
class NavigationSceneConfiguration:
    """Configuration parameters for Navigation Scenes."""

    animation: AnimationConfiguration

    visual_size: float
    """Size of Visual Nodes."""

    background_color: tuple[float, float, float]
    """
    Default color (RGB) of the background. Background color may change depending on
      depth (selection of components).
    """

    max_number_of_points: int
    """Maximum number of Data Points to handle at a time."""

    number_of_detailed_nodes: int
    """Number of Visual Nodes to render with detail."""

    neighborhood_radius: float
    """Radius for local neighborhood."""

    shared_neighborhood: bool
    """
    If `True`, make sure to share points while moving around neighborhoods.

    This may cause bias in sampling. Will also increase number of data points.
    """

    screenshot_prefix: str | None
    """
    Prefix of filenames for screenshots. Characters after the last '/' will be used as
      the prefix of the name of the file. Directories will be created as necessary.
    """

    def split(
        self,
    ) -> tuple[SharedNavigationSceneConfiguration, UniqueNavigationSceneConfiguration]:
        """Generate configuration instances from this one."""
        return (
            SharedNavigationSceneConfiguration(
                animation=self.animation,
                max_number_of_points=self.max_number_of_points,
                number_of_detailed_nodes=self.number_of_detailed_nodes,
                neighborhood_radius=self.neighborhood_radius,
                shared_neighborhood=self.shared_neighborhood,
            ),
            UniqueNavigationSceneConfiguration(
                visual_size=self.visual_size,
                background_color=self.background_color,
                screenshot_prefix=self.screenshot_prefix,
            ),
        )
