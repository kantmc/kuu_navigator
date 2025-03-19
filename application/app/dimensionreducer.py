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
Defines class that handles dimension reduction.

Array type: numpy
"""

from dataclasses import dataclass
from typing import (
    Any,
    cast,
)

import numpy as np

from panda3d.core import LPoint3

import pylog

from .datasource.api import (
    DataPoint,
    DataPointCollection,
    DataSource,
)
from .dimensionreduction import DimensionReducer
from .dimensionreduction.linear import LinearOrientedDimensionReducer
from .mathutils import (
    VISUAL_DIMENSIONS,
    NPArray,
    OptionalVectorPair,
    VectorPair,
    magnitude,
)
from .nodes.camera import Camera
from .nodes.navigationcamera import NavigationCamera
from .nodes.node import Node
from .nodes.visualinformation import VisualInformation
from .nodes.visualnode import VisualNodeCollection

_logger = pylog.getLogger(__name__)


class DimensionReducerCoordinator:
    """
    Defines procedures for dimension reduction.

    This class acts as a Facade (Pattern) to coordinate the use of
    :class:`.dimensionreduction.api.DimensionReducer` subclasses and
    :class:`.datasource.api.DataSource` subclasses.

    :param space_parameters: Parameters for the visual space.
    :param data_source: Data source that provides the data points.
    :param configuration: Configuration of the dimension reduction algorithm.
      The `dict` contains the following keys:

      - "type": Name of algorithm (registered to
        :class:`.dimensionreduction.api.DimensionReducer`).
      - "parameters": Arguments passed to the initializer for the dimension reduction
        class.
    """

    def __init__(
        self,
        space_parameters: 'SpaceConfiguration',
        data_source: DataSource,
        configuration: 'Configuration',
    ) -> None:
        self._space_parameters = space_parameters
        self._data_source = data_source
        self._configuration = configuration
        self._reducer: DimensionReducer | None = None
        self._new_position: VectorPair | None = None
        self._previous_position: VectorPair | None = None

        super().__init__()

    @dataclass
    class SpaceConfiguration:
        """Holds values regarding visual and virtual spaces."""

        navigation_camera: NavigationCamera
        """The :class:`.nodes.camera.Camera` instance."""

        max_number_of_points: int
        """Maximum number of data points to sample."""

        visual_size: float
        """Size of visuals for each data point."""

        neighborhood_radius: float
        """Radius for local neighborhood."""

        shared_neighborhood: bool
        """If `True`, make sure to share points while moving around neighborhoods."""

    @dataclass(frozen=True)
    class Configuration:
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
        local_mode: dict[str, Any]
        """
        Configuration of the dimension reduction algorithm for Local Mode.

        The `dict` has the same format as for `global_mode`.
        """
        random_generator: np.random.Generator
        """A random number generator."""
        visual_information: bool
        """Whether to show Visual Information about dimension reduction."""

    def has_linear_navigation(self) -> bool:
        """
        Return whether linear navigation is supported.

        .. note:: Valid only after call to one of `setup_*()` methods.
        """
        assert self._reducer is not None, "Not set up"
        return isinstance(self._reducer, LinearOrientedDimensionReducer)

    def get_reducer(self) -> DimensionReducer:
        """Get dimension reduction algorithm."""
        assert self._reducer is not None, (
            "`DimensionReducerCoordinator._prepare()` not called."
        )
        return self._reducer

    def _prepare(
        self,
        camera_directions: Camera.Directions3D,
        *,
        is_global: bool,
        previous: DimensionReducer | None,
    ) -> None:
        reducer_configuration = (
            self._configuration.global_mode
            if is_global
            else self._configuration.local_mode
        )
        parameters = reducer_configuration.get("parameters", None)
        self._reducer = DimensionReducer.factory.create(
            reducer_configuration["type"], parameters
        )

        self._reducer.set_previous(previous)

        self._reducer.set_global(is_global=is_global)

        self._reducer.set_camera_directions(
            right_3d=camera_directions.right_3d,
            forward_3d=camera_directions.forward_3d,
            up_3d=camera_directions.up_3d,
        )

    def _prepare_for_global_movement(
        self, camera_directions: Camera.Directions3D
    ) -> None:
        """
        Prepare before dimension reduction for universe.

        :param camera_directions: Tuple of vectors for the right, forward, and up
          directions for the camera.
        """
        self._prepare(camera_directions, is_global=True, previous=self._reducer)

    def _prepare_for_local_movement(
        self, camera_directions: Camera.Directions3D
    ) -> None:
        """
        Prepare before dimension reduction for neighborhood.

        :param camera_directions: Tuple of vectors for the right, forward, and up
          directions for the camera.
        """
        self._prepare(camera_directions, is_global=False, previous=self._reducer)

    @staticmethod
    def _log_extremes(data_points: DataPointCollection) -> None:
        extremes = data_points.get_extremes()
        _logger.debug("Extremes: %r", extremes)

    def setup_global_visual_nodes(
        self,
        directions: Camera.Directions3D | None,
        visual_nodes: VisualNodeCollection,
        *,
        number_of_detailed_nodes: int,
        force_fit: bool,
    ) -> None:
        """
        Prepare Visual Nodes for Global mode.

        :param directions: Directions of the camera. `None` won't change directions.
        :param visual_nodes: Visual Nodes to prepare.
        :param number_of_detailed_nodes: Number of detailed :class:`VisualNode`.
        :param force_fit: Force dimension reduction. May not perform dimension reduction
          if `False`.
        """
        if directions is not None:
            self._prepare_for_global_movement(directions)

        virtual_dimensions = self._data_source.get_dimensions()
        self._move_center(
            position_3d=np.zeros((3,)), position=np.zeros((virtual_dimensions,))
        )

        data_points = self._data_source.get_from_all(
            self._space_parameters.max_number_of_points
        )
        self._log_extremes(data_points)

        visual_nodes.clear()

        visual_nodes.update_data_points(
            data_points,
            visual_size=self._space_parameters.visual_size,
            max_number_of_detailed=number_of_detailed_nodes,
            point_to_camera=self._space_parameters.navigation_camera,
            random_generator=self._configuration.random_generator,
        )

        self._fit_transform(visual_nodes, set_new_position=False, force_fit=force_fit)

    def calculate_data_point_positions(self, data_point: DataPoint) -> VectorPair:
        """
        Calculate positions of a Data Point.

        :param data_point: The :class:`DataPoint`.
        :return: Position in visual space and virtual space.

        ..note:: Only works with Linear Dimension Reducers.
        """
        position_3d = self.calculate_position_3d(data_point)

        position = np.array(data_point.get_coordinates())

        return VectorPair(visual_3d=position_3d, virtual=position)

    def setup_local_visual_nodes(
        self,
        visual_nodes: VisualNodeCollection,
        *,
        new_position: OptionalVectorPair | None,
        number_of_detailed_nodes: int,
        selected_data_point: DataPoint | None,
        point_to_camera: Node,
        new_alpha: float,
        shared_neighborhood: bool | None,
        force_fit: bool,
    ) -> None:
        """
        Prepare Visual Nodes for Local mode.

        :param visual_nodes: Visual Nodes to prepare.
        :param new_position: New position for center. `None`, for no movement. Contains
          the following fields:

            - `visual_3d`: New position in visual space for center.
            - `virtual`: New position in virtual space for center. Can be `None`, if
              linear movement is supported. c.f. :meth:`has_linear_movement()`.

        :param number_of_detailed_nodes: Number of detailed :class:`VisualNode`.
        :param selected_data_point: If set, this will be added to `visual_nodes`, if it
          is within the neighborhood.
        :param point_to_camera: Make visuals face the specified camera.
        :param new_alpha: Opacity for new :class:`VisualNode`.
        :param shared_neighborhood: Whether to share Data Points between neighborhoods
          that are close. `None` to use value from
          `self._space_parameters.shared_neighborhood`.
        :param force_fit: Force dimension reduction. May not perform dimension reduction
          if `False`.

        .. note:: This will remove some nodes from `visual_nodes`, and also release
          them making them useless even if references to the nodes are kept elsewhere.
        """

        def _add_shared_neighborhood(
            data_points: DataPointCollection,
            visual_nodes: VisualNodeCollection,
            new_position: NPArray,
        ) -> None:
            original_data_points = visual_nodes.to_data_point_collection()
            shared_data_points = self._data_source.filter_sphere(
                original_data_points,
                new_position,
                self._space_parameters.neighborhood_radius,
            )
            data_points.update(shared_data_points)

        def _add_selected_if_close(
            data_points: DataPointCollection,
            selected_data_point: DataPoint,
            new_position: NPArray,
        ) -> None:
            if self._data_source.in_sphere(
                selected_data_point,
                new_position,
                self._space_parameters.neighborhood_radius,
            ):
                data_points.add(selected_data_point)

        # `self._reducer` replaced here
        self._prepare_for_local_movement(
            self._space_parameters.navigation_camera.get_directions_3d()
        )

        new_virtual_position = self._move_center(
            None if new_position is None else new_position.visual_3d,
            None if new_position is None else new_position.virtual,
        )

        data_points = self._data_source.get_from_neighborhood(
            new_virtual_position,
            self._space_parameters.neighborhood_radius,
            self._space_parameters.max_number_of_points,
        )
        self._log_extremes(data_points)

        if shared_neighborhood or (
            (shared_neighborhood is None) and self._space_parameters.shared_neighborhood
        ):
            # Might skew distribution
            _add_shared_neighborhood(data_points, visual_nodes, new_virtual_position)

        if selected_data_point is not None:
            # Might skew distribution
            _add_selected_if_close(
                data_points, selected_data_point, new_virtual_position
            )

        visual_nodes.update_data_points(
            data_points,
            visual_size=self._space_parameters.visual_size,
            max_number_of_detailed=number_of_detailed_nodes,
            point_to_camera=point_to_camera,
            new_alpha=new_alpha,
            random_generator=self._configuration.random_generator,
        )

        self._fit_transform(visual_nodes, set_new_position=True, force_fit=force_fit)

        if selected_data_point is not None:
            visual_nodes.select_data_point(selected_data_point)

    def _move_center(
        self, position_3d: NPArray | None, position: NPArray | None
    ) -> NPArray:
        assert self._reducer is not None, "Not set up"

        self._previous_position = self._new_position

        if position_3d is None:
            assert position is None
            assert self._new_position is not None
            position = self._reducer.move_center(
                self._new_position.visual_3d, self._new_position.virtual
            )

        else:
            if position is None:
                assert isinstance(self._reducer, LinearOrientedDimensionReducer)
                position = self._reducer.move_center_3d(position_3d)

            else:
                position = self._reducer.move_center(position_3d, position)

            self._new_position = VectorPair(position_3d, position)

        _logger.info("Moved center: %r", position)
        return position

    def get_center_3d(self) -> NPArray:
        """Get center position in visual space."""
        assert self._new_position is not None, (
            "`DimensionReducerCoordinator.setup_local_visual_nodes()` not called."
        )
        assert len(self._new_position.visual_3d) == VISUAL_DIMENSIONS
        return self._new_position.visual_3d

    def get_center(self) -> NPArray:
        """Get center position in virtual space."""
        assert self._new_position is not None, (
            "`DimensionReducerCoordinator.setup_local_visual_nodes()` not called."
        )
        assert len(self._new_position.virtual) == self._data_source.get_dimensions()
        return self._new_position.virtual

    @staticmethod
    def _set_positions_3d(
        nodes: VisualNodeCollection,
        new_positions_3d: NPArray,
        *,
        set_new_position: bool,
    ) -> None:
        for each, each_point in zip(nodes, new_positions_3d, strict=True):
            if set_new_position:
                each.set_new_pose_3d(LPoint3(*each_point), None)

            else:
                each.set_position_3d(LPoint3(*each_point))

    def calculate_direction_from_3d(
        self,
        vector_3d: NPArray,
    ) -> NPArray:
        """
        Calculate direction in virtual space for vector in visual space.

        Uses axes that were obtained with PCA.

        :param vector_3d: Vector in visual space to convert.
          (Can be rows of vectors.)

        :returns: Converted vector. (Or, rows of converted vectors.)

        .. note:: Only valid with linear dimension reduction.
          c.f. :meth:`DimensionReducerCoordinator.has_linear_navigation()`
        """
        assert self.has_linear_navigation()

        pca = cast(LinearOrientedDimensionReducer, self._reducer)
        assert isinstance(pca, LinearOrientedDimensionReducer)

        return pca.calculate_direction_from_3d(vector_3d)

    def calculate_position_3d(self, data_point: DataPoint) -> NPArray:
        """Calculate the visual coordinates of a Data Point."""
        assert isinstance(self._reducer, LinearOrientedDimensionReducer)

        reducer = self._reducer

        coordinates = np.array(data_point.get_coordinates())
        return reducer.transform(coordinates)

    @dataclass
    class _Statistics:
        sum_squared_dislocation: float | None
        """
        Sum of squared dislocation in visual space of Visual Nodes that were inherited
        from previous iteration of dimensionality reduction. `None`, if all Visual Nodes
        were new.
        """

        variance_dislocated: float | None
        """
        Variance of dislocated Visual Nodes in visual space. `None`, if all Image
        Nodes were new.
        """

        dislocation_count: int
        """Number of dislocated Visual Nodes."""

    @staticmethod
    def _calculate_statistics(
        nodes: VisualNodeCollection,
        new_positions_3d: NPArray,
    ) -> _Statistics:
        def _fill_statistics(
            dislocated_points: list[NPArray], sum_squares: float
        ) -> DimensionReducerCoordinator._Statistics:
            count = len(dislocated_points)

            if count == 0:
                return DimensionReducerCoordinator._Statistics(None, None, count)

            dislocated_points_array = np.array(dislocated_points)

            variances = np.var(dislocated_points_array, axis=0)

            return DimensionReducerCoordinator._Statistics(
                sum_squared_dislocation=sum_squares,
                variance_dislocated=np.sum(variances),
                dislocation_count=count,
            )

        dislocated_points = []
        sum_squares = 0.0
        for each, each_point in zip(nodes, new_positions_3d, strict=True):
            previous_position_3d = each.get_position_3d_if_set()
            if previous_position_3d is not None:
                dislocated_points.append(each_point)
                sum_squares += magnitude(previous_position_3d - each_point)

        return _fill_statistics(dislocated_points, sum_squares)

    def _fit_transform(
        self, nodes: VisualNodeCollection, *, set_new_position: bool, force_fit: bool
    ) -> None:
        """
        Transform nodes based on dimension reduction analysis.

        :param nodes: Nodes to transform.
        :param set_new_position: Whether to set new position for the nodes in
          preparation for animation.
        :param force_fit: Force dimension reduction. May not perform dimension reduction
          if `False`.

        .. note:: Valid only after call to one of `setup_*()` methods.
        """
        assert self._reducer is not None, "Not set up"

        new_positions_3d = self._reducer.fit_transform(nodes, force_fit=force_fit)

        self._set_positions_3d(
            nodes, new_positions_3d, set_new_position=set_new_position
        )

    def get_depth(self) -> float | None:
        """
        Get depth after fitting.

        For PCA, depth is the maximum index for the components that were
        chosen divided by the number of total components.

        :returns: Depth between [0, 1], or `None` if depth not available.

        .. note:: Valid only after calling :meth:`setup_global_visual_nodes()` or
          :meth:`setup_local_visual_nodes()`.
        """
        assert self._reducer is not None, "Not set up"
        return self._reducer.get_depth()

    def get_message(self) -> list[str]:
        """Get message from reducer."""
        if self._reducer is None:
            return []

        return self._reducer.get_message()

    def get_visual_information(self) -> VisualInformation | None:
        """
        Get information to show on screen.

        :returns: `VisualInformation` that represents information about the reducer.
          `None`, if there is nothing to be shown.
        """
        if not self._configuration.visual_information or (self._reducer is None):
            return None

        return self._reducer.get_visual_information()

    def set_visual_size(self, size: float) -> None:
        """Set size of Visual Nodes."""
        self._space_parameters.visual_size = size

    def get_visual_size(self) -> float:
        """Get size of Visual Nodes."""
        return self._space_parameters.visual_size

    def set_number_of_points(self, value: int) -> None:
        """Set the maximum number of Visual Nodes to show."""
        self._space_parameters.max_number_of_points = value

    def set_neighborhood_radius(self, value: float) -> None:
        """Set the radius of neighborhood."""
        self._space_parameters.neighborhood_radius = value

    def set_shared_neighborhood(self, *, value: bool) -> None:
        """Set shared neighborhood flag."""
        self._space_parameters.shared_neighborhood = value
