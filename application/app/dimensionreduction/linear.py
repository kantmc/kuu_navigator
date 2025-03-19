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
Defines base classes for linear dimension reduction.

Array type: numpy
"""

from abc import abstractmethod
from collections.abc import Iterable
from itertools import permutations
from math import inf
from sys import float_info
from types import NoneType
from typing import Protocol, cast

from typing_extensions import override

import numpy as np

from direct.interval.Interval import Interval
from panda3d.core import (
    Geom,
    GeomLines,
    GeomNode,
    GeomVertexData,
    GeomVertexFormat,
    GeomVertexWriter,
    LQuaternion,
    NodePath,
)

import pylog

from ..mathutils import (
    PLANE_DIMENSIONS,
    VISUAL_DIMENSIONS,
    NPArray,
    VectorPair,
    normalize,
    np_all,
)
from ..nodes.camera import Camera
from ..nodes.node import Node
from ..nodes.visualinformation import VisualInformation
from ..nodes.visualnode import VisualNodeCollection
from .algebra import find_orthonormal_2d, generate_orthonormal_3d
from .api import DimensionReducer

_logger = pylog.getLogger(__name__)


_epsilon = float_info.epsilon


class PCAProtocol(Protocol):
    """
    :class:`~typing.Protocol` for PCA classes.

    See :class:`sklearn.decomposition.PCA` for definitions.
    """

    @abstractmethod
    def fit(self, x: NPArray) -> None:
        """Fit data points."""

    @property
    def components_(self) -> NPArray:
        """
        Get the principal components.

        Doesn't exist if PCA was not performed.
        """

    @components_.setter
    def components_(self, components: NPArray) -> None:
        """Set principal components."""

    @property
    def n_features_in_(self) -> int:
        """
        Get number of features in the training data.

        Doesn't exist if PCA was not performed.
        """

    @property
    def explained_variance_ratio_(self) -> NPArray | None:
        """
        Get ratio (0-1) of variance explained by each of the components.

        Will be `None`, when this is a fake PCA class.
        May not exist if PCA was not performed.
        """


def _map(vectors: NPArray, axes: NPArray) -> NPArray:
    """
    Map high-dimensional vector(s) to low-dimensional subspace.

    :param vectors: Multidimensional vectors to map each in a row.
    :param axes: Multidimensional vectors that correspond to axes in the
      subspace.
    :returns: `vectors` mapped to the space spanned by `axes`.
    """
    # Works with any dimensions, but 2 or 3 is expected
    assert axes.shape[0] in {PLANE_DIMENSIONS, VISUAL_DIMENSIONS}

    return vectors @ axes.transpose()


def _expand(vectors: NPArray, axes: NPArray) -> NPArray:
    """
    Map a vector in a low-dimensional subspace to the high-dimensional space.

    :param vectors: Vectors to map each in a row.
    :param axes: Vectors that correspond to axes in the resulting
      high-dimensional space.
    :returns: Mapped vectors each in a row.
    """
    # Works with any dimensions, but 2 or 3 is expected
    assert (vectors.shape in {(PLANE_DIMENSIONS,), (VISUAL_DIMENSIONS,)}) or (
        vectors.shape[1] in {PLANE_DIMENSIONS, VISUAL_DIMENSIONS}
    ), f"`vectors` is {vectors.shape}"
    return vectors @ axes


def _calculate_vector_from_3d(
    vector_3d: NPArray, directions_3d: NPArray, directions: NPArray
) -> NPArray:
    """
    Calculate vector in virtual space for vector in visual space.

    (e.g. Camera space)

    :param vector_3d: Vector in visual space to convert.
      (Can be rows of vectors.)
    :param directions_3d: Directions for 3 axes in visual space.
    :param directions: Directions for each axes in `direction_3d`
      represented in virtual space.

    :returns: Converted vector, or rows of converted vectors.
    """
    movement = _expand(vector_3d @ directions_3d.transpose(), directions)
    return movement


def _find_3_orthonormal(
    original_vectors: NPArray, prioritize: int, axes: NPArray
) -> NPArray:
    """
    Find 3 vectors in the space spanned by 3 axes that are close to 3 vectors.

    :param original_vectors: The 3 specified vectors each in a row.
    :param prioritize: The row index for `original_vectors`, which will have
      the most importance.
    :param axes: The 3 axes each in a row.
    :returns: The 3 close vectors each in a row.
    """

    def _find(original_vectors: NPArray, axes: NPArray) -> NPArray:
        main_vector_3d = _map(original_vectors[0, :], axes)
        temporary_orthonormal_3d = generate_orthonormal_3d(main_vector_3d)
        temporary_axes = _expand(temporary_orthonormal_3d[1:, :], axes)
        mapped_vectors_2d = _map(original_vectors[1:, :], temporary_axes)
        assert mapped_vectors_2d.shape == (
            2,
            2,
        ), f"`mapped_vectors_2d` is {mapped_vectors_2d.shape}"
        close_vectors_2d = find_orthonormal_2d(
            mapped_vectors_2d[0, :], mapped_vectors_2d[1, :]
        )
        return np.vstack(
            (
                _expand(temporary_orthonormal_3d[0, :], axes),
                _expand(np.array(close_vectors_2d), temporary_axes),
            )
        )

    if prioritize == 0:
        order = [0, 1, 2]

    elif prioritize == 1:
        order = [1, 0, 2]

    elif prioritize == 2:  # noqa: PLR2004
        order = [2, 1, 0]

    else:
        raise AssertionError("`prioritize` out of bounds")

    result = _find(original_vectors[order, :], axes)

    return result[order, :]


_DYNAMIC_COMPONENTS_SELECTION_POWER = 2


def _find_matching_axes(
    axes: NPArray, desired_directions: NPArray, weights: NPArray
) -> list[int]:
    """
    Find 3 vectors that can span the desired directions from a selection of axes.

    :param axes: Normal vectors to choose from, one in each row.
    :param desired_directions: The desired directions as normal vectors in each
      row.
    :param weights: Weights for each of the `desired_directions` to specify how
      important each is.

    :returns: Indexes of `axes` for the chosen vectors.
    """
    inner_products = axes @ desired_directions.transpose()
    converted = np.abs(inner_products) ** _DYNAMIC_COMPONENTS_SELECTION_POWER
    weighted_sum = (converted * weights.transpose()).sum(axis=1)

    max_value = -inf
    max_permutation = None
    for each in permutations(range(len(weighted_sum)), 3):
        value = weighted_sum[each, ...].sum()
        if value > max_value:
            max_value = value
            max_permutation = each

    assert max_permutation is not None
    return list(max_permutation)


def _make_axis_geom(
    x1: float,
    y1: float,
    z1: float,
    x2: float,
    y2: float,
    z2: float,
    *,
    color: Iterable[float],
) -> Geom:
    """
    Make an arrow.

    The arguments are coordinates of the diagonals of the square.
    """
    vertex_format = GeomVertexFormat.getV3cp()
    vdata = GeomVertexData('axis', vertex_format, Geom.UHStatic)
    vdata.setNumRows(2)

    vertex_writer = GeomVertexWriter(vdata, 'vertex')
    color_writer = GeomVertexWriter(vdata, 'color')

    vertex_writer.addData3(x1, y1, z1)
    vertex_writer.addData3(x2, y2, z2)

    # adding different colors to the vertex for visibility
    color_writer.addData4f(*color)
    color_writer.addData4f(*color)

    line = GeomLines(Geom.UHStatic)
    line.addVertices(0, 1)

    arrow = Geom(vdata)
    arrow.addPrimitive(line)
    return arrow


class _VisualInformation(VisualInformation):
    """Holds information to show for `LinearOrientedDimensionReducer`."""

    _COLORS = (
        (0.0, 1.0, 1.0, 1.0),
        (1.0, 0.0, 1.0, 1.0),
        (1.0, 1.0, 0.0, 1.0),
    )

    _DEFAULT_AXIS = np.array([0.0, 1.0, 0.0])

    def __init__(self) -> None:
        super().__init__()

        self._axis_radius: float | None = None

        self._components: NPArray | None = None
        self._new_components: NPArray | None = None

        self._arrows: list[Node] | None = None

        self._set_new_transformation = False

        self._camera_directions: VectorPair | None = None

    @override
    def set_width(self, width: float) -> None:
        self._axis_radius = width / 2

    @override
    def get_width(self) -> float:
        assert self._axis_radius is not None

        return self._axis_radius * 2

    def update_components(self, components: NPArray) -> None:
        """
        Update components.

        :param components: Principal components in each row. Set to `np.empty(0)`, if
          dimension reduction was not performed.
        """
        self._new_components = components

    @staticmethod
    def _make_axis_geomnode(*, axis_radius: float, color: Iterable[float]) -> GeomNode:
        snode = GeomNode('axis')
        arrow = _make_axis_geom(
            *(_VisualInformation._DEFAULT_AXIS * -axis_radius),
            *(_VisualInformation._DEFAULT_AXIS * axis_radius),
            color=color,
        )
        snode.addGeom(arrow)

        return snode

    @classmethod
    def _make_axis_nodepath(
        cls,
        direction: str,
        *,
        parent: NodePath,
        axis_radius: float,
        color: Iterable[float],
    ) -> Node:
        arrow = cls._make_axis_geomnode(
            axis_radius=axis_radius,
            color=color,
        )

        nodepath = NodePath(direction)
        nodepath.attachNewNode(arrow)

        rotation_3d = Camera.get_axis_rotation_3d(direction)
        nodepath.setQuat(rotation_3d)

        nodepath.reparentTo(parent)
        nodepath.setRenderModeWireframe(1)

        return Node(nodepath)

    @staticmethod
    def _get_rotation_3d(*, before_3d: NPArray, after_3d: NPArray) -> LQuaternion:
        if np_all(after_3d == np.zeros((3,))):
            return LQuaternion()

        normalized_before_3d = normalize(before_3d)

        normalized_after_3d = normalize(after_3d)

        inner_product = normalized_before_3d @ normalized_after_3d.transpose()

        if inner_product <= -1 + _epsilon:
            orthonormal_vectors_3d = generate_orthonormal_3d(before_3d)
            return LQuaternion(0, *orthonormal_vectors_3d[1, :])

        cross_product = np.cross(normalized_before_3d, normalized_after_3d)

        rotation = LQuaternion(1 + inner_product, *cross_product)
        rotation.normalize()

        return rotation

    def _get_components_3d(self, camera_directions: VectorPair) -> NPArray | list[None]:
        if self._components is None:
            return [None] * 3

        assert self._components.shape[0] >= VISUAL_DIMENSIONS

        principal_components = self._components[:3, :]

        return principal_components @ camera_directions.virtual.transpose()

    def _get_new_components_3d(
        self, camera_directions: VectorPair
    ) -> NPArray | list[None]:
        if self._new_components is None:
            result = self._get_components_3d(camera_directions)
            return result

        if self._new_components.shape[0] == 0:
            return np.empty(0)

        assert self._new_components.shape[0] >= VISUAL_DIMENSIONS

        principal_components = self._new_components[:3, :]

        return principal_components @ camera_directions.virtual.transpose()

    def _update_axes(self) -> None:
        def _prepare_axes() -> list[Node]:
            assert self._axis_radius is not None

            return [
                self._make_axis_nodepath(
                    '+X',
                    axis_radius=self._axis_radius,
                    color=self._COLORS[0],
                    parent=self._nodepath,
                ),
                self._make_axis_nodepath(
                    '+Y',
                    axis_radius=self._axis_radius,
                    color=self._COLORS[1],
                    parent=self._nodepath,
                ),
                self._make_axis_nodepath(
                    '+Z',
                    axis_radius=self._axis_radius,
                    color=self._COLORS[2],
                    parent=self._nodepath,
                ),
            ]

        assert self._camera_directions is not None

        if self._arrows is None:
            self._arrows = _prepare_axes()

        previous_components_3d = self._get_components_3d(self._camera_directions)

        new_components_3d = self._get_new_components_3d(self._camera_directions)

        actual_new_components = []

        if (self._new_components is not None) or (self._components is not None):
            self._set_new_transformation = True

            for (
                index,
                (each_previous_component_3d, each_arrow),
            ) in enumerate(
                zip(
                    previous_components_3d,
                    self._arrows,
                    strict=True,
                )
            ):
                if self._new_components is None:
                    assert self._components is not None
                    each_new_component = self._components[index, :]

                elif self._new_components.shape[0] == 0:
                    rotation_3d = each_arrow.get_rotation_3d()

                    each_arrow.set_new_transformation(rotation_3d, scale=0.0)

                    continue

                else:
                    each_new_component = self._new_components[index, :]

                assert not isinstance(new_components_3d, list)
                each_new_component_3d = new_components_3d[index, :]

                if (each_previous_component_3d is not None) and (
                    each_new_component_3d @ each_previous_component_3d < 0
                ):
                    each_new_component_3d = -each_new_component_3d
                    each_new_component = -each_new_component

                rotation_3d = self._get_rotation_3d(
                    before_3d=self._DEFAULT_AXIS, after_3d=each_new_component_3d
                )

                assert np.linalg.norm(self._DEFAULT_AXIS) == 1
                scale = cast(float, np.linalg.norm(each_new_component_3d))

                each_arrow.set_new_transformation(rotation_3d, scale)

                actual_new_components.append(each_new_component)

        if len(actual_new_components) == 0:
            self._components = None

        else:
            self._components = np.vstack(actual_new_components)

        self._new_components = None

    @override
    def update_directions(self, directions: VectorPair) -> None:
        self._camera_directions = directions

        self._update_axes()

    @override
    def make_intervals(
        self, duration: float, *, clear: bool = True
    ) -> list[Interval] | None:
        assert self._arrows is not None, "`update_directions()` not called at all."

        if not self._set_new_transformation:
            return None

        return [
            each.make_interval_to_new_transformation(duration, clear=clear)
            for each in self._arrows
        ]

    @override
    def apply_new_transformation(self) -> None:
        if not self._set_new_transformation:
            return

        assert self._arrows is not None, "`update_directions()` not called at all."

        for each in self._arrows:
            each.apply_new_transformation()

        self._set_new_transformation = False


class LinearOrientedDimensionReducer(DimensionReducer):
    """
    Class to perform PCA on data points maintaining orientation.

    :param pca: The actual (non-oriented) PCA algorithm to use.

    .. note:: Subclasses also preserve location of 3D position, specified with
        :meth:`move_center()`.
    """

    def __init__(
        self,
        *,
        pca: PCAProtocol,
    ) -> None:
        super().__init__()

        self._virtual_dimensions: int | None = None

        self._pca = pca

        self._visual_information: _VisualInformation | None = None

        self._component_selection: list[int] | None = None

        self._previous: LinearOrientedDimensionReducer | None = None

    def get_reducer(self) -> PCAProtocol:
        """Get the actual dimension reduction algorithm."""
        return self._pca

    @override
    def set_previous(self, previous: DimensionReducer | None) -> None:
        def _set_virtual_dimensions() -> None:
            if (previous is not None) and isinstance(
                previous, LinearOrientedDimensionReducer
            ):
                self._virtual_dimensions = previous._virtual_dimensions  # noqa: SLF001

        def _set_visual_information() -> None:
            if isinstance(self._previous, LinearOrientedDimensionReducer):
                assert self._visual_information is None

                previous_visual_information = self._previous._get_visual_information()  # noqa: SLF001

                if previous_visual_information is None:
                    self._visual_information = self._get_visual_information()

                else:
                    self._visual_information = previous_visual_information

        assert isinstance(previous, (LinearOrientedDimensionReducer, NoneType)), (
            "Previous instance is not `LinearOrientedDimensionReducer`."
        )

        if previous is not None:
            # Release unnecessary instance. Can cause some methods to malfunction on
            # previous instance.
            previous._previous = None  # noqa: SLF001

        self._previous = previous

        _set_virtual_dimensions()

        _set_visual_information()

    @override
    def get_previous(self) -> DimensionReducer | None:
        return self._previous

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
        """
        assert self._camera_directions is not None, (
            "`set_camera_directions()` not called."
        )
        assert self._camera_directions.virtual is not None, (
            "`fit_transform()` not called"
        )

        directions = _calculate_vector_from_3d(
            vector_3d,
            self._camera_directions.visual_3d,
            self._camera_directions.virtual,
        )
        return directions

    def calculate_position_from_3d(
        self,
        position_3d: NPArray,
    ) -> NPArray:
        """
        Calculate position in virtual space for position in visual space.

        Uses axes that were obtained with PCA.

        :param position_3d: Position in visual space to convert.
          (Can be rows of vectors.)

        :returns: Converted position. (Or, rows of converted positions.)
        """
        assert self._camera_directions is not None, (
            "`set_camera_directions()` not called."
        )
        assert self._camera_directions.virtual is not None, (
            "`fit_transform()` not called."
        )
        assert self._center is not None, "`move_center()` not called."

        movement = _calculate_vector_from_3d(
            position_3d - self._center.visual_3d,
            self._camera_directions.visual_3d,
            self._camera_directions.virtual,
        )
        new_position = self._center.virtual + movement
        return new_position

    def move_center_3d(self, position_3d: NPArray) -> NPArray:
        """
        Move center to specified position in visual space.

        Virtual space coordinates are inferred from the specified visual space
        coordinates.

        To be called before fitting. If not called, the center will remain at the
        origin.

        The center is a position that won't change with dimensionality reduction.

        :param position_3d: New position in visual space for center.

        :returns: New position in virtual space.
        """
        assert position_3d.shape == (3,), f"{position_3d.shape=}"

        assert not self._is_global, "`position` cannot be `None` for Global mode."
        assert self._previous is not None
        position = self._previous.calculate_position_from_3d(position_3d)

        self._center = VectorPair(visual_3d=position_3d, virtual=position)

        return position

    def _get_center(self) -> NPArray:
        assert self._center is not None, "`move_center()` not called."
        return self._center.virtual

    def _fit_with_new_dimension_reduction(self, data_point_array: NPArray) -> None:
        """
        Perform dimension reduction analysis without changing the center.

        :param data_point_array: Coordinates for data points in virtual space, each in
          a row.
        """

        def _select_components(directions: NPArray) -> list[int]:
            indexes = _find_matching_axes(
                self._pca.components_, directions, np.array([0.1, 1.0, 0.1])
            )

            _logger.info("component indexes=%s", indexes)
            return indexes

        def _calculate_new_directions() -> NPArray:
            assert self._camera_directions is not None
            assert self._previous is not None

            old_directions = self._previous.calculate_direction_from_3d(
                self._camera_directions.visual_3d
            )
            _logger.info("old_directions=%s", old_directions)
            self._component_selection = _select_components(old_directions)
            components = self._pca.components_[self._component_selection]
            new_directions = _find_3_orthonormal(old_directions, 1, components)
            _logger.info("new_directions=%s", new_directions)

            return new_directions

        assert self._camera_directions is not None, (
            "`set_camera_directions()` not called."
        )
        assert self._center is not None, "`move_center()` not called."

        _logger.info("Fitting PCA")

        number_data_points = data_point_array.shape[0]
        assert number_data_points > VISUAL_DIMENSIONS, (
            f"Too few data points: {number_data_points}"
        )  # FUTURE:

        self._pca.fit(data_point_array)
        _logger.info("PCA components: %s", self._pca.components_)
        if self._pca.explained_variance_ratio_ is not None:
            _logger.info(
                "Explained Variance (%%): %s", self._pca.explained_variance_ratio_ * 100
            )

        if self._is_global:
            # PCA components are treated as X,Y,Z in visual space in Global mode.
            self._camera_directions.virtual = (
                self._camera_directions.visual_3d @ self._pca.components_[range(3)]
            )
            self._component_selection = [0, 1, 2]

        else:
            self._camera_directions.virtual = _calculate_new_directions()

    def _fit_with_old_dimension_reduction(self) -> None:
        """Reuse dimension reduction analysis from previous iteration."""
        assert not self._is_global
        assert self._center is not None, "`move_center()` not called."

        assert self._camera_directions is not None, (
            "`set_camera_directions()` not called."
        )

        assert self._previous is not None, "Maybe too few Data Points."
        # FUTURE: Should calculate principal components with the current dimension
        # reducer, and maybe choose the lacking dimensions from previous reductions.

        self._camera_directions.virtual = self._previous.calculate_direction_from_3d(
            self._camera_directions.visual_3d
        )

        # TODO: Assuming that previous dimension reduction is PCA.
        self._pca.components_ = self._previous._pca.components_  # noqa: SLF001
        self._component_selection = self._previous._component_selection  # noqa: SLF001

    def _fit(self, data_point_array: NPArray) -> bool:
        """
        Perform dimension reduction analysis without changing the center.

        :param data_point_array: Coordinates for data points in virtual space, each in
          a row.
        :returns: Whether dimension reduction could be performed. Otherwise, components
          from the previous dimension reduction is reused.
        """
        if (self._virtual_dimensions is None) and (len(data_point_array) > 0):
            self._virtual_dimensions = data_point_array.shape[1]

        _logger.info("`data_point_array.shape`: %r", data_point_array.shape)
        # Global Mode doesn't work with less than 4 data points
        if data_point_array.shape[0] <= VISUAL_DIMENSIONS:
            _logger.info("Number of Data Points: %d", data_point_array.shape[0])

            self._fit_with_old_dimension_reduction()
            return False

        self._fit_with_new_dimension_reduction(data_point_array)
        return True

    def _transform(
        self, data_point_array: NPArray, *, is_position: bool = True
    ) -> NPArray:
        """
        Transform data points based on analysis of `_fit()`.

        :param nodes: Data points to transform.
        :param is_position: Whether `vectors` represents positions. Positions will be
          adjusted so that `position` set by `DimensionReducer.move_center()` will be
          transformed to `position_3d`. If `vectors` represent directions the adjusting
          will not be performed.
        :returns: Transformed coordinates in each row.
        """

        def _transform_aligned(nparray: NPArray, center_coefficient: int) -> NPArray:
            assert self._camera_directions is not None, (
                "`LinearOrientedDimensionReducer.set_camera_directions()` not called."
            )
            assert self._camera_directions.virtual is not None, (
                "`LinearOrientedDimensionReducer._fit()` not called"
            )
            assert self._center is not None, "`move_center()` not called."

            recentered_points = nparray - self._center.virtual * center_coefficient
            points_in_camera_space_3d = _map(
                recentered_points, self._camera_directions.virtual
            )

            points_in_visual_space_3d: NPArray = (
                _expand(points_in_camera_space_3d, self._camera_directions.visual_3d)
                + self._center.visual_3d * center_coefficient
            )

            return points_in_visual_space_3d

        def _transform_global(array: NPArray, center_coefficient: int) -> NPArray:
            # PCA components are treated as X,Y,Z in visual space in Global mode.
            assert self._center is not None, (
                "`LinearOrientedDimensionReducer.move_center()` not called."
            )

            centered = array - self._center.virtual * center_coefficient
            components = self._pca.components_[
                range(3)
            ]  # Selected components ignored in Global mode.
            transformed: NPArray = (
                centered @ components.transpose()
                + self._center.visual_3d * center_coefficient
            )
            return transformed

        if data_point_array.shape[0] == 0:
            return np.empty((0,))

        center_coefficient = 1 if is_position else 0

        if self._is_global:
            return _transform_global(data_point_array, center_coefficient)

        return _transform_aligned(data_point_array, center_coefficient)

    def transform(self, vectors: NPArray, *, is_position: bool = True) -> NPArray:
        """
        Transform data points based on dimension reduction performed previously.

        :param vectors: Vectors in virtual space to transform.
        :param is_position: Whether `vectors` represents positions. Positions will be
          adjusted so that `position` set by `DimensionReducer.move_center()` will be
          transformed to `position_3d`. If `vectors` represent directions the adjusting
          will not be performed.

        :return: Transformed coordinates.
        """
        return self._transform(vectors, is_position=is_position)

    @override
    def fit_transform(self, nodes: VisualNodeCollection, *, force_fit: bool) -> NPArray:
        data_point_array = nodes.coordinates_to_array()

        if force_fit:
            did_dimension_reduction = self._fit(data_point_array)

        elif hasattr(self._pca, 'components_'):
            did_dimension_reduction = True

        else:
            self._fit_with_old_dimension_reduction()
            did_dimension_reduction = True

        transformed = self._transform(data_point_array)

        visual_information = self._get_visual_information()

        if did_dimension_reduction:
            visual_information.update_components(self._pca.components_)

        else:
            visual_information.update_components(np.empty(0))

        return transformed

    @override
    def get_depth(self) -> float | None:
        assert self._component_selection is not None, "`fit_transform()` not called."
        assert self._virtual_dimensions is not None

        return min(self._component_selection) / self._virtual_dimensions

    @override
    def get_message(self) -> list[str]:
        message = []

        if (
            hasattr(self._pca, 'explained_variance_ratio_')
            and self._pca.explained_variance_ratio_ is not None
        ):
            percentage_str = str(np.round(self._pca.explained_variance_ratio_ * 100, 1))
            percentage_str_list = percentage_str.split('\n')
            message.extend(
                [
                    "PCA: Explained Variance (%): " + percentage_str_list[0],
                    *percentage_str_list[1:],
                ]
            )

        message.append("PCA: Selected components: " + str(self._component_selection))

        return message

    def _get_visual_information(self) -> _VisualInformation:
        if self._visual_information is None:
            self._visual_information = _VisualInformation()

        return self._visual_information

    @override
    def get_visual_information(self) -> VisualInformation | None:
        return self._get_visual_information()
