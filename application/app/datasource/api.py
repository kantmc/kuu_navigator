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
API for data sources.

Array type: numpy
"""

from abc import (
    ABC,
    abstractmethod,
)
from collections.abc import (
    Iterable,
    Iterator,
    Sequence,
)
from dataclasses import dataclass
from enum import Enum
from functools import total_ordering
from io import TextIOWrapper
import json
from math import (
    sqrt,
)
from pathlib import Path

from typing_extensions import override

import numpy as np
from numpy.typing import NDArray

from direct.showbase.ShowBase import Loader
from panda3d.core import (
    CardMaker,
    LVector4,
    MeshDrawer2D,
    NodePath,
    Texture,
)

from pyqoolloop.factory import RegistryFactory

from ..mathutils import NPArray


@dataclass(frozen=True)
class NPExtremes:
    """Class that holds upper and lower bounds in :data:`NPArray`."""

    lower: NPArray
    """Lower bounds."""

    upper: NPArray
    """Upper bounds."""

    @classmethod
    def from_array(cls, array: NPArray, dimensions: int | None = None) -> 'NPExtremes':
        """
        Create :class:`NPExtremes` from numpy array.

        :param array: 2D array with vectors in each row. Each vector is a sequence of
          numbers grouped into `dimensions` elements.
        :param dimensions: Number of elements in a group.  If `None`, use the length of
          each row.

        :return: :class:`NPExtremes` instance with upper and lower bounds. The number of
          each are equal to `dimensions`.
        """
        if len(array) == 0:
            return NPExtremes(lower=np.array([]), upper=np.array([]))

        if dimensions is None:
            dimensions = array.shape[1]

        new_size = [array.shape[0], array.shape[1] // dimensions, dimensions]
        reshaped = array.reshape(new_size)

        minimums = reshaped.min(axis=1).min(axis=0)
        maximums = reshaped.max(axis=1).max(axis=0)
        return NPExtremes(lower=minimums, upper=maximums)


Coordinates = Sequence[float] | NDArray[np.float64]
"""
Sequence type for coordinates.

Should be treated as immutable.

An example of immutable coordinates:

>>> coordinates = np.array([0, 0, 0])
>>> coordinates.setflags(write=False)
>>> coordinates[0] = 1
Traceback (most recent call last):
...
ValueError: assignment destination is read-only
"""
# doctest doesn't test comments for variables


def distance_between(one: Coordinates, another: Coordinates) -> float:
    """Calculate Euclidean distance between 2 coordinates."""
    # FUTURE: Should use Dimension Reducer
    return sqrt(sum(((me - him) ** 2 for me, him in zip(one, another, strict=True))))


def equal_coordinates(one: Coordinates, another: Coordinates) -> bool:
    """Return `True` if the 2 :class:`Coordinates` are equal."""
    for each_in_one, each_in_another in zip(one, another, strict=True):
        if each_in_one != each_in_another:
            return False

    return True


class DataPoint:
    """
    Data point with immutable coordinates in virtual space.

    :param coordinates: Coordinates for this instance.

    .. note:: Intentionally doesn't subclass :class:`Coordinates`, because they
      are usually in a difference space.
    """

    def __init__(self, coordinates: Coordinates) -> None:
        self._coordinates: NPArray = np.array(coordinates)
        self._coordinates.setflags(write=False)

    def __eq__(  # type: ignore[override]  # to avoid unintentional comparison
        self, other: 'DataPoint | None'
    ) -> bool:
        """
        Return `True` if coordinates are equal for `other` :class:`DataPoint`.

        .. note:: Argument is intentionally typed, so that mypy can detect unintentional
          usage.
        """
        if not isinstance(other, DataPoint):
            return False

        return np.array_equal(self._coordinates, other._coordinates)

    def __hash__(self) -> int:
        """Return hash value for the coordinates."""
        return hash(tuple(self._coordinates))

    def __len__(self) -> int:
        """Return number of dimensions in the coordinates."""
        return len(self._coordinates)

    def get_coordinates(
        self,
    ) -> Coordinates:
        """Get coordinates."""
        return self._coordinates

    @total_ordering
    class Detail(Enum):
        """Level of detail."""

        LOW = 1
        """Used as default in Navigation Scene."""

        MEDIUM = 2
        """Used for detailed nodes in Navigation Scene."""

        HIGH = 3
        """Used in Detail Scene."""

        def __gt__(self, other: 'DataPoint.Detail') -> bool:
            assert isinstance(other, DataPoint.Detail)

            return self.value > other.value

    @abstractmethod
    def make_nodepath(
        self, visual_size: float, detail: Detail, loader: Loader
    ) -> NodePath:
        """
        Make a :class:`panda3d.core.NodePath` for this `DataPoint`.

        :param visual_size: Size of visuals. The :class:`panda3d.core.NodePath`
          generated by this method should be roughly included in a cube with an edge of
          this length.
        :param detail: Level of detail.
        :param loader: Loader for texture.
        """
        raise NotImplementedError


class ImageCard(DataPoint):
    """
    Data point drawn with a bitmap image.

    :param coordinates: The coordinates of this data point.
    :param image: Bitmap data for a color 2D image.
    """

    def __init__(self, coordinates: Coordinates, image: NPArray) -> None:
        super().__init__(coordinates)

        self._image = image

    @staticmethod
    def _make_card(image: NPArray, visual_size: float) -> NodePath:
        half_size = visual_size / 2

        card_maker = CardMaker('card')
        card_maker.set_frame(-half_size, half_size, -half_size, half_size)

        card = NodePath('ImageCard')
        card.attachNewNode(card_maker.generate())

        texture = Texture()
        texture.setup2dTexture(
            image.shape[1], image.shape[0], Texture.T_float, Texture.F_rgba32
        )
        texture.setRamImageAs(image.tobytes(), 'RGB')

        card.setTexture(texture)

        return card

    @override
    def make_nodepath(
        self, visual_size: float, detail: DataPoint.Detail, loader: Loader
    ) -> NodePath:
        card = self._make_card(self._image, visual_size)
        return card


class DataPointCollection:
    """Collection of :class:`DataPoint`."""

    def __init__(self) -> None:
        self._set = set[DataPoint]()
        self._dimensions: int | None = None

    def __len__(self) -> int:
        """Return number of :class:`DataPoint` s in the collection."""
        return len(self._set)

    def __iter__(self) -> Iterator[DataPoint]:
        """Return iterator that iterates over all :class:`DataPoint` s."""
        return iter(self._set)

    def __eq__(self, other: object) -> bool:
        """Not implemented."""
        raise NotImplementedError

    def add(self, element: DataPoint) -> None:
        """
        Add data point.

        :param element: Data point to add.
        """
        if self._dimensions is None:
            self._dimensions = len(element)

        else:
            assert self._dimensions == len(element)

        self._set.add(element)

    def update(self, other: 'DataPointCollection') -> None:
        """Add data points in another collection."""
        assert (
            (len(self) == 0)
            or (len(other) == 0)
            or other.get_dimensions() == self.get_dimensions()
        )
        self._set.update(other._set)  # noqa: SLF001

    def get_dimensions(self) -> int:
        """Get dimensions for the data points."""
        assert self._dimensions is not None
        return self._dimensions

    def to_array(self) -> NPArray:
        """
        Convert to numpy array.

        Returns rows in the same order as :meth:`__iter__()`.
        """
        result = []
        for each in self:
            coordinates = each.get_coordinates()
            result.append(coordinates)

        return np.array(result)

    def get_extremes(self) -> NPExtremes:
        """
        Get the min/max values for each of the dimensions in this Collection.

        .. note:: Implementation is not fast.
        """
        array = self.to_array()
        return NPExtremes.from_array(array)


def _open(path: Path) -> TextIOWrapper:
    return path.open(encoding='utf-8')


class DataSource(ABC):
    """
    Provides API for a database.

    :param neighborhood_shape: Shape of the neighborhood. One of the strings for
      `DataSource.Shape`.
    """

    def __init__(self, neighborhood_shape: str) -> None:
        self._neighborhood_shape = DataSource.Shape[neighborhood_shape.upper()]

    factory = RegistryFactory['DataSource']()
    """:class:`pyqoolloop.factory.RegistryFactory` for data sources."""

    Shape = Enum('Shape', 'CUBE SPHERE')

    @staticmethod
    def setup_from_file(path: Path) -> 'DataSource':
        """Instantiate a `DataSource` reading configuration from a file."""
        with _open(path) as file:
            configuration = json.load(file)

        return DataSource.factory.create(
            configuration["type"], configuration["parameters"]
        )

    @abstractmethod
    def get_dimensions(self) -> int:
        """Get number of dimensions for the data points."""

    @abstractmethod
    def get_total_number(self) -> int | None:
        """
        Return total number of data points in data source.

        :returns: Total number of data points. `None`, if not known.
        """

    @abstractmethod
    def get_from_all(self, max_number: int) -> DataPointCollection:
        """
        Sample data points from among all data points.

        :param max_number: Maximum number of data points to sample. Could be less.

        :returns: Collection of sampled data points.
        """

    @abstractmethod
    def _get_from_cube_neighborhood(
        self, center: Coordinates, radius: float, max_number: int
    ) -> DataPointCollection:
        """
        Sample data points within a cube neighborhood of specified point.

        :param center: Center of neighborhood.
        :param radius: Radius for neighborhood. The neighborhood would be a
          multi-dimensional cube of width `2 * radius`.
        :param max_number: A hint for the maximum number of data points to sample.
          This might be ignored.

        :returns: Collection of sampled data points.
        """

    @staticmethod
    def in_sphere(data_point: DataPoint, center: Coordinates, radius: float) -> bool:
        """
        Check whether Data Point is within neighborhood.

        :param data_point: :class:`DataPoint` to check.
        :param center: Center of neighborhood.
        :param radius: Radius for neighborhood.

        :return: `True`, if `data_point` is within the neighborhood.
        """
        return distance_between(data_point.get_coordinates(), center) <= radius

    @classmethod
    def filter_sphere(
        cls, data_points: DataPointCollection, center: Coordinates, radius: float
    ) -> DataPointCollection:
        """
        Choose points within neighborhood.

        :param data_points: Collection of data points to choose from.
        :param center: Center of neighborhood.
        :param radius: Radius for neighborhood.
        """
        new_collection = DataPointCollection()
        for each in data_points:
            if cls.in_sphere(each, center, radius):
                new_collection.add(each)

        return new_collection

    def _get_from_sphere_neighborhood(
        self, center: Coordinates, radius: float, max_number: int
    ) -> DataPointCollection:
        """
        Sample data points within a sphere neighborhood of specified point.

        :param center: Center of neighborhood.
        :param radius: Radius for neighborhood. The neighborhood would be a
          multi-dimensional cube of width `2 * radius`.
        :param max_number: A hint for the maximum number of data points to sample.
          This might be ignored.

        :returns: Collection of sampled data points.
        """
        data_points = self._get_from_cube_neighborhood(center, radius, max_number)
        data_points = self.filter_sphere(data_points, center, radius)
        return data_points

    def get_from_neighborhood(
        self, center: Coordinates, radius: float, max_number: int
    ) -> DataPointCollection:
        """
        Sample data points within neighborhood of specified point.

        :param center: Center of neighborhood.
        :param radius: Approximate radius for neighborhood. The actual distance
          of the returned points may be further from the center than this
          value.
        :param max_number: A hint for the maximum number of data points to sample.
          The actual number returned may be far less.

        :returns: Collection of sampled data points.
        """
        if self._neighborhood_shape == DataSource.Shape.SPHERE:
            data_points = self._get_from_sphere_neighborhood(center, radius, max_number)

        else:
            data_points = self._get_from_cube_neighborhood(center, radius, max_number)

        return data_points


class ColoredDataPointRectangles(DataPoint):
    """
    Data point drawn as colored rectangles.

    :param coordinates: The coordinates of this data point.
    :param attributes:
        [If `raw_color` is `False`]
        Attributes that determine color for this data point with values
        between `mins` and `maxs` in `extremes`. Every 3 elements correspond to R, G, B.
        If `None`, use `coordinates`.
        [If `raw_color` is `True`]
        The actual colors for this data point. Each 3 elements in `color`
        will be treated as the intended colors (RGB).
    :param extremes: Minimum/Maximum possible values for each dimension. (Mandatory
        when `raw_color` is `False`)
    :param raw_color: Whether `attributes` specifies raw RGB color.
    """

    def __init__(
        self,
        coordinates: Coordinates,
        *,
        attributes: Iterable[float],
        extremes: NPExtremes | None = None,
        raw_color: bool = False,
    ) -> None:
        super().__init__(coordinates)
        self._extremes = extremes

        self._coordinates_array: Coordinates = np.array(coordinates)
        self._color: NDArray[np.float64] = (
            np.array(attributes) if attributes is not None else self._coordinates_array
        )
        self._raw_color = raw_color
        if raw_color:
            assert self._color is not None
            assert len(self._color) % 3 == 0

    @override
    def make_nodepath(
        self, visual_size: float, detail: DataPoint.Detail, loader: Loader
    ) -> NodePath:
        def _make_length_3(sequence: NPArray, fill: float) -> Iterable[float]:
            max_length = 3

            if len(sequence) == max_length:
                return sequence

            return list(sequence) + [fill] * (max_length - len(sequence))

        def _make_color(attributes: NPArray, start_index: int) -> LVector4:
            if self._raw_color:
                color: Iterable[float] = attributes[(start_index) : (start_index + 3)]

            else:
                assert self._extremes is not None

                color = [
                    # 0-divide results in `nan`. Seems to be treated like 0 for color.
                    (each_attribute - each_min) / (each_max - each_min)
                    for each_attribute, each_min, each_max in zip(
                        _make_length_3(attributes[start_index : (start_index + 3)], 0),
                        _make_length_3(
                            self._extremes.lower[start_index : (start_index + 3)], 0
                        ),
                        _make_length_3(
                            self._extremes.upper[start_index : (start_index + 3)], 1
                        ),
                        strict=True,
                    )
                ]

            return LVector4(*color, 1)

        def _number_of_colors() -> int:
            return (len(self._color) + 2) // 3

        def _draw_rectangles(
            drawer: MeshDrawer2D, visual_size: float, *, number_of_colors: int
        ) -> None:
            half_size = visual_size / 2

            total_width = 2 * half_size
            interval = total_width / number_of_colors

            for index in range(number_of_colors):
                drawer.rectangleRaw(
                    -half_size + interval * index,
                    -half_size,
                    interval,
                    2 * half_size,
                    0,
                    0,
                    1,
                    1,
                    _make_color(self._color, 3 * index),
                )

        number_of_colors = _number_of_colors()
        budget = (
            number_of_colors
            if detail >= DataPoint.Detail.MEDIUM
            else min(10, number_of_colors)
        )

        drawer = MeshDrawer2D()

        # Crashes in `begin()` if not set. Hangs if number too large.
        drawer.setBudget(budget)
        drawer.begin()
        _draw_rectangles(drawer, visual_size, number_of_colors=budget)
        drawer.end()
        root = drawer.getRoot()
        root.setTwoSided(two_sided=True)  # Will become triangle when `False`
        return root


class DataStorage(DataSource):  # Made public to show in documentation
    """
    A :class:`DataSource` that keeps the data stored in memory.

    :param data_points: Data points in each row.
    :param neighborhood_shape: Shape of neighborhood. One of: 'sphere', 'cube'.
    :param random_seed: A seed for the random number generator. Set an `int` value, if
      using `shuffle()`.
    """

    def __init__(
        self,
        data_points: NPArray,
        *,
        neighborhood_shape: str,
        random_seed: int | None = None,
    ) -> None:
        super().__init__(neighborhood_shape=neighborhood_shape)

        self._dimensions: int = data_points.shape[1]
        self._data_points = data_points

        self._generator = (
            None if random_seed is None else np.random.default_rng(seed=random_seed)
        )

    @override
    def get_dimensions(self) -> int:
        return self._dimensions

    @override
    def get_total_number(self) -> int | None:
        return self._data_points.shape[0]

    def merge(self, other: 'DataStorage') -> None:
        """
        Merge Data Points from another `DataStorage`.

        ..note:: It's usually necessary to call `shuffle()` after all the merges.
        """
        self._data_points = np.append(self._data_points, other._data_points, axis=0)  # noqa: SLF001

    def shuffle(self) -> None:
        """Shuffle Data Points."""
        assert self._generator is not None
        self._generator.shuffle(self._data_points)

    @abstractmethod
    def _make_data_point(self, index: int) -> DataPoint:
        """
        Make :class:`DataPoint` for specified data point.

        :param index: Index for data point.
        :returns: Created :class:`DataPoint`.
        """

    @override
    def get_from_all(self, max_number: int) -> DataPointCollection:
        total_number = self._data_points.shape[0]
        max_number = min(total_number, max_number)

        data_point_collection = DataPointCollection()
        last_stride_count = 0
        for index in range(self._data_points.shape[0]):
            # FUTURE: Little worried about calculation error
            stride_count = int(float(index + 1) / total_number * max_number + 0.5)
            if stride_count == last_stride_count:
                continue

            last_stride_count = stride_count

            data_point = self._make_data_point(index)
            data_point_collection.add(data_point)

        assert len(data_point_collection) <= max_number, (
            f"{len(data_point_collection)=} exceeds {max_number=}."
        )
        return data_point_collection

    @override
    def _get_from_cube_neighborhood(
        self, center: Coordinates, radius: float, max_number: int
    ) -> DataPointCollection:
        def _indexes_to_data_point_collection(
            indexes: NDArray[np.int64],
        ) -> DataPointCollection:
            collection = DataPointCollection()
            for count, each_index in enumerate(indexes):
                if count >= max_number:
                    # FUTURE: Is this biased? But, this is how databases work.
                    break

                data_point = self._make_data_point(each_index)
                collection.add(data_point)

            return collection

        in_neighborhood = np.all(
            np.abs(self._data_points - np.array(center)) < radius, axis=1
        )
        return _indexes_to_data_point_collection(np.nonzero(in_neighborhood)[0])
