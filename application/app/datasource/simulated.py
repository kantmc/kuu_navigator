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
Simulated data sources.

Array type: python
"""

from abc import (
    ABC,
    abstractmethod,
)
from collections.abc import (
    Generator,
    Iterable,
    Sequence,
)
from dataclasses import dataclass
from math import (
    ceil,
    floor,
)
from pathlib import Path
from typing import (
    Any,
    TypeAlias,
    TypeVar,
)

from typing_extensions import override

import numpy as np
from numpy.typing import NDArray

from direct.showbase.ShowBase import Loader
from panda3d.core import (
    CardMaker,
    NodePath,
)

import pylog
from pyqoolloop.factory import RegistryFactory

from ..mathutils import NPArray
from .api import (
    ColoredDataPointRectangles,
    Coordinates,
    DataPoint,
    DataPointCollection,
    DataSource,
    DataStorage,
    NPExtremes,
    distance_between,
)

_logger = pylog.getLogger(__name__)


_NPIntArray: TypeAlias = NDArray[np.int64]
"""Type for numpy array made from int."""


@dataclass(frozen=True)
class _NPIntExtremes:
    """Class that holds minimums and maximums in :data:`_NPIntArray`."""

    mins: _NPIntArray
    """Minimums"""

    maxs: _NPIntArray
    """Maximums"""


T = TypeVar('T')


def _nested_range_loop(ranges: Sequence[Iterable[T]]) -> Generator[list[T], None, None]:
    """
    Return a generator that acts like a nested loop with arbitrary depth.

    :param ranges: `Sequence` of `Iterable` (e.g. `range()`) of values to loop
      through.

    :yields: `list` of concatenated values taken from each of the
        `Iterable`s in `ranges`.
    """
    if len(ranges) == 1:
        for value in ranges[0]:
            yield [value]

    else:
        for each in _nested_range_loop(ranges[1:]):
            for value in ranges[0]:
                yield [value, *each]


def _nested_loop(
    mins: Sequence[int] | _NPIntArray, maxs: Sequence[int] | _NPIntArray
) -> Generator[list[int], None, None]:
    """
    Return a generator that acts like a nested loop with arbitrary depth.

    :param mins: A `Sequence` (e.g. `list`) of minimum values of the loop.
    :param maxs: A `Sequence` of maximum values of the loop.

    :yields: `list` of concatenated values, each element between `mins` and
        `maxs`, boundaries included.

    See tests in `test_datasource.py`.
    """
    ranges = [
        range(minimum, maximum + 1) for minimum, maximum in zip(mins, maxs, strict=True)
    ]
    return _nested_range_loop(ranges)


class _DataPointCard(DataPoint):
    """
    Data point drawn with the same image.

    :param coordinates: The coordinates of this data point.

    Other keyword arguments are ignored.
    """

    def __init__(self, coordinates: Coordinates, **_: Any) -> None:
        super().__init__(coordinates)
        self._image_path = Path('maps/envir-reeds.png')

    @staticmethod
    def _make_card(image_path: Path, visual_size: float, loader: Loader) -> NodePath:
        half_size = visual_size / 2

        card_maker = CardMaker('card')
        card_maker.set_frame(-half_size, half_size, -half_size, half_size)

        card = NodePath('DataPointCard')
        card.attachNewNode(card_maker.generate())

        texture = loader.loadTexture(image_path)
        card.setTexture(texture)

        return card

    @override
    def make_nodepath(
        self, visual_size: float, detail: DataPoint.Detail, loader: Loader
    ) -> NodePath:
        card = self._make_card(self._image_path, visual_size, loader)
        return card


class Metric(ABC):
    """Provides API for calculating distances (metrics)."""

    factory = RegistryFactory['Metric']()
    """:class:`pyqoolloop.factory.RegistryFactory` for metrics."""

    @abstractmethod
    def get_distance(self, one: Coordinates, another: Coordinates) -> float:
        """Calculate distance between 2 coordinates."""


@Metric.factory.register
class Euclidean(Metric):
    """Calculates Euclidean distance."""

    @override
    def get_distance(self, one: Coordinates, another: Coordinates) -> float:
        return distance_between(one, another)


class ColoredRectangleDataStorage(DataStorage):
    """
    A :class:`DataStorage` that shows data points as colored rectangles.

    :param data_points: Data points in each row.
    :param extremes: The minimum/maximum possible value for each dimension (in
        `colors` if it exists and `raw_color = False`, in `data_points` if not).
    :param neighborhood_shape: Shape of neighborhood. One of: 'sphere', 'cube'
    :param colors: Color attributes for each data point with values between
        `mins` and `maxs`. Every 3 elements correspond to R, G, B, if the
        width is the same as for `data_points`. If the width is 3 times the
        dimensions for `data_points`, then each 3 elements in `colors` will be
        treated as the intended colors (RGB) for the corresponding data
        point.
    :param raw_color: Whether `color` specifies raw RGB color.
    :param random_seed: A seed for the random number generator. Set an `int` value, if
      using `shuffle()`.
    """

    def __init__(  # Allow many arguments, because these will be set in parameter file.
        self,
        data_points: NPArray,
        *,
        extremes: NPExtremes,
        neighborhood_shape: str,
        colors: NPArray | None = None,
        raw_color: bool = False,
        random_seed: int | None = None,
    ) -> None:
        super().__init__(
            data_points, neighborhood_shape=neighborhood_shape, random_seed=random_seed
        )

        self._extremes = extremes

        if colors is None:
            self._colors_specified = False
            self._colors = data_points

        else:
            self._colors_specified = True
            self._colors = colors

        self._raw_color = raw_color

        if (
            not raw_color
            and (len(self._colors) > 0)
            and (
                np.any(self._colors < extremes.lower)
                or np.any(extremes.upper < self._colors)
            )
        ):
            _logger.warning("Data Points (or colors) are not within extremes.")

    @override
    def merge(self, other: 'DataStorage') -> None:
        super().merge(other)

        if isinstance(other, ColoredRectangleDataStorage):
            if not self._raw_color and (
                np.any(
                    other._colors < self._extremes.lower  # noqa: SLF001
                )
                or np.any(
                    self._extremes.upper < other._colors  # noqa: SLF001
                )
            ):
                _logger.warning("Data Points are not within extremes.")

            assert (
                self._raw_color == other._raw_color  # noqa: SLF001
            ), "Same format for `colors` needs to be specified for `other`."

            assert self._colors_specified == other._colors_specified, (  # noqa: SLF001
                "`colors` must have been specified for `other`"
                if self._colors_specified
                else "`colors` should not have been specified for `other`."
            )

            self._colors = np.append(self._colors, other._colors, axis=0)  # noqa: SLF001

        else:
            assert not self._colors_specified, (
                "Cannot merge with another class if `colors` was specified."
            )

            self._colors = np.append(self._colors, other._data_points, axis=0)  # noqa: SLF001

    @override
    def _make_data_point(self, index: int) -> DataPoint:
        coordinates = self._data_points[index, :]

        color = self._colors[index]

        data_point = ColoredDataPointRectangles(
            coordinates,
            attributes=color,
            extremes=self._extremes,
            raw_color=self._raw_color,
        )

        return data_point


@DataSource.factory.register
class GridData(DataSource):
    """
    Data Source that provides points on a grid in multi-dimensional space.

    Points will be generated on an integer grid between the minimum and
    maximum values. Then, the grid will be scaled with scaling parameters.

    :param int_mins: Sequence of smallest values for each dimension before scaling.
    :param int_maxs: Sequence of largest values for each dimension before scaling.
    :param scale: Sequence of scaling parameters for each dimension. `None`, for no
      scaling.
    :param random_sampling_threshold: Switch to random sampling, if the number of total
      samples in the population (cube neighborhood in Local mode, or all samples in
      Global mode) exceeds this value.  The same data points will be sampled, otherwise.
    :param random_seed: Seed to use for the random number generator.
    :param neighborhood_shape: Shape of neighborhood. One of: `'sphere'`, `'cube'`
    :param _comments: Placeholder for comments in json files.
    """

    def __init__(
        self,
        *,
        int_mins: Sequence[int],
        int_maxs: Sequence[int],
        scale: Sequence[float] | float | None = None,
        random_sampling_threshold: int = 100000,
        random_seed: int = 1234,
        neighborhood_shape: str = 'sphere',
        _comments: str = '',
    ) -> None:
        assert len(int_mins) == len(int_maxs)

        super().__init__(neighborhood_shape=neighborhood_shape)

        self._dimensions = len(int_mins)

        if scale is None:
            self._scale: NPArray = np.array([1.0] * self._dimensions)

        elif isinstance(scale, float):
            self._scale = np.array([scale] * len(int_mins))

        else:
            assert len(scale) == self._dimensions
            self._scale = np.array(scale)

        self._int_extremes = _NPIntExtremes(
            mins=np.array(int_mins), maxs=np.array(int_maxs)
        )

        self._extremes = NPExtremes(
            lower=np.multiply(int_mins, self._scale),
            upper=np.multiply(int_maxs, self._scale),
        )

        self._random_sampling_threshold = random_sampling_threshold

        self._generator = np.random.default_rng(seed=random_seed)

    @override
    def get_dimensions(self) -> int:
        return self._dimensions

    @staticmethod
    def _calculate_total_number(int_mins: _NPIntArray, int_maxs: _NPIntArray) -> int:
        ranges = int_maxs - int_mins + 1

        # expecting to use `bignum`, because this could get pretty large
        product = 1
        for each in ranges:
            product *= int(each)

        return product

    @override
    def get_total_number(self) -> int:
        return self._calculate_total_number(
            self._int_extremes.mins, self._int_extremes.maxs
        )

    def _make_data_point(
        self, coordinates: Coordinates, extremes: NPExtremes
    ) -> DataPoint:
        """
        Make :class:`DataPoint` for specified coordinates.

        :param coordinates: Coordinates for the Data Point.
        :param extremes: The extreme values for all of the Data Points.
        :returns: Created :class:`DataPoint`.
        """
        data_point = (
            ColoredDataPointRectangles(  # `DataPointRectangles` or `_DataPointCard`
                coordinates, attributes=coordinates, extremes=extremes
            )
        )

        return data_point

    def _make_data_point_collection(
        self, data_point_array: NPArray
    ) -> DataPointCollection:
        data_point_collection = DataPointCollection()
        for each in data_point_array:
            data_point = self._make_data_point(each, self._extremes)
            data_point_collection.add(data_point)

        return data_point_collection

    def _stable_sample(
        self,
        *,
        int_mins: _NPIntArray,
        int_maxs: _NPIntArray,
        scale: NPArray,
        max_number: int,
        total_number: int,
    ) -> NPArray:
        def _generate_data_point_array(
            *,
            int_mins: _NPIntArray,
            int_maxs: _NPIntArray,
            scale: NPArray,
            max_number: int,
            total_number: int,
        ) -> NPArray:
            max_number = min(total_number, max_number)

            data_point_array = np.empty((0, len(int_mins)))

            last_stride_count = 0
            for count, indexes in enumerate(_nested_loop(int_mins, int_maxs)):
                # FUTURE: Little worried about calculation error
                stride_count = int(float(count + 1) / total_number * max_number + 0.5)
                if stride_count == last_stride_count:
                    continue

                last_stride_count = stride_count

                coordinates = np.multiply(np.array(indexes), scale)
                data_point_array = np.append(
                    data_point_array, coordinates.reshape(1, len(coordinates)), axis=0
                )

            return data_point_array

        data_point_array = _generate_data_point_array(
            int_mins=int_mins,
            int_maxs=int_maxs,
            scale=scale,
            max_number=max_number,
            total_number=total_number,
        )

        assert len(data_point_array) <= max_number, (
            f"{len(data_point_array)=} exceeds {max_number=}."
        )

        return data_point_array

    def _random_sample(
        self,
        *,
        int_mins: _NPIntArray,
        int_maxs: _NPIntArray,
        scale: NPArray,
        count: int,
    ) -> NPArray:
        _logger.info("Sampling randomly.")

        data_point_array = (
            self._generator.integers(
                int_mins, int_maxs, size=(count, len(int_mins)), endpoint=True
            )
            * scale
        )

        return data_point_array

    def _get_from_grid(
        self,
        *,
        int_mins: _NPIntArray,
        int_maxs: _NPIntArray,
        scale: NPArray,
        max_number: int,
    ) -> DataPointCollection:
        total_number = GridData._calculate_total_number(int_mins, int_maxs)
        if total_number > self._random_sampling_threshold:
            data_point_array = self._random_sample(
                int_mins=int_mins, int_maxs=int_maxs, scale=scale, count=max_number
            )

        else:
            data_point_array = self._stable_sample(
                int_mins=int_mins,
                int_maxs=int_maxs,
                scale=scale,
                max_number=max_number,
                total_number=total_number,
            )

        return self._make_data_point_collection(data_point_array)

    @override
    def get_from_all(self, max_number: int) -> DataPointCollection:
        return self._get_from_grid(
            int_mins=self._int_extremes.mins,
            int_maxs=self._int_extremes.maxs,
            scale=self._scale,
            max_number=max_number,
        )

    def _get_grid_int_min_max(
        self, center: Coordinates, radius: float
    ) -> tuple[_NPIntArray, _NPIntArray]:
        descaled_center = np.array(center) / self._scale
        descaled_radii = radius / self._scale

        int_mins: _NPIntArray = np.array(
            [
                max(int(ceil(each - radius)), minimum)
                for each, radius, minimum in zip(
                    descaled_center,
                    descaled_radii,
                    self._int_extremes.mins,
                    strict=True,
                )
            ]
        )
        int_maxs: _NPIntArray = np.array(
            [
                min(int(floor(each + radius)), maximum)
                for each, radius, maximum in zip(
                    descaled_center,
                    descaled_radii,
                    self._int_extremes.maxs,
                    strict=True,
                )
            ]
        )

        return int_mins, int_maxs

    @override
    def _get_from_cube_neighborhood(
        self, center: Coordinates, radius: float, max_number: int
    ) -> DataPointCollection:
        assert radius > 0

        def _is_point_in_range(point: Coordinates) -> bool:
            for each_element, minimum, maximum in zip(
                point,
                self._int_extremes.mins * self._scale,
                self._int_extremes.maxs * self._scale,
                strict=True,
            ):
                if maximum < (each_element - radius):
                    return False

                if (each_element + radius) < minimum:
                    return False

            return True

        if not _is_point_in_range(center):
            return DataPointCollection()

        int_mins, int_maxs = self._get_grid_int_min_max(center, radius)

        return self._get_from_grid(
            int_mins=int_mins,
            int_maxs=int_maxs,
            scale=self._scale,
            max_number=max_number,
        )


@DataSource.factory.register
class RandomData(ColoredRectangleDataStorage):
    """
    Data Source with uniform random data points.

    Points will be generated at random between the minimum and maximum
    values.

    :param mins: Sequence of smallest values for each dimension.
    :param maxs: Sequence of largest values for each dimension.
    :param count: Number of data points in all.
    :param neighborhood_shape: Shape of neighborhood. One of: `'sphere'`, `'cube'`.
    :param random_seed: A seed for the random number generator.
    :param _comments: Placeholder for comments in json files.
    """

    def __init__(
        self,
        *,
        mins: Sequence[float],
        maxs: Sequence[float],
        count: int,
        neighborhood_shape: str = 'sphere',
        random_seed: int | None = None,
        _comments: str = '',
    ) -> None:
        assert len(mins) == len(maxs)

        generator = np.random.default_rng(seed=random_seed)

        minimums: NPArray = np.array(mins)
        maximums: NPArray = np.array(maxs)

        self._count = count

        data_points = (
            generator.random(size=(self._count, len(minimums))) * (maximums - minimums)
            + minimums
        )

        super().__init__(
            data_points,
            extremes=NPExtremes(lower=minimums, upper=maximums),
            neighborhood_shape=neighborhood_shape,
        )


@DataSource.factory.register
class Gaussian(ColoredRectangleDataStorage):
    """
    Data storage with multi-dimensional Gaussian distribution.

    :param count: Number of data points.
    :param color_radius: Data point will be colored white if any of the
        dimensions exceed this value.
    :param dimensions: Number of dimensions. If `None`, replaced with `len(scale)`.
    :param scale: Standard deviations for each dimension. The remainder of the
      dimensions will be filled with 1.
    :param neighborhood_shape: Shape of neighborhood. One of: `'sphere'`, `'cube'`.
    :param random_seed: A seed for the random number generator.
    :param _comments: Placeholder for comments in json files.
    """

    def __init__(
        self,
        *,
        count: int,
        color_radius: float,
        dimensions: int | None = None,
        scale: Sequence[float] | None = None,
        neighborhood_shape: str = 'sphere',
        random_seed: int | None = None,
        _comments: str = '',
    ) -> None:
        generator = np.random.default_rng(seed=random_seed)

        def _generate_data_points(
            scale: Sequence[float], count: int, color_radius: float
        ) -> tuple[NPArray, NPArray]:
            primary_colors: _NPIntArray = np.array(
                [
                    [[1, 0, 0], [0, -1, -1]],
                    [[0, 1, 0], [-1, 0, -1]],
                    [[0, 0, 1], [-1, -1, 0]],
                ]
            )

            dimensions = len(scale)

            data_points = generator.normal(scale=scale, size=(count, dimensions))
            colors = np.zeros((count, dimensions * 3))
            for index in range(dimensions):
                color_index = index % 3

                data_points_slice = data_points[:, index] / color_radius

                peripheral = (data_points_slice < -1) | (1 < data_points_slice)
                positives = (0 <= data_points_slice) & ~peripheral
                negatives = (data_points_slice < 0) & ~peripheral

                colors[positives, (index * 3) : (index * 3 + 3)] = (
                    data_points_slice[positives, None]
                    @ primary_colors[None, color_index, 0]
                )
                colors[negatives, (index * 3) : (index * 3 + 3)] = (
                    data_points_slice[negatives, None]
                    @ primary_colors[None, color_index, 1]
                )
                colors[peripheral, (index * 3) : (index * 3 + 3)] = np.array([1, 1, 1])

            return data_points, colors

        def _extremes(dimensions: int, color_radius: float) -> NPExtremes:
            maxs: NPArray = np.array([color_radius] * dimensions)
            mins = -maxs
            return NPExtremes(lower=mins, upper=maxs)

        if scale is None:
            assert dimensions is not None
            scale = [1.0] * dimensions

        elif dimensions is None:
            dimensions = len(scale)

        else:
            scale = list(scale) + [1.0] * (dimensions - len(scale))

        data_points, colors = _generate_data_points(scale, count, color_radius)
        extremes = _extremes(len(scale), color_radius)

        super().__init__(
            data_points,
            extremes=extremes,
            neighborhood_shape=neighborhood_shape,
            colors=colors,
            raw_color=True,
        )


@DataSource.factory.register
class SwissRoll(ColoredRectangleDataStorage):
    """
    Data Source with multi-dimensional swiss roll.

    :param count: The number of points to be generated.
    :param start_radius: Radius of the swiss roll at the largest point.
    :param fall_rate: The rate at which the swiss roll approaches the
      origin.
    :param rotations: How many rotations to make.
    :param widths: The widths of the swiss roll for the 3rd dimension and onwards.
    :param virtual_dimensions: (>= `len(widths) + 2`) The total number of dimensions for
      the data.
    :param neighborhood_shape: Shape of neighborhood. One of: `'sphere'`, `'cube'`.
    :param random_seed: A seed for the random number generator.
    :param _comments: Placeholder for comments in json files.
    """

    def __init__(
        # Allow many arguments, because these will be set in parameter file.
        self,
        *,
        count: int,
        start_radius: float,
        fall_rate: float,
        rotations: float,
        widths: Sequence[float],
        virtual_dimensions: int = 3,
        neighborhood_shape: str = 'sphere',
        random_seed: int | None = None,
        _comments: str = '',
    ) -> None:
        generator = np.random.default_rng(seed=random_seed)

        def _generate_data_points() -> tuple[NPArray, NPArray]:
            phi = generator.random(size=(self._count, 1)) * self._length
            random_components = (
                generator.random(size=(self._count, len(self._widths))) * self._widths
                - self._widths / 2
            )
            fixed_components = np.zeros(
                (self._count, self._dimensions - len(self._widths) - 2)
            )

            data_points: NPArray = np.hstack(
                (
                    self._start_radius
                    * (1 - phi / (2 * np.pi) * self._fall_rate)
                    * np.cos(phi),
                    self._start_radius
                    * (1 - phi / (2 * np.pi) * self._fall_rate)
                    * np.sin(phi),
                    random_components,
                    fixed_components,
                )
            )

            colors: NPArray = np.hstack((phi, random_components))

            permutation = generator.permutation(data_points.shape[0])
            return data_points[permutation, :], colors[permutation, :]

        def _color_extremes() -> NPExtremes:
            mins = np.concatenate(
                [[0], np.ones((len(self._widths),)) * (-self._widths / 2)]
            )

            maxs = np.concatenate(
                [[self._length], np.ones((len(self._widths),)) * (self._widths / 2)]
            )

            return NPExtremes(lower=mins, upper=maxs)

        assert virtual_dimensions >= len(widths) + 2

        self._count = count
        self._start_radius = start_radius
        self._fall_rate = fall_rate
        self._length = rotations * 2 * np.pi
        self._widths = np.array(widths)
        self._dimensions = virtual_dimensions

        data_points, colors = _generate_data_points()
        color_extremes = _color_extremes()

        super().__init__(
            data_points,
            extremes=color_extremes,
            colors=colors,
            neighborhood_shape=neighborhood_shape,
        )


@DataSource.factory.register
class Desk(ColoredRectangleDataStorage):
    """
    A distribution that resembles a desk, if in 3D.

    More specifically, there are 2 hypercubes of N dimensions connected by a 2D plane.

    :param connection_length: Length of connecting distribution.
    :param connection_count: Number of Data Points in the connecting distribution.
    :param cube_lengths: The lengths for each dimension for the hypercubes.
    :param cube_count: Number of Data Points in each cube.
    :param neighborhood_shape: Shape of neighborhood. One of: `'sphere'`, `'cube'`.
    :param random_seed: A seed for the random number generator.
    :param _comments: Placeholder for comments in json files.
    """

    def __init__(
        self,
        *,
        connection_length: float,
        connection_count: int,
        cube_lengths: list[float],
        cube_count: int,
        neighborhood_shape: str,
        random_seed: int,
        _comments: str = '',
    ) -> None:
        def _make_shapes(
            *,
            connection_length: float,
            connection_count: int,
            cube_lengths: list[float],
            cube_count: int,
            neighborhood_shape: str,
            random_seed: int,
        ) -> list[DataStorage]:
            cube_depth = cube_lengths[0]

            cube_width = cube_lengths[1]

            cube_dimensions = len(cube_lengths)

            connection_dimensions = 2

            first_cube = RandomData(
                mins=[0] * cube_dimensions,
                maxs=cube_lengths,
                count=cube_count,
                neighborhood_shape=neighborhood_shape,
                random_seed=random_seed,
            )

            connection = RandomData(
                mins=[cube_width] + [0] * (cube_dimensions - 1),
                maxs=[cube_width + connection_length, cube_depth]
                + [0] * (cube_dimensions - connection_dimensions),
                count=connection_count,
                neighborhood_shape=neighborhood_shape,
                random_seed=random_seed + 1,
            )

            second_cube = RandomData(
                mins=[cube_width + connection_length] + [0] * (cube_dimensions - 1),
                maxs=[2 * cube_width + connection_length, cube_depth]
                + cube_lengths[2:],
                count=cube_count,
                neighborhood_shape=neighborhood_shape,
                random_seed=random_seed + 2,
            )

            return [first_cube, connection, second_cube]

        cube_depth = cube_lengths[0]

        cube_width = cube_lengths[1]

        cube_dimensions = len(cube_lengths)

        connection_dimensions = 2

        assert connection_dimensions <= cube_dimensions

        extremes = NPExtremes(
            lower=np.array([0] * cube_dimensions),
            upper=np.array(
                [2 * cube_width + connection_length, cube_depth] + cube_lengths[2:]
            ),
        )

        super().__init__(
            data_points=np.empty((0, cube_dimensions)),
            extremes=extremes,
            neighborhood_shape=neighborhood_shape,
            random_seed=random_seed - 1,
        )

        shapes = _make_shapes(
            connection_length=connection_length,
            connection_count=connection_count,
            cube_lengths=cube_lengths,
            cube_count=cube_count,
            neighborhood_shape=neighborhood_shape,
            random_seed=random_seed,
        )
        for each in shapes:
            self.merge(each)

        self.shuffle()
