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

"""Module to make math easier."""

from collections.abc import Iterable
from dataclasses import dataclass
from typing import (
    Any,
    TypeAlias,
    cast,
)

import numpy as np
from numpy.typing import NDArray

NPArray: TypeAlias = NDArray[np.float64]
"""Type for numpy array made from float."""


VISUAL_DIMENSIONS = 3
"""Dimensions of Visual Space."""

PLANE_DIMENSIONS = 2
"""Dimensions on a plane."""


@dataclass(frozen=False)
class VectorPair:
    """Class to hold a pair of vectors, one in visual space and one in virtual space."""

    visual_3d: NPArray
    virtual: NPArray

    def __eq__(  # type: ignore[override]  # to avoid unintentional comparison
        self, other: 'VectorPair | None'
    ) -> bool:
        if not isinstance(other, VectorPair):
            return False

        return np_all(self.visual_3d == other.visual_3d) and np_all(
            self.virtual == other.virtual
        )


@dataclass(frozen=False)
class OptionalVectorPair:
    """
    Class to hold a pair of vectors, one in visual space and one in virtual space.

    The one in virtual space is optional.
    """

    visual_3d: NPArray
    """Directions for camera in visual space."""

    virtual: NPArray | None
    """Directions for camera in virtual space (obtained after PCA)."""

    @staticmethod
    def from_vector_pair(vector_pair: VectorPair) -> 'OptionalVectorPair':
        """Create from `VectorPair`."""
        return OptionalVectorPair(
            visual_3d=vector_pair.visual_3d, virtual=vector_pair.virtual
        )


def split_float(value: float) -> tuple[float, int]:
    """
    Split a `float` into components of scientific notation.

    :return: `tuple` of mantissa and exponent.
    """
    decomposed = f'{value:e}'.split('e')

    mantissa = float(decomposed[0])

    exponent = int(decomposed[1])

    return (mantissa, exponent)


def np_all(value: bool | Any) -> bool:  # noqa: ANN401
    """
    Test that all elements are True.

    Same as :func:`numpy.all()`, but returns `bool`.
    """
    return bool(np.all(value))


def weighted_sum(ratio: float, first: NPArray, second: NPArray) -> NPArray:
    """
    Take the weighted sum.

    :param ratio: How much of `first` to use.
    :param first: One of the inputs.
    :param second: Another input.

    :return: A weighted sum of `first` and `second`.
    """
    return first * (1.0 - ratio) + second * ratio


def magnitude(
    vector: NPArray,
) -> float:
    """Calculate length of vector."""
    return cast(float, np.sqrt(np.inner(vector, vector)))


def normalize(vector: NPArray) -> NPArray:
    """Stretch or shrink vector to length of 1."""
    return np.divide(vector, magnitude(vector))


def random_bools(
    generator: np.random.Generator, length: int, true_count: int
) -> Iterable[bool]:
    """
    Generate a random array of bools.

    :param generator: A random number generator.
    :param length: Length of the generated array.
    :param true_count: Number of `True` in the array.

    :return: A random array with `true_count` number of `True`.
    """
    bools = np.zeros((length,))
    selection = generator.choice(length, (true_count,), replace=False)
    bools[selection] = True
    return bools
