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
Module for alorithms for calculating weights for data points.

Array type: numpy
"""

from abc import (
    ABC,
    abstractmethod,
)
from collections.abc import (
    Callable,
)
from typing import (
    cast,
)

from typing_extensions import override

import numpy as np
import scipy.stats

from pyqoolloop.factory import RegistryFactory

from ..datasource.api import (
    distance_between,
)
from ..mathutils import NPArray


class WeightAlgorithm(ABC):
    """Superclass of algorithms for weight calculation."""

    factory = RegistryFactory['WeightAlgorithm']()

    @abstractmethod
    def get_weights(self, data_point_array: NPArray, center: NPArray) -> NPArray | None:
        """
        Calculate weights for each data point.

        :param data_point_array: Coordinates of data points, each in a row.
        :param center: Center of neighborhood.

        :returns: Weights for each data point in `data_points`.
          `None` for equal weights.
        """

    @staticmethod
    def _calculate_weights_from_distance(
        data_point_array: NPArray,
        center: NPArray,
        weight_function: Callable[[float], float],
    ) -> NPArray | None:
        weights = np.zeros((data_point_array.shape[0],))
        for index, each in enumerate(data_point_array):
            distance = distance_between(each, center)
            weights[index] = weight_function(distance)

        return weights


@WeightAlgorithm.factory.register
class Equal(WeightAlgorithm):
    """Class that assigns equal weights for each data point."""

    @override
    def get_weights(self, data_point_array: NPArray, center: NPArray) -> NPArray | None:
        return None


@WeightAlgorithm.factory.register
class Linear(WeightAlgorithm):
    """
    Class that assigns decreasing L1 weights for each data point.

    :param radius: The distance where the weight diminishes to 0.
    """

    def __init__(self, radius: float) -> None:
        self._radius = radius

    @override
    def get_weights(self, data_point_array: NPArray, center: NPArray) -> NPArray | None:
        def _calculate_weight(distance: float) -> float:
            return max(0, 1 - distance / self._radius)

        return super()._calculate_weights_from_distance(
            data_point_array, center, _calculate_weight
        )


@WeightAlgorithm.factory.register
class Gaussian(WeightAlgorithm):
    """
    Class that assigns Gaussian weights for each data point.

    :param sigma: Standard deviation for the Gaussian curve.
    """

    def __init__(self, sigma: float) -> None:
        self._sigma = sigma

    @override
    def get_weights(self, data_point_array: NPArray, center: NPArray) -> NPArray | None:
        def _calculate_weight(distance: float) -> float:
            return cast(float, scipy.stats.norm(0, self._sigma).pdf(distance))

        return super()._calculate_weights_from_distance(
            data_point_array, center, _calculate_weight
        )
