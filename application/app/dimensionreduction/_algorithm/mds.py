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

"""Implementation of Spotlighted Multi-Dimensional Scaling."""

from typing_extensions import override

import numpy as np

from skbio import DistanceMatrix
from skbio.stats.ordination import pcoa

from ...mathutils import NPArray, magnitude
from .api import MDSProtocol


class PCoA(MDSProtocol):
    """
    Classical Multi-Dimensional Scaling.

    :param dimensions: Number of dimensions to reduce to. `None` means equal to the
      number of samples.
    """

    def __init__(self, *, dimensions: int | None) -> None:
        super().__init__()

        self._dimensions = dimensions if dimensions is not None else 0

    @override
    def fit_transform(
        self,
        X: NPArray,  # noqa: N803
        init: NPArray | None = None,
    ) -> NPArray:
        def _calc_dissimilarities(data_points: NPArray) -> NPArray:
            number_of_samples = data_points.shape[0]

            dissimilarity_matrix = np.empty(
                (number_of_samples, number_of_samples)
            )  # TODO: Use `squareform(pdist())` https://gist.github.com/mortonjt/219e3485bdfa797da6e74fb22417771d
            for index_one in range(number_of_samples):
                each_one = data_points[index_one, :]
                for index_another in range(index_one + 1, number_of_samples):
                    each_another = data_points[index_another, :]
                    distance = magnitude(each_one - each_another)
                    dissimilarity_matrix[index_one, index_another] = distance
                    dissimilarity_matrix[index_another, index_one] = distance

                dissimilarity_matrix[index_one, index_one] = 0.0

            # TODO: Can use vector-form arrays for `DistanceMatrix`
            return dissimilarity_matrix

        def _fill_dimensions(array: NPArray, dimensions: int) -> NPArray:
            dimension_difference = dimensions - array.shape[1]
            return np.concatenate(
                (array, np.zeros((array.shape[0], dimension_difference))), axis=1
            )

        dissimilarity_matrix = _calc_dissimilarities(X)
        ordination_result = pcoa(
            DistanceMatrix(dissimilarity_matrix),
            number_of_dimensions=min([X.shape[0], X.shape[1], self._dimensions]),
            # FUTURE: `inplace` for speed up?
            # FUTURE: 'fxvd' is faster with less accuracy
        )

        samples: NPArray = ordination_result.samples.to_numpy()
        return _fill_dimensions(samples, self._dimensions)
