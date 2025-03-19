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

"""Tests for `spotlightedmds` module."""

from typing import cast

import pytest

from numpy.random import default_rng
from scipy.linalg import norm, orthogonal_procrustes
from scipy.spatial.distance import pdist

from skbio import DistanceMatrix
from skbio.stats.ordination import pcoa

import pylog

from ...mathutils import NPArray
from .mds import PCoA

_logger = pylog.getLogger(__name__)


@pytest.mark.parametrize(
    'dimensions, samples, random_scale, seed, allowed_error',
    [
        (3, 5, 1.0, 123, 1e-3),
        (10, 100, 10.0, 555, 0.1),
    ],
)
def test__PCoA__regular(
    *,
    dimensions: int,
    samples: int,
    random_scale: float,
    seed: int,
    allowed_error: float,
) -> None:
    """Test that :class:`PCoA` returns results similar to regular PCoA."""

    def _regular_pcoa(data_points: NPArray) -> NPArray:
        dissimilarities = DistanceMatrix(pdist(data_points))
        result = pcoa(dissimilarities, number_of_dimensions=3)
        return cast(NPArray, result.samples.to_numpy())

    generator = default_rng(seed=seed - 2)
    data_points = generator.random((samples, dimensions)) * random_scale

    regular_result = _regular_pcoa(data_points)

    mds = PCoA(dimensions=3)
    mds_result = mds.fit_transform(data_points, init=regular_result)

    rotation_matrix, _ = orthogonal_procrustes(mds_result, regular_result)
    matched_result = mds_result @ rotation_matrix

    _logger.info("Matched result: %r", matched_result[:10, :])
    _logger.info("Regular MDS: %r", regular_result[:10, :])

    relative_error = norm(mds_result @ rotation_matrix - regular_result, 'fro') / norm(
        regular_result, 'fro'
    )

    assert relative_error < allowed_error
