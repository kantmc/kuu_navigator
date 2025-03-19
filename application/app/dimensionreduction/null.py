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
Defines class that doesn't perform dimension reduction.

Array type: numpy
"""

from typing_extensions import override

import numpy as np

from ..mathutils import NPArray
from .api import DimensionReducer
from .linear import LinearOrientedDimensionReducer, PCAProtocol


class _NullAlgorithm(PCAProtocol):
    """Dimension reduction algorithm just taking data as is."""

    @override
    def fit(self, x: NPArray) -> None:
        """Fit data points."""
        assert len(x) > 0

        self._dimensions = x.shape[1]

        self._components = np.eye(self._dimensions)

    @property
    def components_(self) -> NPArray:
        return self._components

    @components_.setter
    def components_(self, components: NPArray) -> None:
        self._components = components

    @property
    def n_features_in_(self) -> int:
        return self._dimensions

    @property
    def explained_variance_ratio_(self) -> NPArray | None:
        return None


@DimensionReducer.factory.register
class Null(LinearOrientedDimensionReducer):
    """Class that doesn't perform dimension reduction."""

    def __init__(
        self,
    ) -> None:
        reducer = _NullAlgorithm()
        super().__init__(
            pca=reducer,
        )
