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
Module for PCA (Principal Components Analysis).

Array type: numpy
"""

from typing import (
    Any,
    Literal,
    cast,
)

from typing_extensions import override
import wpca

import numpy as np

from sklearn import decomposition

import pylog

from ..mathutils import (
    VISUAL_DIMENSIONS,
    NPArray,
)
from .api import DimensionReducer
from .linear import LinearOrientedDimensionReducer, PCAProtocol
from .weight import WeightAlgorithm

_logger = pylog.getLogger(__name__)


@DimensionReducer.factory.register
class OrientedPCA(LinearOrientedDimensionReducer):
    """
    Class to perform orthodox PCA on data points preserving orientation.

    :param random_state: Seed for PCA. The same seed is used every time. `None` to
      allow randomness (maybe deterministic). `False` to not allow randomness.
    :param kwargs: Other keyword arguments will be passed to the initializer of
      `decomposition.PCA`.
    """

    def __init__(
        self,
        *,
        random_state: int | Literal[False] | None = None,
        **kwargs: dict[str, Any],
    ) -> None:
        if random_state is False:
            pca = decomposition.PCA(svd_solver='full', **kwargs)

        else:
            pca = decomposition.PCA(
                random_state=random_state, **kwargs
            )  # FUTURE: n_components='mle')

        super().__init__(
            pca=pca,
        )


class _AdapterForWpca(PCAProtocol):
    """
    Adapter for PCA classes implemented in `wpca`_.

    :param pca: The actual PCA algorithm to use.
    """

    def __init__(self, pca: wpca.PCA | wpca.WPCA) -> None:
        self._pca = pca

    @override
    def fit(self, x: NPArray) -> None:
        self._pca.fit(x)

    @property
    @override
    def components_(self) -> NPArray:
        return cast(NPArray, self._pca.components_)

    @components_.setter
    def components_(self, components: NPArray) -> None:
        self._pca.components_ = components

    @property
    @override
    def n_features_in_(self) -> int:
        raise AssertionError

    @property
    @override
    def explained_variance_ratio_(self) -> NPArray:
        return cast(NPArray, self._pca.explained_variance_ratio_)


@DimensionReducer.factory.register
class _OrientedPCA(LinearOrientedDimensionReducer):
    """
    Class to perform orthodox PCA on data points preserving orientation.

    Defined for comparison with :class:`OrientedPCA`.

    (PCA implementation from `wpca`_.)

    .. _`wpca`: https://github.com/jakevdp/wpca
    """

    def __init__(
        self,
    ) -> None:
        pca = _AdapterForWpca(pca=wpca.PCA())
        super().__init__(
            pca=pca,
        )


@DimensionReducer.factory.register
class OrientedWeightedPCA(LinearOrientedDimensionReducer):
    """
    Class to perform weighted PCA on data points preserving orientation.

    Uses implementation by `[Delchambre, 2014]`_ in `wpca`_.

    .. _[Delchambre, 2014]: https://arxiv.org/abs/1412.4533

    :param weight: Configuration for weights. The `dict` should have the following
        key/value pairs:

        - "type": Class name for one of the
          :class:`app.dimensionreduction.weight.WeightAlgorithm`.
        - "arguments": Arguments passed to the initializer of the
          :class:`app.dimensionreduction.weight.WeightAlgorithm`.
    """

    def __init__(
        self,
        *,
        weight: dict[str, Any] | None = None,
    ) -> None:
        def _setup_weight_algorithm(
            configuration: dict[str, Any] | None,
        ) -> WeightAlgorithm | None:
            if configuration is None:
                return None

            weight_algorithm = WeightAlgorithm.factory.create(
                configuration["type"], configuration["parameters"]
            )  # FUTURE: `parameters` should be optional.
            _logger.info("Using weights: %s", type(weight_algorithm))

            return weight_algorithm

        pca = OrientedWeightedPCA._WPCAAdapter()  # FUTURE: Can use PCA if no weights
        super().__init__(
            pca=pca,
        )

        self._weight_algorithm = _setup_weight_algorithm(weight)

    class _WPCAAdapter(_AdapterForWpca):
        def __init__(self) -> None:
            super().__init__(pca=wpca.WPCA())
            self._weights: NPArray | None = None

        def set_weights(self, weights: NPArray | None) -> None:
            self._weights = weights

        @override
        def fit(self, x: NPArray) -> None:
            if self._weights is not None:
                indexes = self._weights > 0
                sliced_x = x[indexes, :]
                sliced_weights = self._weights[indexes, None]
                self._pca.fit(
                    sliced_x, weights=np.tile(sliced_weights, sliced_x.shape[1])
                )
            else:
                super().fit(x)

    @override
    def _fit(self, data_point_array: NPArray) -> bool:
        assert isinstance(self._pca, OrientedWeightedPCA._WPCAAdapter)

        if self._weight_algorithm is not None:
            center = self._get_center()
            self._pca.set_weights(
                self._weight_algorithm.get_weights(data_point_array, center)
            )

        return super()._fit(data_point_array)


@DimensionReducer.factory.register
class PCA(OrientedPCA):
    """
    Class to perform PCA on data points without preserving orientation.

    :param random_state: Seed for PCA. The same seed is used every time. `None` to
        allow randomness (maybe deterministic). `False` to not allow randomness.
    """

    # This is a temporary implementation using the least amount of code just for
    # comparison.
    # Should redefine with proper class hierarchy.

    def __init__(
        self,
        *,
        random_state: int | Literal[False] | None = None,
    ) -> None:
        super().__init__(
            random_state=random_state,
        )


@DimensionReducer.factory.register
class OrientedPCALess(LinearOrientedDimensionReducer):
    """
    Class preserves orientation, and uses PCA only for Global Mode or when no movement.

    :param random_state: Seed for PCA. The same seed is used every time. `None` to
        allow randomness (maybe deterministic). `False` to not allow randomness.

    ..note:: Note that this is not the same as using PCA in Global mode and Null in
      Local mode. This is because, the PCA components are preserved in Local mode
      with this implementation. This means the principal component in Local mode is the
      same as for Global mode.
    """

    def __init__(
        self,
        *,
        random_state: int | Literal[False] | None = None,
    ) -> None:
        if random_state is False:
            pca = decomposition.PCA(svd_solver='full')

        else:
            pca = decomposition.PCA(
                random_state=random_state
            )  # FUTURE: n_components='mle')

        super().__init__(
            pca=pca,
        )

    @override
    def _fit(self, data_point_array: NPArray) -> bool:
        if (self._virtual_dimensions is None) and (len(data_point_array) > 0):
            self._virtual_dimensions = data_point_array.shape[1]

        if (self._previous is None) or (
            (
                self._center == self._previous._center  # noqa: SLF001
            )
            and (data_point_array.shape[0] > VISUAL_DIMENSIONS)
        ):
            self._fit_with_new_dimension_reduction(data_point_array)

            return True

        self._fit_with_old_dimension_reduction()

        return False
