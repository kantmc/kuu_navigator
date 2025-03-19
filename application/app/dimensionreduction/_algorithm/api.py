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

"""Defines common classes for dimension reduction algorithms."""

from typing import Protocol

from ...mathutils import NPArray


class MDSProtocol(Protocol):
    """Protocol for Multi-Dimensional Scaling classes."""

    def fit_transform(
        self,
        X: NPArray,  # noqa: N803
        init: NPArray | None = None,
    ) -> NPArray:
        """
        See MDS documentation: MDS_fit_transform_ .

        .. _MDS_fit_transform: https://scikit-learn.org/stable/modules/generated/sklearn.manifold.MDS.html#sklearn.manifold.MDS.fit_transform>`_
        """
