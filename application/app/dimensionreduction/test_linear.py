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

"""Tests for `linear` module."""

from collections.abc import Sequence

import pytest

import numpy as np

from panda3d.core import LVector3

from ..datasource.api import distance_between
from .linear import _VisualInformation


@pytest.mark.parametrize(
    'before_3d, after_3d',
    [
        # parallel to axes
        ((1, 0, 0), (1, 0, 0)),
        ((1, 0, 0), (0, 1, 0)),
        ((1, 0, 0), (0, 0, 1)),
        ((0, 1, 0), (1, 0, 0)),
        ((0, 1, 0), (0, 1, 0)),
        ((0, 1, 0), (0, 0, 1)),
        ((1, 0, 0), (-1, 0, 0)),
        ((0, 1, 0), (0, -1, 0)),
        ((0, 0, 1), (0, 0, -1)),
        ((1, 0, 0), (0, -1, 0)),
        ((0, 1, 0), (-1, 0, 0)),
        ((0, 1, 0), (0, 0, -1)),
        ((1, 0, 0), (0.5, 0, 0)),
        ((1, 0, 0), (0, 0.5, 0)),
        ((0, 1, 0), (0.5, 0, 0)),
        ((0, 1, 0), (0, 0, 0.5)),
        ((0, 1, 0), (0, 0, 0)),
        ((-1, 0, 0), (0.5, 0, 0)),
        ((0, -1, 0), (0, 0.5, 0)),
        ((0, 0, -1), (0, 0, 0.5)),
        ((-1, 0, 0), (0, 0.5, 0)),
        ((0, -1, 0), (0.5, 0, 0)),
        ((0, -1, 0), (0, 0, 0.5)),
        ((0, -1, 0), (0, 0, 0)),
        # not parallel to axes
        ((0, 0, 0), (0, 0, 0)),
        ((1, 0, 1), (1, 0, 1)),
        ((1, 0, 1), (0, 1, 1)),
        ((1, 1, 0), (1, 1, 0)),
        ((1, 1, 0), (1, 0, 1)),
        ((1, 0, 1), (-1, 0, -1)),
        ((1, 1, 0), (-1, -1, 0)),
        ((0, 1, 1), (0, -1, -1)),
        ((1, 0, 1), (0, -1, 1)),
    ],
)
def test__get_rotation_3d(
    before_3d: Sequence[float], after_3d: Sequence[float]
) -> None:
    """Test `_get_rotation_3d()`."""
    before_3d_vector = LVector3(*before_3d)

    rotation = _VisualInformation._get_rotation_3d(  # noqa: SLF001
        before_3d=np.array(before_3d), after_3d=np.array(after_3d)
    )

    expected = LVector3(*after_3d)

    expected_length = distance_between(after_3d, (0, 0, 0))

    eps = 1.6858739404357614e-07
    assert (
        distance_between(
            rotation.xform(before_3d_vector.normalized()) * expected_length, expected
        )
        <= eps
    )
