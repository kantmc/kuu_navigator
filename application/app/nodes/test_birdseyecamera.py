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

"""Test `birdseyecamera` module."""

import pytest

from panda3d.core import LVector3

from ..datasource.api import distance_between
from ..scenes.navigationscene import NavigationScene
from .camera import Camera


@pytest.mark.parametrize(
    'index, axis_direction', enumerate(NavigationScene.BIRDS_EYE_VIEWS)
)
def test__get_axis_rotation_3d(index: int, axis_direction: str) -> None:
    """Test that text for Bird's Eye View matches direction."""
    text_directions = {
        '[forward]': (0, 1, 0),
        '[above]': (0, 0, 1),
        '[right]': (1, 0, 0),
        '[left]': (-1, 0, 0),
        '[backward]': (0, -1, 0),
        '[below]': (0, 0, -1),
    }

    rotation_3d = Camera.get_axis_rotation_3d(axis_direction)

    text = NavigationScene._BIRDS_EYE_DIRECTION[index]  # noqa: SLF001
    text = text[: text.find(']') + 1]
    direction_vector = LVector3(*text_directions[text])

    rotation_of_navigation_camera_3d = rotation_3d.conjugate()

    eps = 1.0580858110250448e-07
    assert (
        distance_between(
            rotation_of_navigation_camera_3d.getForward(), direction_vector
        )
        <= eps
    )
