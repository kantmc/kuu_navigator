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

"""Tests for the `camera` module."""

import math
from sys import float_info

import pytest

from panda3d.core import LPoint3, LQuaternion, LVecBase2, LVector3

from pyqoolloop.testutils import combine_lists

from .camera import Camera


def _make_corners(half_box_lengths: LVector3) -> tuple[LPoint3, ...]:
    corners = (
        LPoint3(
            +half_box_lengths.getX(),
            +half_box_lengths.getY(),
            +half_box_lengths.getZ(),
        ),
        LPoint3(
            +half_box_lengths.getX(),
            +half_box_lengths.getY(),
            -half_box_lengths.getZ(),
        ),
        LPoint3(
            +half_box_lengths.getX(),
            -half_box_lengths.getY(),
            +half_box_lengths.getZ(),
        ),
        LPoint3(
            +half_box_lengths.getX(),
            -half_box_lengths.getY(),
            -half_box_lengths.getZ(),
        ),
        LPoint3(
            -half_box_lengths.getX(),
            +half_box_lengths.getY(),
            +half_box_lengths.getZ(),
        ),
        LPoint3(
            -half_box_lengths.getX(),
            +half_box_lengths.getY(),
            -half_box_lengths.getZ(),
        ),
        LPoint3(
            -half_box_lengths.getX(),
            -half_box_lengths.getY(),
            +half_box_lengths.getZ(),
        ),
        LPoint3(
            -half_box_lengths.getX(),
            -half_box_lengths.getY(),
            -half_box_lengths.getZ(),
        ),
    )

    return corners


def _check_corners(
    rotation: LQuaternion,
    tolerance: float,
    fov_degrees: LVecBase2,
    position: LPoint3,
    corners: tuple[LPoint3, ...],
) -> None:
    count_on_edge = 0
    for each in corners:
        direction_vector = each - position
        print(f"{direction_vector=}")

        horizontal_angle = math.atan2(
            direction_vector.dot(rotation.getRight()),
            direction_vector.dot(rotation.getForward()),
        )

        vertical_angle = math.atan2(
            direction_vector.dot(rotation.getUp()),
            direction_vector.dot(rotation.getForward()),
        )

        assert math.degrees(abs(horizontal_angle)) <= fov_degrees.getX() / 2 + tolerance

        assert math.degrees(abs(vertical_angle)) <= fov_degrees.getY() / 2 + tolerance

        if (
            math.degrees(abs(horizontal_angle)) >= fov_degrees.getX() / 2 - tolerance
        ) or (math.degrees(abs(vertical_angle)) >= fov_degrees.getY() / 2 - tolerance):
            count_on_edge += 1

    minimum_number_on_edge = 4
    assert count_on_edge >= minimum_number_on_edge


@pytest.mark.parametrize(
    'box_lengths_tuple, axis_direction',
    combine_lists(
        [(1, 2, 3), (2, 3, 1), (3, 1, 2), (3, 2, 1), (2, 1, 3), (1, 3, 2)],
        ['+X', '+Y', '+Z', '-X', '-Y', '-Z'],
    ),
)
def test__get_pose_to_view_box(
    box_lengths_tuple: tuple[float, float, float],
    axis_direction: str,
) -> None:
    """Test `_get_pose_to_view_box()` so that the whole box is visible from position."""
    tolerance = math.sqrt(float_info.epsilon) * 1000

    box_lengths = LVector3(*box_lengths_tuple)

    half_box_lengths = box_lengths / 2

    # fov_degrees = LVecBase2(39.3201, 30)  # Taken from Panda3D
    fov_degrees = LVecBase2(40, 30)

    position, rotation = Camera._get_pose_to_view_box_with_fov(  # noqa: SLF001
        box_lengths,
        axis_direction,
        fov_radians=fov_degrees / 180 * math.pi,
        step_back=0,
    )

    corners = _make_corners(half_box_lengths)

    _check_corners(rotation, tolerance, fov_degrees, position, corners)
