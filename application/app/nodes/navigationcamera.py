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

"""Implementation of Navigation Camera."""

import math

from typing_extensions import override

from direct.showbase.ShowBase import ShowBase
from panda3d.core import LPoint3, LQuaternion, LVector3, NodePath

from ..window import CanvasWindow
from .camera import Camera
from .visualnode import VisualNodeCollection


class NavigationCamera(Camera):
    """
    A camera node used for navigation.

    :param scene_root: The root node of the scene that this camera belongs to.
    :param window: Window in which the view from this camera is to be shown.
    :param base: Application framework for Panda3D.
    """

    def __init__(
        self,
        scene_root: NodePath,
        *,
        window: CanvasWindow,
        base: ShowBase,
    ) -> None:
        self._shift_speed = 1.0

        self._rotation_speed = 0.2

        self._is_main = window.is_main()

        if self._is_main:
            assert base.camera is not None
            camera_nodepath: NodePath = base.cam

            cam_node = base.camNode  # != base.camera.getNode(0)

        else:
            graphicswindow = window.get_graphicswindow()
            camera_nodepath = base.makeCamera(graphicswindow, camName='another camera')

            cam_node = camera_nodepath.getNode(0)

        self._camera_nodepath = camera_nodepath

        super().__init__(
            camera_nodepath=camera_nodepath,
            cam_node=cam_node,
            parent=scene_root,
            scene_root=scene_root,
        )

    def set_shift_speed(self, value: float) -> None:
        """Set the speed of translation movement of the `Camera`."""
        self._shift_speed = value

    def get_shift_speed(self) -> float:
        """Get the speed of translation movement of the `Camera`."""
        return self._shift_speed

    def set_rotation_speed(self, value: float) -> None:
        """Set the rotation speed of the `Camera`."""
        self._rotation_speed = value

    def get_rotation_speed(self) -> float:
        """Get the rotation speed of the `Camera`."""
        return self._rotation_speed

    def get_forward_3d(self) -> LPoint3:
        """Get the forward direction."""
        return self._camera.getQuat().getForward()

    def get_right_3d(self) -> LPoint3:
        """Get the direction to the right."""
        return self._camera.getQuat().getRight()

    def get_up_3d(self) -> LPoint3:
        """Get the up direction."""
        return self._camera.getQuat().getUp()

    def make_pose_for_move(self, direction: LPoint3) -> Camera.Pose:
        """Make new pose in the specified direction."""
        return Camera.Pose(
            position_3d=self._camera.getPos() + direction * self._shift_speed,
            rotation_3d=None,
        )

    def make_pose_for_rotate(self, axis: LVector3, direction: int) -> Camera.Pose:
        """
        Make new pose rotated around the specified axis.

        :param axis: The vector to rotate around.
        :param direction: The direction to rotate. +1 to rotate camera clockwise. -1 for
          anticlockwise.
        """
        cos = math.cos(direction * self._rotation_speed / 2)
        sin = math.sin(direction * self._rotation_speed / 2)
        rotation = LQuaternion(
            cos, axis.getX() * sin, axis.getY() * sin, axis.getZ() * sin
        )
        return Camera.Pose(
            position_3d=None, rotation_3d=self._camera.getQuat() * rotation
        )

    def set_new_pose_for_look_at(self, point: LPoint3, up: LVector3) -> None:
        """
        Set new pose to face a specified point.

        :param point: Specified point.
        :param up: The up direction.
        """
        backup = self._camera.getQuat()
        self._camera.lookAt(point, up)
        new_pose = Camera.Pose(position_3d=None, rotation_3d=self._camera.getQuat())
        self.set_new_pose_3d(new_pose.position_3d, new_pose.rotation_3d)
        self._camera.setQuat(backup)

    @override
    def set_visual_node_properties(self, visual_nodes: VisualNodeCollection) -> None:
        # May cause problem with `Node.make_interval_to_new_position()` if fading in is
        # enabled.

        for each in visual_nodes:
            each.set_properties(self, self.get_near())

    @override
    def set_new_pose_for_visual_information(
        self, *, will_be_global_mode: bool, camera_rotation: bool = True
    ) -> None:
        # match original pose

        if self._visual_information is None:
            return

        # FUTURE: Using `BirdsEyeCamera` for Show All may make this simpler.
        if will_be_global_mode:
            if camera_rotation:
                new_rotation_3d = self.get_new_rotation_3d()
                assert new_rotation_3d is not None

            else:
                new_rotation_3d = self.get_rotation_3d()

                assert self.get_new_rotation_3d() is None

        else:
            # Because Visual Information is calculated with rotation included.
            new_rotation_3d = LQuaternion.identQuat()

        self._visual_information_parent_node.set_new_pose_3d(
            None, new_rotation_3d.conjugate()
        )
