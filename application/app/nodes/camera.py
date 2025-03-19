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
Module for camera.

Array type: Panda3D
"""

from dataclasses import dataclass
import math
from typing import (
    cast,
)

from typing_extensions import override

import numpy as np
from numpy.typing import NDArray

from direct.interval.Interval import Interval
from panda3d.core import (
    Camera as PandaCamera,
)
from panda3d.core import (
    CollisionHandlerQueue,
    CollisionNode,
    CollisionRay,
    CollisionTraverser,
    DisplayRegion,
    GeomNode,
    Lens,
    LPoint2,
    LPoint3,
    LQuaternion,
    LVecBase2,
    LVector3,
    NodePath,
)

import pylog

from .node import Node
from .visualinformation import VisualInformation
from .visualnode import VisualNodeCollection

_logger = pylog.getLogger(__name__)


class Camera(Node):
    """
    Superclass for different kinds of cameras.

    :param camera_nodepath: The `NodePath` for this camera.
    :param cam_node: The Panda3D `Camera` instance for this camera.
    :param displayregion: The `DisplayRegion` to associate with this camera. `None` to
      associate with `DisplayRegion` associated with `cam_node`.
    :param scene_root: The root of the scene to be shown with this camera.
    """

    def __init__(
        self,
        *,
        camera_nodepath: NodePath,
        cam_node: PandaCamera,
        displayregion: DisplayRegion | None = None,
        parent: NodePath,
        scene_root: NodePath,
    ) -> None:
        super().__init__(camera_nodepath)

        self._camera = camera_nodepath

        self._camera.reparentTo(parent)

        assert isinstance(cam_node, PandaCamera), f"{cam_node=}"
        self._cam_node = cam_node

        self._lens: Lens = self._cam_node.get_lens()

        self._bottom_right_node = NodePath('BottomRight')
        self._bottom_right_node.reparentTo(self._camera)
        self._update_bottom_right_node()

        self._visual_information: VisualInformation | None = None

        self._visual_information_parent_node = Node(NodePath('VisualInformationParent'))
        # Position not set
        self._set_bottom_right_node(self._visual_information_parent_node.get_nodepath())

        self._picker = Camera._Picker(scene_root, self)

        if (displayregion is None) and (self._cam_node.getNumDisplayRegions() > 0):
            displayregion = self._cam_node.getDisplayRegion(0)

        self._displayregion = displayregion

    class _Picker:
        def __init__(self, scene_root: Node, camera: 'Camera') -> None:
            self._scene_root = scene_root

            self._cam_node = camera._cam_node  # noqa: SLF001

            self._collisiontraverser = CollisionTraverser('picker traverser')

            self._collisionhandlerqueue = CollisionHandlerQueue()

            picker_collisionnode = CollisionNode('pickerRay')
            picker_collisionnode.setFromCollideMask(GeomNode.getDefaultCollideMask())
            picker_collider: NodePath = camera._camera.attachNewNode(  # noqa: SLF001
                picker_collisionnode
            )

            self._picker_collisionray = CollisionRay()
            picker_collisionnode.addSolid(self._picker_collisionray)

            self._collisiontraverser.addCollider(
                picker_collider, self._collisionhandlerqueue
            )

        def get_node_at(
            self, position: LPoint2, *, pickable_tag: str, pickable_value: str | None
        ) -> NodePath | None:
            # https://docs.panda3d.org/1.10/python/programming/collision-detection/clicking-on-3d-objects#clicking-on-3d-objects
            self._picker_collisionray.setFromLens(
                self._cam_node, position.x, position.y
            )
            self._collisiontraverser.traverse(self._scene_root)
            if self._collisionhandlerqueue.getNumEntries() > 0:
                self._collisionhandlerqueue.sortEntries()

                for each_collisionentry in self._collisionhandlerqueue.entries:
                    each_nodepath = each_collisionentry.getIntoNodePath()
                    top_node: NodePath = each_nodepath.findNetTag(pickable_tag)
                    tag_value = top_node.getTag(pickable_tag)
                    if (pickable_value is None) or (tag_value == pickable_value):
                        return each_nodepath

            return None

    @dataclass
    class Pose:
        """Data class to hold pose of `Camera`."""

        position_3d: LPoint3 | None
        rotation_3d: LQuaternion | None

    @staticmethod
    def _get_bounds(
        coordinates_3d: NDArray[np.float64], margin: float
    ) -> list[list[float]] | None:
        if coordinates_3d.shape[0] == 0:
            return None

        minimums = coordinates_3d.min(axis=0)
        _logger.info("minimums: %s", minimums)

        maximums = coordinates_3d.max(axis=0)
        _logger.info("maximums: %s", maximums)

        bounds = [
            coordinates_3d.min(axis=0) - margin,
            coordinates_3d.max(axis=0) + margin,
        ]
        assert bounds[0].shape == (3,), str(bounds[0].shape)
        assert bounds[1].shape == (3,), str(bounds[0].shape)
        return bounds

    def get_lens(self) -> Lens:
        """Get the Pand3d :class:`Lens` for this `Camera`."""
        return self._lens

    def get_displayregion(self) -> DisplayRegion | None:
        """Get :class:`panda3d.core.DisplayRegion` associated with this camera."""
        return self._displayregion

    @dataclass(frozen=False)
    class Directions3D:
        """Directions for the camera."""

        right_3d: LVector3
        """Right direction of the camera."""

        forward_3d: LVector3
        """Forward direction of the camera."""

        up_3d: LVector3
        """Up direction of the camera."""

    @staticmethod
    def get_axis_rotation_3d(axis_direction: str) -> LQuaternion:
        """
        Get the quaternion that represents rotation toward the specified direction.

        :param axis_direction: One of the following:

            - '+X': TOWARDS +X direction
            - '+Y': TOWARDS +Y direction
            - '+Z': TOWARDS +Z direction
            - '-X': TOWARDS -X direction
            - '-Y': TOWARDS -Y direction
            - '-Z': TOWARDS -Z direction

        :returns: A quaternion that represents rotating a vector pointing to the +Y
          direction toward `axis_direction`.
        """
        match axis_direction:
            case '+X':
                # By default in Panda3D, the X axis points to the right, the Y axis is
                # forward, and Z is up.
                # https://docs.panda3d.org/1.10/python/programming/scene-graph/common-state-changes#state-change-cheat-sheet
                direction_3d = LQuaternion()
                # rotate -90 degrees anticlockwise looking toward the axis
                direction_3d.setFromAxisAngle(-90, LVector3(0, 0, 1))

            case '-X':
                direction_3d = LQuaternion()
                direction_3d.setFromAxisAngle(90, LVector3(0, 0, 1))

            case '+Y':
                direction_3d = LQuaternion.identQuat()

            case '-Y':
                direction_3d = LQuaternion()
                direction_3d.setFromAxisAngle(180, LVector3(1, 0, 1).normalized())

            case '+Z':
                direction_3d = LQuaternion()
                direction_3d.setFromAxisAngle(90, LVector3(1, 0, 0))

            case '-Z':
                direction_3d = LQuaternion()
                direction_3d.setFromAxisAngle(-90, LVector3(1, 0, 0))

            case _:
                raise AssertionError("not supported")

        return direction_3d

    @staticmethod
    def get_axis_directions_3d(axis_direction: str) -> Directions3D:
        """
        Get directions of axes when facing specified direction.

        :param axis_direction: Which direction to look.
          See :meth:`set_new_pose_for_show_all()` for examples.
        """
        rotation_3d = Camera.get_axis_rotation_3d(axis_direction)

        return Camera.Directions3D(
            right_3d=rotation_3d.getRight(),
            forward_3d=rotation_3d.getForward(),
            up_3d=rotation_3d.getUp(),
        )

    def get_directions_3d(
        self,
    ) -> Directions3D:
        """
        Get the directions of axes of the camera.

        :returns: Right, forward, up vectors in 3D.
        """
        quaternion_3d = self._camera.getQuat()
        directions = Camera.Directions3D(
            right_3d=quaternion_3d.getRight(),
            forward_3d=quaternion_3d.getForward(),
            up_3d=quaternion_3d.getUp(),
        )
        return directions

    def get_new_directions_3d(
        self,
    ) -> Directions3D:
        """
        Get the directions of axes of the camera in its new pose.

        :returns: Right, forward, up vectors in 3D.
        """
        new_quaternion_3d = self.get_new_rotation_3d()

        if new_quaternion_3d is None:
            new_quaternion_3d = self.get_rotation_3d()

        directions = Camera.Directions3D(
            right_3d=new_quaternion_3d.getRight(),
            forward_3d=new_quaternion_3d.getForward(),
            up_3d=new_quaternion_3d.getUp(),
        )
        return directions

    def get_near(
        self,
    ) -> (
        float
    ):  # FUTURE: What to do with near plane, when distribution of data points is small
        """Get distance to near plane."""
        return cast(float, self._lens.getNear())

    def get_near_plane_width(self) -> float:
        """Get width of the visible area of the near plane."""
        return math.tan(self._get_fov_radians().getX()) * self.get_near()

    def make_active(self) -> None:
        """Make the view from this camera show on the associated :class:`Window`."""
        if self._displayregion is not None:
            self._displayregion.setCamera(self._camera)

    def is_active(self) -> bool:
        """Return whether this `Camera` is active."""
        if self._displayregion is None:
            return False

        # 'is' doesn't work
        return cast(bool, self._displayregion.getCamera() == self._camera)

    def look_at(self, point: LPoint3, up: LVector3) -> None:
        """
        Make this camera face a specified point.

        :param point: Specified point.
        :param up: The up direction.
        """
        self._camera.lookAt(point, up)

    def set_visual_node_properties(self, visual_nodes: VisualNodeCollection) -> None:
        """
        Set properties of Visual Nodes based on properties of this camera.

        :param camera: The camera.
        """

    def get_nodepath_at(
        self, position: LPoint2, *, pickable_tag: str, pickable_value: str | None
    ) -> NodePath | None:
        """
        Pick `NodePath` at specific location in the camera view.

        This will ignore `NodePath` that do not have `'Y'` for the specified Tag
        for any of its parents.

        :param position: 2D coordinates of the location.
        :param pickable_tag: Name for Tag that specifies whether each `NodePath` is
          pickable.
        :param pickable_value: Value for Tag. `None`, if value is not relevant.

        :returns: A reference to the object. `None`, if not found.
        """
        return self._picker.get_node_at(
            position, pickable_tag=pickable_tag, pickable_value=pickable_value
        )

    def _get_fov_radians(self) -> LVecBase2:
        return self._lens.getFov() / 180 * math.pi

    def _update_bottom_right_node(self) -> None:
        """Update position of bottom right node."""
        fov = self._get_fov_radians()

        near = self._lens.getNear()

        self._bottom_right_node.setPos(
            math.tan(fov.getX() / 2) * near, near, -math.tan(fov.getY() / 2) * near
        )

    def _set_bottom_right_node(self, node: NodePath | None) -> None:
        """Set `NodePath` to show at the bottom right corner."""
        if node is None:
            self._bottom_right_node.clear()

        else:
            node.reparentTo(self._bottom_right_node)

    def get_visual_information_width(self) -> float:
        """Get the preferred width for Visual Information."""
        return self.get_near_plane_width() / 20

    def update_visual_information_position(self) -> None:
        """Update position of Visual Information."""
        if self._visual_information is not None:
            width = self.get_visual_information_width()

            self._visual_information.set_width(width)
            self._visual_information_parent_node.set_position_3d(
                LPoint3(-width / 2, +width / 2, +width / 2)
            )

        self._update_bottom_right_node()

    def set_visual_information(
        self, visual_information: VisualInformation | None
    ) -> None:
        """Set Visual Information to show in this Camera."""

        def _remove_children(nodepath: NodePath) -> None:
            for each in nodepath.getChildren():
                each.detachNode()

        self._visual_information = visual_information

        visual_information_parent_nodepath = (
            self._visual_information_parent_node.get_nodepath()
        )

        if visual_information is None:
            _remove_children(visual_information_parent_nodepath)

        else:
            node = visual_information.get_nodepath()
            node.reparentTo(visual_information_parent_nodepath)

        self.update_visual_information_position()

    def set_new_pose_for_visual_information(
        self, *, will_be_global_mode: bool, camera_rotation: bool = True
    ) -> None:
        """
        Set new pose for parent of Visual Information.

        :param will_be_global_mode: Whether Scene will be in Global mode or not.
        :param camera_rotation: Whether there is rotation of this camera.
        """

    def move_visual_information_to_new_pose(self) -> None:
        """Apply new pose for parent of Visual Information."""
        self._visual_information_parent_node.move_to_new_pose()

    def make_interval_for_visual_information_pose(
        self, duration: float, *, clear: bool = True
    ) -> Interval | None:
        """
        Make Interval for Visual Information.

        :param duration: Length in time for movement.
        :param clear: Whether to clear new pose.
        :returns: Interval for Visual Information. `None`, if not necessary
        """
        interval = self._visual_information_parent_node.make_interval_to_new_pose(
            duration, clear=clear
        )

        return interval

    @staticmethod
    def _get_pose_to_view_box_with_fov(
        box_lengths: LVector3,
        axis_direction: str,
        *,
        fov_radians: LVecBase2,
        step_back: float,
    ) -> tuple[LPoint3, LQuaternion]:
        """
        Get the pose of the camera to view the whole box centered at the origin.

        :param box_lengths: The lengths of the edges of the box.
        :param axis_direction: Which direction to look.
          See :meth:`set_new_pose_for_show_all()` for examples.
        :param fov_radians: Field of view for the camera.
        :param step_back: Amount to step back, so that whole box is more inside the
          view.
        """

        def _get_position(
            box_lengths: LVector3,
            axis_direction: str,
            *,
            fov_radians: LVecBase2,
            step_back: float,
        ) -> LPoint3:
            match axis_direction:
                case '+X':
                    position = -LPoint3(
                        max(
                            box_lengths.getY() / 2 / math.tan(fov_radians.getX() / 2),
                            box_lengths.getZ() / 2 / math.tan(fov_radians.getY() / 2),
                        )
                        + box_lengths.getX() / 2
                        + step_back,
                        0,
                        0,
                    )

                case '+Y':
                    position = -LPoint3(
                        0,
                        max(
                            box_lengths.getZ() / 2 / math.tan(fov_radians.getY() / 2),
                            box_lengths.getX() / 2 / math.tan(fov_radians.getX() / 2),
                        )
                        + box_lengths.getY() / 2
                        + step_back,
                        0,
                    )

                case '+Z':
                    position = -LPoint3(
                        0,
                        0,
                        max(
                            box_lengths.getX() / 2 / math.tan(fov_radians.getX() / 2),
                            box_lengths.getY() / 2 / math.tan(fov_radians.getY() / 2),
                        )
                        + box_lengths.getZ() / 2
                        + step_back,
                    )

                case '-X':
                    position = LPoint3(
                        max(
                            box_lengths.getY() / 2 / math.tan(fov_radians.getX() / 2),
                            box_lengths.getZ() / 2 / math.tan(fov_radians.getY() / 2),
                        )
                        + box_lengths.getX() / 2
                        + step_back,
                        0,
                        0,
                    )

                case '-Y':
                    position = LPoint3(
                        0,
                        max(
                            box_lengths.getZ() / 2 / math.tan(fov_radians.getX() / 2),
                            box_lengths.getX() / 2 / math.tan(fov_radians.getY() / 2),
                        )
                        + box_lengths.getY() / 2
                        + step_back,
                        0,
                    )

                case '-Z':
                    position = LPoint3(
                        0,
                        0,
                        max(
                            box_lengths.getX() / 2 / math.tan(fov_radians.getX() / 2),
                            box_lengths.getY() / 2 / math.tan(fov_radians.getY() / 2),
                        )
                        + box_lengths.getZ() / 2
                        + step_back,
                    )

                case _:
                    raise AssertionError("not supported")

            return position

        position = _get_position(
            box_lengths, axis_direction, fov_radians=fov_radians, step_back=step_back
        )

        direction = Camera.get_axis_rotation_3d(axis_direction)

        return position, direction

    def _get_pose_to_view_box(
        self,
        box_lengths: LVector3,
        axis_direction: str,
        *,
        margin: float,
    ) -> tuple[LPoint3, LQuaternion]:
        """
        Get the pose of the camera to view the whole box centered at the origin.

        :param box_lengths: The lengths of the edges of the box.
        :param axis_direction: Which direction to look.
          See :meth:`set_new_pose_for_show_all()` for examples.
        :param margin: Amount of extra space.
        """
        fov = self._get_fov_radians()
        _logger.info("FOV: %s", fov)

        step_back = (
            self._lens.getNear() + margin * 2
        )  # TODO: Should take maximum of `getNear()` and nearest face

        new_position, new_direction = self._get_pose_to_view_box_with_fov(
            box_lengths,
            axis_direction,
            fov_radians=fov,
            step_back=step_back,
        )

        assert new_position.length() >= self._lens.getNear(), (
            "Don't forget `step_back`."
        )

        return new_position, new_direction

    # TODO: Merge with `BirdsEyeCamera._make_pose_for_birds_eye_view()`
    def _make_pose_for_show_all(
        self,
        visual_nodes: VisualNodeCollection,
        axis_direction: str,
        *,
        margin: float,
    ) -> tuple[Pose, LPoint3 | None]:
        def _get_new_pose(
            coordinates_3d: NDArray[np.float64],
            axis_direction: str,
            *,
            margin: float,
        ) -> tuple[LPoint3 | None, LQuaternion | None, LPoint3 | None]:
            # FUTURE: Should actually be calculating with perspective.

            bounds = self._get_bounds(coordinates_3d, margin)

            if bounds is None:
                return None, None, None

            center = LPoint3(
                (bounds[1][0] + bounds[0][0]) / 2,
                (bounds[1][1] + bounds[0][1]) / 2,
                (bounds[1][2] + bounds[0][2]) / 2,
            )

            box_lengths = LVector3(
                bounds[1][0] - bounds[0][0],
                bounds[1][1] - bounds[0][1],
                bounds[1][2] - bounds[0][2],
            )

            new_position, new_direction = self._get_pose_to_view_box(
                box_lengths, axis_direction, margin=margin
            )

            return new_position + center, new_direction, center

        coordinates_3d = visual_nodes.coordinates_to_array_3d()
        new_position, rotation, look_at = _get_new_pose(
            coordinates_3d, axis_direction, margin=margin
        )
        _logger.info(
            "position: %r, rotation: %r, look_at: %r", new_position, rotation, look_at
        )
        return Camera.Pose(position_3d=new_position, rotation_3d=rotation), look_at

    # TODO: Merge with `BirdsEyeCamera.set_new_pose_for_birds_eye_view()`
    def set_new_pose_for_show_all(
        self,
        axis_direction: str,
        visual_nodes: VisualNodeCollection,
        *,
        margin: float,
    ) -> LPoint3 | None:
        """
        Move camera to show all Visual Nodes.

        :param axis_direction: One of the following:

          - '+X': show all objects TOWARD +X direction
          - '+Y': show all objects TOWARD +Y direction
          - '+Z': show all objects TOWARD +Z direction
          - '-X': show all objects TOWARD -X direction
          - '-Y': show all objects TOWARD -Y direction
          - '-Z': show all objects TOWARD -Z direction

        :param visual_nodes: The collection of `VisualNode` to show.
        :param margin: Amount of extra space.

        :return: Point that the Camera watches. `None`, if there is no `VisualNode`.
        """
        new_pose, look_at = self._make_pose_for_show_all(
            visual_nodes, axis_direction, margin=margin
        )
        self.set_new_pose_3d(new_pose.position_3d, new_pose.rotation_3d)

        return look_at

    @override
    def move_to_new_pose(self) -> None:
        super().move_to_new_pose()

        self.move_visual_information_to_new_pose()

    def show_all(
        self, visual_nodes: VisualNodeCollection, axis_direction: str, *, margin: float
    ) -> None:
        """
        Set camera pose to show all objects.

        :param axis_direction: Which direction to look.
          See :meth:`set_new_pose_for_show_all()` for examples.
        :param visual_nodes: Image nodes to show.
        """
        new_pose, _ = self._make_pose_for_show_all(
            visual_nodes, axis_direction, margin=margin
        )
        self.set_pose_3d(new_pose.position_3d, new_pose.rotation_3d)
