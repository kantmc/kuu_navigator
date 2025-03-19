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

"""Implementation for Bird's Eye Camera."""

from typing_extensions import override

import numpy as np

from direct.showbase.ShowBase import ShowBase
from panda3d.core import (
    Camera as PandaCamera,
)
from panda3d.core import (
    LPoint3,
    LQuaternion,
    LVector3,
    NodePath,
    PerspectiveLens,
)

from ..mathutils import NPArray
from ..window import CanvasWindow
from .camera import Camera
from .navigationcamera import NavigationCamera
from .visualnode import VisualNodeCollection


class BirdsEyeCamera(Camera):
    """
    Camera for Bird's Eye View.

    :param navigating_camera: The Navigation Camera to attach this Bird's Eye Camera to.
    :param window: Window that will show the view from this camera.
    :param base: The `ShowBase` instance.
    :param scene_root: The root of the scene.
    """

    def __init__(
        self,
        *,
        navigation_camera: NavigationCamera,
        window: CanvasWindow,
        base: ShowBase,
        scene_root: NodePath,
    ) -> None:
        camera_nodepath = self._make_camera(
            'birds eye camera', base=base, window=window
        )

        cam_node = camera_nodepath.getNode(0)

        parent_nodepath = navigation_camera.get_nodepath()

        super().__init__(
            displayregion=navigation_camera.get_displayregion(),
            camera_nodepath=camera_nodepath,
            cam_node=cam_node,
            parent=parent_nodepath,
            scene_root=scene_root,
        )

        self._navigation_camera = navigation_camera

    def _make_camera(
        self, name: str, *, base: ShowBase, window: CanvasWindow
    ) -> PandaCamera:
        lens = PerspectiveLens()

        aspect_ratio = base.getAspectRatio(window.get_graphicswindow())
        lens.setAspectRatio(aspect_ratio)

        cam_node = PandaCamera(name)

        cam_node.setLens(lens)

        camera = NodePath(cam_node)
        return camera

    def _make_pose_for_birds_eye_view(
        self,
        visual_nodes: VisualNodeCollection,
        axis_direction: str,
        *,
        margin: float,
    ) -> Camera.Pose:
        def _get_navigation_camera_directions_3d() -> NPArray:
            directions_3d = self._navigation_camera.get_directions_3d()
            array = np.vstack(
                [directions_3d.right_3d, directions_3d.forward_3d, directions_3d.up_3d]
            )
            return array

        def _get_new_pose(
            camera_coordinates_3d: NPArray,
            axis_direction: str,
            *,
            margin: float,
        ) -> tuple[LPoint3 | None, LQuaternion | None]:
            # FUTURE: Should actually be calculating with perspective.

            bounds = self._get_bounds(camera_coordinates_3d, margin)

            if bounds is None:
                return None, None

            half_box_lengths = LVector3(
                max(bounds[1][0], -bounds[0][0]),
                max(bounds[1][1], -bounds[0][1]),
                max(bounds[1][2], -bounds[0][2]),
            )

            new_position, new_direction = self._get_pose_to_view_box(
                half_box_lengths * 2, axis_direction, margin=margin
            )

            return new_position, new_direction

        navigation_camera_position_3d = np.array(
            self._navigation_camera.get_position_3d()
        )

        navigation_camera_directions_3d = _get_navigation_camera_directions_3d()
        navigation_coordinates_3d = visual_nodes.coordinates_to_camera_coordinates_3d(
            navigation_camera_position_3d, navigation_camera_directions_3d
        )
        new_position, new_rotation = _get_new_pose(
            navigation_coordinates_3d, axis_direction, margin=margin
        )
        return Camera.Pose(position_3d=new_position, rotation_3d=new_rotation)

    def set_new_pose_for_birds_eye_view(
        self, axis_direction: str, visual_nodes: VisualNodeCollection, *, margin: float
    ) -> None:
        """
        Set new pose for this camera to view all the neighborhood.

        :param axis_direction: The direction to view the :class:`NavigationCamera`.
          One of the following:

            - '+X': TOWARDS +X direction
            - '+Y': TOWARDS +Y direction
            - '+Z': TOWARDS +Z direction
            - '-X': TOWARDS -X direction
            - '-Y': TOWARDS -Y direction
            - '-Z': TOWARDS -Z direction

        :param visual_nodes: `VisualNode` in the neighborhood.
        :param margin: Extra space to show around the neighborhood.

        ..note:: New pose will not be set, if the neighborhood is empty.
        """
        new_pose = self._make_pose_for_birds_eye_view(
            visual_nodes,
            axis_direction,
            margin=margin,
        )
        self.set_new_pose_3d(new_pose.position_3d, new_pose.rotation_3d)

    @override
    def set_visual_node_properties(self, visual_nodes: VisualNodeCollection) -> None:
        for each in visual_nodes:
            each.set_properties(self, None)

    @override
    def set_new_pose_for_visual_information(
        self, *, will_be_global_mode: bool, camera_rotation: bool = True
    ) -> None:
        assert camera_rotation

        # Align Visual Information parent Node with Navigation Camera.

        assert self.get_nodepath().getParent() == self._navigation_camera.get_nodepath()

        new_rotation_3d = self.get_new_rotation_3d()
        if new_rotation_3d is not None:
            # May be `None`, when neighborhood is empty.
            self._visual_information_parent_node.set_new_pose_3d(
                None, new_rotation_3d.conjugate()
            )
