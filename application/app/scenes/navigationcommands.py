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

"""Implementation of commands for navigation."""

from abc import ABC, abstractmethod

from typing_extensions import override

import numpy as np

from direct.directtools.DirectGeometry import qSlerp
from direct.interval.FunctionInterval import Func
from direct.interval.Interval import Interval
import direct.interval.IntervalGlobal as PandaInterval
from direct.interval.LerpInterval import LerpFunc
from direct.interval.MetaInterval import MetaInterval
from panda3d.core import (
    LPoint3,
    LQuaternion,
    LVector3,
    lookAt,
)

import pylog

from ..datasource.api import DataPoint
from ..dimensionreduction.linear import LinearOrientedDimensionReducer
from ..mathutils import (
    NPArray,
    OptionalVectorPair,
    VectorPair,
    magnitude,
    np_all,
    weighted_sum,
)
from ..nodes.birdseyecamera import BirdsEyeCamera
from ..nodes.camera import Camera
from ..nodes.navigationcamera import NavigationCamera
from ..nodes.visualnode import VisualNodeCollection
from .command import Command
from .navigationscene import NavigationScene

_logger = pylog.getLogger(__name__)


class NavigationCommand(Command, ABC):
    # Allow commands to access protected members of `Scene`.
    # flake8: noqa: SLF001
    """
    Abstract class for navigation commands.

    :param scene: Scene for this command.
    :param command_str: `str` representing the command. Same as for
      :meth:`NavigationScene.process_command()`.
    """

    def __init__(self, *, scene: 'NavigationScene', command_str: str) -> None:
        super().__init__(scene=scene, command_str=command_str)

        self._is_prepared: bool | None = None
        self._look_at: LPoint3 | None = None

        self._will_be_global_mode: bool = scene.is_global_mode()

        with scene._visual_nodes as visual_nodes:
            self._sequence_list = self._make_sequence_list(visual_nodes)

    def _set_global_mode(self, mode: str | None) -> None:
        """
        Set Global mode for the `Scene`.

        See :meth:`NavigationScene.get_global_mode()`.
        """
        self._scene._set_global_mode(mode)

    def _set_birds_eye_mode(
        self, axis_direction: str | None, *, visual_nodes: VisualNodeCollection
    ) -> None:
        """
        Set Birds Eye mode for the `Scene`.

        Will switch cameras as necessary.
        See :meth:`NavigationScene.get_birds_eye_mode()`.
        """
        if axis_direction is None:
            self._get_navigation_camera().set_visual_node_properties(visual_nodes)

            self._get_navigation_camera().make_active()

            self._scene._set_birds_eye_mode(None)

        else:
            self._scene._set_birds_eye_mode(axis_direction)

            self._get_birds_eye_camera().set_visual_node_properties(visual_nodes)

            self._get_birds_eye_camera().make_active()

    def _get_navigation_camera(self) -> NavigationCamera:
        """Get the `NavigationCamera` for the `Scene`."""
        return self._scene._get_navigation_camera()

    def _get_birds_eye_camera(self) -> BirdsEyeCamera:
        """Get the `BirdsEyeCamera` for the `Scene`."""
        return self._scene._get_birds_eye_camera()

    def _get_moving_camera(self) -> Camera:
        """Get the Camera that will move by this Command."""
        return self._scene._get_active_camera()

    def _get_active_camera(self) -> Camera:
        """Get the Camera that is active (displayed on screen)."""
        return self._scene._get_active_camera()

    def _assert_camera_position(self) -> None:
        """Check that `Camera` position and `Scene` center are equal in Local mode."""
        if not self._scene.is_global_mode():
            current_camera_position_3d = np.array(
                self._get_navigation_camera().get_position_3d()
            )

            current_position_3d = self._scene.get_center_3d()
            assert np.allclose(current_camera_position_3d, current_position_3d), (
                f"Camera: {current_camera_position_3d}, Scene: {current_position_3d}"
            )

    def _setup_global_visual_nodes(
        self,
        *,
        axis_direction: str | None,
        visual_nodes: VisualNodeCollection,
        force_fit: bool,
    ) -> None:
        """
        Prepare Visual Nodes for Global mode.

        See :meth:`..dimensionreducer.DimensionReducerCoordinator.setup_global_visual_nodes()`.
        """  # noqa: E501
        self._will_be_global_mode = True

        self._scene._setup_global_visual_nodes(
            axis_direction,
            visual_nodes,
            number_of_detailed_nodes=self._configuration.number_of_detailed_nodes,
            force_fit=force_fit,
        )

    def _setup_local_visual_nodes(
        self,
        visual_nodes: VisualNodeCollection,
        *,
        new_position: OptionalVectorPair | None,
        selected_data_point: DataPoint | None,
        shared_neighborhood: bool | None,
        force_fit: bool,
    ) -> None:
        """
        Prepare Visual Nodes for Local mode.

        See :meth:`..dimensionreducer.DimensionReducerCoordinator.setup_local_visual_nodes()`.
        """  # noqa: E501
        self._will_be_global_mode = False

        self._scene._setup_local_visual_nodes(
            visual_nodes,
            new_position=new_position,
            selected_data_point=selected_data_point,
            shared_neighborhood=shared_neighborhood,
            force_fit=force_fit,
            configuration=self._configuration,
        )

    def _prepare_visual_information(self, *, camera_rotation: bool) -> None:
        self._scene._prepare_visual_information(
            will_be_global_mode=self._will_be_global_mode,
            camera_rotation=camera_rotation,
        )

    @abstractmethod
    def _prepare(self, visual_nodes: VisualNodeCollection) -> bool:
        """
        Prepare for navigation.

        Called in :meth:`make_movement()`.

        :param visual_nodes: :class:`.nodes.visual_node.VisualNode` to process.
        :returns: Whether preparation succeeded.
        """

    def _make_watch_interval(self, camera: Camera) -> Interval | None:
        """
        Make `Interval` for the `Camera` to keep watching a certain point.

        :param camera: The `Camera` to move.
        :returns: `Interval` for the `Camera`. `None`, if not necessary.

        .. note:: The `Camera` will keep looking at point set in `self._look_at`.
        """

        def _look_at(data: float) -> None:
            assert self._look_at is not None
            rotation_3d = qSlerp(old_direction, new_direction, data)
            camera.look_at(self._look_at, rotation_3d.getUp())

        old_direction = camera.get_rotation_3d()
        new_direction = camera.get_new_rotation_3d()
        if new_direction is None:
            return None

        return LerpFunc(
            _look_at,
            duration=self._get_scene_configuration().animation.animation_interval_secs,
        )

    def _make_intervals_for_visual_nodes(
        self,
        visual_nodes: VisualNodeCollection,
    ) -> list[Interval]:
        fade_in_count = 0
        result = []
        for each in visual_nodes:
            interval = each.make_interval_to_new_position(
                self._get_scene_configuration().animation.animation_interval_secs,
                clear=False,
            )
            if interval is not None:
                result.append(interval)

            if isinstance(interval, LerpFunc):
                fade_in_count += 1

        _logger.info("# Fading in : %d", fade_in_count)
        _logger.info("# Moving `VisualNode`s: %d", len(result))
        return result

    def _make_intervals_for_camera(self, *, clear: bool) -> list[Interval]:
        """Make Interval for camera."""
        camera = self._get_moving_camera()
        interval = camera.make_interval_to_new_pose(
            self._get_scene_configuration().animation.animation_interval_secs,
            clear=clear,
        )

        if interval is None:
            return []

        return [interval]

    def _make_intervals_for_visual_information(
        self,
        *,
        clear: bool,
    ) -> list[Interval]:
        """Make Intervals for Visual Information."""
        visual_information = self._get_reducer_coordinator().get_visual_information()

        if visual_information is None:
            return []

        intervals = visual_information.make_intervals(
            self._get_scene_configuration().animation.animation_interval_secs,
            clear=clear,
        )

        if intervals is None:
            return []

        return intervals

    def _make_intervals_regarding_camera_movement(self) -> list[Interval]:
        """
        Make Panda3D Intervals for this command caused by Camera movement.

        Called in :meth:`make_intervals()`.

        :returns: List of intervals to run in parallel.
        """
        if self._get_scene_configuration().animation.camera_animation:
            camera_intervals = self._make_intervals_for_camera(clear=False)

            camera = self._get_active_camera()
            visual_information_pose_interval = (
                camera.make_interval_for_visual_information_pose(
                    self._get_scene_configuration().animation.animation_interval_secs,
                    clear=False,
                )
            )

        else:
            camera_intervals = []

            visual_information_pose_interval = None

        result = camera_intervals

        if visual_information_pose_interval is not None:
            result.append(visual_information_pose_interval)

        return result

    def _make_intervals(self, visual_nodes: VisualNodeCollection) -> list[Interval]:
        """
        Make Panda3D Intervals for this command after call to :meth:`_prepare()`.

        :param visual_nodes: :class:`.nodes.visual_node.VisualNode` to process.
        :returns: List of intervals to run in parallel. `None`, if no movement is
          needed.
        """
        result = self._make_intervals_regarding_camera_movement()

        if self._get_scene_configuration().animation.node_animation:
            visual_information_intervals = self._make_intervals_for_visual_information(
                clear=False
            )
            result += visual_information_intervals

            visual_node_intervals = self._make_intervals_for_visual_nodes(visual_nodes)
            _logger.debug(
                "Visual Node 1 interval: %r",
                visual_node_intervals[0] if len(visual_node_intervals) >= 1 else None,
            )
            result += visual_node_intervals

        return result

    def _make_sequence_list(
        self, visual_nodes: VisualNodeCollection
    ) -> list[MetaInterval]:
        """
        Make Panda3D Intervals for this command.

        :param visual_nodes: :class:`.nodes.visual_node.VisualNode` to process.  This
          will not be used after returning from this method.
        :returns: list of intervals to run.
        """
        self._assert_camera_position()

        self._is_prepared = self._prepare(visual_nodes)
        if not self._is_prepared:
            return []

        intervals = self._make_intervals(visual_nodes)
        _logger.info("Number of intervals: %d", len(intervals))

        if not self._configuration.has_animation():
            assert len(intervals) == 0

        if len(intervals) == 0:
            return []

        all_movement = PandaInterval.Parallel(*intervals)

        sequence_list = [all_movement]

        return sequence_list

    def _force_movement(self, visual_nodes: VisualNodeCollection) -> None:
        # Somehow, animation doesn't always move the object to the final destination.

        self._get_navigation_camera().move_to_new_pose()
        self._get_birds_eye_camera().move_to_new_pose()

        visual_information = self._get_reducer_coordinator().get_visual_information()
        if visual_information is not None:
            visual_information.apply_new_transformation()

        for each in visual_nodes:
            each.move_to_new_position()

    def _set_visual_node_properties(self, visual_nodes: VisualNodeCollection) -> None:
        """Set properties for Visual Nodes depending on the location of the cameras."""
        camera = self._get_active_camera()
        camera.set_visual_node_properties(visual_nodes)

    def _reset_selection(self, visual_nodes: VisualNodeCollection) -> None:
        scene = self._scene

        scene.select_data_point(None, my_visual_nodes=visual_nodes)

        if scene._event_processor is not None:
            scene._event_processor.process_event('select-data-point', None, sender=self)

    def _reset_selection_on_goal(self, visual_nodes: VisualNodeCollection) -> None:
        scene = self._scene

        if scene.is_on_selection():
            self._reset_selection(visual_nodes)

    @override
    def _finalize(self) -> None:
        with self._scene._visual_nodes as visual_nodes:
            self._force_movement(visual_nodes)

            self._reset_selection_on_goal(visual_nodes)

            # FUTURE: doesn't really need to set some properties with rotation of camera
            self._set_visual_node_properties(visual_nodes)

        self._scene.set_background()

        assert self._scene.is_global_mode() == self._will_be_global_mode

        super()._finalize()

    @override
    def start(self) -> None:
        if self._configuration.has_animation():
            self._sequence_list.append(Func(self._finalize))

            self._total_sequence = PandaInterval.Sequence(*self._sequence_list)
            self._total_sequence.start()

        else:
            assert len(self._sequence_list) == 0
            self._finalize()


class _MoveCommand(NavigationCommand):
    """Superclass for shift movements."""

    @override
    def _get_moving_camera(self) -> Camera:
        return self._get_navigation_camera()

    def _get_new_pose_for_camera(
        self,
    ) -> NavigationCamera.Pose:
        """
        Get the new pose for the Navigation Camera.

        :returns: New `Pose`.
        """
        camera = self._get_navigation_camera()

        match self._command_str:
            case 'mF':
                pose = camera.make_pose_for_move(camera.get_forward_3d())
            case 'mB':
                pose = camera.make_pose_for_move(-camera.get_forward_3d())
            case 'mR':
                pose = camera.make_pose_for_move(camera.get_right_3d())
            case 'mL':
                pose = camera.make_pose_for_move(-camera.get_right_3d())
            case 'mU':
                pose = camera.make_pose_for_move(camera.get_up_3d())
            case 'mD':
                pose = camera.make_pose_for_move(-camera.get_up_3d())
            case _:
                raise AssertionError(f"Unsupported camera command: {self._command_str}")

        return pose

    def _set_new_pose_for_camera(self) -> None:
        """Set new pose for the Navigation Camera."""
        pose = self._get_new_pose_for_camera()
        self._get_navigation_camera().set_new_pose_3d(
            pose.position_3d, pose.rotation_3d
        )


@Command.global_command_factory.register('m')
class _GlobalMoveCommand(_MoveCommand):
    @override
    def _prepare(self, visual_nodes: VisualNodeCollection) -> bool:
        if not self._scene.has_linear_navigation():
            return False

        self._set_new_pose_for_camera()

        self._prepare_visual_information(camera_rotation=False)

        return True

    @override
    def _finalize(self) -> None:
        self._set_global_mode('')

        return super()._finalize()


class _MoveToLocalCommand(_MoveCommand, ABC):
    @abstractmethod
    def _calculate_new_positions(
        self, visual_nodes: VisualNodeCollection
    ) -> OptionalVectorPair | None:
        """
        Calculate new position in visual and virtual space.

        :param visual_nodes: Visual Nodes in the Scene.
        :returns: New position in visual and virtual space. `None`, for no movement.
          Contains following fields:

          - `visual_3d`: Position in visual space.
          - `virtual`: Position in virtual space. This can be `None`, if this is to be
            inferred from the position in visual space
            (with linear dimension reduction).
        """

    @override
    def _prepare(self, visual_nodes: VisualNodeCollection) -> bool:
        new_position = self._calculate_new_positions(visual_nodes)
        if new_position is None:
            return False

        selected_data_point = self._scene.get_selected_data_point()
        self._setup_local_visual_nodes(
            visual_nodes,
            new_position=new_position,
            selected_data_point=selected_data_point,
            shared_neighborhood=None,
            force_fit=True,
        )

        if new_position is not None:
            assert np_all(self._scene.get_center_3d() == new_position.visual_3d)
            self._get_navigation_camera().set_new_pose_3d(
                LPoint3(*new_position.visual_3d), None
            )

            self._prepare_visual_information(camera_rotation=False)

        return True


@Command.local_command_factory.register('m')
class _LocalMoveCommand(_MoveToLocalCommand):
    @override
    def _calculate_new_positions(
        self, visual_nodes: VisualNodeCollection
    ) -> OptionalVectorPair | None:
        def _get_new_camera_move_position_3d() -> NPArray:
            new_pose = self._get_new_pose_for_camera()

            _logger.info("Move Linear")

            position_3d = new_pose.position_3d
            assert position_3d is not None

            position_array_3d: NPArray = np.array(position_3d)
            return position_array_3d

        def _adjust_movement(
            move_distance: float,
            position: VectorPair,
        ) -> tuple[NPArray, NPArray]:
            # Current position is in `self._new_position`, because `self._move_center()`
            # not called yet with new position.
            current_center_3d = self._scene.get_center_3d()
            current_center = self._scene.get_center()

            total_distance = magnitude(position.virtual - current_center)
            move_ratio = min(1.0, move_distance / total_distance)

            new_position_3d = weighted_sum(
                move_ratio, current_center_3d, position.visual_3d
            )
            new_position = weighted_sum(move_ratio, current_center, position.virtual)

            return new_position_3d, new_position

        def _get_new_camera_shift_position_3d(
            selected_data_point: DataPoint,
        ) -> OptionalVectorPair | None:
            if self._command_str == 'mF':
                move_distance = self._get_navigation_camera().get_shift_speed()
                _logger.info("Move Towards")

            elif self._command_str == 'mB':
                move_distance = -self._get_navigation_camera().get_shift_speed()
                _logger.info("Move Away")

            else:
                return None

            position = self._scene.get_data_point_position(
                selected_data_point, my_visual_nodes=visual_nodes
            )
            if position is None:
                return None

            new_position_3d, new_position = _adjust_movement(move_distance, position)

            return OptionalVectorPair(visual_3d=new_position_3d, virtual=new_position)

        selected_data_point = self._scene.get_selected_data_point()

        if selected_data_point is None:
            if not self._scene.has_linear_navigation():
                return None

            new_position_3d = _get_new_camera_move_position_3d()
            return OptionalVectorPair(visual_3d=new_position_3d, virtual=None)

        return _get_new_camera_shift_position_3d(selected_data_point)


@Command.global_command_factory.register('R')
class _GlobalRefreshCommand(NavigationCommand):
    @override
    def _prepare(self, visual_nodes: VisualNodeCollection) -> bool:
        self._setup_global_visual_nodes(
            axis_direction=None, visual_nodes=visual_nodes, force_fit=True
        )

        return True


@Command.local_command_factory.register('R')
class _LocalRefreshCommand(NavigationCommand):
    @override
    def _prepare(self, visual_nodes: VisualNodeCollection) -> bool:
        self._setup_local_visual_nodes(
            visual_nodes,
            new_position=None,
            selected_data_point=None,
            shared_neighborhood=False,
            force_fit=True,
        )

        self._prepare_visual_information(camera_rotation=False)

        return True


@Command.global_command_factory.register('S')
class _GlobalRedrawCommand(NavigationCommand):
    @override
    def _prepare(self, visual_nodes: VisualNodeCollection) -> bool:
        self._setup_global_visual_nodes(
            axis_direction=None,
            visual_nodes=visual_nodes,
            force_fit=False,
        )

        return True


@Command.local_command_factory.register('S')
class _LocalRedrawCommand(NavigationCommand):
    @override
    def _prepare(self, visual_nodes: VisualNodeCollection) -> bool:
        self._setup_local_visual_nodes(
            visual_nodes,
            new_position=None,
            selected_data_point=None,
            shared_neighborhood=False,
            force_fit=False,
        )

        return True


def choose_direction(axis: NPArray, direction: NPArray) -> NPArray:
    """
    Choose a direction on the axis that is close to specified direction.

    :param axis: Direction of the axis.
    :param direction: Specified direction.
    :returns: Returns either `axis` or `-axis`, whichever is closer to `direction`.
    """
    product = direction @ axis.transpose()

    if product >= 0:
        return axis

    return -axis


def _set_new_pose_to_face_principal_component(
    components: NPArray,
    camera: NavigationCamera,
    reducer: LinearOrientedDimensionReducer,
) -> None:
    def _calculate_new_rotation(
        camera: NavigationCamera, reducer: LinearOrientedDimensionReducer
    ) -> LQuaternion:
        coordinate_system = camera.get_lens().getCoordinateSystem()

        components_3d = reducer.transform(components[:2], is_position=False)

        directions_3d = camera.get_directions_3d()

        forward_3d = choose_direction(components_3d[0], directions_3d.forward_3d)

        up_3d = choose_direction(components_3d[1], directions_3d.up_3d)

        result = LQuaternion()
        lookAt(
            quat=result,
            fwd=LVector3(*forward_3d),
            up=LVector3(*up_3d),
            cs=coordinate_system,
        )

        return result

    new_camera_rotation_3d = _calculate_new_rotation(camera, reducer)

    new_camera_position_3d = camera.get_new_position_3d()

    camera.set_new_pose_3d(new_camera_position_3d, new_camera_rotation_3d)


@Command.local_command_factory.register('d')
class _LocalDriftCommand(_MoveToLocalCommand):
    @override
    def _calculate_new_positions(  # noqa: C901 https://github.com/astral-sh/ruff/issues/4384
        self, visual_nodes: VisualNodeCollection
    ) -> OptionalVectorPair | None:
        def _choose_movement(
            camera: NavigationCamera,
            reducer: LinearOrientedDimensionReducer,
            command_str: str,
        ) -> NPArray | None:
            components = self._get_components()

            camera_directions_3d = camera.get_directions_3d()

            match command_str:
                case 'dF':
                    component_index = 0
                    camera_direction_3d = camera_directions_3d.forward_3d

                case 'dB':
                    component_index = 0
                    camera_direction_3d = -camera_directions_3d.forward_3d

                case 'dR':
                    component_index = 2
                    camera_direction_3d = camera_directions_3d.right_3d

                case 'dL':
                    component_index = 2
                    camera_direction_3d = -camera_directions_3d.right_3d

                case 'dU':
                    component_index = 1
                    camera_direction_3d = camera_directions_3d.up_3d

                case 'dD':
                    component_index = 1
                    camera_direction_3d = -camera_directions_3d.up_3d

                case _:
                    raise AssertionError("Illegal command str")

            if (components is None) or (len(components) <= component_index):
                return None

            direction = reducer.calculate_direction_from_3d(
                np.array(camera_direction_3d)
            )
            _logger.info("Camera direction: %r\n%r", direction, camera_direction_3d)

            result = choose_direction(components[component_index], direction)
            return result

        def _calculate_new_positions(
            direction: NPArray, reducer: LinearOrientedDimensionReducer
        ) -> tuple[NPArray, NPArray]:
            current_position = self._scene.get_center()
            new_position = current_position + direction

            new_position_3d = reducer.transform(new_position)

            return new_position_3d, new_position

        reducer = self._scene.get_reducer()
        if not isinstance(reducer, LinearOrientedDimensionReducer):
            return None

        movement_direction = _choose_movement(
            self._get_navigation_camera(), reducer, self._command_str
        )
        if movement_direction is None:
            return None

        new_position_3d, new_position = _calculate_new_positions(
            movement_direction, reducer
        )

        return OptionalVectorPair(visual_3d=new_position_3d, virtual=new_position)

    @override
    def _prepare(self, visual_nodes: VisualNodeCollection) -> bool:
        result = super()._prepare(visual_nodes)
        if not result:
            return False

        camera = self._get_navigation_camera()

        reducer = self._scene.get_reducer()
        assert isinstance(reducer, LinearOrientedDimensionReducer)

        components = self._get_components()

        if components is None:
            return False

        _set_new_pose_to_face_principal_component(components, camera, reducer)

        self._prepare_visual_information(camera_rotation=True)

        return True


# FUTURE: _GlobalDriftCommand


@Command.global_command_factory.register('z')
@Command.local_command_factory.register('z')
class _ZoomInCommand(_MoveToLocalCommand):
    @override
    def _calculate_new_positions(
        self, visual_nodes: VisualNodeCollection
    ) -> OptionalVectorPair | None:
        selected_data_point = self._scene.get_selected_data_point()

        if selected_data_point is None:
            return None

        position = self._scene.get_data_point_position(
            selected_data_point, my_visual_nodes=visual_nodes
        )
        if position is None:
            # FUTURE: Jump between neighborhoods with nonlinear dimension reduction
            return None

        return OptionalVectorPair.from_vector_pair(position)

    @override
    def _finalize(self) -> None:
        if self._is_prepared:
            self._set_global_mode(None)

        super()._finalize()


@Command.global_command_factory.register('r')
@Command.local_command_factory.register('r')
class _RotateCommand(NavigationCommand):
    @override
    def _get_moving_camera(self) -> Camera:
        return self._get_navigation_camera()

    def _set_new_pose_for_camera(self) -> None:
        camera = self._get_navigation_camera()

        match self._command_str:
            case 'rU':
                pose = camera.make_pose_for_rotate(camera.get_right_3d(), +1)
            case 'rD':
                pose = camera.make_pose_for_rotate(camera.get_right_3d(), -1)
            case 'rA':
                pose = camera.make_pose_for_rotate(camera.get_forward_3d(), +1)
            case 'rC':
                pose = camera.make_pose_for_rotate(camera.get_forward_3d(), -1)
            case 'rL':
                pose = camera.make_pose_for_rotate(camera.get_up_3d(), +1)
            case 'rR':
                pose = camera.make_pose_for_rotate(camera.get_up_3d(), -1)
            case _:
                raise AssertionError(f"Unsupported camera command: {self._command_str}")

        camera.set_new_pose_3d(pose.position_3d, pose.rotation_3d)

    @override
    def _prepare(self, visual_nodes: VisualNodeCollection) -> bool:
        self._set_new_pose_for_camera()
        self._prepare_visual_information(camera_rotation=True)
        return True

    @override
    def _finalize(self) -> None:
        if self._scene.is_global_mode():
            self._set_global_mode('')

        return super()._finalize()


@Command.global_command_factory.register('c')
@Command.local_command_factory.register('c')
class _CenterSelectedCommand(_RotateCommand):
    def _get_selected_position_3d(
        self, *, visual_nodes: VisualNodeCollection
    ) -> LPoint3 | None:
        reducer_coordinator = self._get_reducer_coordinator()

        if reducer_coordinator.has_linear_navigation():
            selected_data_point = self._scene.get_selected_data_point()

            if selected_data_point is None:
                return None

            position = self._scene.get_data_point_position(
                selected_data_point, my_visual_nodes=visual_nodes
            )
            assert position is not None
            selected_position_3d = LPoint3(*position.visual_3d)

        else:
            selected_node = self._scene.get_selected_visual_node(
                my_visual_nodes=visual_nodes
            )

            if selected_node is None:
                return None

            selected_position_3d = selected_node.get_position_3d()

        return selected_position_3d

    def _set_new_pose_for_camera_to_selected(
        self, visual_nodes: VisualNodeCollection
    ) -> bool:
        selected_position_3d = self._get_selected_position_3d(visual_nodes=visual_nodes)

        if selected_position_3d is None:
            _logger.info("No selection.")
            return False

        if self._get_navigation_camera().get_position_3d() == selected_position_3d:
            _logger.info("Camera on Selected Node.")
            return False

        up = self._get_navigation_camera().get_up_3d()

        self._get_navigation_camera().set_new_pose_for_look_at(selected_position_3d, up)

        return True

    @override
    def _prepare(self, visual_nodes: VisualNodeCollection) -> bool:
        if not self._set_new_pose_for_camera_to_selected(visual_nodes):
            return False

        self._prepare_visual_information(camera_rotation=True)
        return True


@Command.global_command_factory.register('f')
@Command.local_command_factory.register('f')
class _FacePrincipalComponentCommand(_RotateCommand):
    @override
    def _prepare(self, visual_nodes: VisualNodeCollection) -> bool:
        camera = self._get_navigation_camera()

        reducer = self._scene.get_reducer()
        assert isinstance(reducer, LinearOrientedDimensionReducer)

        components = self._get_components()

        if components is None:
            return False

        _set_new_pose_to_face_principal_component(components, camera, reducer)

        self._prepare_visual_information(camera_rotation=True)

        return True

    @override
    def _make_intervals_for_camera(self, *, clear: bool) -> list[Interval]:
        camera_interval = self._get_navigation_camera().make_interval_to_new_pose(
            self._get_scene_configuration().animation.animation_interval_secs,
            clear=clear,
        )
        assert camera_interval is not None
        return [camera_interval]


class _ShowAllCommand(NavigationCommand):
    def __init__(self, *, scene: NavigationScene, command_str: str) -> None:
        if len(command_str) == 1:
            command_str = self._rotate_global_mode(scene)
            _logger.info("New Global View: %s", command_str)

        super().__init__(scene=scene, command_str=command_str)

    @override
    def _prepare(self, visual_nodes: VisualNodeCollection) -> bool:
        self._set_birds_eye_mode(axis_direction=None, visual_nodes=visual_nodes)

        navigation_camera = self._get_navigation_camera()

        self._look_at = navigation_camera.set_new_pose_for_show_all(
            self._command_str[1:],
            visual_nodes,
            margin=self._get_unique_scene_configuration().visual_size / 2,
        )

        if self._look_at is None:
            return False

        self._prepare_visual_information(camera_rotation=True)

        return True

    @override
    def _get_moving_camera(self) -> Camera:
        return self._get_navigation_camera()

    @override
    def _make_intervals_for_camera(self, *, clear: bool) -> list[Interval]:
        result = super()._make_intervals_for_camera(clear=clear)

        camera = self._get_navigation_camera()
        watch_interval = self._make_watch_interval(camera)
        if watch_interval is not None:
            result.append(watch_interval)

        return result

    @override
    def _finalize(self) -> None:
        self._set_global_mode(self._command_str[1:])

        super()._finalize()

    def _rotate_global_mode(self, scene: NavigationScene) -> str:
        current_mode = scene.get_global_mode()
        new_mode = self._rotate(
            current=current_mode,
            initial=scene.INITIAL_AXIS_DIRECTION,
            choices=scene.GLOBAL_VIEWS,
        )

        return 'a' + new_mode


@Command.global_command_factory.register('a')
class _GlobalShowAllCommand(_ShowAllCommand):
    """
    Command to rotate view in Global Mode.

    :param scene: Scene for this command.
    :param command_str: `str` representing the command. The following commands are
      supported:

          -  'a' : Switch between different directions in Global Mode (switch to Global Mode)
          -  'a+X': show all Visual Nodes TOWARD +X direction (switch to Global Mode)
          -  'a+Y': show all Visual Nodes TOWARD +Y direction (switch to Global Mode)
          -  'a+Z': show all Visual Nodes TOWARD +Z direction (switch to Global Mode)
          -  'a-X': show all Visual Nodes TOWARD -X direction (switch to Global Mode)
          -  'a-Y': show all Visual Nodes TOWARD -Y direction (switch to Global Mode)
          -  'a-Z': show all Visual Nodes TOWARD -Z direction (switch to Global Mode)

      See :meth:`NavigationScene.process_command()`.
    """  # noqa: E501


@Command.local_command_factory.register('a')
class _ShowAllFromLocalCommand(_ShowAllCommand):
    @override
    def _prepare(self, visual_nodes: VisualNodeCollection) -> bool:
        self._reset_selection(visual_nodes)

        self._setup_global_visual_nodes(
            axis_direction=self._command_str[1:],
            visual_nodes=visual_nodes,
            force_fit=True,
        )
        return super()._prepare(visual_nodes)


@Command.local_command_factory.register('b')
class _SwitchBirdsEyeCommand(NavigationCommand):
    """
    Command to rotate view in Bird's Eye Mode.

    :param scene: Scene for this command.
    :param command_str: `str` representing the command. The following commands are
      supported:

        -  'b'  : Switch between different directions in Bird's Eye Mode (switch to Bird's Eye Mode)
        -  'b+X': show neighborhood TOWARD +X direction (switch to Bird's Eye Mode)
        -  'b+Y': show neighborhood TOWARD +Y direction (switch to Bird's Eye Mode)
        -  'b+Z': show neighborhood TOWARD +Z direction (switch to Bird's Eye Mode)
        -  'b-X': show neighborhood TOWARD -X direction (switch to Bird's Eye Mode)
        -  'b-Y': show neighborhood TOWARD -Y direction (switch to Bird's Eye Mode)
        -  'b-Z': show neighborhood TOWARD -Z direction (switch to Bird's Eye Mode)

      See :meth:`NavigationScene.process_command()`.
    """  # noqa: E501

    def __init__(self, *, scene: NavigationScene, command_str: str) -> None:
        if len(command_str) == 1:
            command_str = self._rotate_birds_eye_mode(scene)
            _logger.info("New Bird's Eye View: %s", command_str)

        super().__init__(scene=scene, command_str=command_str)

    def _rotate_birds_eye_mode(self, scene: NavigationScene) -> str:
        assert not scene.is_global_mode()

        current_axis_direction = scene.get_birds_eye_mode(long=False)
        new_axis_direction = self._rotate(
            current=current_axis_direction,
            initial=scene.BIRDS_EYE_VIEWS[0],
            choices=scene.BIRDS_EYE_VIEWS,
        )

        return 'b' + new_axis_direction

    @override
    def _prepare(self, visual_nodes: VisualNodeCollection) -> bool:
        def _move_birds_eye_camera_to_navigating_camera() -> None:
            direction = LQuaternion.identQuat()
            self._get_birds_eye_camera().set_pose_3d(
                position_3d=LPoint3(0, 0, 0), rotation_3d=direction
            )

        def _set_new_pose(axis_direction: str) -> None:
            self._get_birds_eye_camera().set_new_pose_for_birds_eye_view(
                axis_direction,
                visual_nodes,
                margin=self._get_unique_scene_configuration().visual_size / 2,
            )

            self._get_birds_eye_camera().set_new_pose_for_visual_information(
                will_be_global_mode=False
            )

            self._look_at = LPoint3(0, 0, 0)  # Position of `NavigationCamera`

        if len(visual_nodes) == 0:
            return False

        if not self._scene.is_birds_eye_mode():
            _move_birds_eye_camera_to_navigating_camera()

        axis_direction = self._command_str[1:]

        self._set_birds_eye_mode(
            axis_direction=axis_direction, visual_nodes=visual_nodes
        )

        _set_new_pose(axis_direction)

        return True

    @override
    def _get_moving_camera(self) -> Camera:
        return self._get_birds_eye_camera()

    @override
    def _make_intervals_for_camera(self, *, clear: bool) -> list[Interval]:
        result = super()._make_intervals_for_camera(clear=clear)

        camera = self._get_birds_eye_camera()
        watch_interval = self._make_watch_interval(camera)
        if watch_interval is not None:
            result.append(watch_interval)

        return result


@Command.local_command_factory.register('x')
class _ExitBirdsEyeCommand(NavigationCommand):
    @override
    def _prepare(self, visual_nodes: VisualNodeCollection) -> bool:
        def _set_new_pose() -> None:
            direction = LQuaternion.identQuat()
            self._get_birds_eye_camera().set_new_pose_3d(
                # Position of Navigation Camera
                position_3d=LPoint3(0, 0, 0),
                rotation_3d=direction,
            )

            self._get_birds_eye_camera().set_new_pose_for_visual_information(
                will_be_global_mode=True
            )

        if not self._scene.is_birds_eye_mode():
            return False

        _set_new_pose()

        # Can jerk at last moment, if set to position of Navigation Camera
        self._look_at = None

        return True

    @override
    def _get_moving_camera(self) -> Camera:
        return self._get_birds_eye_camera()

    @override
    def _finalize(self) -> None:
        with self._scene._visual_nodes as visual_nodes:
            self._set_birds_eye_mode(axis_direction=None, visual_nodes=visual_nodes)

        super()._finalize()
