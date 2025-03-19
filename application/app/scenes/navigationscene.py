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
Defines class that manages the visual space.

Array type: Panda3D
"""

from abc import ABC
from collections.abc import Callable
from copy import deepcopy
import math
from types import NoneType
from typing import cast

from typing_extensions import override

import numpy as np

from direct.showbase.ShowBase import ShowBase
from panda3d.core import (
    LPoint2,
    LVector2i,
    NodePath,
)

import pylog

from ..datasource.api import DataPoint, DataSource, distance_between
from ..dimensionreducer import DimensionReducerCoordinator
from ..dimensionreduction import DimensionReducer
from ..eventprocessing.eventprocessor import (
    EventHandlerRegistry,
    EventProcessor,
    EventSubscriber,
    NavigationEventSubscriber,
    ValueType,
)
from ..mathutils import NPArray, OptionalVectorPair, VectorPair, np_all
from ..nodes.birdseyecamera import BirdsEyeCamera
from ..nodes.camera import Camera
from ..nodes.navigationcamera import NavigationCamera
from ..nodes.visualinformation import VisualInformation
from ..nodes.visualnode import VisualNode, VisualNodeCollection
from ..window import CanvasWindow
from . import command
from .configuration import (
    ReducerConfiguration,
    SharedNavigationSceneConfiguration,
    UniqueNavigationSceneConfiguration,
    WindowConfiguration,
)
from .scene import Scene

_logger = pylog.getLogger(__name__)


class NavigationScene(Scene, NavigationEventSubscriber, ABC):
    """
    A world that has visuals and a camera.

    :param shared_scene_configuration: Configuration parameters shared between
      Navigation Scenes.
    :param unique_scene_configuration: Configuration parameters unique to this
      Navigation Scene. Instance will be copied.
    :param data_source: Data Source that holds the data points.
    :param reducer_configuration: Configuration of the dimension reduction algorithms.
    :param window_configuration: Configuration for the window.
    :param is_main: Whether this is the main window. If not, a new window will be
      created.
    :param event_processor: The processor that handles App Events.
    :param base: The :class:`direct.showbase.ShowBase.ShowBase` instance.
    """

    def __init__(
        self,
        shared_scene_configuration: SharedNavigationSceneConfiguration,
        unique_scene_configuration: UniqueNavigationSceneConfiguration,
        data_source: DataSource,
        reducer_configuration: ReducerConfiguration,
        window_configuration: WindowConfiguration,
        *,
        is_main: bool,
        event_processor: EventProcessor | None,
        base: ShowBase,
    ) -> None:
        def _dimension_reducer_configuration(
            reducer_configuration: ReducerConfiguration,
        ) -> DimensionReducerCoordinator.Configuration:
            local_dimension_reduction = (
                {"type": "Null"}
                if reducer_configuration.local_mode is None
                else reducer_configuration.local_mode
            )

            return DimensionReducerCoordinator.Configuration(
                global_mode=reducer_configuration.global_mode,
                local_mode=local_dimension_reduction,
                random_generator=reducer_configuration.random_generator,
                visual_information=reducer_configuration.visual_information,
            )

        window_size = (
            self.NAVIGATION_SCENE_DEFAULT_WINDOW_SIZE
            if window_configuration.window_size is None
            else window_configuration.window_size
        )
        super().__init__(
            window_origin=window_configuration.window_origin,
            window_size=window_size,
            title=window_configuration.title,
            is_main=is_main,
            parent=window_configuration.parent,
            screenshot_prefix=unique_scene_configuration.screenshot_prefix,
            base=base,
        )

        self._birds_eye_camera = BirdsEyeCamera(
            navigation_camera=self._get_navigation_camera(),
            window=self._visuals.window,
            base=base,
            scene_root=self._visuals.scene_root_nodepath,
        )
        self._shared_configuration = shared_scene_configuration
        self._unique_configuration = deepcopy(unique_scene_configuration)
        self._command_processor = command.CommandProcessor(self)
        self._global_mode: str | None = None
        self._birds_eye_mode: str | None = None
        self._visual_information: VisualInformation | None = None
        self._show_display = True

        with shared_scene_configuration.with_lock() as configuration:
            visual_space = DimensionReducerCoordinator.SpaceConfiguration(
                self._get_navigation_camera(),
                configuration.max_number_of_points,
                unique_scene_configuration.visual_size,
                configuration.neighborhood_radius,
                configuration.shared_neighborhood,
            )

            dimension_reducer_configuration = _dimension_reducer_configuration(
                reducer_configuration
            )
            self._reducer_coordinator = DimensionReducerCoordinator(
                visual_space,
                data_source,
                dimension_reducer_configuration,
            )

            number_of_detailed_nodes = configuration.number_of_detailed_nodes

        self._event_processor = event_processor
        if event_processor is not None:
            event_processor.add_subscriber(self)

        self._show_all(
            axis_direction=NavigationScene.INITIAL_AXIS_DIRECTION,
            number_of_detailed_nodes=number_of_detailed_nodes,
        )

        self._set_visual_node_properties()

        self.set_background()

        self.get_window().set_on_focus_callback(self._on_focus)
        self._on_focus()  # First time gets missed, because window is already shown.

    NAVIGATION_SCENE_DEFAULT_WINDOW_SIZE = LVector2i(1280, 640)
    """Default window size for Navigation Scene."""

    _STABILIZE = True
    """Reuse 3d coordinates of Visual Nodes, without recalculation, if possible."""

    @override
    def get_title(self) -> str:
        """Get the title of the Window."""
        return self.get_window().get_title()

    def _get_help(self) -> list[str]:
        linear_navigation = self.has_linear_navigation()

        text = [
            "[a]: Show All",
            "[b]: Bird's Eye View <L>",
            "[Esc]: Exit Bird's Eye View <L>",
            "[r]: Force Dimensionality Reduction <L>",
            "[z]: Zoom in to selected node",
        ]

        if linear_navigation:
            text.extend(["[c]: Center Selected Node", "[8]: Move Forward"])

        text.extend(
            [
                "[4]: Rotate Left",
                "[6]: Rotate Right",
            ]
        )

        if linear_navigation:
            text.extend(
                [
                    "[2]: Move Back",
                    "[Shift+8]: Move Up",
                    "[Shift+4]: Move Left",
                    "[Shift+6]: Move Right",
                    "[Shfit+2]: Move Down",
                ]
            )

        text.extend(
            [
                "[Ctrl+8]: Rotate Up",
                "[Ctrl+4]: Rotate Anti-clockwise",
                "[Ctrl+6]: Rotate Clockwise",
                "[Ctrl+2]: Rotate Down",
                "[8,4,6,2] can be replace with [I,J,K,M]",
                "<L> is for <Local> mode only.",
            ]
        )

        return text

    def _on_focus(self) -> None:
        if self._event_processor is not None:
            self._event_processor.set_target(self)

    @override
    def _on_close(self) -> bool:
        if self._command_processor.is_processing():
            _logger.info("Command is processing.")
            return False

        if self._event_processor is not None:
            self._event_processor.remove_subscriber(self)

        return super()._on_close()

    @override
    def _on_size(self, width: int, height: int) -> None:
        aspect_ratio = width / height

        self._set_aspect_ratio(self._get_navigation_camera(), aspect_ratio)

        self._set_aspect_ratio(self._get_birds_eye_camera(), aspect_ratio)

        self._get_navigation_camera().update_visual_information_position()
        self._get_birds_eye_camera().update_visual_information_position()

    @override
    def _make_camera(
        self, *, window: CanvasWindow, scene_root: NodePath, base: ShowBase
    ) -> Camera:
        return NavigationCamera(
            scene_root,
            window=window,
            base=base,
        )

    def _show_all(self, *, axis_direction: str, number_of_detailed_nodes: int) -> None:
        with self._visual_nodes as visual_nodes:
            self._setup_global_visual_nodes(
                axis_direction,
                visual_nodes,
                number_of_detailed_nodes=number_of_detailed_nodes,
                force_fit=True,
            )
            self._show_all_visual_nodes(
                visual_nodes, self._unique_configuration.visual_size, axis_direction
            )
            self._update_visual_information()

        self._set_global_mode(axis_direction)

    def get_number_of_visuals(self) -> int:
        """Get the number of visuals currently being handled."""
        with self._visual_nodes as visual_nodes:
            return len(visual_nodes)

    def _set_visual_node_properties(self) -> None:
        # FUTURE: lock not necessary within initializer
        with self._visual_nodes as visual_nodes:
            self._get_navigation_camera().set_visual_node_properties(visual_nodes)

    def _color_map(self, value: float | None) -> tuple[float, float, float]:
        if value is None:
            return self._unique_configuration.background_color

        inverse = 1.0 - value
        return cast(
            tuple[float, float, float],
            tuple(
                each * inverse for each in self._unique_configuration.background_color
            ),
        )

    @override
    def set_background(self) -> None:
        # FUTURE: animate, or set before movement
        color = self._color_map(self._reducer_coordinator.get_depth())
        self._set_background_color(*color)

    def _get_reducer_coordinator(self) -> DimensionReducerCoordinator:
        """Get the `DimensionReducerCoordinator`."""
        return self._reducer_coordinator

    def _get_reducer_message(self) -> list[str]:
        """Get message from the dimension reducer."""
        return self._reducer_coordinator.get_message()

    # "Global View" refers to direction.
    # Make sure there are no 180 degree rotations, because it is ambiguous
    GLOBAL_VIEWS = ('+Y', '-Z', '-X', '-Y', '+Z', '+X')
    GLOBAL_VIEW_AXES = (
        "[+Y]",
        "[-Z]",
        "[-X]",
        "[-Y]",
        "[+Z]",
        "[+X]",
    )
    INITIAL_AXIS_DIRECTION = '+Y'
    """View direction, when :class:`NavigationScene` is instantiated."""

    def is_global_mode(self) -> bool:
        """
        Return whether this :class:`Scene` is in Global Mode or not.

        :returns: `True`, when this :class:`Scene` is in global mode.

        "Global Mode" is when all the nodes are shown (although sparse).
        The antonym "Local Mode" is when only the neighborhood around a point
        is shown.
        """
        return self._global_mode is not None

    def get_global_mode(self) -> str | None:
        """
        Return the direction of view in Global mode.

        One of the following:

        - '+X': Showing all TOWARD +X direction.
        - '+Y': Showing all TOWARD +Y direction.
        - '+Z': Showing all TOWARD +Z direction.
        - '-X': Showing all TOWARD -X direction.
        - '-Y': Showing all TOWARD -Y direction.
        - '-Z': Showing all TOWARD -Z direction.
        - '': Direction not determined (relaxed).
        - `None`: In Local mode.
        """
        return self._global_mode

    def _set_global_mode(self, mode: str | None) -> None:
        assert (mode is None) or (mode == '') or (mode in self.GLOBAL_VIEWS)

        self._global_mode = mode

    # Direction of the camera is smooth when Bird's Eye View starts with '+Y'.
    BIRDS_EYE_VIEWS = (
        # Make sure there are no 180 degree rotations, because it is ambiguous
        '+Y',
        '-Z',
        '-X',
        '-Y',
        '+Z',
        '+X',
    )
    _BIRDS_EYE_DIRECTION = (
        '[forward] +',
        '[above] ^',
        '[right] >',
        '[backward] -',
        '[below] >',
        '[left] ^',
    )

    def is_birds_eye_mode(self) -> bool:
        """
        Return whether this :class:`Scene` is in Bird's Eye Mode or not.

        :returns: `True`, when this :class:`Scene` is in Bird's Eye Mode.

        "Bird's Eye Mode" is when all the nodes in the neighborhood are shown.
        """
        return self._birds_eye_mode is not None

    def get_birds_eye_mode(self, *, long: bool) -> str | None:
        """
        Return the current Bird's Eye mode.

        If `long == False`, returns one of the following:

        - '+X': Showing neighbors TOWARD +X direction of the Navigation Camera.
        - '+Y': Showing neighbors TOWARD +Y direction of the Navigation Camera.
        - '+Z': Showing neighbors TOWARD +Z direction of the Navigation Camera.
        - '-X': Showing neighbors TOWARD -X direction of the Navigation Camera.
        - '-Y': Showing neighbors TOWARD -Y direction of the Navigation Camera.
        - '-Z': Showing neighbors TOWARD -Z direction of the Navigation Camera.
        - `None`: Not in Bird's Eye mode.

        If `long == True`, returns a more human readable string.
        """
        if long:
            return self._BIRDS_EYE_DIRECTION[
                self.BIRDS_EYE_VIEWS.index(self._birds_eye_mode)
            ]

        return self._birds_eye_mode

    def _set_birds_eye_mode(self, mode: str | None) -> None:
        visual_information = self._visual_information

        if self.is_birds_eye_mode() ^ (mode is not None):
            if self.is_birds_eye_mode():
                camera: Camera = self._get_navigation_camera()

            else:
                camera = self._get_birds_eye_camera()

            camera.set_visual_information(visual_information)

        self._birds_eye_mode = mode

    def has_linear_navigation(self) -> bool:
        """Return whether linear navigation is supported."""
        return self._reducer_coordinator.has_linear_navigation()

    def get_reducer(self) -> DimensionReducer:  # FUTURE: necessary?
        """Get the dimension reduction algorithm."""
        return self._reducer_coordinator.get_reducer()

    def process_command(
        self, command_str: str, *, callback: Callable[[], None] | None = None
    ) -> None:
        """
        Execute a navigation command.

        The command will be ignored, if movement is in progress, or movement is
        not possible.

        :param command_str:
          One of the following:

          -  'mF': move forward
          -  'mB': move backward
          -  'mR': move right
          -  'mL': move left
          -  'mU': move up
          -  'mD': move down
          -  'dF': move along 1st principal component
          -  'dB': move against 1st principal component
          -  'dR': move along 2nd principal component
          -  'dL': move against 2nd principal component
          -  'rU': rotate up
          -  'rD': rotate down
          -  'rA': rotate anti-clockwise
          -  'rC': rotate clockwise
          -  'rL': rotate left
          -  'rR': rotate right
          -  'a' : Switch between different directions in Global Mode (switch to Global Mode)
          -  'a+X': show all Visual Nodes TOWARD +X direction (switch to Global Mode)
          -  'a+Y': show all Visual Nodes TOWARD +Y direction (switch to Global Mode)
          -  'a+Z': show all Visual Nodes TOWARD +Z direction (switch to Global Mode)
          -  'a-X': show all Visual Nodes TOWARD -X direction (switch to Global Mode)
          -  'a-Y': show all Visual Nodes TOWARD -Y direction (switch to Global Mode)
          -  'a-Z': show all Visual Nodes TOWARD -Z direction (switch to Global Mode)
          -  'b'  : Switch between different directions in Bird's Eye Mode (switch to Bird's Eye Mode)
          -  'b+X': show neighborhood TOWARD +X direction (switch to Bird's Eye Mode)
          -  'b+Y': show neighborhood TOWARD +Y direction (switch to Bird's Eye Mode)
          -  'b+Z': show neighborhood TOWARD +Z direction (switch to Bird's Eye Mode)
          -  'b-X': show neighborhood TOWARD -X direction (switch to Bird's Eye Mode)
          -  'b-Y': show neighborhood TOWARD -Y direction (switch to Bird's Eye Mode)
          -  'b-Z': show neighborhood TOWARD -Z direction (switch to Bird's Eye Mode)
          -  'zI': zoom in to selected Visual Nodes (switch to Local mode)
          -  'cS': center selected Visual Node
          -  'fP': face principal component
          -  'S': show Visual Nodes, when more will be added or removed
          -  'R': redo calculations (performs dimension reduction without moving)

        :param callback: Function to call, after movement has finished.
        :raises KeyError: Command not found.
        """  # noqa: E501
        self._command_processor.process_command(command_str, callback)

    def push_command(
        self, command_str: str, callback: Callable[[], None] | None = None
    ) -> None:
        """
        Pushes a command into the queue, and executes it, if it is first in the queue.

        :param command_str: See :meth:`process_command()`.
        :param callback: Function to call, after movement has finished.
        """
        self._command_processor.push_command(command_str, callback)

    def select_data_point(
        self,
        selection: DataPoint | None,
        *,
        my_visual_nodes: VisualNodeCollection | None = None,
    ) -> None:
        """
        Select :class:`VisualNode`for :class:`DataPoint``.

        :param selection: `DataPoint` of `VisualNode` to be selected. A `VisualNode`
          with the same coordinates will be selected, if it exists. If `None`, this will
          cause selection to be cancelled.
        :param my_visual_nodes: Set to Visual Nodes of this `Scene`, if lock is already
          acquired.
        """
        if my_visual_nodes is None:
            with self._visual_nodes as visual_nodes:
                self.select_data_point(selection, my_visual_nodes=visual_nodes)
                return

        super().select_data_point(selection)

        if selection is None:
            my_visual_nodes.deselect()
            _logger.info("Deselected")
            return

        my_visual_nodes.select_data_point(selection)

    def get_selected_visual_node(
        self, *, my_visual_nodes: VisualNodeCollection | None = None
    ) -> VisualNode | None:
        """
        Get selected Visual Node.

        :param my_visual_nodes: Set to Visual Nodes of this `Scene`, if lock is already
          acquired.
        :return: Selected Visual Node. May not be a node that is in the
          :class:`Scene`.
        """
        if my_visual_nodes is None:
            with self._visual_nodes as visual_nodes:
                return self.get_selected_visual_node(my_visual_nodes=visual_nodes)

        return my_visual_nodes.get_selected()

    def get_center_3d(self) -> NPArray:
        """Get current position in Visual Space."""
        return self._reducer_coordinator.get_center_3d()

    def get_center(self) -> NPArray:
        """Get current position in Virtual Space."""
        return self._reducer_coordinator.get_center()

    def is_on_selection(self) -> bool:
        """Return whether current Navigation Camera position is on the selection."""
        selected_data_point = self.get_selected_data_point()

        return (
            (selected_data_point is not None)
            and not self.is_global_mode()
            and np_all(
                self.get_center() == np.array(selected_data_point.get_coordinates())
            )
        )

    def _get_navigation_camera(self) -> NavigationCamera:
        """Get the `NavigationCamera` for this `Scene`."""
        assert isinstance(self._visuals.camera, NavigationCamera)
        return self._visuals.camera

    def is_processing_command(self) -> bool:
        """Return whether the scene is processing a command."""
        return self._command_processor.is_processing()

    def _get_birds_eye_camera(self) -> BirdsEyeCamera:
        """Get the `BirdsEyeCamera` for this `Scene`."""
        return self._birds_eye_camera

    def _get_active_camera(self) -> Camera:
        if self._get_birds_eye_camera().is_active():
            return self._get_birds_eye_camera()

        if self._get_navigation_camera().is_active():
            return self._get_navigation_camera()

        raise AssertionError("No camera is active.")

    def _get_node_at(self, position: LPoint2) -> VisualNode | None:
        """
        Obtain the object that is shown at a specific location in the camera view.

        :param position: 2D coordinates of the location.

        :return: The :class:`VisualNode` at the specific position.
        """
        camera = self._get_active_camera()
        new_nodepath: NodePath = camera.get_nodepath_at(
            position,
            pickable_tag=VisualNode.PICKABLE_TAG,
            pickable_value=None,
            # TODO: Make Visual Nodes behind near ones pickable. `pickable_value='Y'`
        )
        _logger.info("Selected: %r", new_nodepath)

        with self._visual_nodes as visual_nodes:
            new_visual_node = visual_nodes.get_visual_node_from_nodepath(new_nodepath)
            _logger.info("newVisualNode: %r", new_visual_node)

        return new_visual_node

    def get_data_point_at(self, position: LPoint2) -> DataPoint | None:
        """
        Obtain Data Point for the node that is shown at a specific visual location.

        :param position: 2D coordinates of the location.

        :return: The :class:`DataPoint` for the :class:`VisualNode` at the specific
          position.
        """
        visual_node = self._get_node_at(position)

        if visual_node is None:
            return None

        return visual_node.get_data_point()

    def calculate_distance_to_neighbor(self, data_point: DataPoint) -> float:
        """Calculate the virtual distance to the closest Visual Node in this `Scene`."""
        # FUTURE: Should use Dimension Reducer
        minimum_distance = math.inf
        with self._visual_nodes as visual_nodes:
            for each in visual_nodes:
                if each.get_data_point() != data_point:
                    distance = distance_between(
                        each.get_data_point().get_coordinates(),
                        data_point.get_coordinates(),
                    )
                    minimum_distance = min(minimum_distance, distance)

        return minimum_distance

    def get_data_point_position(
        self,
        data_point: DataPoint,
        *,
        my_visual_nodes: VisualNodeCollection,
    ) -> VectorPair | None:
        """
        Obtain position in visual space of a Data Point.

        Uses position of the corresponding Visual Node, if available in the Scene.

        :param data_point: The :class:`DataPoint`.
        :param my_visual_nodes: Set to Visual Nodes of this `Scene`.
        :return: Position in visual space and virtual space. `None`, if dimension
          reducer is nonlinear and Data Point is not in the neighborhood.
        """
        if self._STABILIZE:
            visual_node = my_visual_nodes.get_visual_node_from_data_point(data_point)
            if visual_node is not None:
                return VectorPair(
                    visual_3d=np.array(visual_node.get_position_3d()),
                    virtual=np.array(visual_node.get_data_point().get_coordinates()),
                )

        if not self._reducer_coordinator.has_linear_navigation():
            return None

        return self._reducer_coordinator.calculate_data_point_positions(data_point)

    def _setup_global_visual_nodes(
        self,
        axis_direction: str | None,
        visual_nodes: VisualNodeCollection,
        *,
        number_of_detailed_nodes: int,
        force_fit: bool,
    ) -> None:
        directions = (
            None
            if axis_direction is None
            else self._get_navigation_camera().get_axis_directions_3d(axis_direction)
        )
        self._reducer_coordinator.setup_global_visual_nodes(
            directions,
            visual_nodes,
            number_of_detailed_nodes=number_of_detailed_nodes,
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
        configuration: SharedNavigationSceneConfiguration.Data,
    ) -> None:
        camera = self._get_active_camera()

        self._reducer_coordinator.setup_local_visual_nodes(
            visual_nodes,
            new_position=new_position,
            number_of_detailed_nodes=configuration.number_of_detailed_nodes,
            selected_data_point=selected_data_point,
            point_to_camera=camera,
            new_alpha=0.0 if configuration.animation.node_animation else 1.0,
            shared_neighborhood=shared_neighborhood,
            force_fit=force_fit,
        )

    def _post_movement_process(self) -> None:
        if self._event_processor is not None:
            self._event_processor.process_event('update-display', None, sender=self)

    _event_handlers = EventHandlerRegistry['NavigationScene']()

    def _process_reflection(self, name: str, value: ValueType) -> None:
        assert self._event_processor is not None

        self._event_processor.process_reflection(name, value)

    @override
    def process_event(
        self,
        name: str,
        value: ValueType,
        *,
        sender: object,
        target: EventSubscriber | None,
        dont_raise: bool = False,
    ) -> bool:
        return self._event_handlers.process_event(
            self,
            name,
            value,
            sender=sender,
            target=target,
            dont_raise=dont_raise,
        )

    @_event_handlers.register('target')
    def _on_update_target(
        self,
        value: ValueType,
        _sender: object,
        _target: EventSubscriber | None,
    ) -> None:
        if _target is not self:
            return

        assert value is self

        reflection_events = (
            ('visual-size', self._reducer_coordinator.get_visual_size()),
            ('screenshot-prefix', self._screenshot_prefix),
            ('shift-speed', self._get_navigation_camera().get_shift_speed()),
            ('rotation-speed', self._get_navigation_camera().get_rotation_speed()),
        )
        for event_name, event_value in reflection_events:
            self._process_reflection(event_name, event_value)

    @_event_handlers.register('shift-speed')
    def _on_shift_speed_update(
        self,
        value: ValueType,
        _sender: object,
        _target: EventSubscriber | None,
    ) -> None:
        if _target is not self:
            return

        assert isinstance(value, float)

        self._get_navigation_camera().set_shift_speed(value)

    @_event_handlers.register('rotation-speed')
    def _on_rotation_speed_update(
        self,
        value: ValueType,
        _sender: object,
        _target: EventSubscriber | None,
    ) -> None:
        if _target is not self:
            return

        assert isinstance(value, float)

        self._get_navigation_camera().set_rotation_speed(value)

    @_event_handlers.register('visual-size')
    def _on_visual_size_update(
        self,
        value: ValueType,
        _sender: object,
        _target: EventSubscriber | None,
    ) -> None:
        if _target is not self:
            return

        assert isinstance(value, float)

        self._reducer_coordinator.set_visual_size(value)

        with self._visual_nodes as visual_nodes:
            for each in visual_nodes:
                each.set_visual_size(value)

            camera = self._get_navigation_camera()
            camera.set_visual_node_properties(visual_nodes)

    @_event_handlers.register('number-of-points')
    def _on_number_of_points_update(
        self,
        value: ValueType,
        _sender: object,
        _target: EventSubscriber | None,
    ) -> None:
        assert isinstance(value, int)

        with self._shared_configuration.with_lock() as configuration:
            configuration.max_number_of_points = value

        self._reducer_coordinator.set_number_of_points(value)

        self.push_command('S', callback=self._post_movement_process)

    @_event_handlers.register('neighborhood-radius')
    def _on_neighborhood_radius_update(
        self,
        value: ValueType,
        _sender: object,
        _target: EventSubscriber | None,
    ) -> None:
        assert isinstance(value, float)

        self._reducer_coordinator.set_neighborhood_radius(value)

        if not self.is_global_mode():
            self.push_command('S', callback=self._post_movement_process)

    @_event_handlers.register('shared-neighborhood')
    def _on_shared_neighborhood_update(
        self,
        value: ValueType,
        _sender: object,
        _target: EventSubscriber | None,
    ) -> None:
        assert isinstance(value, bool)

        self._reducer_coordinator.set_shared_neighborhood(value=value)

    @_event_handlers.register('select-data-point')
    def _on_select_data_point(
        self,
        value: ValueType,
        _sender: object,
        _target: EventSubscriber | None,
    ) -> None:
        # Note that this cannot be evoked, when `self._visual_nodes` is locked.
        assert isinstance(value, (DataPoint, NoneType))

        self.select_data_point(value)

    @_event_handlers.register('screenshot-prefix')
    def _on_screenshot_prefix_update(
        self,
        value: ValueType,
        _sender: object,
        _target: EventSubscriber | None,
    ) -> None:
        if _target is not self:
            return

        assert isinstance(value, str)

        self._screenshot_prefix = value

    @_event_handlers.register('save-screenshot')
    def _on_screenshot(
        self,
        value: ValueType,
        _sender: object,
        _target: EventSubscriber | None,
    ) -> None:
        if _target is not self:
            return

        assert value == self._screenshot_prefix

        self.save_screenshot()

    @_event_handlers.register('calculate-nearest-neighbor')
    def _on_calculate_nearest_neighbor(
        self,
        value: ValueType,
        _sender: object,
        _target: EventSubscriber | None,
    ) -> None:
        if _target is not self:
            return

        assert value is None

        selected_data_point = self.get_selected_data_point()
        if selected_data_point is not None:
            distance = self.calculate_distance_to_neighbor(selected_data_point)

            assert self._event_processor is not None
            self._event_processor.process_reflection('nearest-neighbor', distance)

    @_event_handlers.register('help')
    def _on_help(
        self,
        value: ValueType,
        _sender: object,
        _target: EventSubscriber | None,
    ) -> None:
        if _target is not self:
            return

        assert value is None

        help_text = "\n".join(self._get_help())

        assert self._event_processor is not None
        self._event_processor.process_reflection('help', help_text)

    def _get_status(self) -> list[str]:
        def _get_mode_str() -> str:
            number_of_nodes = self.get_number_of_visuals()

            if self.is_global_mode():
                mode_str = f"<Global>[{number_of_nodes}]"

                view = self.get_global_mode()
                assert view is not None
                if view != '':
                    mode_str += (
                        "  "
                        + NavigationScene.GLOBAL_VIEW_AXES[
                            NavigationScene.GLOBAL_VIEWS.index(view)
                        ]
                    )

            elif self.is_birds_eye_mode():
                mode_str = f"<Bird's Eye>[{number_of_nodes}]"

                view = self.get_birds_eye_mode(long=True)
                assert view is not None
                mode_str += " " + view

            else:
                mode_str = f"<Local>[{number_of_nodes}]"

            return mode_str

        def _get_current_position_str() -> str:
            if self.is_on_selection():
                return f"ON SELECTION: {self.get_center()}"

            return f"CURRENT: {self.get_center()}"

        mode_str = _get_mode_str()
        text = [mode_str]

        reducer_message = self._get_reducer_message()
        text.extend(reducer_message)

        if not self.is_global_mode():
            position_str = _get_current_position_str()
            text.append(position_str)

        return text

    @override
    def _update_caption(self) -> None:
        window = self.get_window()

        if self._show_display:
            text = self._get_status()

        else:
            text = []

        window.display_text(text)

    def _set_visual_information(
        self, visual_information: VisualInformation | None
    ) -> None:
        camera = self._get_navigation_camera()
        camera.set_visual_information(visual_information)

    def _set_new_transformation_for_visual_information(
        self, visual_information: VisualInformation, *, will_be_global_mode: bool
    ) -> None:
        """Update Visual Information with new Navigation Camera pose."""

        def _directions_3d_for_global_mode() -> NPArray:
            return np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

        def _directions_3d_for_local_mode(camera: NavigationCamera) -> NPArray:
            new_directions_3d = camera.get_new_directions_3d()
            return np.vstack(
                [
                    new_directions_3d.right_3d,
                    new_directions_3d.forward_3d,
                    new_directions_3d.up_3d,
                ]
            )

        camera = self._get_navigation_camera()

        if will_be_global_mode:
            new_visual_3d = _directions_3d_for_global_mode()

        else:
            new_visual_3d = _directions_3d_for_local_mode(camera)

        new_virtual = self._reducer_coordinator.calculate_direction_from_3d(
            new_visual_3d
        )
        visual_information.update_directions(
            directions=VectorPair(visual_3d=new_visual_3d, virtual=new_virtual)
        )

    def _prepare_visual_information(
        self, *, will_be_global_mode: bool, camera_rotation: bool
    ) -> None:
        """
        Update new pose and transformation for Visual Information.

        :param will_be_global_mode: Whether it will Global mode when Command is done.
        :param camera_rotation: Whether there is camera rotation.
        """
        visual_information = self._reducer_coordinator.get_visual_information()

        if visual_information is None:
            return

        self._set_new_transformation_for_visual_information(
            visual_information, will_be_global_mode=will_be_global_mode
        )

        navigation_camera = self._get_navigation_camera()
        navigation_camera.set_new_pose_for_visual_information(
            will_be_global_mode=will_be_global_mode, camera_rotation=camera_rotation
        )

    def _set_transformation_for_visual_information(self) -> None:
        visual_information = self._visual_information

        if visual_information is None:
            return

        if self.is_global_mode():
            self._prepare_visual_information(
                will_be_global_mode=True, camera_rotation=False
            )

            visual_information.apply_new_transformation()

            navigation_camera = self._get_navigation_camera()
            navigation_camera.move_visual_information_to_new_pose()

    def _update_visual_information(self) -> None:
        visual_information = self._reducer_coordinator.get_visual_information()

        if visual_information is not self._visual_information:
            self._set_visual_information(visual_information)

            self._visual_information = visual_information

        if self._visual_information is not None:
            self._visual_information.set_visibility(show=self._show_display)

        self._set_transformation_for_visual_information()

    @_event_handlers.register('update-display')
    def _on_update_display(
        self,
        value: ValueType,
        _sender: object,
        _target: EventSubscriber | None,
    ) -> None:
        assert value is None

        self._update_caption()

        self._update_visual_information()

    @_event_handlers.register('toggle-display')
    def _on_toggle_display(
        self,
        value: ValueType,
        _sender: object,
        _target: EventSubscriber | None,
    ) -> None:
        if _target is not self:
            return

        assert value is None

        self._show_display = not self._show_display

        self._update_caption()

        self._update_visual_information()


class FirstNavigationScene(NavigationScene):
    """
    The first :class:`NavigationScene`.

    :param shared_scene_configuration: Configuration parameters shared between
      Navigation Scenes.
    :param unique_scene_configuration: Configuration parameters unique to this
      Navigation Scene. Instance will be copied.
    :param data_source: Data Source that holds the data points.
    :param reducer_configuration: Configuration of the dimension reduction algorithms.
      See :class:`ReducerConfiguration`.
    :param window_configuration: Configuration for the window.
    :param event_processor: The processor that handles App Events.
    :param base: The :class:`direct.showbase.ShowBase.ShowBase` instance.
    """

    def __init__(
        self,
        shared_scene_configuration: SharedNavigationSceneConfiguration,
        unique_scene_configuration: UniqueNavigationSceneConfiguration,
        data_source: DataSource,
        reducer_configuration: ReducerConfiguration,
        window_configuration: WindowConfiguration,
        *,
        event_processor: EventProcessor | None,
        base: ShowBase,
    ) -> None:
        super().__init__(
            shared_scene_configuration,
            unique_scene_configuration,
            data_source,
            reducer_configuration,
            window_configuration,
            is_main=True,
            event_processor=event_processor,
            base=base,
        )


class AdditionalNavigationScene(NavigationScene):
    """
    Additional :class:`NavigationScene`.

    :param shared_scene_configuration: Configuration parameters shared between
      Navigation Scenes.
    :param unique_scene_configuration: Configuration parameters unique to this
      Navigation Scene. Instance will be copied.
    :param data_source: Data Source that holds the data points.
    :param reducer_configuration: Configuration of the dimension reduction algorithms.
      See :class:`ReducerConfiguration`.
    :param window_configuration: Configuration for the window.
    :param event_processor: The processor that handles App Events.
    :param base: The :class:`direct.showbase.ShowBase.ShowBase` instance.
    """

    def __init__(
        self,
        shared_scene_configuration: SharedNavigationSceneConfiguration,
        unique_scene_configuration: UniqueNavigationSceneConfiguration,
        data_source: DataSource,
        reducer_configuration: ReducerConfiguration,
        window_configuration: WindowConfiguration,
        *,
        event_processor: EventProcessor | None,
        base: ShowBase,
    ) -> None:
        super().__init__(
            shared_scene_configuration,
            unique_scene_configuration,
            data_source,
            reducer_configuration,
            window_configuration,
            is_main=False,
            event_processor=event_processor,
            base=base,
        )
