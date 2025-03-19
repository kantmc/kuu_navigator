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

"""Defines the application class."""

from types import NoneType
from typing import Any, Literal, cast

from typing_extensions import override

from direct.showbase.ShowBase import ShowBase
from panda3d.core import LVector2i

from pyqoolloop.parallel import Guard

from .datasource.api import DataPoint, DataSource
from .eventprocessing.eventprocessor import (
    EventHandlerRegistry,
    EventProcessor,
    EventSubscriber,
    ValueType,
)
from .guiplatform import KeyProcessor, MouseProcessor, PlatformApp
from .scenes.configuration import (
    NavigationSceneConfiguration,
    ReducerConfiguration,
    SharedNavigationSceneConfiguration,
    UniqueNavigationSceneConfiguration,
    WindowConfiguration,
)
from .scenes.detailscene import DetailScene
from .scenes.navigationscene import (
    AdditionalNavigationScene,
    FirstNavigationScene,
    NavigationScene,
)
from .scenes.scene import Scene
from .toolboxwindow import ToolBoxWindow
from .window import Window


class App(EventSubscriber):
    """
    Application to navigate through data points placed in 3D space.

    :param scene_configuration: Configuration parameters for Navigation Scenes.
    :param data_source: Data source that provides the data points.
    :param reducer_configuration: Configuration parameters for dimension reduction.
    :param inspection_tool: Whether to use Inspection Tool for wxPython.
    """

    def __init__(
        self,
        scene_configuration: NavigationSceneConfiguration,
        data_source: DataSource,
        reducer_configuration: ReducerConfiguration,
        *,
        inspection_tool: bool,
    ) -> None:
        # Panda3d says:
        #  For some reason, wx needs to be initialized before the graphics window
        # https://github.com/panda3d/panda3d/blob/4f9092d568bc499e6f26241ee68c5e1a10eb470c/direct/src/showbase/ShowBase.py#L399C9-L399C81
        self._base = ShowBase(windowType='none')

        self._platform_app = PlatformApp(base=self._base, inspect=inspection_tool)

        self._event_processor = EventProcessor()
        self._event_processor.add_subscriber(self)

        self._tool_box_window = ToolBoxWindow(
            max_navigation_scenes=self._MAX_NAVIGATING_SCENES,
            event_processor=self._event_processor,
            base=self._base,
        )

        self._cumulative_navigating_scene_count = 0

        shared_scene_configuration, unique_scene_configuration = (
            scene_configuration.split()
        )
        self._common_scene_parameters = {
            'shared_scene_configuration': shared_scene_configuration,
            'unique_scene_configuration': unique_scene_configuration,
            'data_source': data_source,
            'reducer_configuration': reducer_configuration,
        }

        scene_arguments = self._get_scene_arguments(current_scene=None)
        navigating_scene = FirstNavigationScene(**scene_arguments)

        navigating_scene.set_on_close_callback(self._on_close_navigation_scene)

        self._navigating_scenes: list[NavigationScene] = [navigating_scene]

        self._detailed_scene: DetailScene | None = None

        self._selected_data_point: DataPoint | None = None

        self._oobe_mode = False

        self._base.disableMouse()

        with shared_scene_configuration.with_lock() as configuration:
            has_animation = configuration.has_animation()

        self._device_processors = self._setup_input_devices(has_animation=has_animation)

        self._update_display()

        self._event_processor.process_event('SETUP', None, sender=self)

    def _get_scene_arguments(
        self, *, current_scene: NavigationScene | None
    ) -> dict[str, Any]:
        self._cumulative_navigating_scene_count += 1

        if current_scene is None:
            title = (
                App._NAVIGATING_SCENE_TITLE_FORMAT
                % self._cumulative_navigating_scene_count
            )

            window_configuration = WindowConfiguration(
                title=title,
                parent=self._tool_box_window,
                window_origin=self._calculate_new_window_position([]),
            )
            scene_arguments = dict(
                window_configuration=window_configuration,
                event_processor=self._event_processor,
                base=self._base,
                **self._common_scene_parameters,
            )

        else:
            title = (
                App._NAVIGATING_SCENE_TITLE_FORMAT
                % self._cumulative_navigating_scene_count
            )

            current_window = current_scene.get_window()
            window_size = current_window.get_size()

            window_configuration = WindowConfiguration(
                title=title,
                parent=self._tool_box_window,
                window_origin=self._calculate_new_window_position([current_scene]),
                window_size=window_size,
            )
            scene_arguments = dict(
                window_configuration=window_configuration,
                event_processor=self._event_processor,
                base=self._base,
                **self._common_scene_parameters,
            )

        return scene_arguments

    _NAVIGATING_SCENE_TITLE_FORMAT = "Navigation #%d"

    _MAX_NAVIGATING_SCENES = 2

    def _calculate_new_window_position(
        self, scenes: list[NavigationScene]
    ) -> LVector2i:
        # relative to parent window (ToolBox)
        def _get_right(window: Window) -> int:
            return cast(int, window.get_origin().x + window.get_size().x)

        def _get_max_right(scenes: list[NavigationScene]) -> int:
            max_right = _get_right(self._tool_box_window)

            for each in scenes:
                max_right = max(max_right, _get_right(each.get_window()))

            return max_right

        display_width = self._platform_app.get_display_size().x

        new_position = _get_max_right(scenes)

        overflow = (
            new_position + NavigationScene.NAVIGATION_SCENE_DEFAULT_WINDOW_SIZE.x
        ) - display_width
        if overflow > 0:
            new_position -= overflow

        return LVector2i(new_position, self._tool_box_window.get_origin().y + 40)

    def _update_display(self) -> None:
        self._event_processor.process_event('update-display', None, sender=self)

    _navigation_key_mapping = (
        # key, (raw command, shift, alt, alt-shift, control)
        ('8', ('mF', 'mU', 'dF', 'dU', 'rU')),
        ('i', ('mF', 'mU', 'dF', 'dU', 'rU')),
        ('4', ('rL', 'mL', '', 'dL', 'rA')),
        ('j', ('rL', 'mL', '', 'dL', 'rA')),
        ('6', ('rR', 'mR', '', 'dR', 'rC')),
        ('k', ('rR', 'mR', '', 'dR', 'rC')),
        ('2', ('mB', 'mD', 'dB', 'dD', 'rD')),
        ('m', ('mB', 'mD', 'dB', 'dD', 'rD')),
    )

    _one_time_key_mapping = (
        # key, raw command ...
        ('a', 'a'),
        ('b', 'b'),
        ('c', ('cS', '', 'fP', '', '')),
        ('r', 'R'),
        ('z', 'zI'),
        ('escape', 'x'),
    )

    def _on_close_navigation_scene(self, scene: Scene) -> bool:
        assert isinstance(scene, NavigationScene)

        self._navigating_scenes.remove(scene)

        return True

    def _open_detail_scene(self) -> DetailScene | None:
        window_origin = self._calculate_new_window_position(self._navigating_scenes)

        # Shouldn't really take from `UniqueNavigationSceneConfiguration`.
        scene_configuration = cast(
            UniqueNavigationSceneConfiguration,
            self._common_scene_parameters['unique_scene_configuration'],
        )
        background_color = scene_configuration.background_color

        window = DetailScene(
            window_origin=window_origin,
            parent=self._tool_box_window,
            event_processor=self._event_processor,
            base=self._base,
            background_color=background_color,
        )
        window.set_on_close_callback(self._on_close_detail_scene)

        return window

    def _on_close_detail_scene(self, _scene: Scene) -> bool:
        self._detailed_scene = None

        return True

    def _select_data_point(self, selection: DataPoint | None) -> None:
        if selection is not None:
            self._selected_data_point = selection

            if self._detailed_scene is None:
                self._detailed_scene = self._open_detail_scene()

        else:
            self._selected_data_point = None

        self._event_processor.process_event(
            'select-data-point', self._selected_data_point, sender=self
        )

        self._update_display()

    def _on_click(self) -> None:
        selection: Literal[False] | DataPoint | None = False

        for each_scene in self._navigating_scenes:
            window = each_scene.get_window()
            mouse_watcher = window.get_input_device_watcher()
            mouse_position = mouse_watcher.get_mouse()
            if mouse_position is not None:
                selection = each_scene.get_data_point_at(mouse_position)

        if selection is not False:
            self._select_data_point(selection)

    def _post_movement_process(self) -> None:
        self._update_display()

    def _setup_input_devices(
        self,
        *,
        has_animation: bool,
    ) -> tuple[KeyProcessor, KeyProcessor, MouseProcessor]:
        if has_animation:
            command_function = self._event_processor.push_command

            continuous_mode = KeyProcessor.ContinuousMode.YES

        else:
            command_function = self._event_processor.process_command

            continuous_mode = KeyProcessor.ContinuousMode.TIMER

        navigation_key_processor = KeyProcessor(
            commands=self._navigation_key_mapping,
            positional_key=True,
            command_function=command_function,
            continuous_mode=continuous_mode,
            callback=self._post_movement_process,
            event_processor=self._event_processor,
            base=self._base,
        )

        one_time_key_processor = KeyProcessor(
            commands=self._one_time_key_mapping,
            positional_key=False,
            command_function=self._event_processor.process_command,
            continuous_mode=KeyProcessor.ContinuousMode.NO,
            callback=self._post_movement_process,
            event_processor=self._event_processor,
            base=self._base,
        )

        mouse_processor = MouseProcessor(self._on_click, self._base)

        return (navigation_key_processor, one_time_key_processor, mouse_processor)

    def run(self) -> None:
        """Run the app."""
        self._platform_app.run()

    _event_handlers = EventHandlerRegistry['App']()

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
            self, name, value, sender=sender, target=target, dont_raise=dont_raise
        )

    def _with_shared_scene_configuration(
        self,
    ) -> Guard[SharedNavigationSceneConfiguration.Data]:
        shared_configuration = cast(
            SharedNavigationSceneConfiguration,
            self._common_scene_parameters['shared_scene_configuration'],
        )
        return shared_configuration.with_lock()

    @_event_handlers.register('SETUP')
    def _on_setup(
        self,
        value: ValueType,
        _sender: object,
        _target: EventSubscriber | None,
    ) -> None:
        assert value is None

        with self._with_shared_scene_configuration() as configuration:
            reflection_events = (
                ('animation-interval', configuration.animation.animation_interval_secs),
                ('number-of-points', configuration.max_number_of_points),
                ('neighborhood-radius', configuration.neighborhood_radius),
                ('shared-neighborhood', configuration.shared_neighborhood),
            )

        for event_name, event_value in reflection_events:
            self._event_processor.process_reflection(event_name, event_value)

    @_event_handlers.register('animation-interval')
    def _on_animation_interval_update(
        self,
        value: ValueType,
        _sender: object,
        _target: EventSubscriber | None,
    ) -> None:
        assert isinstance(value, float)
        assert value > 0

        with self._with_shared_scene_configuration() as configuration:
            configuration.animation.animation_interval_secs = value

    @_event_handlers.register('number-of-points')
    def _on_number_of_points_update(
        self,
        value: ValueType,
        _sender: object,
        _target: EventSubscriber | None,
    ) -> None:
        assert isinstance(value, int)

        with self._with_shared_scene_configuration() as configuration:
            configuration.max_number_of_points = value

    @_event_handlers.register('neighborhood-radius')
    def _on_neighborhood_radius_update(
        self,
        value: ValueType,
        _sender: object,
        _target: EventSubscriber | None,
    ) -> None:
        assert isinstance(value, float)

        with self._with_shared_scene_configuration() as configuration:
            configuration.neighborhood_radius = value

    @_event_handlers.register('select-data-point')
    def _on_select_data_point(
        self,
        value: ValueType,
        _sender: object,
        _target: EventSubscriber | None,
    ) -> None:
        assert isinstance(value, (DataPoint, NoneType))

        self._selected_data_point = value

    def _new_navigation_scene(self, *, current_scene: NavigationScene | None) -> None:
        if len(self._navigating_scenes) < self._MAX_NAVIGATING_SCENES:
            scene_arguments = self._get_scene_arguments(current_scene=current_scene)
            new_scene = AdditionalNavigationScene(**scene_arguments)
            new_scene.set_on_close_callback(self._on_close_navigation_scene)
            new_scene.select_data_point(self._selected_data_point)

            self._navigating_scenes.append(new_scene)

            self._update_display()

    @_event_handlers.register('new-navigation-scene')
    def _on_new_navigation_scene(
        self,
        value: ValueType,
        _sender: object,
        _target: EventSubscriber | None,
    ) -> None:
        assert isinstance(value, (NavigationScene | NoneType))

        self._new_navigation_scene(current_scene=value)
