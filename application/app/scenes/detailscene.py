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
Defines the :class:`Scene` that hold a detailed view of a Visual Node.

Array type: Panda3D
"""

from types import NoneType

from typing_extensions import override

import numpy as np

from direct.showbase.ShowBase import ShowBase
from panda3d.core import LPoint2i, LPoint3, LVector2i, NodePath

from ..datasource.api import DataPoint
from ..eventprocessing.eventprocessor import (
    EventHandlerRegistry,
    EventProcessor,
    EventSubscriber,
    ValueType,
)
from ..nodes.camera import Camera
from ..nodes.navigationcamera import NavigationCamera
from ..window import CanvasWindow, Window
from .scene import Scene


class DetailScene(Scene, EventSubscriber):
    """
    A class for holding a defailed view of a Data Point.

    :param window_origin: Coordinates of the origin of the window.
    :param parent: The parent window.
    :param event_processor: The processor that handles App Events.
    :param base: The :class:`direct.showbase.ShowBase.ShowBase` instance.
    """

    def __init__(
        self,
        *,
        window_origin: LPoint2i,
        parent: Window,
        event_processor: EventProcessor | None,
        base: ShowBase,
        background_color: tuple[float, float, float],
    ) -> None:
        super().__init__(
            window_origin=window_origin,
            window_size=LVector2i(1280, 640),
            title="Detail",
            is_main=False,
            parent=parent,
            screenshot_prefix=None,
            base=base,
        )

        self._visual_size = 1.0

        self._event_processor = event_processor
        if event_processor is not None:
            event_processor.add_subscriber(self)

        self._show_caption = True

        self._background_color = background_color

        self.set_background()

    @override
    def _make_camera(
        self, *, window: CanvasWindow, scene_root: NodePath, base: ShowBase
    ) -> Camera:
        return NavigationCamera(  # TODO: Temporary implementation
            scene_root,
            window=window,
            base=base,
        )

    @override
    def _on_size(self, width: int, height: int) -> None:
        aspect_ratio = width / height

        camera = self._visuals.camera
        self._set_aspect_ratio(camera, aspect_ratio)

    @override
    def set_background(self) -> None:
        self._set_background_color(*self._background_color)

    def select_data_point(self, selection: DataPoint | None) -> None:
        """
        Select :class:`DataPoint`.

        :param selection: `DataPoint` of `VisualNode` that was selected. If `None`, this
          will cause selection to be cancelled.
        """
        super().select_data_point(selection)

        with self._visual_nodes as visual_nodes:
            visual_nodes.clear()

            if selection is None:
                return

            visual_nodes.add_data_points(
                [selection],
                visual_size=self._visual_size,
                detail=DataPoint.Detail.HIGH,
                number_of_new_detailed=1,
                point_to_camera=self._visuals.camera,
            )

            for only_one in visual_nodes:
                only_one.set_position_3d(LPoint3(0, 0, 0))

            self._show_all_visual_nodes(visual_nodes, self._visual_size, '-X')

    _event_handlers = EventHandlerRegistry['DetailScene']()

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

    @_event_handlers.register('select-data-point')
    def _on_select_data_point(
        self,
        value: ValueType,
        _sender: object,
        _target: EventSubscriber | None,
    ) -> None:
        assert isinstance(value, (DataPoint, NoneType))

        self.select_data_point(value)

    def _get_selection_str(self) -> str:
        selected_data_point = self.get_selected_data_point()

        if selected_data_point is None:
            return "no selection"

        coordinates = np.round(selected_data_point.get_coordinates(), 2)
        return f"SELECTED: {coordinates}"

    @override
    def _update_caption(self) -> None:
        window = self.get_window()

        if self._show_caption:
            text = [self._get_selection_str()]

        else:
            text = []

        window.display_text(text)

    @_event_handlers.register('update-display')
    def _on_update_caption(
        self,
        value: ValueType,
        _sender: object,
        _target: EventSubscriber | None,
    ) -> None:
        assert value is None

        self._update_caption()
