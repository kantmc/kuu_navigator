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

"""Defines a window with UI components for Navigation Scenes."""

from types import NoneType
from typing import Any

from typing_extensions import override

from direct.showbase.ShowBase import ShowBase
from panda3d.core import LPoint2i, LVector2i

from .datasource.api import DataPoint
from .eventprocessing.eventprocessor import (
    EventHandlerRegistry,
    EventProcessor,
    EventSubscriber,
    NavigationEventSubscriber,
    ValueType,
)
from .guiplatform import GUIWindow, GUIWindowWrapper, ToolBox
from .window import Window


class ToolBoxWindow(Window, GUIWindowWrapper, EventSubscriber):
    """
    A window with UI components.

    :param max_navigation_scenes: Maximum number of Navigation Scenes allowed.
    :param event_processor: The Event Processor instance.
    :param base: The `ShowBase` instance.
    """

    def __init__(
        self,
        *,
        max_navigation_scenes: int,
        event_processor: EventProcessor,
        base: ShowBase,
    ) -> None:
        wx_tool_box = ToolBox(
            max_navigation_scenes=max_navigation_scenes,
            event_processor=event_processor,
            base=base,
        )

        self._wx_tool_box = wx_tool_box

        event_processor.add_subscriber(self)
        self._event_processor = event_processor

    @override
    def on_add_event_subscriber(self, subscriber: EventSubscriber) -> None:
        if isinstance(subscriber, NavigationEventSubscriber):
            title = subscriber.get_title()
            self._wx_tool_box.register_target_window(title)

    @override
    def on_remove_event_subscriber(self, subscriber: EventSubscriber) -> None:
        if isinstance(subscriber, NavigationEventSubscriber):
            title = subscriber.get_title()
            self._wx_tool_box.remove_target_window(title)

    _event_handlers = EventHandlerRegistry['ToolBoxWindow']()

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

    @_event_handlers.register('target')
    def _on_set_target(
        self,
        value: ValueType,
        _sender: object,
        _target: EventSubscriber | None,
    ) -> None:
        assert isinstance(value, NavigationEventSubscriber)

        title = value.get_title()
        self._wx_tool_box.set_target_window(title)

    @_event_handlers.register('select-data-point')
    def _on_select_data_point(
        self,
        value: ValueType,
        _sender: object,
        _target: EventSubscriber | None,
    ) -> None:
        assert isinstance(value, (DataPoint, NoneType))

        self._wx_tool_box.on_select_data_point(value)

    @override
    def on_process_reflection(self, name: str, value: Any) -> None:
        self._wx_tool_box.process_reflection(name, value)

    @override
    def get_gui_window(self) -> GUIWindow:
        return self._wx_tool_box

    @override
    def get_origin(self) -> LPoint2i:
        position = self._wx_tool_box.GetPosition()
        return LPoint2i(position.x, position.y)

    @override
    def get_size(self) -> LVector2i:
        size = self._wx_tool_box.GetSize()
        return LVector2i(
            size.GetWidth(),  # type: ignore[attr-defined] # mypy 1.8.0
            size.GetHeight(),  # type: ignore[attr-defined] # mypy 1.8.0
        )
