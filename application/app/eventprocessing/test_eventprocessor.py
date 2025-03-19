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

"""Tests to test `eventprocessor.py`."""

from collections.abc import Callable

import pytest
from typing_extensions import override

from ..window import CanvasWindow
from .eventprocessor import (
    EventHandlerRegistry,
    EventProcessor,
    EventSubscriber,
    NavigationEventSubscriber,
    ValueType,
)


class _EventProcessor:
    def __init__(self) -> None:
        self.name: str | None = None
        self.value: ValueType | None = None

    _event_handlers = EventHandlerRegistry['_EventProcessor']()

    def process_event(
        self,
        name: str,
        value: ValueType,
        *,
        dont_raise: bool = False,
    ) -> bool:
        return self._event_handlers.process_event(
            self, name, value, sender=self, target=None, dont_raise=dont_raise
        )

    @_event_handlers.register('event')
    def on_event(
        self,
        value: ValueType,
        _sender: object,
        _target: EventSubscriber | None,
    ) -> None:
        self.value = value
        self.sender = _sender
        self.target = _target

    @_event_handlers.register('raise')
    def on_raise(
        self,
        value: ValueType,
        _sender: object,
        _target: EventSubscriber | None,
    ) -> None:
        raise AssertionError(f"Received {value}")


def test__EventHandlerRegistry__process_event() -> None:
    """Test `EventHandlerRegistry.process_event()` relays event to event handler."""
    instance = _EventProcessor()

    processed = instance.process_event('event', 1)

    assert processed
    assert instance.value == 1

    """Test exceptions for `EventHandlerRegistry.process_event()`."""


def test__EventHandlerRegistry__process_event__raise() -> None:
    """Test that `EventHandlerRegistry.process_event()` raises exception."""
    instance = _EventProcessor()

    with pytest.raises(KeyError):
        instance.process_event('no_event', 1)


def test__EventHandlerRegistry__process_event__dont_raise() -> None:
    """Test that `EventHandlerRegistry.process_event()` doesn't raise exception."""
    instance = _EventProcessor()

    processed = instance.process_event('no_event', 1, dont_raise=True)

    assert not processed


def test__EventHandlerRegistry__process_event__relay_exception() -> None:
    """Test with exception raised in handler."""
    instance = _EventProcessor()

    with pytest.raises(AssertionError):
        instance.process_event('raise', 1, dont_raise=True)


def test__EventProcessor__remove_subscriber() -> None:
    """Test that target is reset with `EventProcessor.remove_subscriber()`."""

    class Subscriber(NavigationEventSubscriber):
        @override
        def get_title(self) -> str:
            raise NotImplementedError

        @override
        def get_window(self) -> CanvasWindow:
            raise NotImplementedError

        @override
        def process_command(
            self, command_str: str, *, callback: Callable[[], None] | None = None
        ) -> None:
            raise NotImplementedError

        @override
        def push_command(
            self, command_str: str, callback: Callable[[], None] | None = None
        ) -> None:
            raise NotImplementedError

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
            return True

    subscriber = Subscriber()

    processor = EventProcessor()

    processor.add_subscriber(subscriber)

    assert processor.get_target() is None

    processor.set_target(subscriber)

    assert processor.get_target() is subscriber

    processor.remove_subscriber(subscriber)

    assert processor.get_target() is None
