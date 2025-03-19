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

"""Functionality for processing of events."""

from collections.abc import Callable
from typing import (
    Generic,
    Protocol,
    TypeAlias,
    TypeVar,
    runtime_checkable,
)

from mypy_extensions import Arg, NamedArg

import pylog
from pyqoolloop.decorators import log_calls
from pyqoolloop.factory import FunctionRegistryFactory

from ..datasource.api import DataPoint
from ..window import CanvasWindow

_logger = pylog.getLogger(__name__)
_logger.setLevel(pylog.INFO)


class EventSubscriber(Protocol):
    """Listens to events."""

    def on_add_event_subscriber(
        self,
        subscriber: 'EventSubscriber',  # noqa: ARG002
    ) -> None:
        """Event handler for when a subscriber is added to `EventProcessor`."""
        return

    def on_remove_event_subscriber(
        self,
        subscriber: 'EventSubscriber',  # noqa: ARG002
    ) -> None:
        """Event handler for when subscriber is removed from `EventProcessor`."""
        return

    def on_process_reflection(self, name: str, value: 'ValueType') -> None:  # noqa: ARG002
        """Event handler for reflection of values."""
        return

    def process_event(
        self,
        name: str,
        value: 'ValueType',  # noqa: ARG002
        *,
        sender: object,  # noqa: ARG002
        target: 'EventSubscriber | None',  # noqa: ARG002
        dont_raise: bool = False,
    ) -> bool:
        """
        Process event, such as change in value.

        :param name: Name of event. `SETUP` is used once when the program has started,
          so that subscribers can send back Reflection Events.
        :param value: Parameter for event.
        :param sender: Originator of this event.
        :param target: Target for this event.
        :raises KeyError: `name` was not found.
        """
        if dont_raise:
            return False

        raise KeyError(f"name ({name}) not found")


@runtime_checkable
class NavigationEventSubscriber(EventSubscriber, Protocol):
    """Listens to App Events and Navigation Commands."""

    def get_title(self) -> str:
        """Get the title."""

    def get_window(self) -> CanvasWindow:
        """Get :class:`Window` associated with this subscriber."""

    def process_command(
        self, command_str: str, *, callback: Callable[[], None] | None = None
    ) -> None:
        """
        Execute a navigation command.

        See `NavigationScene.process_command()` for details.
        """

    def push_command(
        self, command_str: str, *, callback: Callable[[], None] | None = None
    ) -> None:
        """
        Pushes a command into the queue.

        See `NavigationScene.push_command()` for details.
        """


ValueType: TypeAlias = str | float | NavigationEventSubscriber | DataPoint | None
"""Type for values in update/reflection events."""

T_contra = TypeVar('T_contra', contravariant=True)

# mypy 1.10.1 won't be able to handle this yet.

# class EventHandler(Protocol[T_contra]):
#     """Protocol for functions that handle events."""

#     def __call__(
#         _,
#         self: T_contra,
#         value: ValueType,
#         *,
#         _name: str | None = None,
#     ) -> None:
#         """
#         Process an event.

#         :param value: The new value.
#         :param name: Name of value. Can be omitted.
#         """

T = TypeVar('T')


EventHandler: TypeAlias = Callable[
    # FUTURE: Should use `type` instead of `TypeVar` with Python 3.12+, when wxPython
    #  supports it. https://github.com/wxWidgets/Phoenix/issues/2455
    # Pylance v2024.6.1 can't handle `NamedArg``
    [
        T_contra,
        Arg(ValueType, 'value'),  # type: ignore[reportInvalidTypeForm, unused-ignore]
        NamedArg(  # type: ignore[reportInvalidTypeForm, unused-ignore]
            object, '_sender'
        ),
        NamedArg(  # type: ignore[reportInvalidTypeForm, unused-ignore]
            EventSubscriber | None, '_target'
        ),
    ],
    None,
]


class EventHandlerRegistry(Generic[T], FunctionRegistryFactory[EventHandler[T]]):
    """A registry for event handlers."""

    def process_event(
        self,
        receiver: T,
        name: str,
        value: ValueType,
        *,
        sender: object,
        target: EventSubscriber | None,
        dont_raise: bool = False,
    ) -> bool:
        """
        Relay an event to a registered event handler.

        :param receiver: The instance to send the event to.
        :param name: Name of event.
        :param value: Parameter for event.
        :param sender: Originator of this event.
        :param target: Target for this event.
        :param dont_raise: `True` to return `False`, when `name` was not found. Can
          raise `KeyError` in other circumstances.

        :return: `False`, if `name` was not found and `dont_raise is True`.

        :raises KeyError: `name` was not found.
        """
        try:
            function = self.create(name)

        except KeyError:
            if dont_raise:
                return False

            raise

        function(receiver, value=value, _sender=sender, _target=target)

        return True


class EventProcessor:
    """Relays App Events and Commands to the appropriate window."""

    def __init__(self) -> None:
        self._subscribers: list[EventSubscriber] = []

        self._target: NavigationEventSubscriber | None = None

    def add_subscriber(self, subscriber: EventSubscriber) -> None:
        """Add subscriber to listen to events."""
        self._subscribers.append(subscriber)

        for each in self._subscribers:
            each.on_add_event_subscriber(subscriber)

    def remove_subscriber(self, subscriber: EventSubscriber) -> None:
        """Remove subscriber."""
        for each in self._subscribers:
            if each is self._target:
                self._target = None

            each.on_remove_event_subscriber(subscriber)

        self._subscribers.remove(subscriber)

    def set_target(self, target: NavigationEventSubscriber) -> None:
        """Set the target Navigation Scene."""
        self._target = target

        self.process_event('target', self._target, sender=self)

    def get_target(self) -> NavigationEventSubscriber | None:
        """Get the target set with :meth:`set_target()`."""
        return self._target

    def process_reflection(
        self,
        name: str,
        value: ValueType,
    ) -> None:
        """
        Process event to reflect values.

        :param name: Name of value.
        :param value: The value.
        """
        for each in self._subscribers:
            each.on_process_reflection(name, value)

    @log_calls(_logger)
    def process_command(
        self, command_str: str, *, callback: Callable[[], None] | None = None
    ) -> None:
        """
        Execute a navigation command in the target window.

        See `NavigationScene.process_command()` for details.
        """
        assert self._target is not None

        self._target.process_command(command_str, callback=callback)

    @log_calls(_logger)
    def push_command(
        self, command_str: str, *, callback: Callable[[], None] | None = None
    ) -> None:
        """
        Pushes a command into the queue of the target window, and executes it.

        See `NavigationScene.push_command()` for details.
        """
        assert self._target is not None

        self._target.push_command(command_str, callback=callback)

    @log_calls(_logger)
    def process_event(self, name: str, value: ValueType, *, sender: object) -> None:
        """
        Process event, such as change in value.

        :param name: Name of event. Call with 'name=SETUP, value=None', after setup is
          completed.
        :param value: Parameter for event.
        :param sender: Originator of this event.
        """
        assert sender is not None

        processed = False

        for each in self._subscribers:
            processed |= each.process_event(
                name,
                value,
                sender=sender,
                target=self._target,
                dont_raise=True,
            )

        assert processed, f"'{name}' wasn't processed."
