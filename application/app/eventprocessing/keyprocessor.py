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

"""Defines class for processing key events."""

from abc import ABC, abstractmethod
from collections.abc import Callable
from enum import Enum, auto
from typing import NamedTuple, TypeAlias

from typing_extensions import Protocol

from direct.showbase.ShowBase import ShowBase
from direct.task import Task

import pylog

from .eventprocessor import EventProcessor
from .inputdevicewatcher import InputDeviceWatcher

_logger = pylog.getLogger(__name__)
_logger.setLevel(pylog.INFO)


class KeyProcessor(ABC):
    """
    Processes key events.

    :param commands:
      Tuple of tuple with elements of the following:

        - key: (str) String that corresponds to a key. See
          https://docs.panda3d.org/1.10/python/programming/hardware-support/keyboard-support
        - commands: (_KeyMapCommandTuple | str)
          Commands that correspond to key event with the following modifiers:

            - No modifier
            - Shift
            - Alt
            - Alt+Shift
            - Control

          If only a `str` is specified, only key presses without modifiers are
          accepted. See :meth:`NavigationScene.process_command()` for a list of
          commands.
    :param positional_key: `True`, if keys are defined by the position on the
        keyboard, not necessarily the character written on the key top.
    :param command_function: Function to use when sending commands.
    :param continuous_mode: Whether/how continuous processing of keys is
        implemented.
    :param callback: Function to call after command is finished.
    :param event_processor: The `EventProcessor` instance.
    :param base: The `ShowBase` instance.
    """

    class MoveMethod(Protocol):
        """Protocol for function for processing commands."""

        def __call__(
            self, command_str: str, *, callback: Callable[[], None] | None
        ) -> None:
            """
            Process command.

            :param command_str: The command.
            :param callback: The function to call after the command is processed.
            """

    class ContinuousMode(Enum):
        """Specifies whether/how continuous processing of keys is implemented."""

        NO = auto()
        YES = auto()
        TIMER = auto()

    def __init__(
        self,
        *,
        commands: tuple[tuple[str, 'KeyProcessor._KeyMapCommandTuple | str'], ...],
        positional_key: bool,
        command_function: MoveMethod,
        continuous_mode: 'KeyProcessor.ContinuousMode',
        callback: Callable[[], None],
        event_processor: 'EventProcessor',
        base: ShowBase,
    ) -> None:
        def _make_canonical(
            commands: tuple[tuple[str, 'KeyProcessor._KeyMapCommandTuple | str'], ...],
        ) -> tuple[tuple[str, 'KeyProcessor._KeyMapCommandTuple'], ...]:
            return tuple(
                (key, (each, 'N', 'N', 'N', 'N') if isinstance(each, str) else each)
                for key, each in commands
            )

        self._commands = _make_canonical(commands)
        self._command_function = command_function
        self._continous_mode = continuous_mode
        self._callback = callback

        self._event_processor = event_processor

        self._base = base

        self._keys_pressed = set[str]()

        self._bind_commands(self._commands, positional_key=positional_key, base=base)

    _KEY_REPEAT_SECS = 0.1  # Valid only when animation is disabled.

    _KeyMapCommandTuple: TypeAlias = tuple[str, str, str, str, str]

    class _KeyMapCommandStrs(NamedTuple):
        raw: str
        shift: str
        alt: str
        alt_shift: str
        control: str

        def get_command(self, watcher: InputDeviceWatcher) -> str:
            """Get the command according to the modifier keys."""
            if watcher.is_alt_down():
                if watcher.is_shift_down():
                    command = self.alt_shift

                else:
                    command = self.alt

            elif watcher.is_shift_down():
                command = self.shift

            elif watcher.is_control_down():
                # control-arrow cannot be detected on Mac
                command = self.control

            else:
                # Actually needs more checks
                command = self.raw

            return command

    @abstractmethod
    def _bind_commands(
        self,
        commands: tuple[tuple[str, _KeyMapCommandTuple], ...],
        *,
        positional_key: bool,
        base: ShowBase,
    ) -> None:
        """
        Set up so that certain keys call `_process_key_down()` and `_process_key_up()`.

        :param commands: Combination of keys and the commands that they invoke. Each
          element in the tuple consists of the following:

            - (str) key without modifiers. Each key is represented by 1 character in
              lower case except the following: 'escape'
            - (tuple[str, str, str, str]) A tuple of commands for each modifier
              combination. Each command is for the following modifiers in the same
              order:

                - none
                - shift
                - alt
                - alt+shift
                - control

        :param positional_key: `True`, if keys are defined by the position on the
            keyboard, not necessarily the character written on the key top.
        :param base: The `ShowBase` instance.
        """

    def _check_key(self) -> None:
        for key, command_strs in self._commands:
            if key in self._keys_pressed:
                self._process_key(
                    self._KeyMapCommandStrs(*command_strs),
                    command_function=self._command_function,
                )

    def _process_key(
        self,
        command_strs: _KeyMapCommandStrs,
        *,
        command_function: MoveMethod,
    ) -> None:
        def _check_key_continuous() -> None:
            self._callback()

            self._check_key()

        def _check_key_continuous_task(_task: Task) -> None:
            _check_key_continuous()

        watcher = self._get_input_device_watcher()
        if watcher is None:
            self._keys_pressed.clear()
            return

        command = command_strs.get_command(watcher)

        if command == '':
            _logger.info("Null Command")
            return

        match self._continous_mode:
            case KeyProcessor.ContinuousMode.NO:
                callback = self._callback

            case KeyProcessor.ContinuousMode.YES:
                callback = _check_key_continuous

            case KeyProcessor.ContinuousMode.TIMER:
                callback = self._callback
                self._base.taskMgr.doMethodLater(
                    self._KEY_REPEAT_SECS,
                    _check_key_continuous_task,
                    "_check_key_continuous",
                )

        try:
            command_function(command, callback=callback)

        except KeyError:
            _logger.warning("Command not implemented.")

    def _get_input_device_watcher(self) -> InputDeviceWatcher | None:
        """
        Get an `InputDeviceWatcher` instance.

        :returns: The `InputDeviceWatcher` instance. `None`, if not available.
        """
        target = self._event_processor.get_target()
        if target is None:
            return None

        window = target.get_window()
        if window is None:
            return None

        watcher = window.get_input_device_watcher()
        return watcher

    def _process_key_down(
        self,
        command_strs: _KeyMapCommandStrs,
        key: str,
    ) -> None:
        """
        Process key down event.

        :param command_strs:
        """
        self._keys_pressed.clear()

        self._keys_pressed.add(key)
        self._process_key(
            command_strs,
            command_function=self._command_function,
        )

    def _process_key_up(self, key: str) -> None:
        self._keys_pressed.discard(key)
