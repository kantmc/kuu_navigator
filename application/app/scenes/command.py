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

"""Defines framework for processing navigation commands."""

from abc import ABC, abstractmethod
from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass

from direct.interval.Interval import Interval

import pylog
from pyqoolloop.factory import RegistryFactory
from pyqoolloop.parallel import Guard

from ..dimensionreducer import DimensionReducerCoordinator
from ..dimensionreduction.linear import LinearOrientedDimensionReducer
from ..mathutils import NPArray
from .configuration import (
    SharedNavigationSceneConfiguration,
    UniqueNavigationSceneConfiguration,
)
from .navigationscene import NavigationScene

_logger = pylog.getLogger(__name__)


class Command(ABC):
    # Allow commands to access protected members of `Scene`.
    # flake8: noqa: SLF001
    """
    Abstract class for all commands.

    :param scene: Scene for this command.
    :param command_str: `str` representing the command. Same as for
      :meth:`NavigationScene.process_command()`.
    """

    def __init__(self, *, scene: 'NavigationScene', command_str: str) -> None:
        self._scene = scene

        with scene._shared_configuration.with_lock() as configuration:
            self._configuration = deepcopy(configuration)

        self._command_str = command_str

        self._callback: Callable[[], None] | None = None

    global_command_factory = RegistryFactory['Command']()
    """Factory for command in Global mode."""

    local_command_factory = RegistryFactory['Command']()
    """Factory for command in Local mode."""

    def _get_scene_configuration(self) -> SharedNavigationSceneConfiguration.Data:
        """Get the configuration shared between Navigation Scenes."""
        return self._configuration

    def _get_unique_scene_configuration(self) -> UniqueNavigationSceneConfiguration:
        """Get the configuration unique to the Navigation Scene."""
        return self._scene._unique_configuration

    def _get_reducer_coordinator(self) -> DimensionReducerCoordinator:
        """Get the coordinator that performs dimension reduction."""
        return self._scene._get_reducer_coordinator()

    def _get_components(self) -> NPArray | None:
        """
        Get principal components of PCA.

        :returns: Principal components in order. `None`, if PCA was not performed.

        ..note:: Only works when using PCA for dimensionality reduction.
        """
        reducer = self._scene.get_reducer()
        assert isinstance(reducer, LinearOrientedDimensionReducer)

        pca = reducer.get_reducer()

        if hasattr(pca, 'components_'):
            return pca.components_

        return None

    @staticmethod
    def _rotate(*, current: str | None, initial: str, choices: tuple[str, ...]) -> str:
        """
        Rotate selection between choices.

        :param current: The current choice. `None` or a value not included in `choices`,
          to return the initial choice.
        :param initial: The initial choice.
        :param choices: The choices to choose from.
        """
        try:
            index = choices.index(current)
            new = choices[(index + 1) % len(choices)]

        except ValueError:
            new = initial

        return new

    @classmethod
    def create(cls, command_str: str, scene: 'NavigationScene') -> 'Command':
        """
        Create a command.

        :param command_str: `str` representing the command. Same as for
          :meth:`NavigationScene.process_command()`.
        :param scene: :class:`Scene` instance.
        :returns: The created command.
        """
        if scene.is_global_mode():
            factory = cls.global_command_factory

        else:
            factory = cls.local_command_factory

        instance = factory.create(
            command_str[0], {"scene": scene, "command_str": command_str}
        )

        return instance

    def set_callback(self, callback: Callable[[], None]) -> None:
        """
        Add a function to call when the command is done.

        It will not be called, if the command was ignored.

        :param callback: The function to call when the command is done.

        ..note:: The actual call to callback later function is to be implemented in
          subclasses.
        """
        assert self._callback is None
        self._callback = callback

    @abstractmethod
    def start(self) -> None:
        """
        Start the command.

        Some commands may be asychronous, meaning that it may not have finished
        processing when control returns from this `start()` method.
        """

    def _finalize(
        self,
    ) -> None:
        """
        Finish up the command.

        To be called when the command is done, but not if the command was ignored.
        """
        if self._callback is not None:
            self._callback()


class CommandProcessor:
    """
    Processes navigation commands.

    :param scene: The :class:`NavigationScene` to process Commands for.
    :param finalizer: The function to call when a Command finishes.
    """

    def __init__(self, scene: 'NavigationScene') -> None:
        self._scene = scene

        self._commands = Guard(CommandProcessor._Movement())

    @dataclass(frozen=False)
    class _Movement:
        current: Interval | None = None
        old_movement: Interval | None = None
        next: tuple[str, Callable[[], None] | None] | None = None

    def is_processing(self) -> bool:
        """Return whether a command is being processed."""
        with self._commands as commands:
            return commands.current is not None

    def _create_command(
        self,
        command_str: str,
        callback: Callable[[], None] | None,
    ) -> Command:
        """
        Create a command.

        :returns: `Command` that was created.
        """

        def _finalize() -> None:
            if callback is not None:
                callback()

            self._start_next_command()

        command = Command.create(command_str, self._scene)

        command.set_callback(_finalize)

        return command

    def _start_next_command(self) -> None:
        current_command: Command | None = None

        with self._commands as commands:
            if commands.next is None:
                commands.current = None

            else:
                _logger.info("Process command: %s", commands.next[0])

                try:
                    current_command = self._create_command(*commands.next)

                    commands.current = current_command

                except KeyError:
                    _logger.warning("Command not implemented.")
                    commands.current = None

                commands.next = None

        if current_command is not None:
            current_command.start()

    def process_command(
        self, command_str: str, callback: Callable[[], None] | None = None
    ) -> None:
        """
        Run a command.

        The command will be ignored, if movement is in progress, or movement is
        not possible.

        :param command_str: A string representing a command. Commands are automatically
          registered in `navigationcommands.py`.  See
          :meth:`NavigationScene.process_command()`.
        :raises KeyError: Command not found.
        """
        current_command: Command | None = None

        with self._commands as commands:
            assert commands.next is None
            if commands.current is None:
                _logger.info("Process command: %s", command_str)
                current_command = self._create_command(command_str, callback)

                commands.current = current_command

            else:
                _logger.warning("command IGNORED: %s", command_str)

        if current_command is not None:
            current_command.start()

    def push_command(
        self, command_str: str, callback: Callable[[], None] | None = None
    ) -> None:
        """
        Pushes a command into the queue and executes it, if it is first in the queue.

        Only one command is allowed in the queue, and subsequent commands will be
        ignored.

        :param command_str: See :meth:`process_command()`.
        :param callback: Function to call, after movement has finished.
        """
        current_command: Command | None = None

        with self._commands as commands:
            if commands.next is None:
                if commands.current is None:
                    _logger.info("IDLE: Process command: %s", command_str)
                    current_command = self._create_command(command_str, callback)
                    _logger.info("Created: %r", current_command)

                    commands.current = current_command

                else:
                    _logger.info("BUSY: Command pushed: %s", command_str)
                    commands.next = (command_str, callback)

            else:
                _logger.info("BUSY: Command IGNORED: %s", command_str)

        if current_command is not None:
            current_command.start()
