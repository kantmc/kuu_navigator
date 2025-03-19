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

"""Defines a class that monitors the mouse and keyboard."""

from typing import Protocol

from panda3d.core import LPoint2


class InputDeviceWatcher(Protocol):
    """A protocol for classes that monitor the mouse and keyboard."""

    def get_mouse(self) -> LPoint2 | None:
        """
        Get x, y coordinates of the mouse cursor relative to this window.

        :return: Coordinates of the mouse cursor. `None`, if outside the window.

        .. note:: Likely will have timing problems with
            :meth:`panda3d.core.MouseWatcher.getMouse()`. (The mouse might move after
            calling `panda3d.core.MouseWatcher.has_mouse()`.)
        """

    def is_alt_down(self) -> bool:
        """Return whether the Alt key is pressed."""

    def is_shift_down(self) -> bool:
        """Return whether the Shift key is pressed."""

    def is_control_down(self) -> bool:
        """Return whether the Control key is pressed."""
