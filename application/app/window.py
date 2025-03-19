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

"""Defines the class to handle graphics windows."""

from abc import ABC, abstractmethod
from collections.abc import Callable
import os
from pathlib import Path
from typing import Any

from typing_extensions import Protocol

from direct.gui.OnscreenText import OnscreenText
from direct.showbase.ShowBase import ShowBase
from panda3d.core import (
    GraphicsWindow,
    LPoint2i,
    LVector2i,
    NodePath,
    TextNode,
)

import pylog

from .eventprocessing.inputdevicewatcher import InputDeviceWatcher

_logger = pylog.getLogger(__name__)


class Window(Protocol):
    """Protocol for all windows."""

    def get_origin(self) -> LPoint2i:
        """Get the top left position of this Window."""

    def get_size(self) -> LVector2i:
        """Get the size of this Window."""


class CanvasWindow(Window, ABC):
    """
    Class to handle graphics window.

    :param is_main: Whether this is the main window. If not, a new
      :class:`panda3d.core.GraphicsWindow` will be created.
    :param base: The :class:`direct.showbase.ShowBase.ShowBase` instance.
    :param graphicswindow: :class:`panda3d.core.GraphicsWindow` for this Window.
    :param wx_frame: :class:`WxPanda3dFrame` for this Window.
    :param aspect2d: :class:`NodePath` for 2D GUI rendering.
    :param top_left_2d: :class:`NodePath` for the top left position in the window.
    :param input_device_watcher: Instance that monitors the mouse and keyboard.

    Named not to conflict with `GraphicsWindow` of Panda3d.

    ..note:: This class does not have enough functionality to open windows. Use
      subclasses `MainWindow` or `SubWindow`.
    """

    def __init__(
        self,
        *,
        is_main: bool,
        base: ShowBase,
        graphicswindow: GraphicsWindow,
        aspect2d: NodePath,
        top_left_2d: NodePath,
        input_device_watcher: InputDeviceWatcher,
        **kwargs: Any,
    ) -> None:
        assert graphicswindow is not None, "Window may not have been opened"

        super().__init__(**kwargs)

        self._base = base
        self._graphicswindow = graphicswindow
        self._aspect2d = aspect2d
        self._top_left_2d = top_left_2d
        self._text = set[OnscreenText]()
        self._is_main = is_main
        self._input_device_watcher = input_device_watcher

        aspect_ratio = self._base.getAspectRatio(graphicswindow)
        self._set_aspect_ratio(aspect_ratio)

        self._on_size_callbacks: list[Callable[[int, int], None]] = []

    @abstractmethod
    def get_title(self) -> str:
        """Get the title of this Window."""

    def is_main(self) -> bool:
        """Return `True`, if this is the main window."""
        return self._is_main

    def get_graphicswindow(self) -> GraphicsWindow:
        """Get the :class:`panda3d.core.GraphicsWindow` for this window."""
        return self._graphicswindow

    def get_input_device_watcher(self) -> InputDeviceWatcher:
        """Get :class:`Window.MouseWatcher` for this window."""
        return self._input_device_watcher

    def _set_aspect_ratio(self, aspect_ratio: float) -> None:
        """Set the aspect ratio for 2D GUI rendering."""
        if aspect_ratio < 1:
            self._aspect2d.setScale(1.0, aspect_ratio, aspect_ratio)
            self._top_left_2d.setPos(-1.0, 0, 1.0 / aspect_ratio)

        else:
            self._aspect2d.setScale(1.0 / aspect_ratio, 1.0, 1.0)
            self._top_left_2d.setPos(-aspect_ratio, 0, 1.0)

    def _on_size(self, width: int, height: int) -> None:
        """Event handler for resizing of window."""
        for each in self._on_size_callbacks:
            each(width, height)

    @abstractmethod
    def set_on_close_callback(self, callback: Callable[[], bool]) -> None:
        """
        Set callback for when `Frame` is to be closed.

        The callback returns `False`, if the `Frame` is not supposed to be closed.
        Sometimes, the `Frame` is closed anyway.
        """

    def add_on_size_callback(self, callback: Callable[[int, int], None]) -> None:
        """
        Add function to call, when `Window` is resized.

        The arguments for the callback function will be (width, height).
        """
        self._on_size_callbacks.append(callback)

    @abstractmethod
    def set_on_focus_callback(self, callback: Callable[[], None]) -> None:
        """Set the function to call when the Window gets focus."""

    def display_text(self, text: list[str]) -> None:
        """
        Display text on the window.

        :param text: List of `str` to show on the window.
        """

        def _add_message(position: float, msg: str) -> None:
            """Put instructions on the screen."""
            text = OnscreenText(
                text=msg,
                style=1,
                fg=(1, 1, 1, 1),
                shadow=(0, 0, 0, 1),
                parent=self._top_left_2d,
                align=TextNode.ALeft,
                pos=(0.08, -position - 0.04),
                scale=0.05,
            )
            self._text.add(text)

        for each in self._text:
            each.destroy()

        y = 0.06
        for each in text:
            _add_message(y, each)
            y += 0.06

    def save_screenshot(self, screenshot_prefix: str) -> None:
        """Save screenshot to a default location."""

        def _make_parent_directories(prefix: str) -> None:
            path = Path(prefix)
            if not prefix.endswith(os.sep):
                path = path.parent

            path.mkdir(parents=True, exist_ok=True)

        _make_parent_directories(screenshot_prefix)

        filename = self._graphicswindow.saveScreenshotDefault(screenshot_prefix)

        if filename:
            _logger.info("Saved screenshot: %s", filename)

        else:
            _logger.error("Screenshot failed.")
