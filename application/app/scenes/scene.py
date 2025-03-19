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
Defines a class that keeps Visual Nodes.

Array type: Panda3D
"""

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass

from direct.showbase.ShowBase import ShowBase
from panda3d.core import (
    DisplayRegion,
    LColor,
    LPoint2i,
    LVector2i,
    NodePath,
)

from pyqoolloop.parallel import Guard

from ..datasource.api import DataPoint
from ..guiplatform import AdditionalCanvasWindow, FirstCanvasWindow
from ..nodes.camera import Camera
from ..nodes.visualnode import VisualNodeCollection
from ..window import CanvasWindow, Window


class Scene(ABC):
    """
    A superclass for keeping Visual Nodes.

    :param window_origin: Coordinates of the origin of the window.
    :param window_size: Size of the window.
    :param title: Title for the window.
    :param is_main: Whether this is the first `Scene` to be created.
    :param parent: The parent :class:`Window`.
    :param screenshot_prefix: Prefix of filenames for screenshots. Characters after the
      last '/' (or `os.sep`) will be used as the prefix of the name of the file.
      Directories will be created as necessary. `None`, if screenshots are not
      necessary.
    :param base: The :class:`direct.showbase.ShowBase.ShowBase` instance.
    """

    def __init__(
        self,
        *,
        window_origin: LPoint2i,
        window_size: LVector2i,
        title: str,
        is_main: bool,
        parent: Window | None,
        screenshot_prefix: str | None,
        base: ShowBase,
    ) -> None:
        self._visuals = self._setup_visuals(
            origin=window_origin,
            size=window_size,
            title=title,
            is_main=is_main,
            parent=parent,
            base=base,
        )

        self._selected_data_point: DataPoint | None = None
        """
        Data Point that is selected globally.

        .. note:: This may not be included in `self._visual_nodes`, in which case
          `self._visual_nodes.get_selected()` may be `None`.
        """

        self._visual_nodes = Guard(
            VisualNodeCollection(self._visuals.scene_root_nodepath, base.loader),
            reentrant=True,
        )

        self._screenshot_prefix = screenshot_prefix

        self._on_close_callback: Callable[[Scene], bool] | None = None

    @dataclass(frozen=True)
    class _VisualsObjects:
        scene_root_nodepath: NodePath
        """Root node of the scene."""

        camera: Camera
        """The :class:`.nodes.camera.Camera` instance."""

        window: CanvasWindow
        """The :class:`Window` where this scene is shown."""

        displayregion: DisplayRegion | None
        """The `DisplayRegion` for this Scene."""

    @staticmethod
    def _set_aspect_ratio(camera: Camera, aspect_ratio: float) -> None:
        """Set aspect ratio for specified `Camera`."""
        lens = camera.get_lens()
        lens.setAspectRatio(aspect_ratio)

    @abstractmethod
    def _make_camera(
        self, *, window: CanvasWindow, scene_root: NodePath, base: ShowBase
    ) -> Camera:
        """
        Instantiate :class:`Camera`.

        :param window: Window where the contents of the camera will be shown.
        :param scene_root: Root node of the scene.
        :param base: The `ShowBase` instance.

        .. note:: Override in subclass.
        """

    def _setup_visuals(
        self,
        *,
        origin: LPoint2i,
        size: LVector2i,
        title: str,
        is_main: bool,
        parent: Window | None,
        base: ShowBase,
    ) -> _VisualsObjects:
        if is_main:
            window: CanvasWindow = FirstCanvasWindow(
                base=base, parent=parent, origin=origin, size=size, title=title
            )

        else:
            assert parent is not None

            window = AdditionalCanvasWindow(
                origin=origin,
                title=title,
                size=size,
                base=base,
                parent=parent,
            )

        window.set_on_close_callback(self._on_close)
        window.add_on_size_callback(self._on_size)

        scene_root = NodePath(title)

        camera = self._make_camera(window=window, scene_root=scene_root, base=base)

        displayregion = camera.get_displayregion()

        return self._VisualsObjects(
            scene_root_nodepath=scene_root,
            camera=camera,
            window=window,
            displayregion=displayregion,
        )

    def get_window(self) -> CanvasWindow:
        """Get :class:`CanvasWindow` associated with this Scene."""
        return self._visuals.window

    def set_on_close_callback(self, callback: Callable[['Scene'], bool]) -> None:
        """
        Set callback for when `Frame` is to be closed.

        The argument for the callback is as follows:
        - (Scene) The Scene that is being closed.
        The callback returns `False`, if the `Frame` is not supposed to be closed.
        Sometimes, the `Frame` is closed anyway.
        """
        self._on_close_callback = callback

    def _on_close(self) -> bool:
        """
        Event handler for closing the Canvas Window.

        :returns: `True`, if the window can be closed.
        """
        if self._on_close_callback is not None:
            return self._on_close_callback(self)

        return True

    @abstractmethod
    def _on_size(self, width: int, height: int) -> None:
        """Event handler for resizing of the window."""

    def _set_background_color(self, red: float, green: float, blue: float) -> None:
        """
        Set the color of the background of the scene.

        :param red: Red component [0.0, 1.0]
        :param green: Green component [0.0, 1.0]
        :param blue: Blue component [0.0, 1.0]
        """
        display_region = self._visuals.displayregion
        if display_region is not None:
            display_region.setClearColorActive(True)
            display_region.setClearColor(LColor(red, green, blue, 0.0))

    @abstractmethod
    def set_background(self) -> None:
        """Set the background color, if necessary."""

    def _show_all_visual_nodes(
        self,
        visual_nodes: VisualNodeCollection,
        visual_size: float,
        axis_direction: str,
    ) -> None:
        """
        Show all Visual Nodes.

        :param visual_nodes: The Visual Nodes to show.
        :param visual_size: Visual size of each :class:`VisualNode`.
        :param axis_direction: The direction to view the Visual Nodes.
        """
        self._visuals.camera.show_all(
            visual_nodes, axis_direction, margin=visual_size / 2
        )

    def select_data_point(
        self,
        selection: DataPoint | None,
    ) -> None:
        """
        Select Data Point.

        :param selection: `DataPoint` of `VisualNode` to be selected. A `VisualNode`
          with the same coordinates will be selected, if it exists. If `None`, this will
          cause selection to be cancelled.
        """
        self._selected_data_point = selection

    def get_selected_data_point(self) -> DataPoint | None:
        """Get the globally selected Data Point."""
        return self._selected_data_point

    @abstractmethod
    def _update_caption(self) -> None:
        """Display text appropriate for the content displayed."""

    def save_screenshot(self) -> None:
        """Save screenshot to a predetermined directory."""
        assert self._screenshot_prefix is not None

        window = self.get_window()
        window.save_screenshot(self._screenshot_prefix)
