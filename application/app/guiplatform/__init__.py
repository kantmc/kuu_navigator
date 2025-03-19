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

"""Defines classes for use of wxPython."""

# ruff: noqa: N802

from abc import abstractmethod
from collections.abc import Callable
import math
import os
from pathlib import Path
from typing import Any, Protocol, TypeAlias, cast, runtime_checkable

from typing_extensions import override
import wx
from wx.lib.inspection import (  # type: ignore[reportMissingImports, unused-ignore]
    # Pylance v2024.5.1 couldn't find `wx.lib.inspection``
    InspectionTool,
)

from direct.showbase.ShowBase import ShowBase
from panda3d.core import (
    ButtonHandle,
    GraphicsWindow,
    KeyboardButton,
    LPoint2,
    LPoint2i,
    LVector2i,
    ModifierButtons,
    NativeWindowHandle,
    NodePath,
    PGTop,
    WindowProperties,
)
from panda3d.core import (
    MouseWatcher as PandaMouseWatcher,
)

import pylog

from ..datasource.api import DataPoint
from ..eventprocessing import keyprocessor, mouseprocessor
from ..eventprocessing.eventprocessor import (
    EventHandlerRegistry,
    EventProcessor,
    EventSubscriber,
    ValueType,
)
from ..eventprocessing.inputdevicewatcher import InputDeviceWatcher
from ..mathutils import split_float
from ..window import CanvasWindow, Window
from .spawnwxloop import spawnWxLoop
from .wxFormBuilder.toolboxframe import ToolBoxFrame

_logger = pylog.getLogger(__name__)


class _InputDeviceWatcher(InputDeviceWatcher):
    """
    A class that monitors the mouse and keyboard.

    :param mousewatcher: :class:`panda3d.core.MouseWatcher` instance.
    """

    def __init__(self, mousewatcher: PandaMouseWatcher) -> None:
        self._mousewatcher = mousewatcher

    @override
    def get_mouse(self) -> LPoint2 | None:
        if self._mousewatcher.hasMouse():
            return self._mousewatcher.getMouse()

        return None

    def _is_button_down(self, buttonhandle: ButtonHandle) -> bool:
        return cast(bool, self._mousewatcher.is_button_down(buttonhandle))

    @override
    def is_alt_down(self) -> bool:
        return (
            self._is_button_down(KeyboardButton.lalt())
            or self._is_button_down(KeyboardButton.ralt())
            or wx.GetKeyState(wx.WXK_ALT)
        )

    @override
    def is_shift_down(self) -> bool:
        return (
            self._is_button_down(KeyboardButton.lshift())
            or self._is_button_down(KeyboardButton.rshift())
            or wx.GetKeyState(wx.WXK_SHIFT)
        )

    @override
    def is_control_down(self) -> bool:
        return (
            self._is_button_down(KeyboardButton.lcontrol())
            or self._is_button_down(KeyboardButton.rcontrol())
            or wx.GetKeyState(wx.WXK_CONTROL)
            or wx.GetKeyState(wx.WXK_RAW_CONTROL)
        )


class KeyProcessor(keyprocessor.KeyProcessor):
    """`KeyProcessor` for wx and Panda3d."""

    @override
    def _bind_commands(
        self,
        commands: tuple[tuple[str, keyprocessor.KeyProcessor._KeyMapCommandTuple], ...],
        *,
        positional_key: bool,
        base: ShowBase,
    ) -> None:
        # List of keys for Panda3d are here:
        # https://docs.panda3d.org/1.10/python/programming/hardware-support/keyboard-support

        for key, command_strs in commands:
            # Somehow, numpad is not detected without 'raw-'
            key_name = 'raw-' + key if positional_key else key

            base.accept(
                key_name,
                self._process_key_down,
                [self._KeyMapCommandStrs(*command_strs), key],
            )
            base.accept(key_name + '-up', self._process_key_up, [key])


GUIWindow: TypeAlias = wx.Frame


@runtime_checkable
class GUIWindowWrapper(Protocol):
    """Protocol for class that holds a window of the GUI platform."""

    def get_gui_window(self) -> GUIWindow:
        """Get the window of the GUI platform."""


class ToolBox(ToolBoxFrame):
    """
    A wxPython window with UI components.

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
        super().__init__(parent=None)  # type: ignore[no-untyped-call]

        self._base = base

        self._event_processor = event_processor

        self._max_navigation_scenes = max_navigation_scenes

        self.m_button_nearestNeighbor.Enable(enable=False)

        self.Show()

        self.Bind(wx.EVT_CLOSE, self._on_close)

    def _get_number_of_targets(self) -> int:
        return self.m_choice_target_window.GetCount()

    def _on_close(self, event: wx.CloseEvent) -> None:
        if self._base.wxTimer is not None:
            self._base.wxTimer.Stop()

        self._base.taskMgr.stop()

        event.Skip()

    _reflection_handlers = EventHandlerRegistry['ToolBox']()

    def on_select_data_point(self, selection: DataPoint | None) -> None:
        """To be called when Data Point is selected."""
        self.m_button_nearestNeighbor.Enable(enable=selection is not None)

    @staticmethod
    def _fix_spinCtrl(spin_ctrl: wx.SpinCtrl | wx.SpinCtrlDouble) -> None:
        # https://github.com/wxWidgets/Phoenix/issues/993
        text_ctrl = spin_ctrl.Children[0]
        assert isinstance(text_ctrl, wx.TextCtrl)
        spin_ctrl.SetValue(text_ctrl.GetValue())

    def _reflect_spinCtrl_pair(
        self,
        mantissa_ctrl: wx.SpinCtrlDouble,
        exponent_ctrl: wx.SpinCtrl,
        value: ValueType,
    ) -> None:
        assert isinstance(value, float)

        mantissa, exponent = split_float(value)

        mantissa_ctrl.SetValue(mantissa)

        exponent_ctrl.SetValue(exponent)

    def _update_spinCtrl_pair(
        self,
        mantissa_ctrl: wx.SpinCtrlDouble,
        exponent_ctrl: wx.SpinCtrl,
        event_name: str,
    ) -> None:
        mantissa = mantissa_ctrl.GetValue()

        exponent = exponent_ctrl.GetValue()

        value = float(f"{mantissa:.14f}e{exponent}")

        self._event_processor.process_event(event_name, value, sender=self)

    @_reflection_handlers.register('visual-size')
    def _reflect_visual_size(
        self,
        value: ValueType,
        _sender: object,
        _target: EventSubscriber | None,
    ) -> None:
        self._reflect_spinCtrl_pair(
            self.m_spinCtrlDouble_visualSize_mantissa,
            self.m_spinCtrl_visualSize_exponent,
            value,
        )

    def _update_visual_size(self) -> None:
        self._update_spinCtrl_pair(
            self.m_spinCtrlDouble_visualSize_mantissa,
            self.m_spinCtrl_visualSize_exponent,
            'visual-size',
        )

    @override
    def m_spinCtrlDouble_visualSize_mantissaOnUpdate(
        self, event: wx.CommandEvent
    ) -> None:
        self._fix_spinCtrl(self.m_spinCtrlDouble_visualSize_mantissa)

        self._update_visual_size()

        super().m_spinCtrlDouble_visualSize_mantissaOnUpdate(  # type: ignore[no-untyped-call]
            event
        )

    @override
    def m_spinCtrl_visualSize_exponentOnUpdate(self, event: wx.CommandEvent) -> None:
        self._fix_spinCtrl(self.m_spinCtrl_visualSize_exponent)

        self._update_visual_size()

        super().m_spinCtrl_visualSize_exponentOnUpdate(  # type: ignore[no-untyped-call]
            event
        )

    @_reflection_handlers.register('shift-speed')
    def _reflect_shift_speed(
        self,
        value: ValueType,
        _sender: object,
        _target: EventSubscriber | None,
    ) -> None:
        self._reflect_spinCtrl_pair(
            self.m_spinCtrlDouble_shiftSpeed_mantissa,
            self.m_spinCtrl_shiftSpeed_exponent,
            value,
        )

    def _update_shift_speed(self) -> None:
        self._update_spinCtrl_pair(
            self.m_spinCtrlDouble_shiftSpeed_mantissa,
            self.m_spinCtrl_shiftSpeed_exponent,
            'shift-speed',
        )

    @override
    def m_spinCtrlDouble_shiftSpeed_mantissaOnUpdate(
        self, event: wx.CommandEvent
    ) -> None:
        self._fix_spinCtrl(self.m_spinCtrlDouble_shiftSpeed_mantissa)

        self._update_shift_speed()

        super().m_spinCtrlDouble_shiftSpeed_mantissaOnUpdate(  # type: ignore[no-untyped-call]
            event
        )

    @override
    def m_spinCtrl_shiftSpeed_exponentOnUpdate(self, event: wx.CommandEvent) -> None:
        self._fix_spinCtrl(self.m_spinCtrl_shiftSpeed_exponent)

        self._update_shift_speed()

        super().m_spinCtrl_shiftSpeed_exponentOnUpdate(  # type: ignore[no-untyped-call]
            event
        )

    @_reflection_handlers.register('rotation-speed')
    def _reflect_rotation_speed(
        self,
        value: ValueType,
        _sender: object,
        _target: EventSubscriber | None,
    ) -> None:
        assert isinstance(value, float)

        degrees = value * 180 / math.pi

        self.m_slider_rotationSpeed.SetValue(round(degrees))

        self.m_spinCtrlDouble_rotationSpeed.SetValue(degrees)

    @override
    def m_slider_rotationSpeedOnSlider(self, event: wx.CommandEvent) -> None:
        degrees = float(self.m_slider_rotationSpeed.GetValue())

        self.m_spinCtrlDouble_rotationSpeed.SetValue(degrees)

        radian = degrees * math.pi / 180
        self._event_processor.process_event('rotation-speed', radian, sender=self)

        super().m_slider_rotationSpeedOnSlider(event)  # type: ignore[no-untyped-call]

    @override
    def m_spinCtrlDouble_shiftSpeed_OnUpdate(self, event: wx.CommandEvent) -> None:
        degrees = self.m_spinCtrlDouble_rotationSpeed.GetValue()

        self.m_slider_rotationSpeed.SetValue(round(degrees))

        radian = degrees * math.pi / 180
        self._event_processor.process_event('rotation-speed', radian, sender=self)

        super().m_spinCtrlDouble_shiftSpeed_OnUpdate(event)  # type: ignore[no-untyped-call]

    @_reflection_handlers.register('neighborhood-radius')
    def _reflect_neighborhood_radius(
        self,
        value: ValueType,
        _sender: object,
        _target: EventSubscriber | None,
    ) -> None:
        self._reflect_spinCtrl_pair(
            self.m_spinCtrlDouble_neighborhoodRadius_mantissa,
            self.m_spinCtrl_neighborhoodRadius_exponent,
            value,
        )

    def _update_neighborhood_radius(self) -> None:
        self._update_spinCtrl_pair(
            self.m_spinCtrlDouble_neighborhoodRadius_mantissa,
            self.m_spinCtrl_neighborhoodRadius_exponent,
            'neighborhood-radius',
        )

    @override
    def m_spinCtrlDouble_neighborhoodRadius_mantissaOnUpdate(
        self, event: wx.CommandEvent
    ) -> None:
        self._fix_spinCtrl(self.m_spinCtrlDouble_neighborhoodRadius_mantissa)

        self._update_neighborhood_radius()

        super().m_spinCtrlDouble_neighborhoodRadius_mantissaOnUpdate(  # type: ignore[no-untyped-call]
            event
        )

    @override
    def m_spinCtrl_neighborhoodRadius_exponentOnUpdate(
        self, event: wx.CommandEvent
    ) -> None:
        self._fix_spinCtrl(self.m_spinCtrl_neighborhoodRadius_exponent)

        self._update_neighborhood_radius()

        super().m_spinCtrl_neighborhoodRadius_exponentOnUpdate(  # type: ignore[no-untyped-call]
            event
        )

    @override
    def m_spinCtrlDouble_animationIntervalOnUpdate(
        self, event: wx.CommandEvent
    ) -> None:
        self._fix_spinCtrl(self.m_spinCtrlDouble_animationInterval)

        value = self.m_spinCtrlDouble_animationInterval.GetValue()
        self._event_processor.process_event('animation-interval', value, sender=self)

        super().m_spinCtrlDouble_animationIntervalOnUpdate(  # type: ignore[no-untyped-call]
            event
        )

    @_reflection_handlers.register('animation-interval')
    def _reflect_animation_interval(
        self,
        value: ValueType,
        _sender: object,
        _target: EventSubscriber | None,
    ) -> None:
        assert isinstance(value, float)

        if value == 0:
            self.m_spinCtrlDouble_animationInterval.Disable()

        else:
            self.m_spinCtrlDouble_animationInterval.SetValue(value)

    @override
    def m_spinCtrlDouble_numberOfPointsOnUpdate(self, event: wx.CommandEvent) -> None:
        self._fix_spinCtrl(self.m_spinCtrlDouble_numberOfPoints)

        value = int(self.m_spinCtrlDouble_numberOfPoints.GetValue())
        self._event_processor.process_event('number-of-points', value, sender=self)

        super().m_spinCtrlDouble_numberOfPointsOnUpdate(  # type: ignore[no-untyped-call]
            event
        )

    @_reflection_handlers.register('number-of-points')
    def _reflect_number_of_points(
        self,
        value: ValueType,
        _sender: object,
        _target: EventSubscriber | None,
    ) -> None:
        assert isinstance(value, int)

        self.m_spinCtrlDouble_numberOfPoints.SetValue(value)

    @_reflection_handlers.register('screenshot-prefix')
    def _reflect_screenshot_folder(
        self,
        value: ValueType,
        _sender: object,
        _target: EventSubscriber | None,
    ) -> None:
        assert isinstance(value, str)

        self.m_textCtrl_screenshotPrefx.SetValue(value)

    @_reflection_handlers.register('shared-neighborhood')
    def _reflect_shared_neighborhood(
        self,
        value: ValueType,
        _sender: object,
        _target: EventSubscriber | None,
    ) -> None:
        assert isinstance(value, bool)

        if value:
            self.m_checkBox_shared_neighborhood.Show()
            self.Layout()

        self.m_checkBox_shared_neighborhood.SetValue(value)

    @override
    def m_checkBox_shared_neighborhoodOnCheckBox(self, event: wx.CommandEvent) -> None:
        checked = event.IsChecked()
        self._event_processor.process_event('shared-neighborhood', checked, sender=self)

        super().m_checkBox_shared_neighborhoodOnCheckBox(  # type: ignore[no-untyped-call]
            event
        )

    @override
    def m_textCtrl_screenshotPrefixOnText(self, event: wx.CommandEvent) -> None:
        screenshot_prefix = self.m_textCtrl_screenshotPrefx.GetValue()
        self._event_processor.process_event(
            'screenshot-prefix', screenshot_prefix, sender=self
        )

        super().m_textCtrl_screenshotPrefixOnText(  # type: ignore[no-untyped-call]
            event
        )

    @override
    def m_button_chooseScreenshotFolderOnButtonClick(
        self, event: wx.CommandEvent
    ) -> None:
        path = Path(self.m_textCtrl_screenshotPrefx.GetValue())
        if not path.is_dir():
            path = path.parent

        dialog = wx.DirDialog(
            self, "Choose folder to save screenshot", defaultPath=str(path)
        )

        try:
            if dialog.ShowModal() == wx.ID_OK:
                path_str = dialog.GetPath()
                self.m_textCtrl_screenshotPrefx.SetValue(path_str + os.sep)

        finally:
            dialog.Destroy()

        super().m_button_chooseScreenshotFolderOnButtonClick(  # type: ignore[no-untyped-call]
            event
        )

    @override
    def m_button_saveScreenshotOnButtonClick(self, event: wx.CommandEvent) -> None:
        screenshot_prefix = self.m_textCtrl_screenshotPrefx.GetValue()

        self._event_processor.process_event(
            'save-screenshot', screenshot_prefix, sender=self
        )

        super().m_button_saveScreenshotOnButtonClick(  # type: ignore[no-untyped-call]
            event
        )

    @override
    def m_button_nearestNeighborOnButtonClick(self, event: wx.CommandEvent) -> None:
        self._event_processor.process_event(
            'calculate-nearest-neighbor', None, sender=self
        )

    @_reflection_handlers.register('nearest-neighbor')
    def _on_nearest_neighbor(
        self,
        value: ValueType,
        _sender: object,
        _target: EventSubscriber | None,
    ) -> None:
        assert isinstance(value, float)

        wx.MessageBox(
            f"Distance to nearest sampled neighbor: {value:f}",
            caption="Information",
            style=wx.OK | wx.ICON_INFORMATION | wx.CENTRE,
            parent=self,
        )

    @override
    def m_button_helpOnButtonClick(self, event: wx.CommandEvent) -> None:
        self._event_processor.process_event('help', None, sender=self)

    @_reflection_handlers.register('help')
    def _on_help(
        self,
        value: ValueType,
        _sender: object,
        _target: EventSubscriber | None,
    ) -> None:
        assert isinstance(value, str)

        dialog = wx.MessageDialog(
            parent=self, message=value, caption="Help for Navigation Window"
        )

        try:
            dialog.ShowModal()

        finally:
            dialog.Destroy()

    @override
    def m_button_newSceneOnButtonClick(self, event: wx.CommandEvent) -> None:
        self._event_processor.process_event(
            'new-navigation-scene', self._event_processor.get_target(), sender=self
        )

    @override
    def m_button_displayOnButtonClick(self, event: wx.CommandEvent) -> None:
        self._event_processor.process_event('toggle-display', None, sender=self)

    def register_target_window(self, title: str) -> None:
        """Add the title of a Window."""
        self.m_choice_target_window.Append(title)

        if self._get_number_of_targets() >= self._max_navigation_scenes:
            self.m_button_newScene.Enable(enable=False)

    def remove_target_window(self, title: str) -> None:
        """Remove the title of a Window."""
        item = self.m_choice_target_window.FindString(title, caseSensitive=True)
        if item != wx.NOT_FOUND:
            self.m_choice_target_window.Delete(item)

        if self._get_number_of_targets() < self._max_navigation_scenes:
            self.m_button_newScene.Enable(enable=True)

        if self._get_number_of_targets() == 0:
            self.m_button_help.Enable(enable=False)

    def set_target_window(self, title: str) -> None:
        """Set the name of the target Window."""
        choice = self.m_choice_target_window

        if self:
            item = choice.FindString(title, caseSensitive=True)
            choice.SetSelection(item)

            self.m_button_help.Enable(enable=True)

    def process_reflection(self, name: str, value: ValueType) -> None:
        """
        Process event to reflect values.

        :param name: Name of value.
        :param value: The value.

        ..note:: Reflection events don't raise exceptions, even if the events were not
          processed.
        """
        if self:
            self._reflection_handlers.process_event(
                self, name, value, sender=self, target=None, dont_raise=True
            )


class KeyEventCallback(Protocol):
    """Protocol for callbacks that respond to key events."""

    def __call__(self, key: str, *, numpad: bool) -> None:
        """
        Respond to key event.

        :param key: Unicode string of the key. ASCII will be upper case.
        :param numpad: Whether the key was on the numpad.
        """


class _WxPanel(wx.Panel):
    """
    A wxPython `Panel`.

    ..note:: Necessary, because `Frame` won't catch most keyboard events on some
      platforms.
    """

    def __init__(self, wx_parent: wx.Window, pos: wx.Point, size: wx.Size) -> None:
        super().__init__(parent=wx_parent, pos=pos, size=size)

        self._on_key_down_callback: KeyEventCallback | None = None
        self._on_key_up_callback: KeyEventCallback | None = None

        self.Bind(wx.EVT_KEY_DOWN, self._on_key_down)
        self.Bind(wx.EVT_KEY_UP, self._on_key_up)

    def set_on_key_down_callback(self, callback: KeyEventCallback) -> None:
        """
        Set callback for key down events.

        Callback not to be called on key repeats.
        """
        assert self._on_key_down_callback is None

        self._on_key_down_callback = callback

    def set_on_key_up_callback(self, callback: KeyEventCallback) -> None:
        """Set callback for key up events."""
        assert self._on_key_up_callback is None

        self._on_key_up_callback = callback

    _NUMPAD_KEYS = frozenset(
        [
            wx.WXK_NUMPAD0,
            wx.WXK_NUMPAD1,
            wx.WXK_NUMPAD2,
            wx.WXK_NUMPAD3,
            wx.WXK_NUMPAD4,
            wx.WXK_NUMPAD5,
            wx.WXK_NUMPAD6,
            wx.WXK_NUMPAD7,
            wx.WXK_NUMPAD8,
            wx.WXK_NUMPAD9,
        ]
    )

    def _on_key_down(self, event: wx.KeyEvent) -> None:
        # FUTURE: Test with Linux https://docs.wxpython.org/wx.KeyEvent.html#wx.KeyEvent.IsAutoRepeat
        if event.IsAutoRepeat():
            return

        if self._on_key_down_callback is not None:
            numpad = event.GetKeyCode() in self._NUMPAD_KEYS
            self._on_key_down_callback(chr(event.GetUnicodeKey()), numpad=numpad)

    def _on_key_up(self, event: wx.KeyEvent) -> None:
        if self._on_key_up_callback is not None:
            numpad = event.GetKeyCode() in self._NUMPAD_KEYS
            self._on_key_up_callback(chr(event.GetUnicodeKey()), numpad=numpad)


class _WxPanda3dFrame(wx.Frame):  # Is ABC
    """
    A wxPython `Frame` with a Panda3d window.

    :param wx_parent: The parent `wx.Window` if this not the Main Window. `None`, if
      this is the Main Window.
    :param title: Title of the window.
    :param origin: Top left of the window in pixels.
    :param size: Size of the Window in pixels.
    :param base: The :class:`direct.showbase.ShowBase.ShowBase` instance.
    """

    def __init__(
        self,
        *,
        wx_parent: wx.Window | None,
        title: str,
        origin: wx.Point,
        size: wx.Size,
        base: ShowBase,
        **kwargs: Any,
    ) -> None:
        super().__init__(wx_parent, title=title, pos=origin, size=size, **kwargs)

        # Make sure title bar is not hidden on MacOS
        self.EnableFullScreenView(enable=False)

        self._base = base

        client_size = self.GetClientSize()
        self._panel = _WxPanel(wx_parent=self, pos=wx.Point(0, 0), size=client_size)

        self.Show()

        self._window_handle = NativeWindowHandle.makeInt(self.GetHandle())

        self._graphicswindow = self._make_graphicswindow()

        self._input_device_watcher = self._make_input_device_watcher()

        self._on_close_callback: Callable[[], bool] | None = None
        self._on_focus_callback: Callable[[], None] | None = None
        self._on_size_callback: Callable[[int, int], None] | None = None

        self._title = title

        self.Bind(wx.EVT_ACTIVATE, self._on_activate)
        self.Bind(wx.EVT_CLOSE, self._on_close)
        self.Bind(wx.EVT_SIZE, self._on_size)

    @abstractmethod
    def _make_graphicswindow(self) -> GraphicsWindow:
        """
        Make a Panda3d `GraphicsWindow`.

        To be overriden Called in `__init__()`.
        """
        raise NotImplementedError

    @abstractmethod
    def _make_input_device_watcher(self) -> _InputDeviceWatcher:
        """Make an instance that monitors the mouse and keyboard."""
        raise NotImplementedError

    def get_input_device_watcher(self) -> _InputDeviceWatcher:
        """Get the instance that monitors the mouse and keyboard."""
        return self._input_device_watcher

    def _clean_up(self) -> None:
        if self._graphicswindow is not None:
            # `Destroy()` could get called twice. Soemtimes closing window fails.

            _logger.info("Close `GraphicsWindow`")

            self._base.closeWindow(self._graphicswindow)

            self._graphicswindow = None

    @override
    def Destroy(self) -> bool:
        self._clean_up()

        return super().Destroy()

    def get_title(self) -> str:
        """Get the title of this `Frame`."""
        return self._title

    def get_graphicswindow(self) -> GraphicsWindow:
        """Get `GraphicsWindow`."""
        return self._graphicswindow

    def set_on_close_callback(self, callback: Callable[[], bool]) -> None:
        """
        Set callback for when `Frame` is to be closed.

        The callback returns `False`, if the `Frame` is not supposed to be closed.
        Sometimes, the `Frame` is closed anyway.
        """
        assert self._on_close_callback is None

        self._on_close_callback = callback

    def set_on_focus_callback(self, callback: Callable[[], None]) -> None:
        """Set callback for when `Frame` gets focus."""
        assert self._on_focus_callback is None

        self._on_focus_callback = callback

    def set_on_key_down_callback(self, callback: KeyEventCallback) -> None:
        """Set callback for key down events."""
        self._panel.set_on_key_down_callback(callback)

    def set_on_key_up_callback(self, callback: KeyEventCallback) -> None:
        """Set callback for key up events."""
        self._panel.set_on_key_up_callback(callback)

    def set_on_size_callback(self, callback: Callable[[int, int], None]) -> None:
        """Set callback for event for window resize."""
        assert self._on_size_callback is None

        self._on_size_callback = callback

    def _make_window_properties(self, width: int, height: int) -> WindowProperties:
        """Make `WindowProperties` with the specified Window size."""
        window_properties = WindowProperties()

        window_properties.set_parent_window(self._window_handle)
        window_properties.set_origin(0, 0)
        window_properties.set_size(width, height)

        return window_properties

    def _on_close(self, event: wx.CloseEvent) -> None:
        """
        Event handler for closing window.

        .. note:: This is not called on child frames.
        """
        if self._on_close_callback is not None:
            can_close = self._on_close_callback()

        if event.CanVeto() and (not can_close):
            event.Veto()
            return

        self.Destroy()

    def _on_activate(self, event: wx.ActivateEvent) -> None:
        if event.GetActive() and self._on_focus_callback is not None:
            self._on_focus_callback()

    def _on_size(self, event: wx.SizeEvent) -> None:  # noqa: ARG002
        assert self._graphicswindow is not None, "Must call `_set_graphicswindow()"

        client_size = self.GetClientSize()
        assert client_size is not None  # FUTURE: Somehow, mypy 1.11.1 assumes `None`

        window_properties = self._make_window_properties(
            client_size.GetWidth(),
            client_size.GetHeight(),
        )

        self._graphicswindow.requestProperties(window_properties)

        if self._on_size_callback is not None:
            self._on_size_callback(client_size.GetWidth(), client_size.GetHeight())


class _FirstWxPanda3dFrame(_WxPanda3dFrame):
    """
    The first wxPython `Frame` with a Panda3d window.

    :param wx_parent: The parent `wx.Window` if this not the Main Window. `None`, if
      this is the Main Window.
    :param title: Title of the window.
    :param origin: Top left of the window in pixels.
    :param size: Size of the Window in pixels.
    :param base: The :class:`direct.showbase.ShowBase.ShowBase` instance.
    """

    def __init__(
        self,
        *,
        wx_parent: wx.Frame | None,
        title: str,
        origin: wx.Point,
        size: wx.Size,
        base: ShowBase,
        **kw: Any,
    ) -> None:
        super().__init__(
            wx_parent=wx_parent, title=title, origin=origin, size=size, base=base, **kw
        )

    @override
    def _make_graphicswindow(self) -> GraphicsWindow:
        assert self.ClientSize is not None
        window_properties = self._make_window_properties(
            self.ClientSize.GetWidth(), self.ClientSize.GetHeight()
        )

        if self._base.pipe is None:
            _logger.info("make pipe.")
            self._base.make_default_pipe()

        # https://discourse.panda3d.org/t/how-to-use-panda3d-in-tkinter-on-mac/29870/10
        self._base.open_default_window(props=window_properties)

        return self._base.win

    @override
    def _make_input_device_watcher(self) -> _InputDeviceWatcher:
        if self._base.buttonThrowers is not None:
            # Don't throw 'alt-c', etc.
            self._base.buttonThrowers[0].node().setModifierButtons(ModifierButtons())

        return _InputDeviceWatcher(self._base.mouseWatcherNode)


class _AdditionalWxPanda3dFrame(_WxPanda3dFrame):
    """
    Additional wxPython `Frame` with a Panda3d window.

    :param wx_parent: The parent `wx.Window` if this not the Main Window. `None`, if
      this is the Main Window.
    :param title: Title of the window.
    :param origin: Top left of the window in pixels.
    :param size: Size of the Window in pixels.
    :param base: The :class:`direct.showbase.ShowBase.ShowBase` instance.
    """

    def __init__(
        self,
        wx_parent: wx.Window | None,
        title: str,
        origin: wx.Point,
        size: wx.Size,
        base: ShowBase,
        **kw: Any,
    ) -> None:
        super().__init__(
            wx_parent=wx_parent, title=title, origin=origin, size=size, base=base, **kw
        )

    @override
    def _make_graphicswindow(self) -> GraphicsWindow:
        def _make_graphicswindow_properties(
            *,
            origin: LPoint2i,
            size: LVector2i,
        ) -> WindowProperties | None:
            result = WindowProperties.getDefault()

            result.set_parent_window(self._window_handle)

            if origin is not None:
                result.set_origin(origin)

            result.set_size(size)

            return result

        def _open_graphicswindow(
            *,
            origin: LPoint2i,
            size: LVector2i,
            base: ShowBase,
        ) -> GraphicsWindow:
            graphicswindow_properties = _make_graphicswindow_properties(
                origin=origin, size=size
            )

            graphicswindow = base.openWindow(
                props=graphicswindow_properties,
                type='none',
                makeCamera=False,
                stereo=False,
                requireWindow=True,
            )

            return graphicswindow

        assert self.ClientSize is not None
        graphicswindow = _open_graphicswindow(
            origin=LPoint2i(0, 0),
            size=LVector2i(self.ClientSize.GetWidth(), self.ClientSize.GetHeight()),
            base=self._base,
        )

        return graphicswindow

    @override
    def _make_input_device_watcher(self) -> _InputDeviceWatcher:
        button_throwers, pointer_watcher_nodes = self._base.setupMouseCB(
            self._graphicswindow
        )
        assert len(pointer_watcher_nodes) == 1

        mousewatcher = pointer_watcher_nodes[0]
        assert isinstance(mousewatcher, PandaMouseWatcher), (
            f"`mousewatcher` is of {type(mousewatcher)}"
        )

        # Don't throw 'alt-c', etc.
        button_throwers[0].node().setModifierButtons(ModifierButtons())

        return _InputDeviceWatcher(mousewatcher)


class _WxCanvasWindow(CanvasWindow, GUIWindowWrapper):
    def __init__(
        self,
        *,
        is_main: bool,
        base: ShowBase,
        graphicswindow: GraphicsWindow,
        wx_frame: _WxPanda3dFrame,
        aspect2d: NodePath,
        top_left_2d: NodePath,
    ) -> None:
        input_device_watcher = wx_frame.get_input_device_watcher()

        super().__init__(
            is_main=is_main,
            base=base,
            graphicswindow=graphicswindow,
            aspect2d=aspect2d,
            top_left_2d=top_left_2d,
            input_device_watcher=input_device_watcher,
            window=wx_frame,
        )

        wx_frame.set_on_key_down_callback(self._on_wx_key_down)
        wx_frame.set_on_key_up_callback(self._on_wx_key_up)
        wx_frame.set_on_size_callback(self._on_size)

        self._wx_frame = wx_frame

    @override
    def set_on_close_callback(self, callback: Callable[[], bool]) -> None:
        self._wx_frame.set_on_close_callback(callback)

    @override
    def set_on_focus_callback(self, callback: Callable[[], None]) -> None:
        self._wx_frame.set_on_focus_callback(callback)

    def _on_wx_key_down(self, key: str, *, numpad: bool) -> None:  # noqa: ARG002
        # This implementation only supports QWERTY keyboards
        lower_key = key.lower()
        keys_to_send = ('raw-' + lower_key, lower_key)

        for each in keys_to_send:
            self._base.messenger.send(each)

    def _on_wx_key_up(self, key: str, *, numpad: bool) -> None:  # noqa: ARG002
        # This implementation only supports QWERTY keyboards
        lower_key = key.lower() + '-up'
        keys_to_send = (lower_key, 'raw-' + lower_key)

        for each in keys_to_send:
            self._base.messenger.send(each)

    @override
    def get_title(self) -> str:
        return self._wx_frame.get_title()

    @override
    def get_origin(self) -> LPoint2i:
        position = self._wx_frame.GetPosition()
        return LPoint2i(position.x, position.y)

    @override
    def get_size(self) -> LVector2i:
        size = self._wx_frame.GetSize()
        return LVector2i(
            size.GetWidth(),  # type: ignore[attr-defined] # mypy 1.8.0
            size.GetHeight(),  # type: ignore[attr-defined] # mypy 1.8.0
        )

    @override
    def get_gui_window(self) -> GUIWindow:
        """Get the window of the GUI platform."""
        return self._wx_frame

    @staticmethod
    def _get_gui_window(window: Window | None) -> wx.Frame | None:
        if window is None:
            wx_frame: wx.Frame | None = None

        elif isinstance(window, GUIWindowWrapper):
            wx_frame = window.get_gui_window()

        else:
            raise AssertionError

        return wx_frame


class FirstCanvasWindow(_WxCanvasWindow):
    """
    The first Canvas Window.

    :param base: The :class:`direct.showbase.ShowBase.ShowBase` instance.
    :param parent: The parent Window.
    :param origin: Top left of the window in pixels.
    :param size: The size of the Window in pixels.
    :param title: The title for the window.
    """

    def __init__(
        self,
        *,
        base: ShowBase,
        parent: Window | None,
        origin: LPoint2i | None,
        size: LVector2i | None,
        title: str,
    ) -> None:
        parent_wx_frame = self._get_gui_window(parent)

        if origin is None:
            origin = wx.Point(0, 0)

        else:
            origin = wx.Point(origin.x, origin.y)

        if size is None:
            size = wx.Size(1280, 640)

        else:
            size = wx.Size(size.x, size.y)

        wx_frame = _FirstWxPanda3dFrame(
            wx_parent=parent_wx_frame,
            title=title,
            origin=origin,
            size=size,
            base=base,
        )

        super().__init__(
            is_main=True,
            base=base,
            graphicswindow=base.win,
            wx_frame=wx_frame,
            aspect2d=base.aspect2d,
            top_left_2d=base.a2dTopLeft,
        )


class AdditionalCanvasWindow(_WxCanvasWindow):
    """
    An additional Canvas Window.

    :param origin: Top left of the window in pixels.
    :param size: Size of the window in pixels.
    :param base: The :class:`direct.showbase.ShowBase.ShowBase` instance.
    :param title: The title for the window.
    :param parent: The parent :class:`Window`.
    """

    def __init__(
        self,
        *,
        origin: LPoint2i,
        size: LVector2i,
        base: ShowBase,
        title: str,
        parent: Window | None,
    ) -> None:
        def _prepare_aspect2d(window: GraphicsWindow) -> NodePath:
            camera2d = base.makeCamera2d(window)

            render2d = NodePath('render2d_for_additional_window')
            render2d.setDepthTest(depth_test=False)
            render2d.setDepthWrite(depth_write=False)
            camera2d.reparentTo(render2d)

            aspect2d = render2d.attachNewNode(PGTop('aspect2d_for_additional_window'))

            return aspect2d

        def _prepare_top_left(aspect2d: NodePath) -> NodePath:
            return aspect2d.attachNewNode('a2dTopLeft_for_another_scene')

        parent_wx_frame = self._get_gui_window(parent)
        wx_frame = _AdditionalWxPanda3dFrame(
            wx_parent=parent_wx_frame,
            title=title,
            origin=wx.Point(origin.x, origin.y),
            size=wx.Size(size.getX(), size.getY()),
            base=base,
        )

        graphicswindow = wx_frame.get_graphicswindow()

        aspect2d = _prepare_aspect2d(graphicswindow)

        top_left_2d = _prepare_top_left(aspect2d)

        super().__init__(
            is_main=False,
            base=base,
            graphicswindow=graphicswindow,
            wx_frame=wx_frame,
            aspect2d=aspect2d,
            top_left_2d=top_left_2d,
        )

    @override
    def _on_size(self, width: int, height: int) -> None:
        aspect_ratio = width / height
        self._set_aspect_ratio(aspect_ratio)

        super()._on_size(width, height)


class PlatformApp(wx.App):
    """
    Subclass of wxPython `App` that forces exit of app when needed.

    :param inspect: Whether to use the wxPython Inspection Tool.
    """

    # Want to keep this class minimal, because we eventually may want to use `ShowBase`
    # implementation, which uses `wx.App` directly.

    def __init__(self, *, base: ShowBase, inspect: bool = False) -> None:
        super().__init__()

        self._base = base

        self.Bind(wx.EVT_IDLE, self._on_idle)

        if inspect:
            self._inspection_tool = InspectionTool()
            self._inspection_tool.Show()

    def _on_idle(self: wx.App, event: wx.Event) -> None:
        assert isinstance(event, wx.IdleEvent)

        # _logger.info(wx.GetTopLevelWindows())

        top_level_windows = wx.GetTopLevelWindows()
        if len(top_level_windows) == 0:
            _logger.warning("Force exit")

            # Sometimes, app doesn't close even if all Top Level Windows are closed.

            # Doesn't work `self.ExitMainLoop()` `wx.Exit()`
            # `self.Destroy()` crashes `App`
            os._exit(os.EX_OK)

    @staticmethod
    def get_display_size() -> LVector2i:
        """Get the width and height of all the display combined."""
        # FUTURE: Multi-display
        display_size = wx.GetDisplaySize()
        return LVector2i(display_size.x, display_size.y)

    def run(self) -> None:
        """Run the GUI loop."""
        spawnWxLoop(self, self._base)

        self._base.run()


class MouseProcessor(mouseprocessor.MouseProcessor):
    """Binds mouse clicks to callable."""

    @override
    def _bind_mouse(self, callback: Callable[[], None]) -> None:
        self._base.accept('mouse1', callback)
