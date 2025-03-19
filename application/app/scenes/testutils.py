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

"""Convenience functions for testing Scenes."""

from collections.abc import Callable
from typing import Any

from typing_extensions import override

import numpy as np
from numpy.testing import assert_allclose

from direct.showbase.ShowBase import Loader, ShowBase
from panda3d.core import NodePath

import pylog

from ..datasource.api import DataPoint, DataSource
from ..datasource.simulated import GridData
from ..dimensionreduction.linear import LinearOrientedDimensionReducer
from ..eventprocessing.eventprocessor import EventProcessor
from ..mathutils import np_all
from ..nodes.camera import Camera
from ..nodes.navigationcamera import NavigationCamera
from ..nodes.visualnode import VisualNode
from ..testutils import UIEnvironment
from .configuration import (
    AnimationConfiguration,
    NavigationSceneConfiguration,
    ReducerConfiguration,
    WindowConfiguration,
)
from .navigationscene import FirstNavigationScene, NavigationScene

_logger = pylog.getLogger(__name__)
_logger.setLevel(pylog.INFO)


class FakeDataPoint(DataPoint):
    """Data Point without :class:`NodePath`."""

    @override
    def make_nodepath(
        self, visual_size: float, detail: DataPoint.Detail, loader: Loader
    ) -> NodePath:
        return NodePath('test_data_point')


def make_global_scene(
    ui_environment: UIEnvironment,
    *,
    scene_configuration: NavigationSceneConfiguration | None = None,
    data_source: DataSource | None = None,
    reducer_configuration: dict[str, Any] | None = None,
    event_processor: EventProcessor | None = None,
) -> FirstNavigationScene:
    """
    Make a Scene in Global mode.

    :param ui_environment: Fixture with variables necessary for testing.
    :param scene_configuration: Configuration parameters. `None` for defaults.
    :param data_source: Data Source that holds the data points. `None` for defaults.
    :param reducer_configuration: Configuration of the dimension reduction algorithms.
      `None` for defaults. See :class:`ReducerConfiguration`.
    :param event_processor: Event Processor that handles events emitted from the Scene.
    """
    if scene_configuration is None:
        animation_configuration = AnimationConfiguration(
            node_animation=False, camera_animation=False, animation_interval_secs=0
        )
        scene_configuration = NavigationSceneConfiguration(
            animation=animation_configuration,
            visual_size=0.1,
            background_color=(0.0, 0.0, 0.0),
            max_number_of_points=1000,
            number_of_detailed_nodes=1,
            neighborhood_radius=1000,
            shared_neighborhood=True,
            screenshot_prefix=None,
        )

    shared_scene_configuration, unique_scene_configuration = scene_configuration.split()

    if data_source is None:
        data_source = GridData(
            int_mins=[-1, -1, -1, -1],
            int_maxs=[1, 1, 1, 1],
            scale=[1, 1, 1, 1],
            neighborhood_shape='sphere',
        )

    if reducer_configuration is None:
        reducer_configuration = {
            "type": "OrientedPCA",
            "parameters": {"random_state": 1234},
        }

    generator = np.random.default_rng(seed=4567)
    all_reducer_configurations = ReducerConfiguration(
        global_mode=reducer_configuration,
        local_mode=reducer_configuration,
        random_generator=generator,
        visual_information=True,
    )

    window_configuration = WindowConfiguration(title="test")

    scene = FirstNavigationScene(
        shared_scene_configuration,
        unique_scene_configuration,
        data_source,
        all_reducer_configurations,
        window_configuration,
        base=ui_environment.base,
        event_processor=event_processor,
    )

    return scene


def assert_on_selection(
    scene: NavigationScene,
    *,
    expected_position: DataPoint,
    original_selection: VisualNode | None,
) -> None:
    """
    Assert that current position is at the expected position.

    :param scene: The Scene in question.
    :param expected_position: The expected position for the current position in the
      Scene.
    :param original_selection: The selected Visual Node before movement, if it is
      supposed to be in the Scene.
    """
    selected_data_point = scene.get_selected_data_point()
    assert selected_data_point is None

    new_position = scene.get_center()
    assert np_all(new_position == np.array(expected_position.get_coordinates()))

    new_position_3d = scene.get_center_3d()

    new_camera_position_3d = get_navigation_camera(scene).get_position_3d()

    reducer = scene.get_reducer()
    if isinstance(reducer, LinearOrientedDimensionReducer):
        expected_position_3d = reducer.transform(
            np.array(expected_position.get_coordinates()), is_position=True
        )
        assert np_all(expected_position_3d == new_position_3d), (
            f"Not equal: {expected_position_3d} == {new_position_3d}"
        )

        assert_allclose(new_camera_position_3d, expected_position_3d)

    if original_selection is not None:
        assert np_all(
            np.array(original_selection.get_data_point().get_coordinates())
            == new_position
        )

        assert original_selection.get_data_point() == expected_position

        assert np_all(
            np.array(original_selection.get_position_3d()) == new_position_3d
        ), f"Not equal: {original_selection.get_position_3d()} == {new_position_3d}"

        assert original_selection.get_position_3d() == new_camera_position_3d, (
            "Not equal: "
            f"{original_selection.get_position_3d()} != {new_camera_position_3d}"
        )


DEFAULT_ZOOM_IN_POINT = FakeDataPoint([1, 1, 1, 1])


def make_local_scene(
    ui_environment: UIEnvironment,
    *,
    zoom_in_point: DataPoint | None = None,
    node_exists: bool = True,
    show_all_command: str | None = None,
    scene_configuration: NavigationSceneConfiguration | None = None,
    data_source: DataSource | None = None,
    reducer_configuration: dict[str, Any] | None = None,
    event_processor: EventProcessor | None = None,
) -> FirstNavigationScene:
    """
    Make a Scene in Local mode.

    :param zoom_in_point: Data Point to Zoom In to to make the Scene in Local mode.
    :param node_exists: Assert that Visual Node exists at `zoom_in_point`.
    :param show_all_command: Command to process in Global mode, before Zoom In.
    :param scene_configuration: Configuration parameters. `None` for defaults.
    :param data_source: Data Source that holds the data points. `None` for defaults.
    :param reducer_configuration: Configuration of the dimension reduction algorithms.
      `None` for defaults. See :class:`ReducerConfiguration`.
    :param event_processor: Event Processor that handles events emitted from the Scene.
    """
    if zoom_in_point is None:
        zoom_in_point = DEFAULT_ZOOM_IN_POINT

    scene = make_global_scene(
        ui_environment=ui_environment,
        scene_configuration=scene_configuration,
        data_source=data_source,
        reducer_configuration=reducer_configuration,
        event_processor=event_processor,
    )

    assert scene.is_global_mode()

    if show_all_command is not None:
        scene.process_command(show_all_command)

        wait_till_processed(scene, base=ui_environment.base)

    assert scene.is_global_mode()

    scene.select_data_point(zoom_in_point)
    selection = scene.get_selected_visual_node()
    assert not node_exists or selection is not None

    scene.process_command('zI')

    wait_till_processed(scene, base=ui_environment.base)

    assert_on_selection(
        scene, expected_position=zoom_in_point, original_selection=selection
    )

    assert not scene.is_global_mode()

    return scene


def get_navigation_camera(scene: NavigationScene) -> NavigationCamera:
    """Get Navigation Camera of Scene."""
    return scene._get_navigation_camera()  # noqa: SLF001


def assert_no_rotation(
    *,
    old_camera_directions_3d: Camera.Directions3D,
    new_camera_directions_3d: Camera.Directions3D,
    direction_3d_atol: float,
) -> None:
    """Assert that there was no rotation."""
    assert_allclose(
        new_camera_directions_3d.forward_3d,
        old_camera_directions_3d.forward_3d,
        atol=direction_3d_atol,
    )
    assert_allclose(
        new_camera_directions_3d.right_3d,
        old_camera_directions_3d.right_3d,
        atol=direction_3d_atol,
    )
    assert_allclose(
        new_camera_directions_3d.up_3d,
        old_camera_directions_3d.up_3d,
        atol=direction_3d_atol,
    )


def wait_till_processed(scene: NavigationScene, *, base: ShowBase) -> None:
    """Wait until command is processed."""
    processed = False
    for _ in range(10):
        base.taskMgr.step()

        if not scene.is_processing_command():
            processed = True
            break

    assert processed


def assert_command_cycle(
    commands: list[str],
    *,
    scene: NavigationScene,
    is_global_mode: bool,
    position_3d_atol: float,
    position_atol: float = 0,
    direction_3d_atol: float,
    must_move: bool = True,
    callback: Callable[[], None] | None = None,
    base: ShowBase,
) -> None:
    """Test that a sequence of Commands comes back to the original position."""
    old_camera_position_3d = get_navigation_camera(scene).get_position_3d()

    old_camera_directions_3d = get_navigation_camera(scene).get_directions_3d()

    old_position = scene.get_center()  # Center doesn't move in Global mode
    _logger.info("%r", old_position)

    assert scene.is_global_mode() == is_global_mode

    for index, each in enumerate(commands):
        scene.process_command(each, callback=callback)

        wait_till_processed(scene, base=base)

        new_camera_position_3d = get_navigation_camera(scene).get_position_3d()

        new_camera_directions_3d = get_navigation_camera(scene).get_directions_3d()

        new_position = scene.get_center()
        _logger.info("%r", new_position)

        if must_move and (index < len(commands) - 1):
            assert (
                (
                    not np.allclose(
                        new_camera_position_3d,
                        old_camera_position_3d,
                        atol=0,
                    )
                    and (is_global_mode or not np.allclose(new_position, old_position))
                )
                or not np.allclose(
                    new_camera_directions_3d.forward_3d,
                    old_camera_directions_3d.forward_3d,
                    atol=0,
                )
                or not np.allclose(
                    new_camera_directions_3d.right_3d,
                    old_camera_directions_3d.right_3d,
                    atol=0,
                )
                or not np.allclose(
                    new_camera_directions_3d.up_3d,
                    old_camera_directions_3d.up_3d,
                    atol=0,
                )
            ), (
                f"{new_camera_position_3d=} : {old_camera_position_3d=}\n\
            {new_camera_directions_3d=} : {old_camera_directions_3d=}"
            )

        assert scene.is_global_mode() == is_global_mode

    assert_allclose(
        new_camera_position_3d, old_camera_position_3d, atol=position_3d_atol
    )

    assert_allclose(new_position, old_position, atol=position_atol)

    assert_no_rotation(
        old_camera_directions_3d=old_camera_directions_3d,
        new_camera_directions_3d=new_camera_directions_3d,
        direction_3d_atol=direction_3d_atol,
    )


def goto_empty_space(
    ui_environment: UIEnvironment,
    selection: DataPoint | None = None,
) -> NavigationScene:
    """
    Go to an empty region.

    :param ui_environment: Variables necessary for UI testing.
    :param selection: Data Point to select while moving into an empty region. `None`, to
      move out into empty space.

    :return: Created Scene.
    """
    animation_configuration = AnimationConfiguration(
        node_animation=False, camera_animation=False, animation_interval_secs=0
    )
    scene_configuration = NavigationSceneConfiguration(
        animation=animation_configuration,
        visual_size=0.1,
        background_color=(0.0, 0.0, 0.0),
        max_number_of_points=1000,
        number_of_detailed_nodes=1,
        neighborhood_radius=1,
        shared_neighborhood=True,
        screenshot_prefix=None,
    )

    data_source = GridData(
        int_mins=[-1, -1, -1],
        int_maxs=[1, 1, 1],
        scale=[1, 1, 1],
        neighborhood_shape='cube',
    )

    zoom_in_point = FakeDataPoint([1, 1, 1])
    scene = make_local_scene(
        ui_environment=ui_environment,
        zoom_in_point=zoom_in_point,
        show_all_command='a-X',
        scene_configuration=scene_configuration,
        data_source=data_source,
    )

    if selection is not None:
        scene.select_data_point(selection)

    scene.process_event(
        'neighborhood-radius', 1.01, sender=goto_empty_space, target=None
    )

    for _ in range(5):
        scene.process_command('mF')

        wait_till_processed(scene, base=ui_environment.base)

        assert not scene.is_global_mode()

        _logger.info("Number of Visuals: %d", scene.get_number_of_visuals())
        if scene.get_number_of_visuals() == 0:
            return scene

    raise AssertionError("Never reached empty space")
