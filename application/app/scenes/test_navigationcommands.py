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

"""Tests for `navigationcommands` module."""

from collections.abc import Sequence
import math
from typing import Any

import pytest
from typing_extensions import override

import numpy as np
from numpy.testing import assert_allclose

from panda3d.core import LPoint3, LVector3

from pyqoolloop.testutils import combine_lists

from app.scenes.testutils import goto_empty_space

from ..datasource.api import Coordinates, DataPoint, distance_between
from ..datasource.simulated import GridData
from ..dimensionreduction.linear import LinearOrientedDimensionReducer
from ..eventprocessing.eventprocessor import EventProcessor, NavigationEventSubscriber
from ..mathutils import NPArray, np_all
from ..testutils import UIEnvironment
from .configuration import (
    AnimationConfiguration,
    NavigationSceneConfiguration,
)
from .navigationcommands import choose_direction
from .navigationscene import (
    NavigationScene,
)
from .testutils import (
    DEFAULT_ZOOM_IN_POINT,
    FakeDataPoint,
    assert_command_cycle,
    assert_no_rotation,
    assert_on_selection,
    get_navigation_camera,
    make_global_scene,
    make_local_scene,
)

_SHOW_ALL_COMMANDS = (
    'show_all_command_str',
    [None, 'a+X', 'a+Y', 'a+Z', 'a-X', 'a-Y', 'a-Z'],
)


def _test__NavigationCommand__relax_global_mode(
    ui_environment: UIEnvironment,
    reducer_configuration: dict[str, Any] | None,
    show_all_command: str,
    command_str: str,
) -> None:
    scene = make_global_scene(
        ui_environment, reducer_configuration=reducer_configuration
    )

    scene.process_command(show_all_command)

    assert scene.is_global_mode()

    scene.process_command(command_str)

    assert scene.get_global_mode() == ''


@pytest.mark.parametrize(
    'show_all_command, command_str',
    [
        ('a', 'mF'),
        ('a-X', 'mB'),
        ('a-Y', 'mR'),
        ('a-Z', 'mL'),
        ('a+X', 'mU'),
        ('a+Y', 'mD'),
        ('a-Z', 'rU'),
        ('a+X', 'rD'),
        ('a+Y', 'rA'),
        ('a+Z', 'rC'),
        ('a', 'rL'),
        ('a-X', 'rR'),
        ('a-Y', 'cS'),  # This gets ignored with no selection
    ],
)
def test__NavigationCommand__relax_global_mode(
    ui_environment: UIEnvironment,
    reducer_configuration: dict[str, Any] | None,
    show_all_command: str,
    command_str: str,
) -> None:
    """Test that some commands set Global Mode to ''."""
    _test__NavigationCommand__relax_global_mode(
        ui_environment, reducer_configuration, show_all_command, command_str
    )


@pytest.mark.parametrize(
    'show_all_command, command_str',
    [
        ('a+Z', 'fP'),
    ],
)
def test__NavigationCommand__relax_global_mode__linear(
    ui_environment: UIEnvironment,
    show_all_command: str,
    command_str: str,
) -> None:
    """Test that some commands with linear dimension reduction set Global Mode to ''."""
    _test__NavigationCommand__relax_global_mode(
        ui_environment, None, show_all_command, command_str
    )


@pytest.mark.parametrize(
    'show_all_command, command_str',
    [
        ('a', 'R'),
        ('a', 'S'),
    ],
)
def test__NavigationCommand__dont_relax_global_mode(
    ui_environment: UIEnvironment,
    show_all_command: str,
    command_str: str,
) -> None:
    """Test that some commands don't change Global Mode."""
    scene = make_global_scene(ui_environment)

    scene.process_command(show_all_command)

    assert scene.is_global_mode()

    scene.process_command(command_str)

    assert scene.is_global_mode()


def test__ZoomInCommand__no_selection(ui_environment: UIEnvironment) -> None:
    """Test `_ZoomInCommand`."""
    scene = make_global_scene(ui_environment)

    original_camera_position_3d = get_navigation_camera(scene).get_position_3d()

    assert scene.is_global_mode()

    scene.process_command('zI')  # ignored, because no Visual Node is selected

    assert scene.is_global_mode()

    new_camera_position_3d = get_navigation_camera(scene).get_position_3d()

    assert original_camera_position_3d == new_camera_position_3d


def test__ZoomInCommand__not_empty(ui_environment: UIEnvironment) -> None:
    """Test that Zoom In doesn't get into empty neighborhood."""
    animation_configuration = AnimationConfiguration(
        node_animation=False, camera_animation=False, animation_interval_secs=0
    )

    neighborhood_radius = 2
    scene_configuration = NavigationSceneConfiguration(
        animation=animation_configuration,
        visual_size=0.1,
        background_color=(0.0, 0.0, 0.0),
        max_number_of_points=1000,
        number_of_detailed_nodes=1,
        neighborhood_radius=neighborhood_radius,
        shared_neighborhood=True,
        screenshot_prefix=None,
    )

    random_sampling_threshold = 100000

    dimensions = int(math.log(random_sampling_threshold, 2 * neighborhood_radius + 1))
    data_source = GridData(
        int_mins=[-5] * dimensions,
        int_maxs=[5] * dimensions,
        random_sampling_threshold=random_sampling_threshold,
    )
    zoom_in_point = FakeDataPoint([-1.0, 1.0, -3.0, 5.0, 0.0, -2.0, 1.0])
    scene = make_local_scene(
        ui_environment,
        zoom_in_point=zoom_in_point,
        node_exists=False,
        scene_configuration=scene_configuration,
        data_source=data_source,
    )

    assert scene.get_number_of_visuals() > 0

    scene.process_event(
        'neighborhood-radius', 1.0, sender=test__ZoomInCommand__not_empty, target=None
    )

    assert scene.get_number_of_visuals() > 0


_COORDINATES_IN_GRID = (
    'selection_coodinates',
    [
        (0, 0, 0, 0),
        (0, 0, 0, 1),
        (0, 0, 1, 0),
        (0, 1, 0, 0),
        (1, 0, 0, 0),
        # (1, 1, 1, 1),  # Zoom in point
        (0, 0, 0, -1),
    ],
)

_COORDINATES_NOT_IN_GRID = (
    'selection_coodinates',
    [
        (10, 5, 0, 0),
        (0, 10, 5, 0),
        (0, 0, 10, 5),
        (5, 0, 0, 10),
    ],
)


@pytest.mark.parametrize(*_COORDINATES_IN_GRID)
def test___COORDINATES_IN_GRID(
    ui_environment: UIEnvironment, selection_coodinates: Coordinates
) -> None:
    """Test that coordinates in `_COORDINATES_IN_GRID` are in default Data Source."""
    scene = make_global_scene(ui_environment)

    data_point_selection = FakeDataPoint(selection_coodinates)

    assert data_point_selection != DEFAULT_ZOOM_IN_POINT

    scene.select_data_point(data_point_selection)

    selected_visual_node = scene.get_selected_visual_node()
    assert selected_visual_node is not None

    assert np_all(
        np.array(selected_visual_node.get_data_point().get_coordinates())
        == np.array(selection_coodinates)
    )


@pytest.mark.parametrize(*_COORDINATES_NOT_IN_GRID)
def test___COORDINATES_NOT_IN_GRID(
    ui_environment: UIEnvironment, selection_coodinates: Coordinates
) -> None:
    """Test that `_COORDINATES_NOT_IN_GRID` are not in default Data Source."""
    scene = make_global_scene(ui_environment)

    data_point_selection = FakeDataPoint(selection_coodinates)
    scene.select_data_point(data_point_selection)

    selected_visual_node = scene.get_selected_visual_node()
    assert selected_visual_node is None


@pytest.mark.parametrize(*_COORDINATES_IN_GRID)
def test__ZoomInCommand__selection_in_global_mode__all_parameters(
    ui_environment: UIEnvironment,
    selection_coodinates: Coordinates,
    reducer_configuration: dict[str, Any] | None,
) -> None:
    """Test `_ZoomInCommand` in Global mode."""
    zoom_in_point = FakeDataPoint(selection_coodinates)
    _scene = make_local_scene(
        ui_environment=ui_environment,
        zoom_in_point=zoom_in_point,
        reducer_configuration=reducer_configuration,
    )
    # Assertion in `make_local_scene()`


_COORDINATES_x_NEIGHBORHOOD_RADIUS = (
    'selection_coodinates, neighborhood_radius',
    combine_lists(_COORDINATES_IN_GRID[1], [1000, 0.1]),
)


@pytest.mark.parametrize(*_COORDINATES_IN_GRID)
def test__ZoomInCommand__selection_in_local_mode(
    ui_environment: UIEnvironment,
    reducer_configuration: dict[str, Any] | None,
    selection_coodinates: Coordinates,
) -> None:
    """Test `_ZoomInCommand` in Local mode."""
    zoom_in_point = FakeDataPoint([-1, -1, -1, -1])

    scene = make_local_scene(
        ui_environment=ui_environment,
        zoom_in_point=zoom_in_point,
        reducer_configuration=reducer_configuration,
    )

    data_point_selection = FakeDataPoint(selection_coodinates)
    scene.select_data_point(data_point_selection)

    original_selection = scene.get_selected_visual_node()

    scene.process_command('zI')

    assert not scene.is_global_mode()

    assert_on_selection(
        scene,
        expected_position=data_point_selection,
        original_selection=original_selection,
    )


def test__zoom_in_outside_neighborhood(ui_environment: UIEnvironment) -> None:
    """Test with few data points."""
    animation_configuration = AnimationConfiguration(
        node_animation=False, camera_animation=False, animation_interval_secs=0
    )
    scene_configuration = NavigationSceneConfiguration(
        animation=animation_configuration,
        visual_size=0.1,
        background_color=(0.0, 0.0, 0.0),
        max_number_of_points=1000,
        number_of_detailed_nodes=1,
        neighborhood_radius=0.5,
        shared_neighborhood=True,
        screenshot_prefix=None,
    )

    # 4 Data Points necessary for Global Mode
    data_source = GridData(
        int_mins=[0, 0, 0],
        int_maxs=[0, 1, 1],
        neighborhood_shape='cube',
    )

    zoom_in_point = FakeDataPoint([0, 0, 0])
    scene = make_local_scene(
        ui_environment=ui_environment,
        zoom_in_point=zoom_in_point,
        show_all_command='a-X',
        scene_configuration=scene_configuration,
        data_source=data_source,
    )

    selected_data_point = FakeDataPoint([0, 0, 1])
    scene.select_data_point(selected_data_point)

    selected_node = scene.get_selected_visual_node()

    scene.push_command('zI')

    assert_on_selection(
        scene, expected_position=selected_data_point, original_selection=selected_node
    )


_EMPTY_SPACE_COMMANDS = ('dF', 'fP', 'b')


def test__empty_space__outside(ui_environment: UIEnvironment) -> None:
    """Test some commands don't crash in empty space outside the data source grid."""
    scene = goto_empty_space(ui_environment)

    for each in _EMPTY_SPACE_COMMANDS:
        print(f"Processing command: {each}")
        scene.process_command(each)


def test__empty_space__inside(ui_environment: UIEnvironment) -> None:
    """Test some commands don't crash in empty space within the data source grid."""
    selection = FakeDataPoint([1, 1, 0])
    scene = goto_empty_space(ui_environment, selection=selection)

    for each in _EMPTY_SPACE_COMMANDS:
        print(f"Processing command: {each}")
        scene.process_command(each)


@pytest.mark.parametrize(
    'mins, maxs, scale, neighborhood_shape, neighborhood_radius, local_node_count',
    [
        ([0, 0, 0], [0, 1, 1], None, 'cube', 0.5, 1),
        ([0, 0, 0], [0, 1, 1], [1, 1, 2], 'cube', 1.5, 2),
        ([0, 0, 0], [0, 1, 1], None, 'sphere', 1.1, 3),
        ([0, 0, 0], [0, 1, 1], None, 'cube', 10, 4),
    ],
)
def test__few_data_points(
    ui_environment: UIEnvironment,
    reducer_configuration: dict[str, Any] | None,
    mins: Sequence[int],
    maxs: Sequence[int],
    scale: Sequence[float] | None,
    neighborhood_shape: str,
    neighborhood_radius: float,
    local_node_count: int,
) -> None:
    """Test with few data points."""
    animation_configuration = AnimationConfiguration(
        node_animation=False, camera_animation=False, animation_interval_secs=0
    )
    scene_configuration = NavigationSceneConfiguration(
        animation=animation_configuration,
        visual_size=0.1,
        background_color=(0.0, 0.0, 0.0),
        max_number_of_points=1000,
        number_of_detailed_nodes=1,
        neighborhood_radius=neighborhood_radius,
        shared_neighborhood=True,
        screenshot_prefix=None,
    )

    # 4 Data Points necessary for Global Mode
    data_source = GridData(
        int_mins=mins,
        int_maxs=maxs,
        scale=scale,
        neighborhood_shape=neighborhood_shape,
    )

    zoom_in_point = FakeDataPoint([0, 0, 0])
    scene = make_local_scene(
        ui_environment=ui_environment,
        zoom_in_point=zoom_in_point,
        show_all_command='a-X',
        scene_configuration=scene_configuration,
        data_source=data_source,
        reducer_configuration=reducer_configuration,
    )

    assert scene.get_number_of_visuals() == local_node_count


def _assert_movement_3d(
    *,
    command_str: str,
    direction_field: str,
    direction_coefficient: float,
    scene: NavigationScene,
    is_global_mode: bool,
) -> None:
    old_camera_position_3d = get_navigation_camera(scene).get_position_3d()

    old_camera_directions_3d = get_navigation_camera(scene).get_directions_3d()

    assert scene.is_global_mode() == is_global_mode

    scene.process_command(command_str)

    assert scene.is_global_mode() == is_global_mode

    new_camera_position_3d = get_navigation_camera(scene).get_position_3d()
    assert new_camera_position_3d is not None

    if direction_coefficient == 0:
        expected_movement_3d = 0.0

    else:
        direction_3d = getattr(old_camera_directions_3d, direction_field)

        expected_movement_3d = (
            direction_3d.normalized()
            * direction_coefficient
            * get_navigation_camera(scene).get_shift_speed()
        )

    assert_allclose(
        old_camera_position_3d + expected_movement_3d,
        new_camera_position_3d,
        rtol=1.1920929e-07,
        atol=1.19209304e-07,
        err_msg=(
            f"{old_camera_position_3d=}\n"
            f"{expected_movement_3d=}\n"
            f"{new_camera_position_3d=}"
        ),
    )

    new_camera_directions_3d = get_navigation_camera(scene).get_directions_3d()

    assert_no_rotation(
        old_camera_directions_3d=old_camera_directions_3d,
        new_camera_directions_3d=new_camera_directions_3d,
        direction_3d_atol=0,
    )


def _assert_movement_consistency(
    *,
    scene: NavigationScene,
    old_camera_position_3d: LPoint3,
    old_position: NPArray,
    atol_3d: float,
) -> None:
    new_camera_position_3d = get_navigation_camera(scene).get_position_3d()

    new_position = scene.get_center()

    movement_3d = new_camera_position_3d - old_camera_position_3d

    movement = new_position - old_position

    reducer = scene.get_reducer()
    assert isinstance(reducer, LinearOrientedDimensionReducer)

    # FUTURE: `is_position` is only used in tests
    expected_movement_3d = reducer.transform(movement, is_position=False)

    assert_allclose(movement_3d, expected_movement_3d, atol=atol_3d)


_DIRECTION_PARAMETERS = (
    'command_str, direction_field, direction_coefficient',
    [
        ('mF', 'forward_3d', +1),
        ('mB', 'forward_3d', -1),
        ('mR', 'right_3d', +1),
        ('mL', 'right_3d', -1),
        ('mU', 'up_3d', +1),
        ('mD', 'up_3d', -1),
    ],
)


@pytest.mark.parametrize(*_DIRECTION_PARAMETERS)
def test__GlobalMoveCommand(
    ui_environment: UIEnvironment,
    command_str: str,
    direction_field: str,
    direction_coefficient: float,
) -> None:
    """Test `_GlobalMoveCommand`."""
    scene = make_global_scene(ui_environment)

    _assert_movement_3d(
        command_str=command_str,
        direction_field=direction_field,
        direction_coefficient=direction_coefficient,
        scene=scene,
        is_global_mode=True,
    )


@pytest.mark.parametrize(*_DIRECTION_PARAMETERS)
def test__GlobalMoveCommand__after_ShowAll(
    ui_environment: UIEnvironment,
    command_str: str,
    direction_field: str,
    direction_coefficient: float,
) -> None:
    """Test `_GlobalMoveCommand`."""
    scene = make_global_scene(ui_environment)

    for each in _SHOW_ALL_COMMANDS[1]:
        if each is not None:
            scene.process_command(each)

        _assert_movement_3d(
            command_str=command_str,
            direction_field=direction_field,
            direction_coefficient=direction_coefficient,
            scene=scene,
            is_global_mode=True,
        )


def test__GlobalRefreshCommand__number_of_points(
    ui_environment: UIEnvironment, reducer_configuration: dict[str, Any] | None
) -> None:
    """Test that the number of points changes with PCA on the spot."""
    new_number_of_points = 10

    scene = make_global_scene(
        ui_environment, reducer_configuration=reducer_configuration
    )

    assert scene.get_number_of_visuals() > new_number_of_points

    scene._on_number_of_points_update(  # noqa: SLF001
        new_number_of_points,
        _sender=test__GlobalRefreshCommand__number_of_points,
        _target=None,
    )

    assert scene.get_number_of_visuals() <= new_number_of_points


@pytest.mark.parametrize('shared_neighborhood', [False, True])
def test__LocalRefreshCommand__number_of_points(
    ui_environment: UIEnvironment,
    reducer_configuration: dict[str, Any] | None,
    shared_neighborhood: bool,  # noqa: FBT001
) -> None:
    """Test that the number of points changes with PCA on the spot."""
    new_number_of_points = 10

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
        shared_neighborhood=shared_neighborhood,
        screenshot_prefix=None,
    )
    scene = make_local_scene(
        ui_environment,
        scene_configuration=scene_configuration,
        reducer_configuration=reducer_configuration,
    )

    assert scene.get_number_of_visuals() > new_number_of_points

    scene._on_number_of_points_update(  # noqa: SLF001
        new_number_of_points,
        _sender=test__LocalRefreshCommand__number_of_points,
        _target=None,
    )

    assert scene.get_number_of_visuals() <= new_number_of_points


@pytest.mark.parametrize(
    'command_str, direction_field, direction_coefficient',
    [
        *_DIRECTION_PARAMETERS[1],
        ('R', '', 0),  # TODO: This isn't `LocalMoveCommand`
        ('S', '', 0),  # TODO: This isn't `LocalMoveCommand`
    ],
)
def test__LocalMoveCommand__no_selection(
    ui_environment: UIEnvironment,
    command_str: str,
    direction_field: str,
    direction_coefficient: float,
) -> None:
    """Test `_LocalMoveCommand` without selected Visual Node."""
    scene = make_local_scene(ui_environment)

    old_camera_position_3d = get_navigation_camera(scene).get_position_3d()
    old_position = scene.get_center()

    _assert_movement_3d(
        command_str=command_str,
        direction_field=direction_field,
        direction_coefficient=direction_coefficient,
        scene=scene,
        is_global_mode=False,
    )

    _assert_movement_consistency(
        scene=scene,
        old_camera_position_3d=old_camera_position_3d,
        old_position=old_position,
        atol_3d=2.38418506e-07,
    )


@pytest.mark.parametrize(
    'command_str, direction_field, direction_coefficient',
    _DIRECTION_PARAMETERS[1],
)
def test__LocalMoveCommand__after_ShowAll(
    ui_environment: UIEnvironment,
    command_str: str,
    direction_field: str,
    direction_coefficient: float,
) -> None:
    """Test `_LocalMoveCommand` without selected Visual Node."""
    for each in _SHOW_ALL_COMMANDS[1]:
        scene = make_local_scene(ui_environment, show_all_command=each)

        old_camera_position_3d = get_navigation_camera(scene).get_position_3d()
        old_position = scene.get_center()

        _assert_movement_3d(
            command_str=command_str,
            direction_field=direction_field,
            direction_coefficient=direction_coefficient,
            scene=scene,
            is_global_mode=False,
        )

        _assert_movement_consistency(
            scene=scene,
            old_camera_position_3d=old_camera_position_3d,
            old_position=old_position,
            atol_3d=2.38418506e-07,
        )


@pytest.mark.parametrize(
    'command_str',
    [
        'mR',
        'mL',
        'mU',
        'mD',
        'R',  # TODO: This isn't `LocalMoveCommand`
        'S',  # TODO: This isn't `LocalMoveCommand`
    ],
)
def test__LocalMoveCommand__ignored_with_selection(
    ui_environment: UIEnvironment,
    command_str: str,
) -> None:
    """Test `_LocalMoveCommand` that don't move with selection of Visual Node."""
    scene = make_local_scene(ui_environment)

    data_point_selection = FakeDataPoint([0, 0, 0, 0])
    scene.select_data_point(data_point_selection)

    assert_command_cycle(
        [command_str],
        scene=scene,
        is_global_mode=False,
        position_3d_atol=0,
        direction_3d_atol=0,
        base=ui_environment.base,
    )


@pytest.mark.parametrize(
    'command_str',
    [
        'cS',
        'mF',
        'mR',
        'mL',
        'mU',
        'mD',
        'R',  # TODO: This isn't `LocalMoveCommand`
        'S',  # TODO: This isn't `LocalMoveCommand`
    ],
)
def test__LocalMoveCommand__ignored_with_selection_outside_neighborhood(
    ui_environment: UIEnvironment,
    nonlinear_reducer_configuration: dict[str, Any],
    command_str: str,
) -> None:
    """
    Test `_LocalMoveCommand` that don't move with selection of Visual Node.

    Selection is outside of local neighborhood.
    """
    neighborhood_radius = 1.0

    animation_configuration = AnimationConfiguration(
        node_animation=False, camera_animation=False, animation_interval_secs=0
    )
    scene_configuration = NavigationSceneConfiguration(
        animation=animation_configuration,
        visual_size=1.0,
        background_color=(0, 0, 0),
        max_number_of_points=1000,
        number_of_detailed_nodes=0,
        neighborhood_radius=neighborhood_radius,
        shared_neighborhood=True,
        screenshot_prefix=None,
    )
    scene = make_local_scene(
        ui_environment,
        scene_configuration=scene_configuration,
        reducer_configuration=nonlinear_reducer_configuration,
    )

    data_point_selection = FakeDataPoint([0, 0, 0, 0])
    assert (
        distance_between(
            data_point_selection.get_coordinates(),
            DEFAULT_ZOOM_IN_POINT.get_coordinates(),
        )
        > neighborhood_radius
    )
    scene.select_data_point(data_point_selection)

    assert_command_cycle(
        [command_str],
        scene=scene,
        is_global_mode=False,
        position_3d_atol=0,
        direction_3d_atol=0,
        base=ui_environment.base,
    )


def test__LocalMoveCommand__forward_with_selection(
    ui_environment: UIEnvironment,
) -> None:
    """Test forward movement of `_LocalMoveCommand` with selection of Visual Node."""

    def _assert_selection_released(scene: NavigationScene) -> None:
        selected = scene.get_selected_data_point()
        assert selected is None

    scene = make_local_scene(ui_environment)

    old_camera_position_3d = get_navigation_camera(scene).get_position_3d()

    old_position = scene.get_center()

    data_point_selection = FakeDataPoint(np.array([0, 0, 0, 0]))
    scene.select_data_point(data_point_selection)

    assert not scene.is_global_mode()

    for _ in range(10):
        scene.process_command('mF')

        assert not scene.is_global_mode()

        new_position = scene.get_center()

        if np_all(data_point_selection.get_coordinates() == new_position):
            new_camera_position_3d = get_navigation_camera(scene).get_position_3d()
            movement_3d = new_camera_position_3d - old_camera_position_3d

            movement = new_position - old_position

            reducer = scene.get_reducer()
            assert isinstance(reducer, LinearOrientedDimensionReducer)
            expected_movement_3d = reducer.transform(movement, is_position=False)

            assert_allclose(expected_movement_3d, movement_3d, atol=1.19209275e-07)

            scene.process_command('cS')
            _assert_selection_released(scene)

            return

    raise AssertionError("Never reached selection.")


@pytest.mark.parametrize(
    'selection, within_neighborhood',
    [([0, 0, 0, 1], True), ([0, 0, 0, 2], False)],
)
def test__LocalMoveCommand__backward_with_selection(
    ui_environment: UIEnvironment,
    selection: Coordinates,
    within_neighborhood: bool,  # noqa: FBT001
) -> None:
    """Test backward movement of `_LocalMoveCommand` with selection of Visual Node."""
    zoom_in_point = FakeDataPoint([0, 0, 0, 0])

    data_source = GridData(
        int_mins=[-1, -1, -1, -2],
        int_maxs=[1, 1, 1, 2],
        scale=[1, 1, 1, 1],
        neighborhood_shape='cube',
    )

    animation_configuration = AnimationConfiguration(
        node_animation=False, camera_animation=False, animation_interval_secs=0
    )
    scene_configuration = NavigationSceneConfiguration(
        animation=animation_configuration,
        visual_size=0.1,
        background_color=(0.0, 0.0, 0.0),
        max_number_of_points=1000,
        number_of_detailed_nodes=0,
        neighborhood_radius=1.5,
        shared_neighborhood=True,
        screenshot_prefix=None,
    )
    scene = make_local_scene(
        ui_environment=ui_environment,
        zoom_in_point=zoom_in_point,
        scene_configuration=scene_configuration,
        data_source=data_source,
    )

    data_point_selection = FakeDataPoint(selection)
    scene.select_data_point(data_point_selection)

    selected_data_point = scene.get_selected_data_point()
    assert selected_data_point == data_point_selection

    selected_visual_node = scene.get_selected_visual_node()
    if within_neighborhood:
        assert selected_visual_node is not None

    else:
        assert selected_visual_node is None

    assert not scene.is_global_mode()

    for _ in range(10):
        scene.process_command('mB')

        assert not scene.is_global_mode()

        with scene._visual_nodes as visual_nodes:  # noqa: SLF001
            if visual_nodes.get_selected() is None:
                # Selected Visual Node is now out of neighborhood
                return

    raise AssertionError("Didn't move far enough.")


def _assert_face_principal_component(
    scene: NavigationScene, *, atol_direction_3d: float
) -> None:
    directions_3d = get_navigation_camera(scene).get_directions_3d()

    reducer = scene.get_reducer()
    assert isinstance(reducer, LinearOrientedDimensionReducer)
    components = reducer.get_reducer().components_
    components_3d = reducer.transform(components, is_position=False)

    forward_3d = choose_direction(
        np.array(directions_3d.forward_3d), components_3d[0, :]
    )
    assert_allclose(forward_3d, components_3d[0, :], atol=atol_direction_3d)

    up_3d = choose_direction(np.array(directions_3d.up_3d), components_3d[1, :])
    assert_allclose(up_3d, components_3d[1, :], atol=atol_direction_3d)

    right_3d = choose_direction(np.array(directions_3d.right_3d), components_3d[2, :])
    assert_allclose(right_3d, components_3d[2, :], atol=atol_direction_3d)


def _assert_drift(
    *,
    command_str: str,
    expected_movement_unsigned: list[float],
    scene: NavigationScene,
    atol_movement: float,
    atol_movement_3d: float,
    atol_direction_3d: float,
) -> None:
    def _assert_movement(
        expected_movement_unsigned: list[float],
        old_position: NPArray,
        scene: NavigationScene,
        atol: float,
    ) -> None:
        new_position = scene.get_center()
        movement = new_position - old_position
        movement = choose_direction(movement, np.array(expected_movement_unsigned))

        assert_allclose(
            movement,
            expected_movement_unsigned,
            atol=atol,
        )

    old_camera_position_3d = get_navigation_camera(scene).get_position_3d()

    old_position = scene.get_center()

    assert not scene.is_global_mode()

    scene.process_command(command_str)

    assert not scene.is_global_mode()

    _assert_movement(expected_movement_unsigned, old_position, scene, atol_movement)

    _assert_movement_consistency(
        scene=scene,
        old_camera_position_3d=old_camera_position_3d,
        old_position=old_position,
        atol_3d=atol_movement_3d,
    )

    _assert_face_principal_component(scene, atol_direction_3d=atol_direction_3d)


@pytest.mark.parametrize(
    'show_all_command, command_str, expected_movement_unsigned',
    combine_lists(
        _SHOW_ALL_COMMANDS[1],
        [
            ['dF', [1, 0, 0, 0]],
            ['dB', [1, 0, 0, 0]],
            ['dR', [0, 0, 1, 0]],
            ['dL', [0, 0, 1, 0]],
            ['dU', [0, 1, 0, 0]],
            ['dD', [0, 1, 0, 0]],
        ],
    ),
)
def test__LocalDriftCommand(
    ui_environment: UIEnvironment,
    show_all_command: str,
    command_str: str,
    expected_movement_unsigned: list[float],
) -> None:
    """Test `_LocalDriftCommand`."""
    data_source = GridData(
        int_mins=[-1, -1, -1, -1],
        int_maxs=[1, 1, 1, 1],
        scale=[4, 3, 2, 1],
        neighborhood_shape='cube',
    )
    zoom_in_point = FakeDataPoint([4, 3, 2, 1])
    scene = make_local_scene(
        ui_environment=ui_environment,
        zoom_in_point=zoom_in_point,
        show_all_command=show_all_command,
        data_source=data_source,
    )

    _assert_drift(
        command_str=command_str,
        expected_movement_unsigned=expected_movement_unsigned,
        scene=scene,
        atol_movement=1.11022303e-16,
        atol_movement_3d=5.96046378e-07,
        atol_direction_3d=2.38420e-07,
    )


@pytest.mark.skip("Currently don't support 'n_components_' parameter for PCA")
@pytest.mark.parametrize('number_of_components', [2, 1])
def test__LocalDriftCommand__decreased_components(
    ui_environment: UIEnvironment, number_of_components: int
) -> None:
    """Test that `_LocalDriftCommand` works when the number of components is few."""
    animation_configuration = AnimationConfiguration(
        node_animation=False, camera_animation=False, animation_interval_secs=0
    )
    scene_configuration = NavigationSceneConfiguration(
        animation=animation_configuration,
        visual_size=0.1,
        background_color=(0.0, 0.0, 0.0),
        max_number_of_points=1000,
        number_of_detailed_nodes=1,
        neighborhood_radius=3,
        shared_neighborhood=True,
        screenshot_prefix=None,
    )

    reducer_configuration = {
        "type": "OrientedPCA",
        "parameters": {
            "random_state": 1234,
            "n_components": number_of_components,
        },
    }

    zoom_in_point = FakeDataPoint([0, 0, 0, 0])
    scene = make_local_scene(
        ui_environment=ui_environment,
        zoom_in_point=zoom_in_point,
        scene_configuration=scene_configuration,
        reducer_configuration=reducer_configuration,
    )

    reducer = scene.get_reducer()
    assert isinstance(reducer, LinearOrientedDimensionReducer)
    pca = reducer.get_reducer()
    assert hasattr(pca, 'components_'), "Too few data points"
    assert pca.components_.shape[0] == number_of_components

    scene.process_command('dF')

    assert not scene.is_global_mode()


_ROTATION_COMMANDS = (
    'commands',
    [
        ['rR'],
        ['rL'],
        ['rU'],
        ['rD'],
        ['rC'],
        ['rA'],
        ['rR', 'rR'],
        ['rL', 'rL'],
        ['rU', 'rU'],
        ['rD', 'rD'],
        ['rC', 'rC'],
        ['rA', 'rA'],
    ],
)


@pytest.mark.parametrize(*_ROTATION_COMMANDS)
def test__RotateCommand__global_no_shift(
    ui_environment: UIEnvironment, commands: list[str]
) -> None:
    """Test that rotation doesn't change position in Global mode."""
    scene = make_global_scene(ui_environment)

    assert_command_cycle(
        commands,
        scene=scene,
        is_global_mode=True,
        position_3d_atol=0,
        direction_3d_atol=math.inf,
        base=ui_environment.base,
    )


@pytest.mark.parametrize(*_ROTATION_COMMANDS)
def test__RotateCommand__local_no_shift(
    ui_environment: UIEnvironment, commands: list[str]
) -> None:
    """Test that rotation doesn't change position in Local mode."""
    scene = make_local_scene(ui_environment)

    assert_command_cycle(
        commands,
        scene=scene,
        is_global_mode=False,
        position_3d_atol=0,
        direction_3d_atol=math.inf,
        base=ui_environment.base,
    )


@pytest.mark.parametrize(*_COORDINATES_IN_GRID)
def test__CenterSelectedCommand__global(
    ui_environment: UIEnvironment,
    reducer_configuration: dict[str, Any] | None,
    selection_coodinates: Coordinates,
) -> None:
    """Test `_CenterSelectedCommand` in Global mode."""
    scene = make_global_scene(
        ui_environment, reducer_configuration=reducer_configuration
    )

    data_point_selection = FakeDataPoint(selection_coodinates)
    scene.select_data_point(data_point_selection)

    assert scene.is_global_mode()

    scene.process_command('cS')

    assert scene.is_global_mode()

    selected_visual_node = scene.get_selected_visual_node()
    assert selected_visual_node is not None

    direction_to_selection_3d = (
        selected_visual_node.get_position_3d()
        - get_navigation_camera(scene).get_position_3d()
    )

    camera_directions_3d = get_navigation_camera(scene).get_directions_3d()
    forward_3d = camera_directions_3d.forward_3d

    assert math.isclose(
        direction_to_selection_3d.normalized().angleRad(forward_3d),
        0.0,
        abs_tol=1.2013789785214612e-07,
    )


@pytest.mark.parametrize('selection_coodinates', _COORDINATES_IN_GRID[1])
def test__CenterSelectedCommand__local(
    ui_environment: UIEnvironment,
    reducer_configuration: dict[str, Any] | None,
    selection_coodinates: Coordinates,
) -> None:
    """Test `_CenterSelectedCommand` in Local mode."""
    scene = make_local_scene(
        ui_environment, reducer_configuration=reducer_configuration
    )

    data_point_selection = FakeDataPoint(selection_coodinates)
    scene.select_data_point(data_point_selection)

    assert not scene.is_global_mode()

    scene.process_command('cS')

    assert not scene.is_global_mode()

    selected_visual_node = scene.get_selected_visual_node()
    assert selected_visual_node is not None

    direction_to_selection_3d = (
        selected_visual_node.get_position_3d()
        - get_navigation_camera(scene).get_position_3d()
    )

    camera_directions_3d = get_navigation_camera(scene).get_directions_3d()
    forward_3d = camera_directions_3d.forward_3d

    assert math.isclose(
        direction_to_selection_3d.normalized().angleRad(forward_3d),
        0.0,
        abs_tol=2.556103027018253e-07,
    )


@pytest.mark.parametrize('selection_coodinates', _COORDINATES_NOT_IN_GRID[1])
def test__CenterSelectedCommand__local__linear(
    ui_environment: UIEnvironment,
    selection_coodinates: Coordinates,
) -> None:
    """Test `_CenterSelectedCommand` in Local mode with nodes not in neighborhood."""
    scene = make_local_scene(ui_environment)

    data_point_selection = FakeDataPoint(selection_coodinates)
    scene.select_data_point(data_point_selection)

    assert not scene.is_global_mode()

    scene.process_command('cS')

    assert not scene.is_global_mode()

    direction_to_selection = (
        np.array(data_point_selection.get_coordinates()) - scene.get_center()
    )

    reducer = scene.get_reducer()
    assert isinstance(reducer, LinearOrientedDimensionReducer)

    direction_to_selection_3d = reducer.transform(
        direction_to_selection, is_position=False
    )

    camera_directions_3d = get_navigation_camera(scene).get_directions_3d()
    forward_3d = camera_directions_3d.forward_3d

    assert math.isclose(
        LVector3(*direction_to_selection_3d).normalized().angleRad(forward_3d),
        0.0,
        abs_tol=1.611807078916172e-07,
    )


@pytest.mark.parametrize(
    'selection_coodinates',
    [
        (0, 0, 0, 0),
        (4.4, 3.3, 2.2, 1.1),
        (-4.4, -3.3, -2.2, -1.1),
        (4.4, 0, 0, 0),
        (0, 3.3, 0, 0),
        (0, 0, 2.2, 0),
        (0, 0, 0, 1.1),
    ],
)
def test__CenterSelectedCommand__on_selection(
    ui_environment: UIEnvironment, selection_coodinates: Coordinates
) -> None:
    """Test `_CenterSelectedCommand` is ignored when on the selected Visual Node."""
    zoom_in_point = FakeDataPoint(selection_coodinates)

    data_source = GridData(
        int_mins=[-1, -1, -1, -1],
        int_maxs=[1, 1, 1, 1],
        scale=[4.4, 3.3, 2.2, 1.1],
        neighborhood_shape='cube',
    )
    scene = make_local_scene(
        ui_environment=ui_environment,
        zoom_in_point=zoom_in_point,
        data_source=data_source,
    )

    assert_command_cycle(
        ['cS'],
        scene=scene,
        is_global_mode=False,
        position_3d_atol=0,
        direction_3d_atol=0,
        base=ui_environment.base,
    )


@pytest.mark.parametrize(*_SHOW_ALL_COMMANDS)
def test__FacePrincipalComponentCommand__global(
    ui_environment: UIEnvironment, show_all_command_str: str | None
) -> None:
    """Test `_FacePrincipalComponentCommand` in Global mode."""
    scene = make_global_scene(ui_environment)

    if show_all_command_str is not None:
        scene.process_command(show_all_command_str)

    assert scene.is_global_mode()

    scene.process_command('fP')

    assert scene.is_global_mode()

    _assert_face_principal_component(scene, atol_direction_3d=0)


@pytest.mark.parametrize(*_SHOW_ALL_COMMANDS)
def test__FacePrincipalComponentCommand__local(
    ui_environment: UIEnvironment, show_all_command_str: str | None
) -> None:
    """Test `_FacePrincipalComponentCommand` in Local mode."""
    zoom_in_point = FakeDataPoint([-1, -1, -1, -1])
    scene = make_local_scene(
        ui_environment=ui_environment,
        zoom_in_point=zoom_in_point,
        show_all_command=show_all_command_str,
    )

    assert not scene.is_global_mode()

    scene.process_command('fP')

    assert not scene.is_global_mode()

    _assert_face_principal_component(scene, atol_direction_3d=2.38418579e-07)


def test__FacePrincipalComponentCommand__line_distribution(
    ui_environment: UIEnvironment,
) -> None:
    """Test `_FacePrincipalComponentCommand` in Local mode."""
    line_data_source = GridData(
        int_mins=[-10, 0, 0, 0],
        int_maxs=[10, 0, 0, 0],
        neighborhood_shape='sphere',
    )

    zoom_in_point = FakeDataPoint([0, 0, 0, 0])
    scene = make_local_scene(
        ui_environment=ui_environment,
        zoom_in_point=zoom_in_point,
        data_source=line_data_source,
    )

    assert not scene.is_global_mode()

    scene.process_command('fP')

    assert not scene.is_global_mode()

    _assert_face_principal_component(scene, atol_direction_3d=2.38418579e-07)


_SHOW_ALL_DIRECTIONS = (
    'command_str, right_3d, forward_3d, up_3d',
    [
        ('a+X', [0, -1, 0], [1, 0, 0], [0, 0, 1]),
        ('a+Y', [1, 0, 0], [0, 1, 0], [0, 0, 1]),
        ('a+Z', [1, 0, 0], [0, 0, 1], [0, -1, 0]),
        ('a-X', [0, 1, 0], [-1, 0, 0], [0, 0, 1]),
        ('a-Y', [0, 0, 1], [0, -1, 0], [1, 0, 0]),
        ('a-Z', [1, 0, 0], [0, 0, -1], [0, 1, 0]),
        # If `NavigationScene.INITIAL_AXIS_DIRECTION == '+Y'`
        ('a', [1, 0, 0], [0, 1, 0], [0, 0, 1]),
    ],
)


@pytest.mark.parametrize(*_SHOW_ALL_DIRECTIONS)
def test__GlobalShowAllCommand__directions(
    ui_environment: UIEnvironment,
    command_str: str,
    right_3d: list[float],
    forward_3d: list[float],
    up_3d: list[float],
) -> None:
    """Test camera directions for `_GlobalShowAllCommand`."""
    atol = 1.23634465e-07

    scene = make_global_scene(ui_environment)

    assert scene.is_global_mode()

    scene.process_command('rR')

    scene.process_command(command_str)

    assert scene.is_global_mode()

    camera = get_navigation_camera(scene)
    directions_3d = camera.get_directions_3d()

    # FUTURE: Just keep `forward_3d`?
    assert_allclose(directions_3d.forward_3d, LVector3(*forward_3d), atol=atol)

    assert_allclose(directions_3d.right_3d, LVector3(*right_3d), atol=atol)
    assert_allclose(directions_3d.up_3d, LVector3(*up_3d), atol=atol)


def test__GlobalShowAllCommand__lose_selection(
    ui_environment: UIEnvironment,
) -> None:
    """Test that Visual Node selection is lost with Show All Command."""

    class _EventProcessor(EventProcessor):
        selected_data_point: DataPoint | None = None

        @override
        def process_event(
            self,
            name: str,
            value: str | float | NavigationEventSubscriber | DataPoint | None,
            *,
            sender: object,
        ) -> None:
            if name == 'select-data-point':
                assert isinstance(value, DataPoint | None)

                self.selected_data_point = value

            return super().process_event(name, value, sender=sender)

    world_radius = 10

    total_number_of_points = (2 * world_radius + 1) ** 3

    data_source = GridData(
        int_mins=[-world_radius, -world_radius, -world_radius],
        int_maxs=[world_radius, world_radius, world_radius],
        neighborhood_shape='cube',
    )

    animation_configuration = AnimationConfiguration(
        node_animation=False, camera_animation=False, animation_interval_secs=0
    )
    scene_configuration = NavigationSceneConfiguration(
        animation=animation_configuration,
        visual_size=0.1,
        background_color=(0.0, 0.0, 0.0),
        max_number_of_points=total_number_of_points,
        number_of_detailed_nodes=1,
        neighborhood_radius=math.sqrt((2 * world_radius) ** 2 * 3) + 1,
        shared_neighborhood=False,
        screenshot_prefix=None,
    )

    event_processor = _EventProcessor()

    zoom_in_point = FakeDataPoint([0, 0, 0])
    scene = make_local_scene(
        ui_environment=ui_environment,
        zoom_in_point=zoom_in_point,
        scene_configuration=scene_configuration,
        data_source=data_source,
        event_processor=event_processor,
    )

    data_point_selection = FakeDataPoint([0, 0, 1])
    event_processor.process_event(
        'select-data-point',
        data_point_selection,
        sender=test__GlobalShowAllCommand__lose_selection,
    )

    selected_visual_node = scene.get_selected_visual_node()
    assert selected_visual_node is not None
    assert selected_visual_node.get_data_point() == data_point_selection

    assert scene.get_selected_data_point() == data_point_selection
    assert event_processor.selected_data_point == data_point_selection

    scene.push_command('a-X')

    assert scene.get_selected_visual_node() is None
    assert scene.get_selected_data_point() is None
    assert event_processor.selected_data_point is None


@pytest.mark.parametrize(*_SHOW_ALL_DIRECTIONS)
def test__LocalShowAllCommand__directions(
    ui_environment: UIEnvironment,
    command_str: str,
    right_3d: list[float],
    forward_3d: list[float],
    up_3d: list[float],
) -> None:
    """Test camera directions for `_LocalShowAllCommand`."""
    atol = 1.23634465e-07

    scene = make_local_scene(ui_environment)

    assert not scene.is_global_mode()

    scene.process_command(command_str)

    assert scene.is_global_mode()

    camera = get_navigation_camera(scene)
    directions_3d = camera.get_directions_3d()

    # FUTURE: Just keep `forward_3d`?
    assert_allclose(directions_3d.forward_3d, LVector3(*forward_3d), atol=atol)

    assert_allclose(directions_3d.right_3d, LVector3(*right_3d), atol=atol)
    assert_allclose(directions_3d.up_3d, LVector3(*up_3d), atol=atol)
