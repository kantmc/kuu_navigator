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

"""Tests for `navigationcommands.py` with multiple commands."""

from math import inf
from typing import Any

import pytest
from typing_extensions import override

from direct.interval.Interval import Interval

from ..datasource.simulated import GridData
from ..nodes.visualnode import VisualNodeCollection
from ..testutils import UIEnvironment
from .command import Command
from .navigationcommands import NavigationCommand
from .testutils import (
    FakeDataPoint,
    assert_command_cycle,
    goto_empty_space,
    make_global_scene,
    make_local_scene,
    wait_till_processed,
)


@pytest.mark.parametrize(
    'commands, position_3d_atol, direction_3d_atol',
    [
        (['mF', 'mB'], 0, 0),
        (['mR', 'mL'], 0, 0),
        (['mU', 'mD'], 0, 0),
        (['mF', 'mF', 'mB', 'mB'], 0, 0),
        (['mR', 'mR', 'mL', 'mL'], 0, 0),
        (['mU', 'mU', 'mD', 'mD'], 0, 0),
        (['mF', 'mR', 'mU', 'mB', 'mL', 'mD'], 7.64274187e-15, 0),
        (['rR', 'rL'], 0, 1.1920929e-07),  # FUTURE: Less error with rotation
        (['rU', 'rD'], 0, 1.1920929e-07),  # FUTURE: Less error with rotation
        (['rC', 'rA'], 0, 1.1920929e-07),  # FUTURE: Less error with rotation
    ],
)
def test_global_command_cycle(
    ui_environment: UIEnvironment,
    commands: list[str],
    position_3d_atol: float,
    direction_3d_atol: float,
) -> None:
    """
    Test sequences of commands that should come back to the original pose.

    This is for Global mode.
    """
    scene = make_global_scene(ui_environment)

    assert_command_cycle(
        commands,
        scene=scene,
        is_global_mode=True,
        position_3d_atol=position_3d_atol,
        direction_3d_atol=direction_3d_atol,
        base=ui_environment.base,
    )


@pytest.mark.parametrize(
    'commands, position_3d_atol, direction_3d_atol',
    [
        (['mF', 'mB'], 0, 0),
        (['mR', 'mL'], 0, 0),
        (['mU', 'mD'], 0, 0),
        (['mF', 'mF', 'mB', 'mB'], 0, 0),
        (['mR', 'mR', 'mL', 'mL'], 0, 0),
        (['mU', 'mU', 'mD', 'mD'], 0, 0),
        (['mF', 'mR', 'mU', 'mB', 'mL', 'mD'], 0, 0),
        (['rR', 'rL'], 0, 1.1920929e-07),  # FUTURE: Less error with rotation
        (['rU', 'rD'], 0, 1.1920929e-07),  # FUTURE: Less error with rotation
        (['rC', 'rA'], 0, 1.1920929e-07),  # FUTURE: Less error with rotation
    ],
)
def test_local_command_cycle(
    ui_environment: UIEnvironment,
    commands: list[str],
    position_3d_atol: float,
    direction_3d_atol: float,
) -> None:
    """
    Test sequences of commands that should come back to the original pose.

    This is for Local mode.
    """
    scene = make_local_scene(ui_environment)

    assert_command_cycle(
        commands,
        scene=scene,
        is_global_mode=False,
        position_3d_atol=position_3d_atol,
        direction_3d_atol=direction_3d_atol,
        base=ui_environment.base,
    )


@pytest.mark.parametrize(
    'commands, position_3d_atol, position_atol, direction_3d_atol',
    [
        (['mF', 'mB'], 0, 0, 0),
        # Error worse in empty space probably b/c of lack of data points in neighborhood
        (['mR', 'mL'], 0, 2.38418277e-07, 0),
        (['mU', 'mD'], 0, 2.38418277e-07, 0),
        (['mF', 'mF', 'mB', 'mB'], 1.1920929e-07, 9.53673663e-07, 0),
        (['mR', 'mR', 'mL', 'mL'], 0, 9.53672881e-07, 0),
        (['mU', 'mU', 'mD', 'mD'], 0, 9.53672881e-07, 0),
        (['mF', 'mR', 'mU', 'mB', 'mL', 'mD'], 0, 7.74859039e-07, 0),
        (['rR', 'rL'], 0, 0, 5.96046448e-08),  # FUTURE: Less error with rotation
        (['rU', 'rD'], 0, 0, 5.96046448e-08),  # FUTURE: Less error with rotation
        (['rC', 'rA'], 0, 0, 1.1920929e-07),  # FUTURE: Less error with rotation
    ],
)
def test__move_back_and_forth_in_empty_space__outside(
    ui_environment: UIEnvironment,
    commands: list[str],
    position_3d_atol: float,
    position_atol: float,
    direction_3d_atol: float,
) -> None:
    """Test commands come back to position in empty space outside data source grid."""
    scene = goto_empty_space(ui_environment)

    assert_command_cycle(
        commands,
        scene=scene,
        is_global_mode=False,
        position_3d_atol=position_3d_atol,
        position_atol=position_atol,
        direction_3d_atol=direction_3d_atol,
        base=ui_environment.base,
    )


@pytest.mark.parametrize(
    'commands, position_3d_atol, position_atol, direction_3d_atol',
    [
        (['mF', 'mB'], 0, 0, 0),
        # Error worse in empty space probably b/c of lack of data points in neighborhood
        (['mR', 'mL'], 0, 2.38418277e-07, 0),
        (['mU', 'mD'], 0, 2.38418277e-07, 0),
        (['mF', 'mF', 'mB', 'mB'], 1.1920929e-07, 9.53673663e-07, 0),
        (['mR', 'mR', 'mL', 'mL'], 0, 9.53672881e-07, 0),
        (['mU', 'mU', 'mD', 'mD'], 0, 9.53672881e-07, 0),
        (['mF', 'mR', 'mU', 'mB', 'mL', 'mD'], 0, 7.74859039e-07, 0),
        (['rR', 'rL'], 0, 0, 5.96046448e-08),  # FUTURE: Less error with rotation
        (['rU', 'rD'], 0, 0, 5.96046448e-08),  # FUTURE: Less error with rotation
        (['rC', 'rA'], 0, 0, 1.1920929e-07),  # FUTURE: Less error with rotation
    ],
)
def test__move_back_and_forth_in_empty_space__inside(
    ui_environment: UIEnvironment,
    commands: list[str],
    position_3d_atol: float,
    position_atol: float,
    direction_3d_atol: float,
) -> None:
    """Test commands come back to position in empty space within data source grid."""
    selection = FakeDataPoint([1, 1, 0])
    scene = goto_empty_space(ui_environment, selection=selection)

    assert_command_cycle(
        commands,
        scene=scene,
        is_global_mode=False,
        position_3d_atol=position_3d_atol,
        position_atol=position_atol,
        direction_3d_atol=direction_3d_atol,
        base=ui_environment.base,
    )


@Command.global_command_factory.register('n')
@Command.local_command_factory.register('n')
class _NullNavigationCommand(NavigationCommand):
    """A command that doesn't do anything, but pretends to be a Movement Command."""

    @override
    def _prepare(self, visual_nodes: VisualNodeCollection) -> bool:
        return True

    @override
    def _make_intervals(self, visual_nodes: VisualNodeCollection) -> list[Interval]:
        return []


@pytest.mark.parametrize(
    'commands, position_3d_atol, direction_3d_atol',
    [
        # These are commands that don't move. Make sure command sequence works.
        (['n', 'n'], 0, 0),
        (['R', 'R'], 0, inf),
        (['S', 'S'], 0, 0),
    ],
)
def test_local_command_cycle_with_animation_on(
    ui_environment: UIEnvironment,
    reducer_configuration: dict[str, Any] | None,
    commands: list[str],
    position_3d_atol: float,
    direction_3d_atol: float,
) -> None:
    """
    Test sequences of commands call callback.

    This is for Local mode and with animation, but currently only commands that don't
    move are supported.
    """

    class _Callback:
        count: int = 0

        def __call__(self) -> None:
            self.count += 1

    zoom_in_point = FakeDataPoint([0, 0, 0])

    data_source = GridData(
        int_mins=[-3, -2, -1],
        int_maxs=[3, 2, 1],
        scale=[1, 0.5, 0.25],
        neighborhood_shape='cube',
    )

    scene = make_local_scene(
        ui_environment,
        zoom_in_point=zoom_in_point,
        reducer_configuration=reducer_configuration,
        data_source=data_source,
    )

    wait_till_processed(scene, base=ui_environment.base)

    with scene._shared_configuration.with_lock() as configuration:  # noqa: SLF001
        configuration.animation.camera_animation = True
        configuration.animation.node_animation = True

    callback = _Callback()

    assert_command_cycle(
        commands,
        scene=scene,
        is_global_mode=False,
        position_3d_atol=position_3d_atol,
        direction_3d_atol=direction_3d_atol,
        must_move=False,
        callback=callback,
        base=ui_environment.base,
    )

    for _ in range(10):
        ui_environment.base.taskMgr.step()

        if callback.count == len(commands):
            break

    assert callback.count == len(commands)


def test__LocalMoveCommand__command_cycle_with_selection(
    ui_environment: UIEnvironment,
) -> None:
    """
    Test that movement with selection comes back to the original pose.

    This is for Local mode.
    """
    scene = make_local_scene(ui_environment)

    data_point_selection = FakeDataPoint([0, 0, 0, 0])
    scene.select_data_point(data_point_selection)

    assert_command_cycle(
        ['mF', 'mB'],
        scene=scene,
        is_global_mode=False,
        position_3d_atol=0.0,
        direction_3d_atol=0.0,
        base=ui_environment.base,
    )
