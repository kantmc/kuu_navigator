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

"""Test fixtures."""

from collections.abc import Iterator
from math import inf
import os
import pathlib
from sys import stderr
from typing import Any, cast

import pytest

from direct.showbase.ShowBase import ShowBase

from pyqoolloop.decorators import cache

from .dimensionreduction import get_json_configuration
from .guiplatform import PlatformApp
from .testutils import TestMode, UIEnvironment


@pytest.fixture
def test_mode(capsys: pytest.CaptureFixture[str]) -> TestMode:
    """
    Returns what is being tested.

    Set environment variable `TEST_TEST=1` to enable testing of tests.
    """

    @cache(expire_time_secs=inf)  # TODO: Use `@runonce`
    def print_test_mode() -> None:
        with capsys.disabled():
            print("Running in Test Test mode!", file=stderr)  # noqa: T201

    if int(os.environ.get('TEST_TEST', '0')) != 0:
        print_test_mode()

        return TestMode.TestTest

    return TestMode.Regular


@pytest.fixture(scope='session')
def base() -> Iterator[ShowBase]:
    """
    `ShowBase` fixture with offscreen buffer.

    Only one is allowed to be instantiated.
    """
    base_ = ShowBase(fStartDirect=False, windowType='offscreen')
    assert base_.loader is not None

    yield base_

    base_.destroy()


@pytest.fixture(scope="session")
def ui_environment(base: ShowBase) -> UIEnvironment:
    """
    Prepare variables necessary for UI testing.

    Only one is allowed to be instantiated.
    """
    wx_app = PlatformApp(base=base)

    environment = UIEnvironment(wx_app, base)

    return environment


def _get_dimension_reduction_configurations(
    file_pattern: str = '*.json',
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for each_file in pathlib.Path().glob(
        'parameters/dimension_reduction/' + file_pattern
    ):
        parameter = get_json_configuration(each_file)
        result.append(parameter)

    assert len(result) > 0

    return result


@pytest.fixture(params=[None, *_get_dimension_reduction_configurations()])
def reducer_configuration(request: pytest.FixtureRequest) -> dict[str, Any] | None:
    """Dimension reduction parameter in 'parameters/dimension_reduction' directory."""
    return cast(dict[str, Any] | None, request.param)


@pytest.fixture(
    params=_get_dimension_reduction_configurations('*_mds.json')
    + _get_dimension_reduction_configurations('*_pcoa.json')
)
def nonlinear_reducer_configuration(
    request: pytest.FixtureRequest,
) -> dict[str, Any]:
    """Nonlinear dimension reduction parameter in 'parameters/dimension_reduction'."""
    return cast(dict[str, Any], request.param)
