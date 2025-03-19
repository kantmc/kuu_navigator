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

"""Test parameter files."""

from pathlib import Path
from typing import cast

import pytest

from main import get_configurations


def _get_parameter_paths() -> list[Path]:
    # Note that path is relative
    return list(Path().glob('parameters/*.txt'))


@pytest.fixture(params=_get_parameter_paths())
def parameter_path(request: pytest.FixtureRequest) -> Path:
    """Path to parameter file."""
    return cast(Path, request.param)


def test_parameters(parameter_path: Path) -> None:
    """Test parameter files are readable."""
    _ = get_configurations(["@" + str(parameter_path)])
