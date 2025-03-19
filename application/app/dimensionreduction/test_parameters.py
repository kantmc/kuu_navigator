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

"""Test for parameter files."""

from pathlib import Path

import pytest

from app.datasource.api import DataSource


@pytest.mark.parametrize("path", Path().glob('parameters/data_source/*.json'))
def test_parameters(path: Path) -> None:
    """Test that parameter files can be read."""
    _data_source = DataSource.setup_from_file(path)
