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

"""Configuration of pytest."""

import logging

import pytest


def pytest_configure(config: pytest.Config) -> None:  # noqa: ARG001
    """
    I don't like it, but I want to silence flake8's internal warnings.

    https://github.com/eisensheng/pytest-catchlog/issues/59
    """
    logging.getLogger('flake8').setLevel(logging.ERROR)
