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

"""Convenience definitions for testing."""

from dataclasses import dataclass
from enum import Enum, auto

from direct.showbase.ShowBase import ShowBase

from .guiplatform import PlatformApp


class TestMode(Enum):
    """Enum that represents the what is being tested."""

    __test__ = False

    Regular = auto()
    """Regular test."""

    TestTest = auto()
    """Testing of tests."""


@dataclass
class UIEnvironment:
    """Dataclass that holds variables necessary for UI testing."""

    wx_app: PlatformApp

    base: ShowBase
