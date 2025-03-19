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

"""Implements functionality for processing mouse events."""

from abc import ABC, abstractmethod
from collections.abc import Callable

from direct.showbase.ShowBase import ShowBase


class MouseProcessor(ABC):
    """
    Binds mouse clicks to callable.

    :param callback: Function that is called when there is a mouse click.
    :param base: The `ShowBase` instance.
    """

    def __init__(self, callback: Callable[[], None], base: ShowBase) -> None:
        self._base = base

        self._bind_mouse(callback)

    @abstractmethod
    def _bind_mouse(self, callback: Callable[[], None]) -> None:
        """Prepare so that mouse clicks call callback."""
