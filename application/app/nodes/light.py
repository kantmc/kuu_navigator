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

"""Module for lighting."""

from panda3d.core import (
    AmbientLight as Panda3dAmbientLight,
)
from panda3d.core import (
    NodePath,
    Vec4,
)


class SelectionLight:
    """
    Light for selected node.

    :param scene_root: Root node of the scene.
    """

    def __init__(self, scene_root: NodePath) -> None:
        light = Panda3dAmbientLight('selection light')
        light.setColor(Vec4(2, 2, 2, 1))

        self._nodepath = scene_root.attachNewNode(light)

    def get_nodepath(self) -> NodePath:
        """Get :class:`panda3d.core.NodePath` of this instance."""
        return self._nodepath
