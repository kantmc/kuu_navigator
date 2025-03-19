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

"""
Module for the `Node` class.

Array type: Panda3D
"""

from direct.interval.Interval import Interval
from direct.interval.LerpInterval import (
    LerpFunc,
    LerpPosInterval,
    LerpPosQuatInterval,
    LerpQuatInterval,
    LerpQuatScaleInterval,
    LerpScaleInterval,
)
from panda3d.core import (
    LPoint3,
    LQuaternion,
    LVector3,
    NodePath,
    TransparencyAttrib,
)


class Node:
    """
    A class that holds a Panda3D `NodePath`.

    :param nodepath: :class:`panda3d.core.NodePath` to hold.
    """

    def __init__(self, nodepath: NodePath) -> None:
        self._nodepath = nodepath
        self._position_is_set = False
        self._new_position_3d: LPoint3 | None = None
        self._new_rotation_3d: LQuaternion | None = None
        self._new_scale: float | None = None

    def get_nodepath(self) -> NodePath:
        """Get :class:`panda3d.core.NodePath` for this `Node`."""
        return self._nodepath

    def _replace_nodepath(self, nodepath: NodePath) -> None:
        """Replace :class:`panda3d.core.NodePath` with new one."""
        self._nodepath.removeNode()

        self._nodepath = nodepath

    def set_alpha(self, alpha: float) -> None:
        """
        Set alpha (opacity) value.

        :param alpha: Opacity (0.0~1.0).
        """
        max_alpha = 1.0
        if alpha >= max_alpha:
            self._nodepath.setTransparency(TransparencyAttrib.M_none)

        else:
            # negative values seem to be treated like 0.
            self._nodepath.setTransparency(TransparencyAttrib.M_alpha)
            self._nodepath.setAlphaScale(alpha)

    def set_position_3d(self, position_3d: LPoint3) -> None:
        """
        Set the position of this node.

        :param position_3d: Position to set.
        """
        assert self._new_position_3d is None, "New position is set"

        self._nodepath.setPos(position_3d)

        self._position_is_set = True

    def get_position_3d(self) -> LPoint3:
        """
        Get position of this node, if it was set.

        :raises AssertionError: If position was not set.

        .. note:: Don't use this, if this :class:`VisualNode` does not belong to the
          :class:`VisualNodeCollection` of current :class:`Scene`.
        """
        assert self._position_is_set
        return self._nodepath.getPos()

    def get_position_3d_if_set(self) -> LPoint3 | None:
        """
        Get position of this node, if it was set.

        :returns: Position of this node. `None`, if not set.
        """
        if self._position_is_set:
            return self._nodepath.getPos()

        return None

    def _set_rotation_3d(self, rotation_3d: LQuaternion) -> None:
        self._nodepath.setQuat(rotation_3d)

    def get_rotation_3d(self) -> LQuaternion:
        """Get rotation of this node."""
        return self._nodepath.getQuat()

    def set_pose_3d(self, position_3d: LPoint3, rotation_3d: LQuaternion) -> None:
        """
        Set the pose of this node.

        :param position_3d: Position to set.
        :param rotation_3d: Rotation to set.
        """
        assert self._new_position_3d is None, "New position was set."
        assert self._new_rotation_3d is None, "New rotation was set."

        self._nodepath.setPosQuat(position_3d, rotation_3d)

        self._position_is_set = True

    def _set_scale(self, scale: float) -> None:
        self._nodepath.setScale(scale)

    def get_up_3d(self) -> LVector3:
        """Get the up direction of this node."""
        return self._nodepath.getQuat().getUp()

    def get_new_position_3d(self) -> LPoint3 | None:
        """
        Get position set by :meth:`set_new_position_3d()` or :meth:`set_new_pose_3d()`.

        :returns: New position. `None`, if not set.
        """
        return self._new_position_3d

    def get_new_rotation_3d(self) -> LQuaternion | None:
        """Get new rotation set by :meth:`set_new_pose_3d()`."""
        return self._new_rotation_3d

    def _get_new_pose_3d(
        self, *, clear: bool
    ) -> tuple[LPoint3 | None, LQuaternion | None]:
        """
        Get new pose for this node.

        :param clear: Whether to clear new position after returning it.
        :returns: The new position and rotation that was set with
          :meth:`set_new_position_3d()` or :meth:`set_new_pose_3d()`. `None` for element
          that was not set.
        """
        new_position_3d = self._new_position_3d
        new_rotation_3d = self._new_rotation_3d

        if clear:
            self._new_position_3d = None
            self._new_rotation_3d = None

        return new_position_3d, new_rotation_3d

    def set_new_pose_3d(
        self, position_3d: LPoint3 | None, rotation_3d: LQuaternion | None
    ) -> None:
        """
        Set new position and rotation for this node.

        :param position_3d: New position. `None`, if not to be changed.
        :param rotation_3d: New rotation. `None`, if not to be changed.
        """
        self._new_position_3d = position_3d
        self._new_rotation_3d = rotation_3d

    def set_new_transformation(
        self, new_rotation_3d: LQuaternion, scale: float
    ) -> None:
        """
        Set new scale for this node.

        :param new_rotation_3d: New rotation.
        :param scale: New scale.
        """
        self._new_rotation_3d = new_rotation_3d
        self._new_scale = scale

    def _get_new_scale(self, *, clear: bool) -> float | None:
        new_scale = self._new_scale

        if clear:
            self._new_scale = None

        return new_scale

    def get_relative_rotation_3d(self, other: 'Node') -> LQuaternion:
        """Get rotation of other Node relative to this."""
        other_nodepath = other.get_nodepath()
        return other_nodepath.getQuat(self.get_nodepath())

    def move_to_new_position(self) -> None:
        """
        Move to new position, if it was set.

        ..note:: Clears new position.
        """
        new_position_3d, new_rotation_3d = self._get_new_pose_3d(clear=True)
        assert new_rotation_3d is None

        if new_position_3d is not None:
            self.set_position_3d(new_position_3d)

    def move_to_new_pose(self) -> None:
        """
        Move to new pose, if it was set.

        ..note:: Clears new position.
        """
        new_position_3d, new_rotation_3d = self._get_new_pose_3d(clear=True)

        if new_position_3d is not None:
            self.set_position_3d(new_position_3d)

        if new_rotation_3d is not None:
            self._set_rotation_3d(new_rotation_3d)

    def apply_new_transformation(self) -> None:
        """
        Apply new transformation.

        ..note:: Clears new transformation.
        """
        new_position_3d, new_rotation_3d = self._get_new_pose_3d(clear=True)
        assert new_position_3d is None, "Currently only for axis transformation."

        assert new_rotation_3d is not None
        self._set_rotation_3d(new_rotation_3d)

        new_scale = self._get_new_scale(clear=True)
        assert new_scale is not None
        self._set_scale(new_scale)

    def make_interval_to_new_position(
        self, duration: float, *, clear: bool
    ) -> Interval | None:
        """
        Make an interval for movement to new position.

        Will change transparency, if the node is new.

        :param duration: Length in time for movement.
        :param clear: Whether to clear new pose.
        :returns: A :class:`direct.interval.Interval.Interval` representing the
          movement. `None`, if new position is not set.
        """
        new_position_3d, _ = self._get_new_pose_3d(clear=clear)
        if new_position_3d is None:
            return None

        if self._position_is_set:
            interval = LerpPosInterval(self._nodepath, duration, new_position_3d)

        else:
            interval = LerpFunc(self.set_alpha, fromData=0, toData=1, duration=duration)

            # Necessary for fading in.
            self._nodepath.setPos(new_position_3d)

        return interval

    def make_interval_to_new_pose(
        self, duration: float, *, bake_in_start: bool = True, clear: bool = True
    ) -> Interval | None:
        """
        Make an interval for movement to new pose.

        :param duration: Length in time for movement.
        :param bake_in_start: If `False`, movement is calculated every frame.
        :param clear: Whether to clear new pose.
        :returns: A :class:`direct.interval.Interval.Interval` representing the
          movement. `None`, if new position/rotation is not set.
        """
        new_position_3d, new_rotation_3d = self._get_new_pose_3d(clear=clear)

        if new_position_3d is None:
            if new_rotation_3d is None:
                return None

            interval = LerpQuatInterval(
                self._nodepath,
                duration,
                new_rotation_3d,
                bakeInStart=1 if bake_in_start else 0,
            )

        else:
            if new_rotation_3d is None:
                interval = LerpPosInterval(self._nodepath, duration, new_position_3d)
                return interval

            interval = LerpPosQuatInterval(
                self._nodepath,
                duration,
                new_position_3d,
                new_rotation_3d,
                bakeInStart=1 if bake_in_start else 0,
            )
        return interval

    def make_interval_to_new_transformation(
        self, duration: float, *, bake_in_start: bool = True, clear: bool = True
    ) -> Interval | None:
        """
        Make an interval for movement to new pose and scale.

        :param duration: Length in time for movement.
        :param bake_in_start: If `False`, movement is calculated every frame.
        :param clear: Whether to clear new pose.
        :returns: A :class:`direct.interval.Interval.Interval` representing the
          movement. `None`, if new position is not set.
        """
        new_position_3d, new_rotation_3d = self._get_new_pose_3d(clear=clear)
        assert new_position_3d is None, "Currently only for axis transformation."

        new_scale = self._get_new_scale(clear=clear)

        if new_scale == 0:
            # FUTURE: Maybe fading out is better, if no dimension reduction.
            interval = LerpScaleInterval(
                self._nodepath,
                duration,
                scale=0,
                bakeInStart=1 if bake_in_start else 0,
            )

        else:
            assert new_rotation_3d is not None, "New rotation not set."

            interval = LerpQuatScaleInterval(
                self._nodepath,
                duration,
                new_rotation_3d,
                new_scale,
                bakeInStart=1 if bake_in_start else 0,
            )

        return interval
