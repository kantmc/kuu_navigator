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
Classes for visual nodes.

Array type: Panda3D, python
"""

from collections.abc import (
    Collection,
    Iterable,
    Iterator,
)
import contextlib
from weakref import (
    ReferenceType,
    ref,
)

from typing_extensions import override

import numpy as np

from direct.showbase.ShowBase import Loader
from panda3d.core import (
    NodePath,
)

import pylog

from ..datasource.api import (
    DataPoint,
    DataPointCollection,
)
from ..mathutils import NPArray, random_bools
from .light import SelectionLight
from .node import Node

_logger = pylog.getLogger(__name__)


class VisualNode(Node):
    """
    A node in 3D space that displays a visual representation of a data point.

    Set position with :meth:`.node.Node.set_position_3d()`.

    :param visual_size: Size of visuals.
    :param data_point: The data point associated with this instance.
    :param scene_root_nodepath: Root node of the scene.
    :param detail: Level of detail to show.
    :param point_to_camera: Make this instance face the specified camera.
    """

    def __init__(
        self,
        *,
        visual_size: float,
        data_point: DataPoint,
        scene_root_nodepath: NodePath,
        detail: DataPoint.Detail,
        loader: Loader,
    ) -> None:
        self._data_point = data_point

        self._detail = detail

        self._loader = loader

        nodepath = self._make_nodepath(
            visual_size, data_point, scene_root_nodepath, detail, loader
        )

        super().__init__(nodepath)

    def _make_nodepath(
        self,
        visual_size: float,
        data_point: DataPoint,
        scene_root_nodepath: NodePath,
        detail: DataPoint.Detail,
        loader: Loader,
    ) -> NodePath:
        nodepath = data_point.make_nodepath(visual_size, detail, loader)
        nodepath.reparentTo(scene_root_nodepath)

        nodepath.setTag(self.PICKABLE_TAG, 'Y')
        return nodepath

    _PICKABLE_ALPHA = 0.9
    """Minimum alpha (opacity) for selectable Visual Nodes."""

    PICKABLE_TAG = 'Pickable'
    """Tag for selectable Visual Nodes."""

    def set_visual_size(self, size: float) -> None:
        """Change size."""
        nodepath = self._make_nodepath(
            visual_size=size,
            data_point=self._data_point,
            scene_root_nodepath=self._nodepath.getParent(),
            detail=self._detail,
            loader=self._loader,
        )

        position_3d = self._nodepath.getPos()

        rotation_3d = self._nodepath.getQuat()

        nodepath.setPosQuat(position_3d, rotation_3d)

        self._replace_nodepath(nodepath)

    def set_selected(self, *, selected: bool, light: SelectionLight) -> None:
        """
        Make visual modifications to show that this `VisualNode` is being selected.

        :param selected: `True`, when selected. `False`, when not.
        :param light: Light for selection.

        ..note:: Intended to be called only from :class:`VisualNodeCollection`.
        """
        if selected:
            self._nodepath.setLight(light.get_nodepath())

        else:
            self._nodepath.setLightOff()

    def get_data_point(self) -> DataPoint:
        """Get the associated data point."""
        return self._data_point

    def get_detail(self) -> DataPoint.Detail:
        """Get level of detail."""
        return self._detail

    def keep_pointing_to_camera(self, camera: Node) -> None:
        """Make `VisualNode` keep facing the specified `Camera`."""
        self._nodepath.setBillboardPointEye(camera.get_nodepath(), 0, fixed_depth=False)

    def set_properties(self, camera: Node, near: float | None) -> None:
        """
        Set properties of this Visual Node based on relation with `Camera`.

        :param camera: `Camera` that shows this Visual Node.
        :param near: Distance to near plane of the `Camera`. Won't set transparency, if
          this is `None`.
        """

        def _set_transparency(camera: Node, near: float) -> None:
            distance = (self.get_position_3d() - camera.get_position_3d()).length()
            if distance >= near:
                alpha = min((distance - near) / near, 1)
                self.set_alpha(alpha)

                if alpha >= self._PICKABLE_ALPHA:
                    self._nodepath.setTag(self.PICKABLE_TAG, 'Y')
                    return

            self._nodepath.setTag(self.PICKABLE_TAG, 'N')

        def _set_opaque() -> None:
            self.set_alpha(1)
            self._nodepath.setTag(self.PICKABLE_TAG, 'Y')

        if near is None:
            _set_opaque()

        else:
            _set_transparency(camera, near)

        self.keep_pointing_to_camera(camera)


class VisualNodeCollection(Iterable[VisualNode]):
    """
    Collection of :class:`VisualNode`.

    :param scene_root_nodepath: Root node of the scene.
    :param loader: :class:`Loader` for loading texture.
    """

    def __init__(self, scene_root_nodepath: NodePath, loader: Loader) -> None:
        self._scene_root = scene_root_nodepath

        self._loader = loader

        self._dict = dict[DataPoint, VisualNodeCollection._VisualNode]()

        # Not making `WeakValueDictionary`, because `delete()` will delete anyway.
        self._all_visual_nodes = dict[str, VisualNodeCollection._VisualNode]()

        self._selected_visual_node: (
            ReferenceType[VisualNodeCollection._VisualNode] | None
        ) = None

        self._selection_light = SelectionLight(scene_root_nodepath)

    class _VisualNode(VisualNode):
        """:class:`VisualNode` that keeps record of created instances."""

        def __init__(
            self,
            *,
            visual_size: float,
            data_point: DataPoint,
            parent: 'VisualNodeCollection',
            scene_root_nodepath: NodePath,
            detail: DataPoint.Detail,
            loader: Loader,
        ) -> None:
            self._parent = ref(parent)

            super().__init__(
                visual_size=visual_size,
                data_point=data_point,
                scene_root_nodepath=scene_root_nodepath,
                detail=detail,
                loader=loader,
            )

            self._register(parent)

        VISUAL_NODE_TAG = '_VisualNode'

        def _register(self, parent: 'VisualNodeCollection') -> None:
            tag_value = self._get_tag_value()
            parent._all_visual_nodes[tag_value] = self  # noqa: SLF001

        def _get_tag_value(self) -> str:
            return str(id(self))

        @override
        def _make_nodepath(
            self,
            visual_size: float,
            data_point: DataPoint,
            scene_root_nodepath: NodePath,
            detail: DataPoint.Detail,
            loader: Loader,
        ) -> NodePath:
            nodepath = super()._make_nodepath(
                visual_size, data_point, scene_root_nodepath, detail, loader
            )

            tag_value = self._get_tag_value()
            nodepath.setTag(self.VISUAL_NODE_TAG, tag_value)

            return nodepath

        def __del__(self) -> None:
            self.delete()

        def delete(self) -> None:
            """
            Prepare for destruction.

            This will remove internal resources held by this instance.
            """
            # Member variables may not exist if exception happened in `__init__()`.
            # Exceptions will be ignored in `__del__()`.

            with contextlib.suppress(AttributeError):
                self._nodepath.removeNode()

            try:
                parent = self._parent()

            except AttributeError:
                pass

            else:
                if parent is not None:
                    # Use `pop()` to ignore double deletion
                    parent._all_visual_nodes.pop(  # noqa: SLF001
                        self._get_tag_value(), None
                    )

    def __iter__(self) -> Iterator[VisualNode]:
        """
        Return iterator for each :class:`VisualNode`.

        The order of iteration is the same each time, unless a :class:`VisualNode` is
        added or removed.
        """
        return iter(self._dict.values())

    def __len__(self) -> int:
        """Get number of image nodes in the collection."""
        return len(self._dict)

    def clear(self) -> None:
        """Make this collection empty."""
        for each in self._dict.values():
            each.delete()

        self._dict.clear()

    def add_data_points(
        self,
        data_points: Collection[DataPoint],
        *,
        visual_size: float,
        detail: DataPoint.Detail,
        number_of_new_detailed: int,
        point_to_camera: Node | None,
        alpha: float = 1.0,
        random_generator: np.random.Generator | None = None,
    ) -> None:
        """
        Add data points as :class:`VisualNode`.

        :param data_points: Data Points to add.
        :param visual_size: Image size for the :class:`VisualNode`.
        :param detail: Level of detail for "detailed" :class:`VisualNode`.
        :param number_of_new_detailed: Number of detailed :class:`VisualNode` to add.
          Will be less if `len(data_points)` is less.
        :param point_to_camera: Whether the :class:`VisualNode` s point to the camera.
        :param alpha: Opacity for the new :class:`VisualNode`.
        :param random_generator: A random number generator. Can be `None`, if
          `number_of_new_detailed == len(data_points)`.
        """
        if random_generator is None:
            assert number_of_new_detailed == len(data_points)

            selection_of_detailed: Iterable[bool] = [True] * len(data_points)

        else:
            actual_number_of_new_detailed = min(
                len(data_points), number_of_new_detailed
            )
            selection_of_detailed = random_bools(
                random_generator, len(data_points), actual_number_of_new_detailed
            )

        for each, detailed in zip(data_points, selection_of_detailed, strict=True):
            visual_node = VisualNodeCollection._VisualNode(
                visual_size=visual_size,
                data_point=each,
                parent=self,
                scene_root_nodepath=self._scene_root,
                detail=detail if detailed else DataPoint.Detail.LOW,
                loader=self._loader,
            )
            visual_node.set_alpha(alpha)

            if point_to_camera is not None:
                visual_node.keep_pointing_to_camera(point_to_camera)

            assert each not in self._dict  # May have to rethink equality
            self._dict[each] = visual_node

    def _remove_data_points(self, data_points: Iterable[DataPoint]) -> None:
        selected = self.get_selected()

        for each in data_points:
            visual_node = self._dict[each]

            if visual_node is selected:
                _logger.info("Deselected")
                self.deselect()

            visual_node.delete()

            del self._dict[each]

    def get_number_of_detailed(self) -> int:
        """Count the number of :class:`VisualNode` that are not of LOW detail."""
        count = 0
        for each in self._dict.values():
            if each.get_detail() != DataPoint.Detail.LOW:
                count += 1

        return count

    def update_data_points(
        self,
        new_data_points: DataPointCollection,
        *,
        visual_size: float,
        max_number_of_detailed: int,
        point_to_camera: Node | None = None,
        new_alpha: float = 1.0,
        random_generator: np.random.Generator,
    ) -> None:
        """
        Change data points contained in this instance.

        :param new_data_points: New collection of data points.
        :param visual_size: Size of visuals for each data point.
        :param max_number_of_detailed: Desired number of detailed :class:`VisualNode`.
          Could be less depending on what was in `self` and `new_data_points`.
        :param point_to_camera: Make visuals face the specified camera.
        :param new_alpha: Opacity for new :class:`VisualNode`.

        .. note:: This will remove some nodes from the collection, and also release
          :class:`NodePath` s from them making them useless even if references to the
          :class:`VisualNode` are kept.
        """

        def _get_points_to_remove(
            new_data_points: Iterable[DataPoint],
        ) -> Iterable[DataPoint]:
            new_data_points_set = set(new_data_points)

            result = set[DataPoint]()
            for each in self._dict:
                if each not in new_data_points_set:
                    result.add(each)

            return result

        def _get_points_to_add(
            new_data_points: Iterable[DataPoint],
        ) -> Collection[DataPoint]:
            result = set[DataPoint]()
            for each in new_data_points:
                if each not in self._dict:
                    result.add(each)

            return result

        points_to_remove = _get_points_to_remove(new_data_points)
        self._remove_data_points(points_to_remove)

        detailed_count = self.get_number_of_detailed()

        points_to_add = _get_points_to_add(new_data_points)
        self.add_data_points(
            points_to_add,
            visual_size=visual_size,
            detail=DataPoint.Detail.MEDIUM,
            number_of_new_detailed=(max_number_of_detailed - detailed_count),
            point_to_camera=point_to_camera,
            alpha=new_alpha,
            random_generator=random_generator,
        )

    def coordinates_to_array(self) -> NPArray:
        """Convert virtual space coordinates to numpy array."""
        result = []
        for each in self:
            coordinates = each.get_data_point().get_coordinates()
            result.append(coordinates)

        return np.array(result)

    def coordinates_to_list_3d(self) -> list[np.typing.ArrayLike | None]:
        """
        Collect visual space coordinates as list.

        .. note:: Elements for nodes that do not have position set will
          become `None`.
        """
        result = []
        for each in self:
            coordinates = each.get_position_3d_if_set()
            result.append(coordinates)

        return result

    def coordinates_to_array_3d(self) -> NPArray:
        """
        Collect visual space coordinates as numpy array.

        :raises AssertionError: If any of the nodes do not have position set.
        """
        result = []
        for each in self:
            coordinates = each.get_position_3d()
            result.append(coordinates)

        return np.array(result)

    def coordinates_to_camera_coordinates_3d(
        self, camera_position_3d: NPArray, camera_directions_3d: NPArray
    ) -> NPArray:
        """
        Convert coordinates to camera coordinates.

        :param camera_directions:
        """
        array_3d = self.coordinates_to_array_3d() - camera_position_3d
        return array_3d @ camera_directions_3d.transpose()

    def to_data_point_collection(self) -> DataPointCollection:
        """Convert Visual Nodes to Data Points."""
        result = DataPointCollection()
        for each in self:
            data_point = each.get_data_point()
            result.add(data_point)

        return result

    def get_visual_node_from_data_point(
        self, data_point: DataPoint
    ) -> VisualNode | None:
        """
        Get the :class:`VisualNode` that has the given :class:`DataPoint`.

        :param data_point: Data Point for the Visual Node to be obtained.

        :return: The Visual Node in this Visual Node Collection with the given
          Data Point. `None`, if none could be found.
        """
        return self._dict.get(data_point, None)

    def get_visual_node_from_nodepath(
        self, child_node: NodePath | None
    ) -> VisualNode | None:
        """
        Find the visual node that holds specified node path.

        :param child_node: The specified :class:`panda3d.core.NodePath` instance.
        :returns: :class:`VisualNode` instance that holds `child_node`. `None`, if none
          are found, or if `child_node` is `None`.
        """
        if child_node is None:
            return None

        tag_value = child_node.getNetTag(
            VisualNodeCollection._VisualNode.VISUAL_NODE_TAG
        )
        _logger.info("Found tag: %s", tag_value)

        visual_node = self._all_visual_nodes.get(tag_value, None)
        return visual_node

    def deselect(self) -> None:
        """Cancel selection."""
        if self._selected_visual_node is not None:
            selected_visual_node = self._selected_visual_node()
            if selected_visual_node is not None:
                selected_visual_node.set_selected(
                    selected=False, light=self._selection_light
                )

        self._selected_visual_node = None

    def _set_selected(self, visual_node: VisualNode) -> None:
        self.deselect()

        assert isinstance(visual_node, VisualNodeCollection._VisualNode), (
            "Not created by `VisualNodeCollection`."
        )

        assert visual_node in self

        self._selected_visual_node = ref(visual_node)

        visual_node.set_selected(selected=True, light=self._selection_light)

    def get_selected(self) -> VisualNode | None:
        """Get selected :class:`VisualNode` instance."""
        if self._selected_visual_node is not None:
            return self._selected_visual_node()

        return None

    def select_data_point(self, data_point: DataPoint | None) -> None:
        """Select Visual Node corresponding to Data Point."""
        if data_point is not None:
            visual_node = self.get_visual_node_from_data_point(data_point)
            if visual_node is not None:
                self._set_selected(visual_node)
                return

        self.deselect()
