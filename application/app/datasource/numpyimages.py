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
Data Source involving numpy arrays.

Array type: python (for interface)
"""

from typing_extensions import override

import numpy as np

import pylog

from .api import DataPoint, DataSource, DataStorage, ImageCard

_logger = pylog.getLogger(__name__)
_logger.setLevel(pylog.INFO)


@DataSource.factory.register
class NumpyImages(DataStorage):
    """
    A :class:`DataStorage` that shows images stored in a numpy array.

    :param vectors_file: Name of `.npy` file that stores a 2D numpy array with virtual
      coordinates in each row.
    :param images_file: Name of `.npy` file that stores a 4D numpy array with 2D color
      images. The image in [i, :, :, :] corresponds to the coordinate in [i, :] in
      `vectors_file`.
    :param invert_images: Whether to invert the images.
    :param _comments: Placeholder for comments in json files.
    """

    def __init__(
        self,
        *,
        vectors_file: str,
        images_file: str,
        invert_images: bool = True,
        neighborhood_shape: str = 'sphere',
        _comments: str = '',
    ) -> None:
        data_points = np.load(vectors_file)
        _logger.info("Loaded data points: %r", data_points.shape)
        super().__init__(data_points, neighborhood_shape=neighborhood_shape)

        self._images = np.load(images_file)
        if invert_images:
            self._images = np.flip(self._images, axis=1)

        assert data_points.shape[0] == self._images.shape[0], (
            "Size of vectors and images do not match."
        )

    @override
    def _make_data_point(self, index: int) -> DataPoint:
        data_point = ImageCard(self._data_points[index], self._images[index, :, :, :])
        return data_point
