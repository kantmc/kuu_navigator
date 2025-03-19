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

"""Defines functions related to linear algebra."""

from sys import float_info

import numpy as np

from ..mathutils import PLANE_DIMENSIONS, NPArray, normalize

_epsilon = float_info.epsilon


def generate_orthonormal_3d(
    first_vector_3d: NPArray,
) -> NPArray:
    """
    Generate 3 vectors that are orthonormal to each other.

    :param first_vector3d: The first 3D vector of the returned vectors.
    :returns: 3 3D vectors that are orthogonormal to each other, inserted row
      by row into a matrix. The first row is `first_vector3d` normalized.
    """
    if abs(np.inner(first_vector_3d, [0, 1, 1])) < _epsilon:
        first_element = np.sign(first_vector_3d[0])
        return np.vstack(
            (np.array([first_element, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1]))
        )

    a, b, c = first_vector_3d
    return np.vstack(
        # `sklearn.preprocessing.normalize()` seems to treat 1.99840144e-15 as 0
        (
            normalize(first_vector_3d),
            normalize(np.array([0, c, -b])),
            normalize(np.array([c * c + b * b, -a * b, -a * c])),
        )
    )


def find_orthonormal_2d(first_vector: NPArray, second_vector: NPArray) -> NPArray:
    """
    Find 2 orthonormal 2D vectors that are closest to two given 2D vectors.

    :param first_vector: One of the given 2D vectors.
    :param second_vector: One of the given 2D vectors.
    :returns: 2 orthonormal vectors that are closest to `first_vector` and
      `second_vector` each in a row in the same order.
    """
    assert len(first_vector) == PLANE_DIMENSIONS
    assert len(second_vector) == PLANE_DIMENSIONS

    if first_vector[0] * second_vector[1] > first_vector[1] * second_vector[0]:
        return np.vstack(
            (
                normalize(
                    np.array(
                        [
                            first_vector[0] + second_vector[1],
                            first_vector[1] - second_vector[0],
                        ]
                    )
                ),
                normalize(
                    np.array(
                        [
                            second_vector[0] - first_vector[1],
                            second_vector[1] + first_vector[0],
                        ]
                    )
                ),
            )
        )

    return np.vstack(
        (
            normalize(
                np.array(
                    [
                        first_vector[0] - second_vector[1],
                        first_vector[1] + second_vector[0],
                    ]
                )
            ),
            normalize(
                np.array(
                    [
                        second_vector[0] + first_vector[1],
                        second_vector[1] - first_vector[0],
                    ]
                )
            ),
        )
    )
