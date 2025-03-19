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

"""Test `numpy` module."""

import pytest

import numpy as np

from .mathutils import random_bools, split_float


@pytest.mark.parametrize(
    'value, expected_mantissa, expected_exponent',
    [
        (0, 0, 0),
        (1, 1, 0),
        (10, 1, 1),
        (0.1, 1, -1),
        (2, 2, 0),
        (20, 2, 1),
        (0.2, 2, -1),
        (-1, -1, 0),
        (-10, -1, 1),
        (-0.1, -1, -1),
        (-2, -2, 0),
        (-20, -2, 1),
        (-0.2, -2, -1),
    ],
)
def test__split_float(
    value: float, expected_mantissa: float, expected_exponent: int
) -> None:
    """Test `split_float()`."""
    mantissa, exponent = split_float(value)

    assert expected_mantissa == mantissa
    assert expected_exponent == exponent


@pytest.mark.parametrize(
    'length, true_count',
    [(1, 0), (1, 1), (2, 0), (2, 1), (2, 2), (3, 0), (3, 1), (3, 2), (3, 3)],
)
def test__random_bools(length: int, true_count: int) -> None:
    """Test `random_bools()`."""
    generator = np.random.default_rng(None)

    result = list(random_bools(generator, length, true_count))

    assert len(result) == length

    assert sum(result) == true_count
