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
Algorithms for dimension reduction.

Array type: numpy

.. autoapi-inheritance-diagram:: app.dimensionreduction.algebra
  app.dimensionreduction.api app.dimensionreduction.linear app.dimensionreduction.mds
  app.dimensionreduction.null app.dimensionreduction.pca app.dimensionreduction.weight
  :parts: 2
"""

from io import TextIOWrapper
import json
from pathlib import Path
from typing import Any

import pylog
from pyqoolloop.inspection import autoimport_modules

from .api import (
    DimensionReducer,
)

_logger = pylog.getLogger(__name__)
_logger.setLevel(pylog.WARNING)


__all__ = ['DimensionReducer']


autoimport_modules(
    __package__, ignore_pattern='(test_.*)|(__init__.py)|(api.py)', logger=_logger
)


def _open(path: Path) -> TextIOWrapper:
    return path.open(encoding='utf-8')


def get_json_configuration(
    path: Path,
) -> dict[str, Any]:
    """
    Read parameters from json file.

    :param path: Path to the json file containing the parameters.
    """
    with _open(path) as file:
        configuration = json.load(file)

    assert isinstance(configuration, dict)
    return configuration
