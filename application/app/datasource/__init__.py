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
Module with classes to get data points from (data sources).

Array type: numpy

.. autoapi-inheritance-diagram:: app.datasource.api app.datasource.numpyimages
  app.datasource.simulated
  :parts: 2
"""

import pylog
from pyqoolloop.inspection import autoimport_modules

_logger = pylog.getLogger(__name__)
_logger.setLevel(pylog.WARNING)


autoimport_modules(
    __package__,
    ignore_pattern=r'(test_.*)|(__init__.py)|(api.py)|(__pycache__)|(\..*)',
    logger=_logger,
)
