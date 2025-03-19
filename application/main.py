#!/usr/bin/env python

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
An application to navigate through visuals placed in 3D space.

Author: Kan Torii
"""

import argparse
from collections.abc import Sequence
from importlib import import_module
from pathlib import Path
from typing import cast

import numpy as np

from panda3d.core import loadPrcFileData

from app.app import App
from app.datasource.api import DataSource
from app.dimensionreduction import get_json_configuration
from app.scenes.configuration import (
    AnimationConfiguration,
    NavigationSceneConfiguration,
    ReducerConfiguration,
)


def _non_negative_float(string: str) -> float:
    value = float(string)
    if value < 0:
        raise argparse.ArgumentTypeError(
            f"Value needs to be a non-negative float: {string}"
        )

    return value


def _positive_float(string: str) -> float:
    value = float(string)
    if value <= 0:
        raise argparse.ArgumentTypeError(
            f"Value needs to be a positive float: {string}"
        )

    return value


def _get_arguments(arguments: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        fromfile_prefix_chars='@',
        description="A visual tool to navigate high-dimensional space.",
    )
    parser.add_argument(
        '--animation-interval',
        type=_non_negative_float,
        default=1.0,
        help=(
            "[Non-negative float] Interval in seconds for animation. Specifying 0"
            " disables animation."
        ),
    )
    parser.add_argument(
        # FUTURE: Might want to make this relative to range of data
        '-i',
        '--visual-size',
        type=_positive_float,
        default=0.1,
        help="Size of each visual node.",
    )
    parser.add_argument(
        '--background-color',
        type=str,
        default="000000",
        help=(
            "Default color (RGB in hexadecimals) of the background. Background color "
            "may change depending on depth (selection of components)."
        ),
    )
    parser.add_argument(
        '-n',
        '--number-of-points',
        type=int,
        default=3000,
        help=(
            "Maximum number of data points to handle at a time. Could be"
            " ignored by some data sources."
        ),
    )
    parser.add_argument(
        '--number-of-detailed-nodes',
        type=int,
        help=(
            "Number of visual nodes to render in detail. All visual nodes will be in "
            "detail, if not specified."
        ),
    )
    parser.add_argument(
        '-r',
        '--neighborhood-radius',
        type=_non_negative_float,
        default=20.0,
        help="Radius for local neighborhood.",
    )
    parser.add_argument(
        '--shared-neighborhood',
        action='store_true',
        help="Make sure to share points while moving around neighborhoods. Note that "
        "this may skew the local distribution, while making the animation look "
        "smoother.",
    )
    parser.add_argument(
        '-s',
        '--data-source',
        required=True,
        help="Path to json file that specifies properties of a simulated data source.",
    )
    parser.add_argument(
        '-g',
        '--dimension-reduction',
        # TODO: '--global-dimension-reduction',
        required=True,
        help=(
            "Path to json file that specifies the Global Mode algorithm to be used for"
            " dimension reduction and its parameters."
        ),
    )
    parser.add_argument(
        '-l',
        '--local-dimension-reduction',
        required=False,
        help=argparse.SUPPRESS,
        # TODO: This is not supported yet.
        # help=(
        #   "Path to json file that specifies the Local Mode algorithm to be used for"
        #   " dimension reduction and its parameters. The same algorithm for Global"
        #   " Mode will be used, if this option is omitted.\n"
        #   " With the current implementation, nonlinear dimension reduction cannot be"
        #   " specified for Global Mode, if Local Mode uses linear dimension reduction."
        #   # FUTURE: Remedy this constraint.
        # ),
    )
    parser.add_argument(
        '--random-seed',
        default=1234,
        type=int,
        help="Seed for random number generator.",
    )
    parser.add_argument(
        '--screenshot-prefix',
        default='screenshots/kuu-navigator',
        help=(
            "Prefix of filename for screenshots. Characters after the last '/' will be"
            " used as the prefix of the name of the file. Directories will be created"
            " as necessary."
        ),
    )
    parser.add_argument(
        '--import',
        dest='import_package',
        help=(
            "Name of package to be imported.\n"
            "New data sources and dimension reducers can be added using this option."
        ),
    )
    parser.add_argument(
        '--inspection-tool',
        action='store_true',
        help=argparse.SUPPRESS,
        # help="Use Inspection Tool for wxPython",
    )
    parser.add_argument(
        '--visual-information',
        action='store_true',
        help="Show Visual Information about dimension reduction.",
    )
    return parser.parse_args(arguments)


def _color_to_tuple(color: str) -> tuple[float, float, float]:
    if len(color) != len("FFFFFF"):
        raise ValueError("Color needs 6 characters.")

    return cast(
        tuple[float, float, float],
        tuple(
            int(color[(2 * index) : (2 * index + 2)], 16) / 255.0 for index in range(3)
        ),
    )


def get_configurations(
    arguments: Sequence[str] | None,
) -> tuple[NavigationSceneConfiguration, DataSource, ReducerConfiguration, bool]:
    """
    Prepare configuration parameters from command line arguments.

    :param arguments: Command line options. Reads from `sys.argv`, if `None`.

    :returns: Tuple containing the following elements:
      - Configuration parameters for the `NavigationScene`.
      - Data Source.
      - Configuration parameters for the dimension reducer.
      - Whether to use the Inspection Tool for wxPython.
    """
    namespace = _get_arguments(arguments=arguments)

    if namespace.import_package is not None:
        import_module(namespace.import_package)

    number_of_detailed_nodes = (
        namespace.number_of_points
        if namespace.number_of_detailed_nodes is None
        else namespace.number_of_detailed_nodes
    )

    enable_animation = namespace.animation_interval > 0
    animation_configuration = AnimationConfiguration(
        node_animation=enable_animation,
        camera_animation=enable_animation,
        animation_interval_secs=namespace.animation_interval,
    )

    scene_configuration = NavigationSceneConfiguration(
        animation=animation_configuration,
        visual_size=namespace.visual_size,
        background_color=_color_to_tuple(namespace.background_color),
        max_number_of_points=namespace.number_of_points,
        number_of_detailed_nodes=number_of_detailed_nodes,
        neighborhood_radius=namespace.neighborhood_radius,
        shared_neighborhood=namespace.shared_neighborhood,
        screenshot_prefix=namespace.screenshot_prefix,
    )

    data_source = DataSource.setup_from_file(Path(namespace.data_source))

    random_generator = np.random.default_rng(seed=namespace.random_seed)

    global_mode = get_json_configuration(Path(namespace.dimension_reduction))
    reducer_configuration = ReducerConfiguration(
        global_mode=global_mode,
        local_mode=(
            global_mode
            if namespace.local_dimension_reduction is None
            else get_json_configuration(Path(namespace.local_dimension_reduction))
        ),
        random_generator=random_generator,
        visual_information=namespace.visual_information,
    )

    return (
        scene_configuration,
        data_source,
        reducer_configuration,
        namespace.inspection_tool,
    )


def _run_app(arguments: Sequence[str] | None = None) -> None:
    """Run the app with the specified command line options."""
    scene_configuration, data_source, reducer_configuration, inspection_tool = (
        get_configurations(arguments)
    )

    app = App(
        scene_configuration,
        data_source,
        reducer_configuration,
        inspection_tool=inspection_tool,
    )

    app.run()


if __name__ == "__main__":
    loadPrcFileData(
        '',
        """audio-library-name\n
        gl-check-errors #f\n
        screenshot-filename %~p%Y-%m-%dT%H-%M-%SF%~f.%~e\n
        wx-main-loop #f""",
    )  # disable audio to avoid errors when starting up

    _run_app()
