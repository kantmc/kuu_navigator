# Kuu navigator

This application is a GUI tool that visualizes data points in multi-dimensional space.

"Kuu" is Japanese for space and nothing, so named with the implication that it is up to the user to provide meaning to the data that is shown with this tool.

# How to run the application

The current version has only been tested on Mac.

If you get an error such as "configure: error: C compiler cannot create executables", when installing pyenv, install latest Xcode from App Store. Then, install pyenv again.

## For downloaded source code

If the source code was downloaded without using `git` the following needs to be executed in the root directory. The second line needs to be executed for each entry in `.gitmodules`.

1. > git init
1. > git submodule add \<string for url\> \<string for path\>
1. > chmod +x main.p

(Please not that the version of the submodules may not be appropriate, because `.gitmodules` does not record the git hashes.)

### After checking out the repository

This repository includes git submodules. Please don't forget to update the submodules, after cloning and checking out.

(In the root directory)
> git submodule update --init --recursive

## Preparing the Python environment

(In the `application` directory)
1. Install `pyenv` ( https://github.com/pyenv/pyenv#installation )
1. (Remove any backslashes, if any)
   > pyenv install \`cat .python-version\`
1. > pyenv local \`cat .python-version\`
1. > python -m venv .venv
1. > . .venv/bin/activate
1. > pip install --upgrade pip
1. > pip install -r requirements.lock

## Running the application

(In the `application` directory)
> ./main.py --animation-interval 1 @parameters/oriented_pca_swiss_roll_3d_in_5d.txt
(This may take some time on the first launch.)

This is a 3D plane rolled up in 5D space (Swiss Roll).

The argument starting with `@parameters/` specifies parameters saved in the `parameters`
directory.  There are several examples in the directory.

## Cleaning up

When finished, run the following:
> deactivate

## The user interface

### Windows

When the application is launched, there will be two windows:
- The Navigation Window: This window visualizes the data as Visual Nodes. How the Visual Nodes are rendered can be implemented differently for each data source. Navigation in the space is done with commands explained in the next section. A Visual Node can be selected with the mouse. In this case, a Detail Window will open to show the Visual Node close up.
- The Toolbox window: A number of parameters can be set in this window:
  - Target window: This shows which Navigation Window the parameters are for.
  - Visual size: The size of the Visual Nodes. (See `--visual-size` command line option.)
  - Shift speed: How fast movement is for navigation.
  - Rotation speed: How fast rotation is.
  - Screenshot button: Saves screenshots of the Navigation Window to the specified directory.
  - Help button: Shows the key bindings for the navigation commands.
  - Nearest neighbor button: Shows the distance to the nearest sampled neighbor.
  - Display button: Toggles the text information shown in the Navigation Window.
  - New Scene: Opens another Navigation Window.
  - Animation interval: How slowly animation is played.  (See `--animation-interval` command line option.)
  - Number of points: Maximum number of data points to be handled at a time. Could be ignored by some data sources. (See `--number-of-points` command line option.)
  - Neighborhood radius: Radius for local neighborhood. (See `--neighborhood-radius` command line option.)

### Commands

#### Show All ('a' key)

Shows all of the (sampled) data points from different points of view. Points of view are switched by pressing the 'a' key again.

Switches to Global view, if executed in Local view.

#### Bird's Eye View ('b' key; Local view)

Shows the (sampled) data points in the local neighborhood from different points of view that are away from the Navigation Camera.  Points of view are switched by pressing the 'b' key again.

Only valid in Local view.

#### Exit Bird's Eye View ('Esc' key; Local view)

Exits Bird's Eye View and returns to the Navigation Camera (first person view) in the
local neighborhood.

Only valid in Local view.

#### Zoom In ('z' key)

Zoom in to the selected Visual Node.

#### Center Selected Node ('c' key)

Rotate the camera so that the selected Visual Node is centered in the window.

#### Move Forward ('8' key)

Move the Navigation Camera forward.

If a Visual Node is selected in Local view, the Navigation Camera will move toward the selected Visual Node.

#### Rotate Left ('4' key)

Rotate the Navigation Camera to the left.

#### Rotate Right ('6' key)

Rotate the Navigation Camera to the right.

#### Move Back ('2' key)

Shift the Navigation Camera backward.

If a Visual Node is selected in Local view, the Navigation Camera will move away from the selected Visual Node.

#### Move Up ('Shift-8')

Shift the Navigation Camera in the upward direction.

#### Move Left ('Shift-4')

Shift the Navigation Camera to the left.

#### Move Right ('Shift-6')

Shift the Navigation Camera to the right.

#### Move Down ('Shift-2')

Shift the Navigation Camera in the downward direction.

#### Rotate Up ('Ctrl-8')

Rotate the Navigation Camera upward.

#### Rotate Anti-clockwise ('Ctrl-4')

Rotate the Navigation Camera anti-clockwise.

#### Rotate Clockwise ('Ctrl-6')

Rotate the Navigation Camera Clockwise.

#### Rotate Down ('Ctrl-2')

Rotate the Navigation Camera downward.


## Extra

### Sphinx documentation

If generating Sphinx documentation from the source code, GraphViz needs to be installed. ( https://graphviz.org/download/ ) Generating documentation can be done by running `make html` in the `docs` directory.

### Extensions

Custom data sources and dimension reducers can be added with the `--import` command line option. See uses of `app.datasource.api.DataSource` and `app.dimensionreduction.api.DimensionReducer` classes.

### VSCode

If you are bothered by ruff giving warnings for the source code, Set "Ruff: Path" to the executable used in the virtual environment. The path can be obtained by running the following in the `application` directory:
> which ruff

## Potential issues

### Known problems

- Implementation for Multidimensional Scaling (MDS) is provided as reference, but is not feature complete. It does not have some commands that PCA has, such as Move Forward without selection of a Visual Node.
- Resizing Navigation Window can crash app.
- The application does not support the case when there are less than 4 data points.
- Data with less than 3 dimensions are not supported.

### Potentially unintuitive behavior

- Visual Nodes can be unstable during navigation, when the distribution is equal in all directions. This is because PCA will not be able to determine a dominant direction for the principal components.
- A Grid distribution for high-dimensions may seem Gaussian when visualized ( `@parameters/oriented_pca_grid_200d_isotropic` ). This is because projection to lower dimensions (3D) is causing phenomena similar to the law of large numbers.
- The Desk Data Source ( `@parameters/oriented_pca_desk_5d` ) is a simulated data set with two high-dimensional cuboids connected with a 2D plane. It appears that the plane is not connected at the "top of the desk". This is because projection to lower dimensions (3D) is causing phenomena similar to the law of large numbers.
- There could be a case when the number of Visual Nodes after Zoom In becomes 0 with the `GridData` data source. It may be surprising that the selected Visual Node has disappeared. When a spherical neighborhood is carved out of the cubic neighborhood, all of the data points in the original cubic neighborhood could be lost since the volume of the sphere could be much less than the volume of the cube in very high dimensions.


---

Copyright (c) 2025 TOYOTA MOTOR CORPORATION. ALL RIGHTS RESERVED.

[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
