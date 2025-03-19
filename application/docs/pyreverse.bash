#! /bin/bash
this_folder=$(dirname $(readlink -f "$0"))
parent_folder=$(dirname ${this_folder})

pushd $parent_folder

default_packages="app.app app.datasource app.dimensionreduction app.eventprocessing app.guiplatform app.nodes app.scenes"

packages=""

output_directory="docs/generated_plantuml"
output_directory_specified=0

options=""

while [[ -n "$@" ]]; do
    case "$1" in
        "--class")
            options="$options --class $2"
            shift
            ;;
        "--output-directory")
            output_directory="$2"
            output_directory_specified=1
            shift
            ;;
        *)
            packages="$@"
            break
            ;;
    esac
    shift
done

if [[ "$options" != "" || "$packages" != "" ]]; then
    if [[ $output_directory_specified -eq 0 ]]; then
        echo "Please specify ``--output-directory`` option, if specifying other options."
        exit 1
    fi
fi

if [[ -z "$packages" ]]; then
    packages="$default_packages"
fi

mkdir -p $output_directory
pyreverse --output plantuml --output-directory $output_directory --colorized --module-names y --show-ancestors 1 --ignore test_api.py,test_simulated.py,test_pca.py,test_visual_node.py $options $packages

popd
