#!/usr/bin/env bash

# Install RTK's Python packages, by CMake, into DEST_DIR (a folder holding the
# ITK Python package). Symlinks hand-written Python to the source (live edits)
# and makes DEST_DIR importable via a .pth. Does not build ITK; ITK_DIR is the
# ITK build tree (with Python wrapping). When RTK_USE_CUDA=ON, CudaCommon's
# wrapping must already be in DEST_DIR.
# Set inputs via environment variables, e.g.:
#   ITK_DIR=/path/to/itk/build DEST_DIR=/path/to/python_packages \
#   rtk_install_python_packages.sh
# Run it from the RTK source directory. Required: ITK_DIR, DEST_DIR.

set -e

RTK_USE_CUDA=${RTK_USE_CUDA:-OFF}
PYTHON=${PYTHON:-python3}
NTHREADS=${NTHREADS:-$(nproc)}
RTK_BUILD_DIR=${RTK_BUILD_DIR:-$PWD/build-rtk-python}

ITK_DIR=${ITK_DIR:?"ITK_DIR must be set (ITK build tree with Python wrapping)"}
DEST_DIR=${DEST_DIR:?"DEST_DIR must be set (folder that already holds the itk package)"}
RTK_SRC=$PWD

echo "ITK build tree: $ITK_DIR"
echo "RTK source:     $RTK_SRC"
echo "RTK build dir:  $RTK_BUILD_DIR"
echo "Install folder: $DEST_DIR"
echo "Python:         $PYTHON"
echo "RTK_USE_CUDA:   $RTK_USE_CUDA"

# The Python wrapping must already be installed in DEST_DIR.
if [ ! -f "$DEST_DIR/itk/__init__.py" ]; then
  echo "Error: no Python wrapping found in $DEST_DIR/itk. Configure ITK with" >&2
  echo "-DITK_WRAP_PYTHON=ON -DPY_SITE_PACKAGES_PATH=$DEST_DIR, build it, then" >&2
  echo "install the wrapping into $DEST_DIR with:" >&2
  echo "  cmake --install $ITK_DIR --component RuntimeLibraries" >&2
  echo "and run this script again." >&2
  exit 1
fi

# Building RTK (an external ITK module) requires an ITK build tree.
if [ ! -f "$ITK_DIR/ITKConfig.cmake" ]; then
  echo "Error: $ITK_DIR is not an ITK build tree (missing ITKConfig.cmake)." >&2
  exit 1
fi

# With CUDA, CudaCommon's wrapping must already be in DEST_DIR.
if [ "$RTK_USE_CUDA" = "ON" ]; then
  if [ ! -f "$DEST_DIR/itk/__init_cudacommon__.py" ]; then
    echo "Error: RTK_USE_CUDA=ON but no CudaCommon Python wrapping in $DEST_DIR/itk." >&2
    echo "Build CudaCommon (inside ITK with -DModule_CudaCommon=ON, or standalone" >&2
    echo "against $ITK_DIR) with -DPY_SITE_PACKAGES_PATH=$DEST_DIR, then install" >&2
    echo "its wrapping with:" >&2
    echo "  cmake --install <itk-or-cudacommon-build-dir> --component RuntimeLibraries" >&2
    echo "and run this script again." >&2
    exit 1
  fi
fi

# Configure, build and install RTK. PY_SITE_PACKAGES_PATH merges the wrapping
# into the itk package in DEST_DIR; RuntimeLibraries matches the itk install.
cmake -S "$RTK_SRC" -B "$RTK_BUILD_DIR" \
  -DITK_DIR="$ITK_DIR" \
  -DRTK_USE_CUDA="$RTK_USE_CUDA" \
  -DRTK_BUILD_APPLICATIONS=OFF \
  -DBUILD_TESTING=OFF \
  -DPY_SITE_PACKAGES_PATH="$DEST_DIR"
cmake --build "$RTK_BUILD_DIR" -j "$NTHREADS"
cmake --install "$RTK_BUILD_DIR" --component RuntimeLibraries

# Symlink hand-written Python (apps and support modules) to the source so edits
# are picked up without reinstalling; generated wrapping still needs a rebuild.
for src in \
    "$RTK_SRC/wrapping/__init_rtk__.py" \
    "$RTK_SRC/wrapping/rtkExtras.py" \
    "$RTK_SRC/applications/rtkargumentparser.py" \
    "$RTK_BUILD_DIR/Wrapping/Generators/Python/rtkConfig.py" \
    "$RTK_SRC"/applications/rtk*_group.py \
    "$RTK_SRC"/applications/rtk*/rtk*.py; do
  [ -f "$src" ] || continue
  ln -sf "$src" "$DEST_DIR/itk/$(basename "$src")"
done
echo "RTK Python modules in $DEST_DIR/itk are symlinked to the RTK source tree, so their edits are taken into account immediately."

# Make DEST_DIR importable for $PYTHON via a .pth in its site-packages.
SITE_PACKAGES=$("$PYTHON" -c "import site; print(site.getsitepackages()[0])")
PTH_FILE="$SITE_PACKAGES/rtk_python_packages.pth"
echo "$DEST_DIR" >> "$PTH_FILE"

echo "Done. $DEST_DIR was added to PYTHONPATH through $PTH_FILE, so no manual export is needed for $PYTHON."
echo "Run the RTK applications as Python modules, e.g.:"
echo "  python -m itk.rtkfdk -g geometry.xml --path . --regexp *.mha -o output.mha"
