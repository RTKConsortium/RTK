# Build and install the RTK Python package via pip (editable mode).
# Must be run from the RTK source directory.
#
# Prerequisites:
#   - An ITK build tree with Python wrapping enabled (set ITK_DIR)
#   - Optionally CUDA (set RTK_USE_CUDA=ON).
#     When CUDA is enabled, CudaCommon must be findable by ITK.
#
# Usage (from the RTK source directory):
#   ITK_DIR=/path/to/itk-build ./utilities/build_rtk_python.sh
#   ITK_DIR=/path/to/itk-build RTK_USE_CUDA=ON ./utilities/build_rtk_python.sh
#
# All paths below can be overridden via environment variables.

set -euo pipefail

RTK_PIP_BUILD_DIR="${RTK_PIP_BUILD_DIR:-build-pip}"
BUILD_TYPE="${BUILD_TYPE:-Release}"
RTK_USE_CUDA="${RTK_USE_CUDA:-OFF}"

if [ -z "$ITK_DIR" ]; then
  echo "ERROR: ITK_DIR must be set to an ITK build tree with Python wrapping."
  exit 1
fi

# Verify that ITK Python wrapping is installed in site-packages. RTK wrapping
# builds on top of the ITK wrapping infrastructure, so ITK's wrapping shared
# libraries must already be present.
ITK_PYTHON_DIR=$(python -c "import itk; import os; print(os.path.dirname(itk.__file__))")
if ! ls "$ITK_PYTHON_DIR"/_ITKCommonPython*.so >/dev/null 2>&1; then
  echo "ERROR: ITK Python wrapping not found in $ITK_PYTHON_DIR."
  echo "Install ITK with ITK_WRAP_PYTHON=ON and run: cmake --install <itk-build> --component RuntimeLibraries"
  exit 1
fi

# CudaCommon is an ITK remote module. When CUDA support is requested, verify
# that its Python wrapping is installed in site-packages.
if [ "$RTK_USE_CUDA" = "ON" ]; then
  if ! ls "$ITK_PYTHON_DIR"/_CudaCommonPython*.so >/dev/null 2>&1; then
    echo "ERROR: CudaCommon Python wrapping not found in $ITK_PYTHON_DIR."
    echo "Install CudaCommon with CudaCommon_WRAP_PYTHON=ON and run: cmake --install <cudacommon-build> --component RuntimeLibraries"
    exit 1
  fi
fi

# Build the RTK Python wrapping via scikit-build-core.
# WRAP_ITK_INSTALL_COMPONENT_IDENTIFIER tells the CMake install (run during wheel assembly)
# which install component to use, so only the files needed for the Python wheel are installed.
PIP_ARGS=(
  --no-build-isolation
  --no-deps
  --config-settings=build-dir="$RTK_PIP_BUILD_DIR"
  --config-settings=cmake.build-type="$BUILD_TYPE"
  --config-settings=cmake.define.ITK_DIR="$ITK_DIR"
  --config-settings=cmake.define.WRAP_ITK_INSTALL_COMPONENT_IDENTIFIER=PythonWheel
  --config-settings=cmake.define.RTK_WRAP_PYTHON=ON
  --config-settings=cmake.define.RTK_USE_CUDA="$RTK_USE_CUDA"
  --config-settings=cmake.define.ITK_USE_PYTHON_LIMITED_API=OFF
)
echo "==> Installing RTK Python package (editable)"
rm -rf "$RTK_PIP_BUILD_DIR"
python -m pip install -e . -vvv "${PIP_ARGS[@]}"
echo "==> Done"
