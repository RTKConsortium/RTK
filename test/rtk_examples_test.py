import sys
import os
import itk
import pytest
import urllib.request
import runpy
from pathlib import Path

# Base examples directory
EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


def run_example(tmp_path, rel_script, *args):
    script = EXAMPLES / rel_script
    # Examples are executed in-process, so restore global state after each run.
    previous_cwd = os.getcwd()
    previous_argv = sys.argv[:]
    try:
        os.chdir(tmp_path)
        sys.argv = [str(script), *map(str, args)]
        runpy.run_path(str(script), run_name="__main__")
    finally:
        os.chdir(previous_cwd)
        sys.argv = previous_argv


@pytest.fixture
def thorax_file(tmp_path):
    url = (
        "https://raw.githubusercontent.com/RTKConsortium/Forbild/refs/heads/main/Thorax"
    )
    thorax_path = tmp_path / "Thorax"
    with urllib.request.urlopen(url) as response, open(thorax_path, "wb") as out_file:
        out_file.write(response.read())
    return thorax_path


# This file wraps the example scripts as pytest tests executed in-process via runpy.
# All outputs are written to a temporary directory using the tmp_path fixture.
if hasattr(itk, "CudaImage"):

    def test_FirstCudaReconstructionExample(tmp_path):
        out_img = tmp_path / "FirstCudaReconstruction.mha"
        out_geom = tmp_path / "FirstCudaReconstruction.xml"
        run_example(
            tmp_path,
            "FirstReconstruction/FirstCudaReconstruction.py",
            out_img,
            out_geom,
        )


def test_InlineReconstructionExample(tmp_path):
    run_example(tmp_path, "InlineReconstruction/InlineReconstruction.py")


def test_AddNoiseExample(tmp_path):
    run_example(tmp_path, "AddNoise/AddNoise.py")


def test_GeometricPhantomExample(tmp_path, thorax_file):
    run_example(tmp_path, "GeometricPhantom/GeometricPhantom.py", thorax_file)


def test_FirstReconstructionExample(tmp_path):
    out_img = tmp_path / "FirstReconstruction.mha"
    out_geom = tmp_path / "FirstReconstruction.xml"
    run_example(
        tmp_path, "FirstReconstruction/FirstReconstruction.py", out_img, out_geom
    )


def test_ConjugateGradient(tmp_path, thorax_file):
    out_img = tmp_path / "ConjugateGradient.mha"
    run_example(
        tmp_path, "ConjugateGradient/ConjugateGradient.py", thorax_file, out_img
    )


@pytest.fixture(scope="session")
def four_d_data_dir(tmp_path_factory):
    data_dir = tmp_path_factory.mktemp("four_d_data")
    run_example(data_dir, "GenerateFourDData/GenerateFourDData.py")
    return data_dir


def test_FourDFDKExample(four_d_data_dir):
    run_example(four_d_data_dir, "FourDFDK/FourDFDK.py")


def test_FourDConjugateGradientExample(four_d_data_dir):
    run_example(four_d_data_dir, "FourDConjugateGradient/FourDConjugateGradient.py")


def test_FourDROOSTERExample(four_d_data_dir):
    run_example(four_d_data_dir, "FourDROOSTER/FourDROOSTER.py")
