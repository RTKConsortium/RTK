#!/usr/bin/env python
import os
import filecmp
import shutil
import subprocess
import pytest
import itk
from itk import RTK as rtk


@pytest.fixture(scope="session")
def rtk_reference(tmp_path_factory):
    d = tmp_path_factory.mktemp("rtk_reference")
    cwd = os.getcwd()
    os.chdir(d)
    try:
        rtk.rtksimulatedgeometry("-n 180 --sid 600 --sdd 1200 -o geometry.xml")
        rtk.rtkprojectshepploganphantom(
            "-g geometry.xml -o projections.mha --spacing 4 --size 64 --phantomscale 64"
        )
        rtk.rtkdrawshepploganphantom(
            "--spacing 2 --size 64 -o reference.mha --phantomscale 64"
        )
    finally:
        os.chdir(cwd)
    return d


@pytest.fixture()
def rtk_workdir(tmp_path, rtk_reference):
    """Per-test working directory populated with cached geometry and projections."""
    for name in ("geometry.xml", "projections.mha", "reference.mha"):
        src = rtk_reference / name
        dst = tmp_path / name
        os.link(src, dst)

    cwd = os.getcwd()
    os.chdir(tmp_path)
    yield tmp_path
    os.chdir(cwd)


def test_fdk_application(rtk_workdir):
    if hasattr(itk, "CudaImage"):
        rtk.rtkfdk(
            "-g geometry.xml -p . -r projections.mha -o fdk.mha --spacing 2 --size 64 --origin=-64 --hardware cuda"
        )
    else:
        rtk.rtkfdk(
            "-g geometry.xml -p . -r projections.mha -o fdk.mha --spacing 2 --size 64 --origin=-64"
        )
    rtk.rtkcheckimagequality("-i reference.mha -j fdk.mha -t 200000")


def test_application_invocation_modes(rtk_workdir):
    """Validate rtksimulatedgeometry equivalence across invocation styles"""

    # Keyword API baseline
    rtk.rtksimulatedgeometry(
        output="geometry_kw.xml",
        nproj=180,
        sid=600,
        sdd=1200,
    )

    # CLI
    exe = shutil.which("rtksimulatedgeometry")
    cli_cmd = [
        exe,
        "-n",
        "180",
        "--sid",
        "600",
        "--sdd",
        "1200",
        "-o",
        "geometry_cli.xml",
    ]
    subprocess.run(cli_cmd, check=True, capture_output=True)

    assert filecmp.cmp("geometry_kw.xml", "geometry.xml", shallow=False)
    assert filecmp.cmp("geometry_cli.xml", "geometry.xml", shallow=False)


def test_conjugategradient_application(rtk_workdir):
    rtk.rtkconjugategradient(
        "-g geometry.xml -p . -r projections.mha -o cg.mha --spacing 2 --size 64 -n 2"
    )
    rtk.rtkcheckimagequality("-i reference.mha -j cg.mha -t 20000")


def test_admmtotalvariation_application(rtk_workdir):
    rtk.rtkadmmtotalvariation(
        "-g geometry.xml -p . -r projections.mha -o admm.mha --spacing 2 --size 64 -n 1 --CGiter 2"
    )
    rtk.rtkcheckimagequality("-i reference.mha -j admm.mha -t 20000")
