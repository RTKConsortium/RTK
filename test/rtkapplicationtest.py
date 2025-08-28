#!/usr/bin/env python
import os
import shutil
import subprocess
import pytest
from itk import RTK as rtk


@pytest.fixture(scope="session")
def rtk_reference(tmp_path_factory):
	d = tmp_path_factory.mktemp("rtk_reference")
	cwd = os.getcwd()
	os.chdir(d)
	try:
		if not os.path.exists("geometry.xml"):
			rtk.rtksimulatedgeometry("-n 180 --sid 600 --sdd 1200 -o geometry.xml")
		if not os.path.exists("projections.mha"):
			rtk.rtkprojectshepploganphantom("-g geometry.xml -o projections.mha --spacing 4 --size 64 --phantomscale 64")
		if not os.path.exists("reference.mha"):
			rtk.rtkdrawshepploganphantom("--spacing 2 --size 64 -o reference.mha --phantomscale 64")
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
	rtk.rtkfdk("-g geometry.xml -p . -r projections.mha -o fdk.mha --spacing 2 --size 64 --origin=-64")
	rtk.rtkcheckimagequality("-i reference.mha -j fdk.mha -t 200000")


def test_application_invocation_modes(rtk_workdir):
	"""Validate rtkfdk equivalence across invocation styles"""

	# Keyword API baseline
	rtk.rtkfdk(
		path=".",
		regexp="projections.mha",
		output="fdk_kw.mha",
		geometry="geometry.xml",
		spacing="2,2,2",
		size=[64, 64, 64],
	)

	# String API
	rtk.rtkfdk("-p . -r projections.mha -o fdk_str.mha -g geometry.xml --spacing 2 --size 64")

	# CLI
	exe = shutil.which("rtkfdk")
	cli_cmd = [exe, "-p", ".", "-r", "projections.mha", "-o", "fdk_cli.mha", "-g", "geometry.xml", "--spacing", "2", "--size", "64"]
	subprocess.run(cli_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

	rtk.rtkcheckimagequality("-i fdk_kw.mha -j fdk_str.mha -t 10")
	rtk.rtkcheckimagequality("-i fdk_kw.mha -j fdk_cli.mha -t 10")
	rtk.rtkcheckimagequality("-i fdk_str.mha -j fdk_cli.mha -t 10")


def test_conjugategradient_application(rtk_workdir):
	rtk.rtkconjugategradient("-g geometry.xml -p . -r projections.mha -o cg.mha --spacing 2 --size 64 -n 2")
	rtk.rtkcheckimagequality("-i reference.mha -j cg.mha -t 20000")


def test_admmtotalvariation_application(rtk_workdir):
	rtk.rtkadmmtotalvariation("-g geometry.xml -p . -r projections.mha -o admm.mha --spacing 2 --size 64 -n 1 --CGiter 2")
	rtk.rtkcheckimagequality("-i reference.mha -j admm.mha -t 20000")
