#!/usr/bin/env python

import itk
from itk import RTK as rtk
import os
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize

# Script parameters
directory = "output/"
spacing = [0.5] * 3
size = [512, 1, 512]
order = 6
reference_attenuation = 0.02
origin = [-(size[0] / 2 + 0.5) * spacing[0], 0.0, -(size[2] / 2 + 0.5) * spacing[2]]

# Generate test data if output directory does not exist
# The original example requires pre-generated attenuation projection files
# from a prior acquisition step. Here we generate them using the core RTK API.
if not os.path.exists(directory):
    print(f"Generating test data in {directory}...")
    os.makedirs(directory, exist_ok=True)

    numberOfProjections = 360
    sid = 1000
    sdd = 1500

    # Create geometry
    geometry = rtk.ThreeDCircularProjectionGeometry.New()
    for i in range(numberOfProjections):
        angle = i * 360.0 / numberOfProjections
        geometry.AddProjection(sid, sdd, angle)
    rtk.write_geometry(geometry, os.path.join(directory, "geometry.xml"))

    # Create individual projection files using Shepp-Logan phantom
    ImageType = itk.Image[itk.F, 3]
    for i in range(numberOfProjections):
        geo_single = rtk.ThreeDCircularProjectionGeometry.New()
        angle = i * 360.0 / numberOfProjections
        geo_single.AddProjection(sid, sdd, angle)

        proj_source = rtk.constant_image_source(
            size=[size[0], size[2], 1],
            spacing=[spacing[0], spacing[1], spacing[2]],
            ttype=[ImageType],
            origin=[origin[0], origin[1], origin[2]],
        )
        proj = rtk.shepp_logan_phantom_filter(
            proj_source,
            geometry=geo_single,
            phantom_scale=150,
        )
        itk.imwrite(proj, os.path.join(directory, f"attenuation{i:03d}.mha"))
    print(f"Generated {numberOfProjections} projections in {directory}")

# List of filenames
file_names = list()
for file in os.listdir(directory):
    if file.startswith("attenuation") and file.endswith(".mha"):
        file_names.append(directory + file)

if not file_names:
    raise FileNotFoundError(
        f"No attenuation projection files found in {directory}. "
        "Please generate test data first or provide projection files."
    )

# Read in full geometry
geometry = rtk.read_geometry(os.path.join(directory, "geometry.xml"))

# Crate template image
ImageType = itk.Image[itk.F, 3]
projections_reader = rtk.ProjectionsReader[ImageType].New(file_names=file_names)
projections = projections_reader.GetOutput()
constant_image_filter = rtk.ConstantImageSource[ImageType].New(
    origin=origin, spacing=spacing, size=size
)
constant_image = constant_image_filter.GetOutput()
template = rtk.draw_ellipsoid_image_filter(
    constant_image, density=reference_attenuation, axis=[100, 0, 100]
)
itk.imwrite(template, "template.mha")
template = itk.array_from_image(template).flatten()

# Create weights (where the template should match)
weights = rtk.draw_ellipsoid_image_filter(
    constant_image, density=1.0, axis=[125, 0, 125]
)
weights = rtk.draw_ellipsoid_image_filter(weights, density=-1.0, axis=[102, 0, 102])
weights = rtk.draw_ellipsoid_image_filter(weights, density=1.0, axis=[98, 0, 98])
itk.imwrite(weights, "weights.mha")
weights = itk.array_from_image(weights).flatten()

# Create reconstructed images
wpcoeffs = np.zeros(order + 1)
fdks = [None] * (order + 1)
for o in range(0, order + 1):
    wpcoeffs[o - 1] = 0.0
    wpcoeffs[o] = 1.0
    water_precorrection = rtk.water_precorrection_image_filter(
        projections, coefficients=wpcoeffs, in_place=False
    )
    fdk = rtk.fdk_cone_beam_reconstruction_filter(
        constant_image, water_precorrection, geometry=geometry
    )
    itk.imwrite(fdk, f"fdk{o}.mha")
    fdks[o] = itk.array_from_image(fdk).flatten()

# Create and solve the linear system of equation B.c= a to find the coeffs c
a = np.zeros(order + 1)
B = np.zeros((order + 1, order + 1))
for i in range(0, order + 1):
    a[i] = np.sum(weights * fdks[i] * template)
    for j in np.arange(i, order + 1):
        B[i, j] = np.sum(weights * fdks[i] * fdks[j])
        B[j, i] = B[i, j]
c = np.linalg.solve(B, a)

water_precorrection = rtk.water_precorrection_image_filter(projections, coefficients=c)
fdk = rtk.fdk_cone_beam_reconstruction_filter(
    constant_image, water_precorrection, geometry=geometry
)
itk.imwrite(fdk, "fdk.mha")

fdk = itk.imread("fdk.mha")
fdk1 = itk.imread("fdk1.mha")

plt.plot(itk.array_from_image(fdk)[:, 0, 256], label="Corrected")
plt.plot(itk.array_from_image(fdk1)[:, 0, 256], label="Uncorrected")
plt.legend()
plt.xlabel("Pixel number")
plt.ylabel("Attenuation")
plt.xlim(0, 512)
plt.savefig("profile.png")
