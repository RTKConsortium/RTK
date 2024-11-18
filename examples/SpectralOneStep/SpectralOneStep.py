#!/usr/bin/env python
import sys
import itk
from itk import RTK as rtk
import spekpy as sp
import numpy as np
import xraylib as xrl

# Simulation, decomposition and reconstruction parameters
niterations = 10
nsubsets = 4
thresholds = [20.0, 40.0, 60.0, 80.0, 100.0, 120.0]
nbins = len(thresholds) - 1
sdd = 1000.0
sid = 500.0
npix_x = 32
npix_y = npix_x // 8
spacing = 256.0 / npix_x
nmat = 2
origin_x = -0.5 * (npix_x - 1) * spacing
origin_y = -0.5 * (npix_y - 1) * spacing
nproj = int(np.pi / 2.0 * npix_x)
spacing_proj = sdd / sid * spacing
origin_proj_x = sdd / sid * origin_x
origin_proj_y = sdd / sid * origin_y
materials = []
materials.append(
    {
        "material": xrl.GetCompoundDataNISTByIndex(xrl.NIST_COMPOUND_WATER_LIQUID),
        "centers": [[0.0, 0.0, 0.0], [-50.0, 0.0, 0.0], [50.0, 0.0, 0.0]],
        "semi_axes": [[100.0, 0.0, 100.0], [20.0, 0.0, 20.0], [20.0, 0.0, 20.0]],
        "fractions": [1.0, -1.0, -1.0],
        "regularization_weight": 5.0e1,
    }
)
materials.append(
    {
        "material": xrl.GetCompoundDataNISTByIndex(
            xrl.NIST_COMPOUND_BONE_CORTICAL_ICRP
        ),
        "centers": [[50.0, 0.0, 0.0]],
        "semi_axes": [[20.0, 0.0, 20.0]],
        "fractions": [1.0],
        "regularization_weight": 1.0e2,
    }
)
nmat = len(materials)
has_gpu_capability = hasattr(itk, "CudaImage")

# Defines the RTK geometry object using nproj, sdd, sid
geometry = rtk.ThreeDCircularProjectionGeometry.New()
angles = np.linspace(0.0, 360.0, nproj, endpoint=False)
for angle in angles:
    geometry.AddProjection(sid, sdd, angle)
rtk.write_geometry(geometry, "geometry.xml")

# Generate a source spectrum using SkekPy, see https://doi.org/10.1002/mp.14945 and
# assume the same spectrum for all pixels in the projection (no bow-tie, no heel effect, etc.)
s = sp.Spek(kvp=120, th=10.5, z=62.56, dk=1.0, shift=0.5, mas=0.01)
s.filter("Al", 2)
energies = s.get_spectrum()[0]
fluence = s.get_spectrum()[1]
energies = np.pad(energies, [1, 0], constant_values=1)
energies = np.concatenate((energies, np.arange(121, 151)))
nenergies = energies.size
fluence = np.pad(fluence, [1, 30])
SpectrumImageType = itk.Image[itk.F, 3]
spectrum = SpectrumImageType.New()
spectrum.SetRegions([nenergies, npix_x, npix_y])
spectrum.AllocateInitialized()
spectrum_np = itk.array_view_from_image(spectrum)
for i in range(npix_y):
    for j in range(npix_x):
        spectrum_np[i, j, :] = fluence
spectrum.SetSpacing([1.0, spacing_proj, spacing_proj])
spectrum.SetOrigin([0.0, origin_proj_x, origin_proj_y])
itk.imwrite(spectrum, "spectrum.mha")

# Create material images and corresponding gpu_image
DecomposedImageType = itk.Image[itk.Vector[itk.F, nmat], 3]
recon_ref = DecomposedImageType.New()
recon_ref.SetRegions([npix_x, int(npix_y * 1.5), npix_x])
recon_ref.AllocateInitialized()
recon_ref.SetOrigin([origin_x, -0.5 * (int(npix_y * 1.5) - 1) * spacing, origin_x])
recon_ref.SetSpacing([spacing] * 3)
recon_ref_np = itk.array_view_from_image(recon_ref)
ImageType = itk.Image[itk.F, 3]
for i, m in zip(range(nmat), materials):
    mat_ref = rtk.constant_image_source(
        information_from_image=recon_ref, ttype=ImageType
    )
    mat_proj = rtk.constant_image_source(
        origin=[origin_proj_x, origin_proj_y, 0.0],
        size=[npix_x, npix_y, nproj],
        spacing=[spacing_proj] * 3,
        ttype=ImageType,
    )
    for c, a, d in zip(m["centers"], m["semi_axes"], m["fractions"]):
        mat_ref = rtk.draw_ellipsoid_image_filter(mat_ref, center=c, axis=a, density=d)
        mat_proj = rtk.ray_ellipsoid_intersection_image_filter(
            mat_proj, geometry=geometry, center=c, axis=a, density=d
        )
    itk.imwrite(mat_proj, f"mat{i}_proj.mha")
    mat_proj_np = itk.array_view_from_image(mat_proj)
    m["projections"] = mat_proj_np
    recon_ref_np[:, :, :, i] = itk.array_view_from_image(mat_ref)[:, :, :]
itk.imwrite(recon_ref, "recon_ref.mha")

# Detector response matrix
drm = rtk.constant_image_source(
    size=[energies.size, energies.size], spacing=[1.0] * 2, ttype=itk.Image[itk.F, 2]
)
drm_np = itk.array_view_from_image(drm)
for i in range(energies.size):
    drm_np[:i, i] = np.arange(i) / i
itk.imwrite(drm, "drm.mha")

# Image of materials basis (linear attenuation coefficients)
mat_basis = itk.vnl_matrix[itk.F](nenergies, nmat, 0.0)
for e in range(int(np.argwhere(energies == thresholds[0]).item()), energies.size):
    for i, m in zip(range(nmat), materials):
        mat_basis.put(
            e,
            i,
            0.1
            * xrl.CS_Total_CP(m["material"]["name"], energies[e])
            * m["material"]["density"],
        )
mat_basis_np = itk.array_from_vnl_matrix(mat_basis)
itk.imwrite(itk.image_from_array(mat_basis_np), "mat_basis.mha")

# Spectral mixture using line integral of materials and linear attenuation coefficient
# Also record the binned detector response in parallel
CountsImageType = itk.Image[itk.Vector[itk.F, nbins], 3]
counts = CountsImageType.New()
counts.SetRegions([npix_x, npix_y, nproj])
counts.AllocateInitialized()
counts.SetOrigin([origin_proj_x, origin_proj_y, 0.0])
counts.SetSpacing([spacing_proj] * 3)
counts_np = itk.array_view_from_image(counts)
binned_drm = itk.vnl_matrix[itk.F](nbins, nenergies, 0.0)
for t in range(nbins):
    for eimp in range(drm_np.shape[0]):
        # Attenuation at energy eimp
        att = np.zeros(materials[0]["projections"].shape)
        for i, m in zip(range(nmat), materials):
            att += m["projections"] * mat_basis_np[eimp, i]
        att = np.exp(-att)

        # Spectral mixing
        selected_energy_indices = np.argwhere(
            (energies >= thresholds[t]) & (energies <= thresholds[t + 1])
        ).flatten()
        for edet in selected_energy_indices:
            if drm_np[edet, eimp] == 0.0:
                continue

            # Calculate the weight for the trapezoidal rule
            if (
                edet == selected_energy_indices[0]
                or edet == selected_energy_indices[-1]
            ):
                w = 0.5
            else:
                w = 1.0
            counts_np[:, :, :, t] += w * fluence[eimp] * drm_np[edet, eimp] * att

            # Binned detector response
            binned_drm.put(t, eimp, binned_drm(t, eimp) + w * float(drm_np[edet, eimp]))
itk.imwrite(counts, "counts.mha")

# Create initialization for iterative decomposition
zero_recon = rtk.constant_image_source(
    information_from_image=recon_ref, ttype=DecomposedImageType
)
itk.imwrite(zero_recon, "zero_recon.mha")

# Regularization weights
regulWeights = itk.Vector[itk.F, nmat]()
for m in range(nmat):
    regulWeights[m] = materials[m]["regularization_weight"]

# Mask during reconstruction
mask = rtk.constant_image_source(information_from_image=zero_recon, ttype=ImageType)
mask = rtk.draw_ellipsoid_image_filter(
    mask, center=[0.0] * 3, axis=[105.0, 0.0, 105.0], density=1.0
)
itk.imwrite(mask, "mask.mha")

# Spatial regul weights
proj_ones = rtk.constant_image_source(
    origin=[origin_proj_x, origin_proj_y, 0.0],
    size=[npix_x, npix_y, nproj],
    spacing=[spacing_proj] * 3,
    ttype=ImageType,
    constant=1.0,
)
reg_weights = rtk.constant_image_source(
    information_from_image=zero_recon, ttype=ImageType
)
reg_weights = rtk.back_projection_image_filter(
    reg_weights, proj_ones, geometry=geometry
)
reg_weights_np = itk.array_view_from_image(reg_weights)
reg_weights_np[reg_weights_np != 0] = nproj / reg_weights_np[reg_weights_np != 0]
reg_weights_np[reg_weights_np == 0] = nproj
reg_weights = itk.image_from_array(reg_weights_np)
reg_weights.CopyInformation(zero_recon)
itk.imwrite(reg_weights, "reg_weights.mha")

# Callback to write each iteration
iteration_number = 0


def callback():  # write the result before the end of the reconstruction
    global iteration_number
    iteration_number += 1
    itk.image_file_writer(
        mechlem.GetOutput(),
        file_name=f"decomposed_iteration{iteration_number//nsubsets:03d}_subset{iteration_number%nsubsets:01d}.mha",
        ttype=DecomposedImageType,
    )
    print(
        f"Iteration #{iteration_number//nsubsets} subset #{iteration_number%nsubsets} written.",
        end="\r",
    )


# One-step decomposition and reconstruction
if has_gpu_capability:
    zero_recon = itk.cuda_image_from_image(zero_recon)
    counts = itk.cuda_image_from_image(counts)
    spectrum = itk.cuda_image_from_image(spectrum)
    mask = itk.cuda_image_from_image(mask)
MechlemFilterType = rtk.MechlemOneStepSpectralReconstructionFilter[
    type(zero_recon), type(counts), type(spectrum)
]
mechlem = MechlemFilterType.New()
if has_gpu_capability:
    mechlem.SetForwardProjectionFilter(
        MechlemFilterType.ForwardProjectionType_FP_CUDARAYCAST
    )
    mechlem.SetBackProjectionFilter(
        MechlemFilterType.BackProjectionType_BP_CUDAVOXELBASED
    )
else:
    mechlem.SetForwardProjectionFilter(
        MechlemFilterType.ForwardProjectionType_FP_JOSEPH
    )
    mechlem.SetBackProjectionFilter(MechlemFilterType.BackProjectionType_BP_JOSEPH)
mechlem.SetInputMaterialVolumes(zero_recon)
mechlem.SetInputSpectrum(spectrum)
mechlem.SetBinnedDetectorResponse(binned_drm)
mechlem.SetMaterialAttenuations(mat_basis)
mechlem.SetNumberOfIterations(niterations)
mechlem.SetNumberOfSubsets(nsubsets)
mechlem.SetRegularizationRadius(1)
mechlem.SetRegularizationWeights(regulWeights)
mechlem.SetInputPhotonCounts(counts)
mechlem.SetSupportMask(mask)
if has_gpu_capability:
    reg_weights = itk.cuda_image_from_image(reg_weights)
mechlem.SetSpatialRegularizationWeights(reg_weights)
mechlem.SetGeometry(geometry)
mechlem.AddObserver(itk.IterationEvent(), callback)
mechlem.Update()

# GPU to CPU
itk.image_file_writer(
    mechlem.GetOutput(), file_name="decomposed.mha", ttype=DecomposedImageType
)
