#!/usr/bin/env python
import sys
import itk
from itk import RTK as rtk
import spekpy as sp
import numpy as np
import xraylib as xrl

# Simulation, decomposition and reconstruction parameters
thresholds = [20.,40.,60.,80.,119.]
sdd=1000.
sid=500.
spacing=4.
npix=64
nmat=2
origin=-0.5*(npix-1)*spacing
nproj=16
spacing_proj=sdd/sid*spacing
origin_proj=sdd/sid*origin
materials=[]
materials.append({'material': xrl.GetCompoundDataNISTByIndex(xrl.NIST_COMPOUND_WATER_LIQUID),
                  'centers':  [[0.,0.,0.],     [-50.,0.,0.], [50.,0.,0.]],
                  'semi_axes': [[100.,0.,100.], [20.,0.,20.], [20.,0.,20.]],
                  'fractions': [1.,             -1.,          -1.],
                  'init': 100})
materials.append({'material': xrl.GetCompoundDataNISTByIndex(xrl.NIST_COMPOUND_BONE_CORTICAL_ICRP),
                  'centers':  [[50.,0.,0.]],
                  'semi_axes': [[20.,0.,20.]],
                  'fractions': [1.],
                  'init': .1})

# Defines the RTK geometry object using nproj, sdd, sid
geometry = rtk.ThreeDCircularProjectionGeometry.New()
angles = np.linspace(0., 360., nproj, endpoint=False)
for angle in angles:
  geometry.AddProjection(sid, sdd, angle)

# Generate a source spectrum using SkekPy, see https://doi.org/10.1002/mp.14945 and
# assume the same spectrum for all pixels in the projection (no bow-tie, no heel effect, etc.)
s = sp.Spek(kvp=120, th=10.5, z=62.56, dk=1.)
s.filter('Al', 2)
energies = s.get_spectrum()[0]
fluence = s.get_spectrum()[1]
SpectrumImageType = itk.VectorImage[itk.F,2]
spectrum = SpectrumImageType.New()
spectrum.SetRegions([npix,npix])
spectrum.SetVectorLength(energies.size)
spectrum.Allocate()
spectrum_np = itk.array_view_from_image(spectrum)
for i in range(npix):
    for j in range(npix):
        spectrum_np[i,j,:] = fluence
spectrum.SetSpacing([spacing_proj]*2)
spectrum.SetOrigin([origin_proj]*2)
itk.imwrite(spectrum, 'spectrum.mha')

# Create material images and corresponding projections
ImageType = itk.Image[itk.F,3]
for i,m in zip(range(len(materials)), materials):
    mat_ref = rtk.constant_image_source(origin=[origin]*3, size=[npix]*3, spacing=[spacing]*3, ttype=ImageType)
    mat_proj = rtk.constant_image_source(origin=[origin_proj,origin_proj,0.], size=[npix,npix,nproj], spacing=[spacing_proj]*3, ttype=ImageType)
    for c,a,d in zip(m['centers'], m['semi_axes'], m['fractions']):
        mat_ref = rtk.draw_ellipsoid_image_filter(mat_ref, center=c, axis=a, density=d)
        mat_proj = rtk.ray_ellipsoid_intersection_image_filter(mat_proj, geometry=geometry, center=c, axis=a, density=d)
    itk.imwrite(mat_ref, f'mat{i}_ref.mha')
    itk.imwrite(mat_proj, f'mat{i}_proj.mha')
    mat_proj_np = itk.array_view_from_image(mat_proj)
    m['projections'] = mat_proj_np

# Detector response matrix
drm = rtk.constant_image_source(size=[energies.size,energies.size], spacing=[1.]*2, ttype=itk.Image[itk.F,2])
drm_np = itk.array_view_from_image(drm)
for i in range(energies.size):
    # Assumes a perfect photon counting detector
    drm_np[i,i]=1.
itk.imwrite(drm, 'drm.mha')

# Spectral mixture using line integral of materials and linear attenuation coefficient
CountsImageType = itk.VectorImage[itk.F,3]
counts = CountsImageType.New()
counts.SetRegions([npix,npix,nproj])
counts.SetVectorLength(len(thresholds)-1)
counts.Allocate()
counts.SetOrigin([origin_proj,origin_proj,0.])
counts.SetSpacing([spacing_proj]*3)
counts_np = itk.array_view_from_image(counts)
counts_np[:,:,:,:]=0.
for t in range(len(thresholds)-1):
    selected_energy_indices = np.argwhere((energies>=thresholds[t]) & (energies<thresholds[t+1])).flatten()
    for edet in selected_energy_indices:
        att = 0. * materials[0]['projections']
        for m in materials:
            mu_mat = xrl.CS_Total_CP(m['material']['name'], energies[edet]) * m['material']['density']
            mu_mat *= 0.1 # to / mm
            att += m['projections'] * mu_mat
        for eimp in range(drm_np.shape[0]):
            counts_np[:,:,:,t] += fluence[eimp] * drm[eimp, edet] * np.exp(-att)
itk.imwrite(counts, 'counts.mha')

# Create initialization for iterative decomposition
DecomposedImageType = itk.VectorImage[itk.F,3]
decomposed_init = DecomposedImageType.New()
decomposed_init.SetRegions([npix,npix,nproj])
decomposed_init.SetVectorLength(nmat)
decomposed_init.Allocate()
decomposed_init.SetOrigin([origin_proj,origin_proj,0.])
decomposed_init.SetSpacing([spacing_proj]*3)
decomposed_init_np = itk.array_view_from_image(decomposed_init)
for i in range(len(materials)):
    decomposed_init_np[:,:,:,i] = materials[i]['init']
itk.imwrite(decomposed_init, 'decomposed_init.mha')

# Image of materials basis (linear attenuation coefficients)
mat_basis = rtk.constant_image_source(size=[nmat, energies.size], spacing=[1.]*2, ttype=itk.Image[itk.F,2])
mat_basis_np = itk.array_view_from_image(mat_basis)
for e in range(energies.size):
    for i,m in zip(range(len(materials)), materials):
        mat_basis_np[e,i] = xrl.CS_Total_CP(m['material']['name'], energies[e]) * m['material']['density']
        mat_basis_np[e,i] *= 0.1 # to / mm
itk.imwrite(mat_basis, 'mat_basis.mha')

# Thresholds in an itk.VariableLengthVector
thresholds_itk=itk.VariableLengthVector[itk.D](len(thresholds))
for i in range(thresholds_itk.GetSize()):
    thresholds_itk[i] = thresholds[i]

# Projection-based decomposition
decomposed = rtk.simplex_spectral_projections_decomposition_image_filter(input_decomposed_projections=decomposed_init,
                                                                         guess_initialization=False,
                                                                         input_measured_projections=counts,
                                                                         input_incident_spectrum=spectrum,
                                                                         detector_response=drm,
                                                                         material_attenuations=mat_basis,
                                                                         thresholds=thresholds_itk,
                                                                         number_of_iterations=1000,
                                                                         optimize_with_restarts=True,
                                                                         log_transform_each_bin=False,
                                                                         IsSpectralCT=True)
itk.imwrite(decomposed[0], 'decomposed.mha')

# Reconstruct each material image with FDK
for m in range(len(materials)):
    decomp_mat_proj = itk.image_from_array(itk.array_from_image(decomposed[0])[:,:,:,m].copy())
    decomp_mat_proj.CopyInformation(decomposed[0])
    mat_recon = rtk.constant_image_source(origin=[origin]*3, size=[npix]*3, spacing=[spacing]*3, ttype=ImageType)
    mat_recon = itk.fdk_cone_beam_reconstruction_filter(mat_recon, decomp_mat_proj, geometry=geometry)
    itk.imwrite(mat_recon, f'mat{m}_recon.mha')
