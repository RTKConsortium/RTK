#!/usr/bin/env python
import sys
import itk
from itk import RTK as rtk
import spekpy as sp
import numpy as np
import xraylib as xrl

# Simulation, decomposition and reconstruction parameters
niterations = 1000
thresholds = [20.,40.,60.,80.,100.,120.]
nbins = len(thresholds)-1
sdd=1000.
sid=500.
npix=64
spacing=256./npix
nmat=2
origin=-0.5*(npix-1)*spacing
nproj=int(np.pi/2.*npix)
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
xmlWriter = rtk.ThreeDCircularProjectionGeometryXMLFileWriter.New()
xmlWriter.SetFilename ( 'geometry.xml' )
xmlWriter.SetObject ( geometry )
xmlWriter.WriteFile()


# Generate a source spectrum using SkekPy, see https://doi.org/10.1002/mp.14945 and
# assume the same spectrum for all pixels in the projection (no bow-tie, no heel effect, etc.)
s = sp.Spek(kvp=120, th=10.5, z=62.56, dk=1., shift=0.5)
s.filter('Al', 2)
energies = s.get_spectrum()[0]
fluence = s.get_spectrum()[1]
energies = np.pad(energies, [1,0], constant_values=1)
energies = np.concatenate((energies, np.arange(121,151)))
nenergies = energies.size
fluence = np.pad(fluence, [1,30])
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
for i,m in zip(range(nmat), materials):
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
counts.SetVectorLength(nbins)
counts.AllocateInitialized()
counts.SetOrigin([origin_proj,origin_proj,0.])
counts.SetSpacing([spacing_proj]*3)
counts_np = itk.array_view_from_image(counts)
for t in range(nbins):
    selected_energy_indices = np.argwhere((energies>=thresholds[t]) & (energies<=thresholds[t+1])).flatten()
    for edet in selected_energy_indices:
        att = 0. * materials[0]['projections']
        for m in materials:
            mu_mat = xrl.CS_Total_CP(m['material']['name'], energies[edet]) * m['material']['density']
            mu_mat *= 0.1 # to / mm
            att += m['projections'] * mu_mat
        att = np.exp(-att)
        for eimp in range(drm_np.shape[0]):
            if drm[eimp, edet] == 0.:
                continue
            # Calculate the weight for the trapezoidal rule
            if edet==selected_energy_indices[0] or edet==selected_energy_indices[-1]:
                w=0.5
            else:
                w=1.
            counts_np[:,:,:,t] += w * fluence[eimp] * drm[eimp, edet] * att
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
for i in range(nmat):
    decomposed_init_np[:,:,:,i] = materials[i]['init']
itk.imwrite(decomposed_init, 'decomposed_init.mha')

# Image of materials basis (linear attenuation coefficients)
mat_basis = rtk.constant_image_source(size=[nmat, energies.size], spacing=[1.]*2, ttype=itk.Image[itk.F,2])
mat_basis_np = itk.array_view_from_image(mat_basis)
for e in range(energies.size):
    for i,m in zip(range(nmat), materials):
        mat_basis_np[e,i] = xrl.CS_Total_CP(m['material']['name'], energies[e]) * m['material']['density']
        mat_basis_np[e,i] *= 0.1 # to / mm
itk.imwrite(mat_basis, 'mat_basis.mha')

# Thresholds in an itk.VariableLengthVector
thresholds_itk=itk.VariableLengthVector[itk.D](nbins+1)
for i in range(thresholds_itk.GetSize()):
    thresholds_itk[i] = thresholds[i]

# Spectral mixture using line integral of materials and linear attenuation coefficient
DecomposedImageType = itk.VectorImage[itk.F,3]
decomposed_ref = DecomposedImageType.New()
decomposed_ref.SetRegions([npix,npix,nproj])
decomposed_ref.SetVectorLength(nmat)
decomposed_ref.Allocate()
decomposed_ref.SetOrigin([origin_proj,origin_proj,0.])
decomposed_ref.SetSpacing([spacing_proj]*3)
decomposed_ref_np = itk.array_view_from_image(decomposed_ref)
for i,m in zip(range(nmat), materials):
    decomposed_ref_np[:,:,:,i] = m['projections']
itk.imwrite(decomposed_ref, 'decomposed_ref.mha')

CountsImageType = itk.VectorImage[itk.F,3]
counts_forward = CountsImageType.New()
counts_forward.SetRegions([npix,npix,nproj])
counts_forward.SetVectorLength(nbins)
counts_forward.AllocateInitialized()
counts_forward.SetOrigin([origin_proj,origin_proj,0.])
counts_forward.SetSpacing([spacing_proj]*3)
counts_forward = rtk.spectral_forward_model_image_filter(input_decomposed_projections=decomposed_ref,
                                                         input_measured_projections=counts_forward,
                                                         input_incident_spectrum=spectrum,
                                                         detector_response=drm,
                                                         material_attenuations=mat_basis,
                                                         number_of_energies=energies.size,
                                                         number_of_materials=nmat,
                                                         thresholds=thresholds_itk,
                                                         IsSpectralCT=True)
itk.imwrite(counts_forward[0], 'counts_forward.mha')

# Projection-based decomposition
decomposed = rtk.simplex_spectral_projections_decomposition_image_filter(input_decomposed_projections=decomposed_init,
                                                                         guess_initialization=False,
                                                                         input_measured_projections=counts,
                                                                         input_incident_spectrum=spectrum,
                                                                         detector_response=drm,
                                                                         material_attenuations=mat_basis,
                                                                         thresholds=thresholds_itk,
                                                                         number_of_iterations=niterations,
                                                                         optimize_with_restarts=True,
                                                                         log_transform_each_bin=False,
                                                                         IsSpectralCT=True)
itk.imwrite(decomposed[0], 'decomposed.mha')

# Reconstruct each material image with FDK
for m in range(nmat):
    decomp_mat_proj = itk.image_from_array(itk.array_from_image(decomposed[0])[:,:,:,m].copy())
    decomp_mat_proj.CopyInformation(decomposed[0])
    mat_recon = rtk.constant_image_source(origin=[origin]*3, size=[npix]*3, spacing=[spacing]*3, ttype=ImageType)
    mat_recon = itk.fdk_cone_beam_reconstruction_filter(mat_recon, decomp_mat_proj, geometry=geometry)
    itk.imwrite(mat_recon, f'mat{m}_recon.mha')
