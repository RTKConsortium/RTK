#!/usr/bin/env python
import sys
import itk
from itk import RTK as rtk
import spekpy as sp
import numpy as np
import xraylib as xrl

# Simulation, decomposition and reconstruction parameters
niterations = 100
nsubsets = 4
thresholds = [20.,40.,60.,80.,100.,120.]
nbins = len(thresholds)-1
sdd=1000.
sid=500.
npix=128
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
                  'regularization_weight': 1.e3})
materials.append({'material': xrl.GetCompoundDataNISTByIndex(xrl.NIST_COMPOUND_BONE_CORTICAL_ICRP),
                  'centers':  [[50.,0.,0.]],
                  'semi_axes': [[20.,0.,20.]],
                  'fractions': [1.],
                  'regularization_weight': 1.e3})
nmat=len(materials)

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
SpectrumImageType = itk.Image[itk.F,3]
spectrum = SpectrumImageType.New()
spectrum.SetRegions([nenergies,npix,npix])
spectrum.AllocateInitialized()
spectrum_np = itk.array_view_from_image(spectrum)
for i in range(npix):
    for j in range(npix):
        spectrum_np[i,j,:] = fluence
spectrum.SetSpacing([1.,spacing_proj,spacing_proj])
spectrum.SetOrigin([0.,origin_proj,origin_proj])
itk.imwrite(spectrum, 'spectrum.mha')

# Create material images and corresponding gpu_image
DecomposedImageType = itk.Image[itk.Vector[itk.F,nmat],3]
decomposed_ref = DecomposedImageType.New()
decomposed_ref.SetRegions([npix,int(npix*1.5),npix])
decomposed_ref.AllocateInitialized()
decomposed_ref.SetOrigin([origin,origin*1.5,origin])
decomposed_ref.SetSpacing([spacing]*3)
decomposed_ref_np = itk.array_view_from_image(decomposed_ref)
ImageType = itk.Image[itk.F,3]
for i,m in zip(range(nmat), materials):
    mat_ref = rtk.constant_image_source(information_from_image=decomposed_ref, ttype=ImageType)
    mat_proj = rtk.constant_image_source(origin=[origin_proj,origin_proj,0.], size=[npix,npix,nproj], spacing=[spacing_proj]*3, ttype=ImageType)
    for c,a,d in zip(m['centers'], m['semi_axes'], m['fractions']):
        mat_ref = rtk.draw_ellipsoid_image_filter(mat_ref, center=c, axis=a, density=d)
        mat_proj = rtk.ray_ellipsoid_intersection_image_filter(mat_proj, geometry=geometry, center=c, axis=a, density=d)
    itk.imwrite(mat_proj, f'mat{i}_proj.mha')
    mat_proj_np = itk.array_view_from_image(mat_proj)
    m['projections'] = mat_proj_np
    decomposed_ref_np[:,:,:,i] = itk.array_view_from_image(mat_ref)[:,:,:]
itk.imwrite(decomposed_ref, 'decomposed_ref.mha')

# Detector response matrix
drm = rtk.constant_image_source(size=[energies.size,energies.size], spacing=[1.]*2, ttype=itk.Image[itk.F,2])
drm_np = itk.array_view_from_image(drm)
for i in range(energies.size):
    # Assumes a perfect photon counting detector
    drm_np[i,i]=1.
itk.imwrite(drm, 'drm.mha')

# Spectral mixture using line integral of materials and linear attenuation coefficient
# Also record the binned detector response in parallel
CountsImageType = itk.Image[itk.Vector[itk.F,nbins],3]
counts = CountsImageType.New()
counts.SetRegions([npix,npix,nproj])
counts.AllocateInitialized()
counts.SetOrigin([origin_proj,origin_proj,0.])
counts.SetSpacing([spacing_proj]*3)
counts_np = itk.array_view_from_image(counts)
binned_drm = itk.vnl_matrix[itk.F](nbins, nenergies, 0.)
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
            binned_drm.put(t, eimp, binned_drm(t,eimp) + w*drm[eimp, edet])

itk.imwrite(itk.image_from_array(itk.array_from_vnl_matrix(binned_drm)), 'binned_drm.mha')
itk.imwrite(counts, 'counts.mha')

# Create initialization for iterative decomposition
decomposed_init = rtk.constant_image_source(information_from_image=decomposed_ref, ttype=DecomposedImageType)
itk.imwrite(decomposed_init, 'decomposed_init.mha')

# Image of materials basis (linear attenuation coefficients)
mat_basis = itk.vnl_matrix[itk.F](nenergies, nmat, 0.)
for e in range(int(np.argwhere(energies==thresholds[0])), energies.size):
    for i,m in zip(range(nmat), materials):
        mat_basis.put(e, i, 0.1 * xrl.CS_Total_CP(m['material']['name'], energies[e]) * m['material']['density'])
itk.imwrite(itk.image_from_array(itk.array_from_vnl_matrix(mat_basis)), 'mat_basis.mha')

# Regularization weights
regulWeights = itk.Vector[itk.F, nmat]()
for m in range(nmat):
    regulWeights[m] = materials[i]['regularization_weight']

# Mask during reconstruction
mask = rtk.constant_image_source(information_from_image=decomposed_init, ttype=ImageType)
mask = rtk.draw_ellipsoid_image_filter(mask, center=[0.]*3, axis=[105.,0.,105.], density=1.)
itk.imwrite(mask, 'mask.mha')

# Spatial regul weights
proj_ones = rtk.constant_image_source(origin=[origin_proj,origin_proj,0.], size=[npix,npix,nproj], spacing=[spacing_proj]*3, ttype=ImageType, constant=1.)
regularization_weights = rtk.constant_image_source(information_from_image=decomposed_init, ttype=ImageType)
regularization_weights = rtk.back_projection_image_filter(regularization_weights, proj_ones, geometry=geometry)
regularization_weights_np = itk.array_view_from_image(regularization_weights)
regularization_weights_np = nproj / regularization_weights_np
regularization_weights_np = np.nan_to_num(regularization_weights_np, nan=nproj, posinf=nproj)
regularization_weights = itk.image_from_array(regularization_weights_np)
regularization_weights.CopyInformation(decomposed_init)
itk.imwrite(regularization_weights, 'regularization_weights.mha')

# Function to convert CPU to GPU images
def cpu_to_gpu_image(cpu_image):
    dimension = cpu_image.GetImageDimension()
    pixel_type = itk.F
    vector_length = cpu_image.GetNumberOfComponentsPerPixel()
    if vector_length>1:
        pixel_type = itk.Vector[itk.F, vector_length]
    gpu_image_type = itk.CudaImage[pixel_type, dimension].New()
    gpu_image = gpu_image_type.New()
    gpu_image.SetPixelContainer(cpu_image.GetPixelContainer())
    gpu_image.CopyInformation(cpu_image)
    gpu_image.SetBufferedRegion(cpu_image.GetBufferedRegion())
    gpu_image.SetRequestedRegion(cpu_image.GetRequestedRegion())
    return gpu_image

# Function to convert CPU to GPU images
def gpu_to_cpu_image(gpu_image):
    dimension = gpu_image.GetImageDimension()
    pixel_type = itk.F
    vector_length = gpu_image.GetNumberOfComponentsPerPixel()
    if vector_length>1:
        pixel_type = itk.Vector[itk.F, vector_length]
    cpu_image_type = itk.Image[pixel_type, dimension].New()
    cpu_image = cpu_image_type.New()
    cpu_image.SetPixelContainer(gpu_image.GetPixelContainer())
    cpu_image.CopyInformation(gpu_image)
    cpu_image.SetBufferedRegion(gpu_image.GetBufferedRegion())
    cpu_image.SetRequestedRegion(gpu_image.GetRequestedRegion())
    return cpu_image

# Callback to write each iteration
iteration_number = 0
def callback(): # write the result before the end of the reconstruction
    global iteration_number
    iteration_number += 1
    itk.imwrite(gpu_to_cpu_image(mechlem.GetOutput()), f'decomposed_iteration{iteration_number//nsubsets:03d}_subset{iteration_number%nsubsets:01d}.mha')
    print(f'Iteration #{iteration_number} written.', end='\r')

# One-step decomposition and reconstruction
decomposed_init_gpu = cpu_to_gpu_image(decomposed_init)
counts_gpu = cpu_to_gpu_image(counts)
spectrum_gpu = cpu_to_gpu_image(spectrum)
MechlemFilterType = rtk.MechlemOneStepSpectralReconstructionFilter[type(decomposed_init_gpu), type(counts_gpu), type(spectrum_gpu)]
mechlem = MechlemFilterType.New()
mechlem.SetForwardProjectionFilter(MechlemFilterType.ForwardProjectionType_FP_CUDARAYCAST)
#mechlem.SetForwardProjectionFilter(MechlemFilterType.ForwardProjectionType_FP_JOSEPH)
mechlem.SetBackProjectionFilter(MechlemFilterType.BackProjectionType_BP_CUDAVOXELBASED)
#mechlem.SetBackProjectionFilter(MechlemFilterType.BackProjectionType_BP_JOSEPH)
mechlem.SetInputMaterialVolumes( decomposed_init_gpu )
mechlem.SetInputSpectrum( spectrum_gpu )
mechlem.SetBinnedDetectorResponse(binned_drm)
mechlem.SetMaterialAttenuations( mat_basis )
mechlem.SetNumberOfIterations( niterations )
mechlem.SetNumberOfSubsets( nsubsets )
mechlem.SetRegularizationRadius( 1 )
mechlem.SetRegularizationWeights( regulWeights )
mechlem.SetInputPhotonCounts( counts_gpu )
mechlem.SetSupportMask( cpu_to_gpu_image(mask) )
mechlem.SetSpatialRegularizationWeights( cpu_to_gpu_image(regularization_weights) )
mechlem.SetGeometry(geometry)
mechlem.AddObserver(itk.IterationEvent(), callback)
mechlem.Update()

# GPU to CPU
itk.imwrite(gpu_to_cpu_image(mechlem.GetOutput()), 'decomposed.mha')
