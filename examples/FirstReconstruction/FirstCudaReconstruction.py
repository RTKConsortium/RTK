#!/usr/bin/env python
import os
import sys
import itk
from itk import RTK as rtk

if len ( sys.argv ) < 3:
  print( "Usage: FirstReconstruction <outputimage> <outputgeometry>" )
  sys.exit ( 1 )

# Import Windows CUDA_PATH for dll (required for some Python versions,
# https://docs.python.org/3/whatsnew/3.8.html#bpo-36085-whatsnew)
if sys.platform == 'win32':
  os.add_dll_directory(os.path.join(os.environ['CUDA_PATH'], 'bin'))

# Defines the image type
GPUImageType = rtk.CudaImage[itk.F,3]
CPUImageType = rtk.Image[itk.F,3]

# Defines the RTK geometry object
geometry = rtk.ThreeDCircularProjectionGeometry.New()
numberOfProjections = 360
firstAngle = 0.
angularArc = 360.
sid = 600 # source to isocenter distance
sdd = 1200 # source to detector distance
for x in range(0,numberOfProjections):
  angle = firstAngle + x * angularArc / numberOfProjections
  geometry.AddProjection(sid,sdd,angle)

# Writing the geometry to disk
xmlWriter = rtk.ThreeDCircularProjectionGeometryXMLFileWriter.New()
xmlWriter.SetFilename ( sys.argv[2] )
xmlWriter.SetObject ( geometry );
xmlWriter.WriteFile();

# Create a stack of empty projection images
ConstantImageSourceType = rtk.ConstantImageSource[GPUImageType]
constantImageSource = ConstantImageSourceType.New()
origin = [ -127, -127, 0. ]
sizeOutput = [ 128, 128,  numberOfProjections ]
spacing = [ 2.0, 2.0, 2.0 ]
constantImageSource.SetOrigin( origin )
constantImageSource.SetSpacing( spacing )
constantImageSource.SetSize( sizeOutput )
constantImageSource.SetConstant(0.)

REIType = rtk.RayEllipsoidIntersectionImageFilter[CPUImageType, CPUImageType]
rei = REIType.New()
semiprincipalaxis = [ 50, 50, 50]
center = [ 0, 0, 10]
# Set GrayScale value, axes, center...
rei.SetDensity(2)
rei.SetAngle(0)
rei.SetCenter(center)
rei.SetAxis(semiprincipalaxis)
rei.SetGeometry( geometry )
rei.SetInput(constantImageSource.GetOutput())

# Create reconstructed image
constantImageSource2 = ConstantImageSourceType.New()
sizeOutput = [ 128, 128, 128 ]
origin = [ -63.5, -63.5, -63.5 ]
spacing = [ 1.0, 1.0, 1.0 ]
constantImageSource2.SetOrigin( origin )
constantImageSource2.SetSpacing( spacing )
constantImageSource2.SetSize( sizeOutput )
constantImageSource2.SetConstant(0.)

# Graft the projections to an itk::CudaImage
projections = GPUImageType.New()
rei.Update()
projections.SetPixelContainer(rei.GetOutput().GetPixelContainer())
projections.CopyInformation(rei.GetOutput())
projections.SetBufferedRegion(rei.GetOutput().GetBufferedRegion())
projections.SetRequestedRegion(rei.GetOutput().GetRequestedRegion())

# FDK reconstruction
print("Reconstructing...")
FDKGPUType = rtk.CudaFDKConeBeamReconstructionFilter
feldkamp = FDKGPUType.New()
feldkamp.SetInput(0, constantImageSource2.GetOutput());
feldkamp.SetInput(1, projections);
feldkamp.SetGeometry(geometry);
feldkamp.GetRampFilter().SetTruncationCorrection(0.0);
feldkamp.GetRampFilter().SetHannCutFrequency(0.0);

# Field-of-view masking
FOVFilterType = rtk.FieldOfViewImageFilter[CPUImageType, CPUImageType]
fieldofview = FOVFilterType.New()
fieldofview.SetInput(0, feldkamp.GetOutput())
fieldofview.SetProjectionsStack(rei.GetOutput())
fieldofview.SetGeometry(geometry)

# Writer
print("Writing output image...")
WriterType = rtk.ImageFileWriter[CPUImageType]
writer = WriterType.New();
writer.SetFileName(sys.argv[1]);
writer.SetInput(fieldofview.GetOutput());
writer.Update();

print("Done!")

