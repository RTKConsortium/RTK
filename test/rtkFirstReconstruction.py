#!/usr/bin/env python
from __future__ import print_function
import itk
from itk import RTK as rtk
import sys
import os

if len ( sys.argv ) < 2:
  print( "Usage: rtkFirstReconstruction.py <output.mha>" )
  sys.exit ( 1 )

TImageType = itk.Image[itk.F,3]

# Defines the RTK geometry object
geometry = rtk.ThreeDCircularProjectionGeometry.New()
numberOfProjections = 360
firstAngle = 0
angularArc = 360
sid = 600 # source to isocenter distance in mm
sdd = 1200 # source to detector distance in mm
isox = 0 # X coordinate on the projection image of isocenter
isoy = 0 # Y coordinate on the projection image of isocenter
for x in range(0,numberOfProjections):
  angle = firstAngle + x * angularArc / numberOfProjections
  geometry.AddProjection(sid,sdd,angle,isox,isoy)

constantImageSource = rtk.ConstantImageSource[TImageType].New()
origin = [ -127.5, -127.5, 0. ]
sizeOutput = [ 256, 256,  numberOfProjections ]
spacing = [ 1.0, 1.0, 1.0 ]
constantImageSource.SetOrigin( origin )
constantImageSource.SetSpacing( spacing )
constantImageSource.SetSize( sizeOutput )
constantImageSource.SetConstant(0.0)
source = constantImageSource.GetOutput()

rei = rtk.RayEllipsoidIntersectionImageFilter[TImageType, TImageType].New()
semiprincipalaxis = [ 50, 50, 50]
center = [ 0, 0, 0]
# Set GrayScale value, axes, center...
rei.SetDensity(20)
rei.SetAngle(0)
rei.SetCenter(center)
rei.SetAxis(semiprincipalaxis)
rei.SetGeometry( geometry )
rei.SetInput(source)
reiImage = rei.GetOutput()

# Create reconstructed image
constantImageSource2 = rtk.ConstantImageSource[TImageType].New()
origin = [ -63.5, -63.5, -63.5 ]
sizeOutput = [ 128, 128, 128 ]
constantImageSource2.SetOrigin( origin )
constantImageSource2.SetSpacing( spacing )
constantImageSource2.SetSize( sizeOutput )
constantImageSource2.SetConstant(0.0)
source2 = constantImageSource2.GetOutput()

print("Performing reconstruction")
feldkamp = rtk.FDKConeBeamReconstructionFilter[TImageType].New()
feldkamp.SetGeometry( geometry )
#feldkamp.SetTruncationCorrection(0.0)
#feldkamp.SetHannCutFrequency(0.0)
feldkamp.SetInput(0, source2)
feldkamp.SetInput(1, reiImage)
image = feldkamp.GetOutput()

print("Masking field-of-view")
fov = rtk.FieldOfViewImageFilter[TImageType, TImageType].New()
fov.SetGeometry(geometry)
fov.SetProjectionsStack(reiImage)
fov.SetInput(image)
image = fov.GetOutput()

writer = itk.ImageFileWriter[TImageType].New()
writer.SetFileName ( sys.argv[1] )
writer.SetInput(image)
writer.Update()
