#!/usr/bin/env python
import itk
from itk import RTK as rtk


def test_MaximumIntensity():
    TImageType = itk.Image[itk.F, 3]
    # Defines the RTK geometry object
    geometry = rtk.ThreeDCircularProjectionGeometry.New()
    numberOfProjections = 1
    geometry.AddProjection(700, 800, 0)
    # Constant image sources
    # Create MIP Forward Projector volume
    volInput = rtk.ConstantImageSource[TImageType].New()
    origin = [0.0, 0.0, 0.0]
    size = [64, 64, 64]
    sizeOutput = [200, 200, numberOfProjections]
    spacing = [4.0, 4.0, 4.0]
    spacingOutput = [1.0, 1.0, 1.0]
    volInput.SetOrigin(origin)
    volInput.SetSpacing(spacing)
    volInput.SetSize(size)
    volInput.SetConstant(1.0)
    volInput.UpdateOutputInformation()
    volInputSource = volInput.GetOutput()
    # Initialization Imager volume
    projInput = rtk.ConstantImageSource[TImageType].New()
    projInput.SetOrigin(origin)
    projInput.SetSpacing(spacingOutput)
    projInput.SetSize(sizeOutput)
    projInput.SetConstant(0.0)
    projInput.Update()
    projInputSource = projInput.GetOutput()
    # MIP Forward Projection filter
    mip = rtk.MaximumIntensityProjectionImageFilter[TImageType, TImageType].New()
    mip.SetGeometry(geometry)
    mip.SetInput(volInputSource)
    mip.SetInput(1, projInputSource)
    mipImage = mip.GetOutput()
