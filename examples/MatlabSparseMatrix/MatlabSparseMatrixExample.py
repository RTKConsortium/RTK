#!/usr/bin/env python

import itk
import numpy as np

# Define types
OutputPixelType = itk.F
Dimension = 3
OutputImageType = itk.Image[OutputPixelType, Dimension]

# Parameters
numberOfProjections = 10
sid = 600  # source to isocenter distance
sdd = 1200  # source to detector distance

# Set up the geometry
GeometryType = itk.ThreeDCircularProjectionGeometry
geometry = GeometryType.New()
for i in range(numberOfProjections):
    angle = (360.0 * i) / numberOfProjections
    geometry.AddProjectionInRadians(0, 0, angle * np.pi / 180.0, sid, sdd, 0, 0)

# Create projection images
projectionsSource = itk.ConstantImageSource[OutputImageType].New()
projectionsSource.SetSize([512, 1, numberOfProjections])
projectionsSource.SetSpacing([1.0, 1.0, 1.0])
projectionsSource.SetOrigin([-256.0, 0.0, 0.0])
projectionsSource.SetConstant(1.0)
projectionsSource.Update()

# Create volume
volumeSource = itk.ConstantImageSource[OutputImageType].New()
volumeSource.SetSize([32, 32, 32])
volumeSource.SetSpacing([2.0, 2.0, 2.0])
volumeSource.SetOrigin([-32.0, -32.0, -32.0])
volumeSource.SetConstant(0.0)
volumeSource.Update()

# Back-project with matrix capture
BackProjectionType = itk.JosephBackProjectionImageFilter[
    OutputImageType,
    OutputImageType,
    itk.Functor.InterpolationWeightMultiplicationBackProjection[OutputPixelType, OutputPixelType],
    itk.Functor.StoreSparseMatrixSplatWeightMultiplication[OutputPixelType, itk.D, OutputPixelType]
]
backProjection = BackProjectionType.New()
backProjection.SetInput(volumeSource.GetOutput())
backProjection.SetInput(1, projectionsSource.GetOutput())
backProjection.SetGeometry(geometry)

# Initialize and configure matrix capture
backProjection.GetSplatWeightMultiplication().GetVnlSparseMatrix().resize(
    projectionsSource.GetOutput().GetLargestPossibleRegion().GetNumberOfPixels(),
    volumeSource.GetOutput().GetLargestPossibleRegion().GetNumberOfPixels()
)
backProjection.GetSplatWeightMultiplication().SetProjectionsBuffer(
    projectionsSource.GetOutput().GetBufferPointer()
)
backProjection.GetSplatWeightMultiplication().SetVolumeBuffer(
    volumeSource.GetOutput().GetBufferPointer()
)

backProjection.Update()

# Export to Matlab format
systemMatrix = backProjection.GetSplatWeightMultiplication().GetVnlSparseMatrix()
matlabMatrix = itk.MatlabSparseMatrix[OutputImageType].New()
matlabMatrix.SetMatrix(systemMatrix)
matlabMatrix.SetOutput(volumeSource.GetOutput())

with open("backprojection_matrix.mat", "wb") as outputFile:
    matlabMatrix.Save(outputFile)

# Print matrix information
matlabMatrix.Print()

print("Matrix successfully saved to backprojection_matrix.mat")
