import itk
from itk import RTK as rtk

OutputPixelType = itk.F
Dimension = 3

CPUOutputImageType = itk.Image[OutputPixelType, Dimension]

if hasattr(itk, "CudaImage"):
    OutputImageType = itk.CudaImage[OutputPixelType, Dimension]
else:
    OutputImageType = CPUOutputImageType

# Read geometry, projections and signal
geometry = rtk.read_geometry("four_d_geometry.xml")

projectionsReader = rtk.ProjectionsReader[CPUOutputImageType].New()
fileNames = ["four_d_projections.mha"]
projectionsReader.SetFileNames(fileNames)
projectionsReader.Update()

# Part specific to 4D
selector = rtk.SelectOneProjectionPerCycleImageFilter[CPUOutputImageType].New()
selector.SetInput(projectionsReader.GetOutput())
selector.SetInputGeometry(geometry)
selector.SetSignalFilename("four_d_signal.txt")

# Create one frame of the reconstructed image
constantImageSource = rtk.ConstantImageSource[OutputImageType].New()

constantImageSource.SetOrigin([-63.0, -31.0, -63.0])
constantImageSource.SetSpacing([4.0, 4.0, 4.0])
constantImageSource.SetSize([32, 16, 32])
constantImageSource.SetConstant(0.0)
constantImageSource.Update()

# FDK reconstruction filtering
if hasattr(itk, "CudaImage"):
    cuda_projections = itk.CudaImageFromImageFilter[CPUOutputImageType].New()
    cuda_projections.SetInput(selector.GetOutput())

    feldkamp = rtk.CudaFDKConeBeamReconstructionFilter.New()
    feldkamp.SetInput(1, cuda_projections.GetOutput())
else:
    feldkamp = rtk.FDKConeBeamReconstructionFilter[
        OutputImageType, OutputImageType, OutputPixelType
    ].New()
    feldkamp.SetInput(1, selector.GetOutput())

feldkamp.SetInput(0, constantImageSource.GetOutput())
feldkamp.SetGeometry(selector.GetOutputGeometry())

# Create empty 4D image
FourDOutputImageType = itk.Image[OutputPixelType, Dimension + 1]
fourDSource = rtk.ConstantImageSource[FourDOutputImageType].New()
fourDSource.SetOrigin([-63.0, -31.0, -63.0, 0.0])
fourDSource.SetSpacing([4.0, 4.0, 4.0, 1.0])
fourDSource.SetSize([32, 16, 32, 8])
fourDSource.SetConstant(0.0)
fourDSource.Update()

# Go over each frame, reconstruct 3D frame and paste with iterators in 4D image
it4D = itk.GetArrayViewFromImage(fourDSource.GetOutput())
for f in range(8):
    selector.SetPhase(f / 8.0)
    feldkamp.UpdateLargestPossibleRegion()

    it3D = itk.GetArrayFromImage(feldkamp.GetOutput())
    it4D[f] = it3D

# Write
writer = itk.ImageFileWriter[FourDOutputImageType].New()
writer.SetFileName("fourdfdk.mha")
writer.SetInput(fourDSource.GetOutput())
writer.Update()
