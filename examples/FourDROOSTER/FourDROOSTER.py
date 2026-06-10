import itk
from itk import RTK as rtk

OutputPixelType = itk.F
CpuProjectionStackType = itk.Image[OutputPixelType, 3]
CpuVolumeSeriesType = itk.Image[OutputPixelType, 4]
DVFVectorType = itk.CovariantVector[OutputPixelType, 3]
CpuDVFSequenceImageType = itk.Image[DVFVectorType, 4]

if hasattr(itk, "CudaImage"):
    ProjectionStackType = itk.CudaImage[OutputPixelType, 3]
    VolumeSeriesType = itk.CudaImage[OutputPixelType, 4]
else:
    ProjectionStackType = CpuProjectionStackType
    VolumeSeriesType = CpuVolumeSeriesType


# Generate the input volume series, used as initial estimate by 4D conjugate gradient
four_d_source = rtk.constant_image_source(
    ttype=[VolumeSeriesType],
    origin=[-63.0, -31.0, -63.0, 0.0],
    spacing=[4.0, 4.0, 4.0, 1.0],
    size=[32, 16, 32, 8],
)

# Read geometry, projections and signal
geometry = rtk.read_geometry("four_d_geometry.xml")

projections_reader = rtk.ProjectionsReader[CpuProjectionStackType].New()
file_names = ["four_d_projections.mha"]
projections_reader.SetFileNames(file_names)
projections_reader.Update()

signal = rtk.read_signal_file("four_d_signal.txt")

# Re-order geometry and projections
# In the new order, projections with identical phases are packed together
reorder = rtk.ReorderProjectionsImageFilter[
    ProjectionStackType, ProjectionStackType
].New()
if hasattr(itk, "CudaImage"):
    reorder.SetInput(itk.cuda_image_from_image(projections_reader.GetOutput()))
else:
    reorder.SetInput(projections_reader.GetOutput())
reorder.SetInputGeometry(geometry)
reorder.SetInputSignal(signal)
reorder.Update()

# Release the memory holding the stack of original projections
projections_reader.GetOutput().ReleaseData()

# Compute the interpolation weights
signal_to_interpolation_weights = rtk.SignalToInterpolationWeights.New()
signal_to_interpolation_weights.SetSignal(reorder.GetOutputSignal())
signal_to_interpolation_weights.SetNumberOfReconstructedFrames(
    four_d_source.GetLargestPossibleRegion().GetSize()[3]
)
signal_to_interpolation_weights.Update()

# Set the forward and back projection filters to be used
ROOSTERFilterType = rtk.FourDROOSTERConeBeamReconstructionFilter[
    VolumeSeriesType, ProjectionStackType
]
rooster = ROOSTERFilterType.New()

rooster.SetInputVolumeSeries(four_d_source)
rooster.SetCG_iterations(2)
rooster.SetMainLoop_iterations(2)
if hasattr(itk, "CudaImage"):
    rooster.SetCudaConjugateGradient(True)
    rooster.SetUseCudaCyclicDeformation(True)
    rooster.SetForwardProjectionFilter(
        ROOSTERFilterType.ForwardProjectionType_FP_CUDARAYCAST
    )
    rooster.SetBackProjectionFilter(
        ROOSTERFilterType.BackProjectionType_BP_CUDAVOXELBASED
    )
else:
    rooster.SetForwardProjectionFilter(
        ROOSTERFilterType.ForwardProjectionType_FP_JOSEPH
    )
    rooster.SetBackProjectionFilter(ROOSTERFilterType.BackProjectionType_BP_VOXELBASED)

# Set the newly ordered arguments
rooster.SetInputProjectionStack(reorder.GetOutput())
rooster.SetGeometry(reorder.GetOutputGeometry())
rooster.SetWeights(signal_to_interpolation_weights.GetOutput())
rooster.SetSignal(reorder.GetOutputSignal())

# For each optional regularization step, set whether or not
# it should be performed, and provide the necessary inputs

# Positivity
rooster.SetPerformPositivity(True)

# Motion mask
rooster.SetPerformMotionMask(False)

# Spatial TV
rooster.SetGammaTVSpace(0.1)
rooster.SetTV_iterations(4)
rooster.SetPerformTVSpatialDenoising(True)

# Spatial wavelets
rooster.SetPerformWaveletsSpatialDenoising(False)

# Temporal TV
rooster.SetGammaTVTime(0.1)
rooster.SetTV_iterations(4)
rooster.SetPerformTVTemporalDenoising(True)

# Temporal L0
rooster.SetPerformL0TemporalDenoising(False)

# Total nuclear variation
rooster.SetPerformTNVDenoising(False)

# Warping
rooster.SetPerformWarping(True)
rooster.SetUseNearestNeighborInterpolationInWarping(False)

dvf_reader = itk.ImageFileReader[CpuDVFSequenceImageType].New()
dvf_reader.SetFileName("four_d_dvf.mha")
dvf = dvf_reader.GetOutput()
if hasattr(itk, "CudaImage"):
    dvf = itk.cuda_image_from_image(dvf)
rooster.SetDisplacementField(dvf)

rooster.SetComputeInverseWarpingByConjugateGradient(False)
idvf_reader = itk.ImageFileReader[CpuDVFSequenceImageType].New()
idvf_reader.SetFileName("four_d_idvf.mha")
idvf = idvf_reader.GetOutput()
if hasattr(itk, "CudaImage"):
    idvf = itk.cuda_image_from_image(idvf)
rooster.SetInverseDisplacementField(idvf)


class VerboseIterationCommand:
    def __init__(self):
        self.count = 0

    def callback(self):
        self.count += 1
        print(f"Iteration {self.count}", end="\r")


class VerboseEndCommand:
    def callback(self):
        print("")


cmd = VerboseIterationCommand()
rooster.AddObserver(itk.IterationEvent(), cmd.callback)
cmd_end = VerboseEndCommand()
rooster.AddObserver(itk.EndEvent(), cmd_end.callback)

rooster.Update()

# Write
writer = itk.ImageFileWriter[CpuVolumeSeriesType].New()
writer.SetFileName("fourdrooster.mha")
writer.SetInput(rooster.GetOutput())
writer.Update()
