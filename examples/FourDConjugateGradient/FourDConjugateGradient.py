import itk
from itk import RTK as rtk

OutputPixelType = itk.F
CpuProjectionStackType = itk.Image[OutputPixelType, 3]
CpuVolumeSeriesType = itk.Image[OutputPixelType, 4]

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
ConjugateGradientFilterType = rtk.FourDConjugateGradientConeBeamReconstructionFilter[
    VolumeSeriesType, ProjectionStackType
]
fourdconjugategradient = ConjugateGradientFilterType.New()
if hasattr(itk, "CudaImage"):
    fourdconjugategradient.SetCudaConjugateGradient(True)
    fourdconjugategradient.SetForwardProjectionFilter(
        ConjugateGradientFilterType.ForwardProjectionType_FP_CUDARAYCAST
    )
    fourdconjugategradient.SetBackProjectionFilter(
        ConjugateGradientFilterType.BackProjectionType_BP_CUDAVOXELBASED
    )
else:
    fourdconjugategradient.SetForwardProjectionFilter(
        ConjugateGradientFilterType.ForwardProjectionType_FP_JOSEPH
    )
    fourdconjugategradient.SetBackProjectionFilter(
        ConjugateGradientFilterType.BackProjectionType_BP_VOXELBASED
    )

fourdconjugategradient.SetInputVolumeSeries(four_d_source)
fourdconjugategradient.SetNumberOfIterations(2)

# Set the newly ordered arguments
fourdconjugategradient.SetInputProjectionStack(reorder.GetOutput())
fourdconjugategradient.SetGeometry(reorder.GetOutputGeometry())
fourdconjugategradient.SetWeights(signal_to_interpolation_weights.GetOutput())
fourdconjugategradient.SetSignal(reorder.GetOutputSignal())


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
fourdconjugategradient.AddObserver(itk.IterationEvent(), cmd.callback)
cmd_end = VerboseEndCommand()
fourdconjugategradient.AddObserver(itk.EndEvent(), cmd_end.callback)

fourdconjugategradient.Update()

# Write
writer = itk.ImageFileWriter[CpuVolumeSeriesType].New()
writer.SetFileName("fourdconjugategradient.mha")
writer.SetInput(fourdconjugategradient.GetOutput())
writer.Update()
