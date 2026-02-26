import itk
from itk import RTK as rtk

Dimension = 3
OutputPixelType = itk.F
OutputImageType = itk.Image[OutputPixelType, Dimension]

niterations = 10
numberOfProjections = 180
angularArc = 360.0
sid = 600
sdd = 1200
scale = 2.0

# Using RTK applications python APIs to create the geometry and projections
rtk.rtksimulatedgeometry(
    nproj=numberOfProjections,
    arc=angularArc,
    sid=sid,
    sdd=sdd,
    output="geometry.xml",
)
rtk.rtkprojectgeometricphantom(
    geometry="geometry.xml",
    output="projections.mha",
    phantomfile="Thorax",
    phantomscale=scale,
    rotation="1,0,0,0,0,-1,0,1,0",
    spacing=2,
    size=128,
)

projections = itk.imread("projections.mha")
geometry = rtk.read_geometry("geometry.xml")

# Create a constant image source to initialize the reconstruction
conjugate_gradient_source = rtk.constant_image_source(
    ttype=[OutputImageType],
    origin=[-127.0, -127.0, -127.0],
    size=[128, 128, 128],
    spacing=[2.0] * 3,
)

# Set the forward and back projection filters to be used
if hasattr(itk, "CudaImage"):
    OutputCudaImageType = itk.CudaImage[itk.F, Dimension]
    ConjugateGradientFilterType = rtk.ConjugateGradientConeBeamReconstructionFilter[
        OutputCudaImageType, OutputCudaImageType
    ]
    conjugategradient = ConjugateGradientFilterType.New()
    conjugategradient.SetCudaConjugateGradient(True)
    conjugategradient.SetInput(itk.cuda_image_from_image(conjugate_gradient_source))
    conjugategradient.SetInput(1, itk.cuda_image_from_image(projections))
    conjugategradient.SetBackProjectionFilter(
        ConjugateGradientFilterType.BackProjectionType_BP_CUDAVOXELBASED
    )
    conjugategradient.SetForwardProjectionFilter(
        ConjugateGradientFilterType.ForwardProjectionType_FP_CUDARAYCAST
    )

else:
    ConjugateGradientFilterType = rtk.ConjugateGradientConeBeamReconstructionFilter[
        OutputImageType, OutputImageType
    ]
    conjugategradient = ConjugateGradientFilterType.New()
    conjugategradient.SetInput(conjugate_gradient_source)
    conjugategradient.SetInput(1, projections)
    conjugategradient.SetBackProjectionFilter(
        ConjugateGradientFilterType.BackProjectionType_BP_VOXELBASED
    )
    conjugategradient.SetForwardProjectionFilter(
        ConjugateGradientFilterType.ForwardProjectionType_FP_JOSEPH
    )

conjugategradient.SetGeometry(geometry)
conjugategradient.SetNumberOfIterations(niterations)


# Create a command to observe the iterations
class VerboseIterationCommand:
    def __init__(self):
        self.count = 0

    def callback(self):
        self.count = self.count + 1
        print(f"Iteration {self.count}", end="\r")


class VerboseEndCommand:
    def callback(self):
        print("")


cmd = VerboseIterationCommand()
conjugategradient.AddObserver(itk.IterationEvent(), cmd.callback)
cmd = VerboseEndCommand()
conjugategradient.AddObserver(itk.EndEvent(), cmd.callback)
conjugategradient.Update()

# Write
writer = itk.ImageFileWriter[OutputImageType].New()
writer.SetFileName("conjugategradient.mha")
writer.SetInput(conjugategradient.GetOutput())
writer.Update()
