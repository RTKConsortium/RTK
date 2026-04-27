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

# Create geometry using core RTK API (replaces rtk.rtksimulatedgeometry)
geometry = rtk.ThreeDCircularProjectionGeometry.New()
for i in range(numberOfProjections):
    angle = i * angularArc / numberOfProjections
    geometry.AddProjection(sid, sdd, angle)
rtk.write_geometry(geometry, "geometry.xml")

# Create projections using Shepp-Logan phantom
# (replaces rtk.rtkprojectgeometricphantom which is an application-level function)
projections_source = rtk.constant_image_source(
    ttype=[OutputImageType],
    origin=[-127.0, -127.0, 0.0],
    size=[128, 128, numberOfProjections],
    spacing=[2.0] * 3,
)
projections = rtk.shepp_logan_phantom_filter(
    projections_source,
    geometry=geometry,
    phantom_scale=scale,
)
itk.imwrite(projections, "projections.mha")

# Create a constant image source to initialize the reconstruction
conjugate_gradient_source = rtk.constant_image_source(
    ttype=[OutputImageType],
    origin=[-127.0, -127.0, -127.0],
    size=[128, 128, 128],
    spacing=[2.0] * 3,
)

# Set up conjugate gradient reconstruction
# Note: wrapped filter only accepts 1 template parameter
ConjugateGradientFilterType = rtk.ConjugateGradientConeBeamReconstructionFilter[
    OutputImageType
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
