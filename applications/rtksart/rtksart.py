#!/usr/bin/env python3
import argparse
import itk
from itk import RTK as rtk


def build_parser():
    # argument parsing
    parser = rtk.RTKArgumentParser(
        description="Reconstructs a 3D volume from a sequence of projections with Simultaneous Algebraic Reconstruction Technique [Andersen, 1984]."
    )

    parser.add_argument(
        "--geometry", "-g", help="XML geometry file name", type=str, required=True
    )
    parser.add_argument(
        "--output", "-o", help="Output file name", type=str, required=True
    )
    parser.add_argument(
        "--niterations", "-n", help="Number of iterations", type=int, default=5
    )
    parser.add_argument(
        "--lambda",
        "-l",
        dest="lambdaval",
        help="Convergence factor",
        type=float,
        default=0.3,
    )
    parser.add_argument(
        "--positivity",
        help="Enforces positivity during the reconstruction",
        action="store_true",
    )
    parser.add_argument("--input", "-i", help="Input volume", type=str)
    parser.add_argument(
        "--nprojpersubset",
        help=(
            "Number of projections processed between each update of the reconstructed volume "
            "(1 for SART, several for OSSART, all for SIRT)"
        ),
        type=int,
        default=1,
    )
    parser.add_argument(
        "--nodisplaced",
        help="Disable the displaced detector filter",
        action="store_true",
    )
    parser.add_argument(
        "--divisionthreshold",
        help="Denominator threshold below which denominator pixels are zero",
        type=float,
    )
    parser.add_argument(
        "--reset_nesterov",
        help="Reset Nesterov after a number of subset (1 means no momentum)",
        type=int,
        default=1,
    )

    # Phase gating
    phase = parser.add_argument_group("Phase gating")
    phase.add_argument(
        "--signal", help="File containing the phase of each projection", type=str
    )
    phase.add_argument(
        "--windowcenter",
        "-c",
        help="Target reconstruction phase",
        type=float,
        default=0.0,
    )
    phase.add_argument(
        "--windowwidth",
        "-w",
        help="Tolerance around the target phase to determine in-phase and out-of-phase projections",
        type=float,
        default=1.0,
    )
    phase.add_argument(
        "--windowshape",
        "-s",
        help="Shape of the gating window",
        choices=["Rectangular", "Triangular"],
        default="Rectangular",
    )

    # RTK specific groups
    rtk.add_rtkinputprojections_group(parser)
    rtk.add_rtkprojectors_group(parser)
    rtk.add_rtk3Doutputimage_group(parser)

    # Parse the command line arguments
    return parser


def process(args_info: argparse.Namespace):
    # Types
    OutputPixelType = itk.F
    Dimension = 3
    OutputImageType = itk.Image[OutputPixelType, Dimension]

    # Projections reader
    reader = rtk.ProjectionsReader[OutputImageType].New()
    rtk.SetProjectionsReaderFromArgParse(reader, args_info)

    # Geometry
    if args_info.verbose:
        print(f"Reading geometry information from {args_info.geometry}...")
    geometry = rtk.read_geometry(args_info.geometry)

    # Phase gating weights reader
    phaseGating = rtk.PhaseGatingImageFilter[OutputImageType].New()
    if args_info.signal:
        phaseGating.SetPhasesFileName(args_info.signal)
        phaseGating.SetGatingWindowWidth(args_info.windowwidth)
        phaseGating.SetGatingWindowCenter(args_info.windowcenter)
        # Rectangular=0, Triangular=1
        if args_info.windowshape == "Triangular":
            phaseGating.SetGatingWindowShape(1)
        else:
            phaseGating.SetGatingWindowShape(0)
        phaseGating.SetInputProjectionStack(reader.GetOutput())
        phaseGating.SetInputGeometry(geometry)

        phaseGating.Update()

    # Create input: either an existing volume read from a file or a blank image
    if args_info.input:
        if args_info.verbose:
            print(f"Reading input volume from {args_info.input}...")
        inputFilter = itk.imread(args_info.input)
    else:
        # Create new empty volume
        constantImageSource = rtk.ConstantImageSource[OutputImageType].New()
        rtk.SetConstantImageSourceFromArgParse(constantImageSource, args_info)
        constantImageSource.SetConstant(0.0)
        inputFilter = constantImageSource

    if hasattr(itk, "CudaImage"):
        CudaImageType = itk.CudaImage[OutputPixelType, Dimension]
        sart = rtk.SARTConeBeamReconstructionFilter[CudaImageType, CudaImageType].New()
        sart.SetInput(itk.cuda_image_from_image(inputFilter.GetOutput()))
        if args_info.signal:
            sart.SetInput(1, itk.cuda_image_from_image(phaseGating.GetOutput()))
            sart.SetGeometry(phaseGating.GetOutputGeometry())
            sart.SetGatingWeights(
                itk.cuda_image_from_image(
                    phaseGating.GetGatingWeightsOnSelectedProjections()
                )
            )
        else:
            sart.SetInput(1, itk.cuda_image_from_image(reader.GetOutput()))
            sart.SetGeometry(geometry)
    else:
        sart = rtk.SARTConeBeamReconstructionFilter[
            OutputImageType, OutputImageType
        ].New()
        sart.SetInput(inputFilter.GetOutput())
        if args_info.signal:
            sart.SetInput(1, phaseGating.GetOutput())
            sart.SetGeometry(phaseGating.GetOutputGeometry())
            sart.SetGatingWeights(phaseGating.GetGatingWeightsOnSelectedProjections())
        else:
            sart.SetInput(1, reader.GetOutput())
            sart.SetGeometry(geometry)

    sart.SetNumberOfIterations(args_info.niterations)
    sart.SetNumberOfProjectionsPerSubset(args_info.nprojpersubset)
    sart.SetLambda(args_info.lambdaval)
    sart.SetDisableDisplacedDetectorFilter(args_info.nodisplaced)
    sart.SetResetNesterovEvery(args_info.reset_nesterov)
    if args_info.positivity:
        sart.SetEnforcePositivity(True)

    if args_info.divisionthreshold is not None:
        sart.SetDivisionThreshold(float(args_info.divisionthreshold))

    # Set the forward and back projection filters
    rtk.SetForwardProjectionFromArgParse(args_info, sart)
    rtk.SetBackProjectionFromArgParse(args_info, sart)
    # Progress
    if args_info.verbose:
        progress = rtk.PercentageProgressCommand(sart)
        sart.AddObserver(itk.ProgressEvent(), progress.callback)
        sart.AddObserver(itk.EndEvent(), progress.End)

    sart.Update()

    # Write
    if args_info.verbose:
        print("Reconstructing and writing...")
    WriterType = itk.ImageFileWriter[OutputImageType].New()
    WriterType.SetFileName(args_info.output)
    WriterType.SetInput(sart.GetOutput())
    WriterType.Update()


def main(argv=None):
    parser = build_parser()
    args_info = parser.parse_args(argv)
    process(args_info)


if __name__ == "__main__":
    main()
