#!/usr/bin/env python
import argparse
import sys
import itk
from itk import RTK as rtk


def build_parser():
    parser = rtk.RTKArgumentParser(
        description=(
            "Reconstructs a 4D sequence of volumes from a sequence of projections "
            "with a 4D version of the Simultaneous Algebraic Reconstruction Technique [Andersen, 1984]."
        )
    )

    parser.add_argument(
        "--geometry", "-g", help="XML geometry file name", type=str, required=True
    )
    parser.add_argument(
        "--output", "-o", help="Output file name", type=str, required=True
    )
    parser.add_argument(
        "--niterations",
        "-n",
        help="Number of iterations",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--lambda",
        "-l",
        help="Convergence factor",
        type=float,
        default=0.3,
        dest="lambda_val",
    )
    parser.add_argument(
        "--positivity",
        help="Enforces positivity during the reconstruction",
        action="store_true",
    )
    parser.add_argument("--input", "-i", help="Input volume", type=str)
    parser.add_argument(
        "--nprojpersubset",
        help="Number of projections processed between each update of the reconstructed volume (1 for SART, several for OSSART, all for SIRT)",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--nodisplaced",
        help="Disable the displaced detector filter",
        action="store_true",
    )

    # Phase gating
    parser.add_argument(
        "--signal",
        help="File containing the phase of each projection",
        type=str,
        required=True,
    )

    # RTK common groups
    rtk.add_rtkinputprojections_group(parser)
    rtk.add_rtk4Doutputimage_group(parser)
    rtk.add_rtkprojectors_group(parser)
    rtk.add_rtkiterations_group(parser)

    return parser


def process(args_info: argparse.Namespace):
    OutputPixelType = itk.F
    VolumeSeriesType = itk.Image[OutputPixelType, 4]
    ProjectionStackType = itk.Image[OutputPixelType, 3]

    # Check CUDA-related options vs. build capabilities
    if hasattr(itk, "CudaImage"):
        CudaVolumeSeriesType = itk.CudaImage[OutputPixelType, 4]
        CudaProjectionStackType = itk.CudaImage[OutputPixelType, 3]

    # Projections reader
    reader = rtk.ProjectionsReader[ProjectionStackType].New()
    rtk.SetProjectionsReaderFromArgParse(reader, args_info)

    # Geometry
    if args_info.verbose:
        print(f"Reading geometry information from {args_info.geometry}...")
    geometry = rtk.read_geometry(args_info.geometry)

    # Create input: either an existing volume read from a file or a blank image
    if args_info.input:
        # Read existing 4D volume
        inputFilter = itk.ImageFileReader[VolumeSeriesType].New()
        inputFilter.SetFileName(args_info.input)
    else:
        # Constant 4D volume
        constantImageSource = rtk.ConstantImageSource[VolumeSeriesType].New()
        rtk.SetConstantImageSourceFromArgParse(constantImageSource, args_info)
        # Set 4th dimension (frames)
        size = constantImageSource.GetSize()
        size[3] = args_info.frames
        constantImageSource.SetSize(size)
        inputFilter = constantImageSource

    inputFilter.Update()
    inputFilter.ReleaseDataFlagOn()

    # Read the phases file
    phaseReader = rtk.PhasesToInterpolationWeights.New()
    phaseReader.SetFileName(args_info.signal)
    phaseReader.SetNumberOfReconstructedFrames(
        inputFilter.GetOutput().GetLargestPossibleRegion().GetSize()[3]
    )
    phaseReader.Update()

    # 4D SART reconstruction filter
    if hasattr(itk, "CudaImage"):
        fourdsart = rtk.FourDSARTConeBeamReconstructionFilter[
            CudaVolumeSeriesType, CudaProjectionStackType
        ].New()
        fourdsart.SetInputVolumeSeries(
            itk.cuda_image_from_image(inputFilter.GetOutput())
        )
        fourdsart.SetInputProjectionStack(itk.cuda_image_from_image(reader.GetOutput()))
    else:
        fourdsart = rtk.FourDSARTConeBeamReconstructionFilter[
            VolumeSeriesType, ProjectionStackType
        ].New()
        fourdsart.SetInputVolumeSeries(inputFilter.GetOutput())
        fourdsart.SetInputProjectionStack(reader.GetOutput())

    # Set the forward and back projection filters
    rtk.SetForwardProjectionFromArgParse(args_info, fourdsart)
    rtk.SetBackProjectionFromArgParse(args_info, fourdsart)

    fourdsart.SetGeometry(geometry)
    fourdsart.SetNumberOfIterations(args_info.niterations)
    fourdsart.SetNumberOfProjectionsPerSubset(args_info.nprojpersubset)
    fourdsart.SetWeights(phaseReader.GetOutput())
    fourdsart.SetSignal(rtk.read_signal_file(args_info.signal))
    fourdsart.SetLambda(args_info.lambda_val)
    fourdsart.SetDisableDisplacedDetectorFilter(args_info.nodisplaced)

    if args_info.positivity:
        fourdsart.SetEnforcePositivity(True)

    rtk.SetIterationsReportFromArgParse(args_info, fourdsart)

    fourdsart.Update()

    # Write
    if args_info.verbose:
        print(f"Writing output to {args_info.output}...")
    writer = itk.ImageFileWriter[itk.Image[OutputPixelType, 4]].New()
    writer.SetFileName(args_info.output)
    writer.SetInput(fourdsart.GetOutput())
    writer.Update()


def main(argv=None):
    parser = build_parser()
    args_info = parser.parse_args(argv)
    process(args_info)


if __name__ == "__main__":
    main()
