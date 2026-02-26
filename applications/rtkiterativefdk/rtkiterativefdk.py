#!/usr/bin/env python
import argparse
import sys
import itk
from itk import RTK as rtk


def build_parser():
    parser = rtk.RTKArgumentParser(
        description=(
            "Reconstructs a 3D volume from a sequence of projections "
            "[Feldkamp, David, Kress, 1984]."
        )
    )

    parser.add_argument(
        "--geometry", "-g", help="XML geometry file name", type=str, required=True
    )
    parser.add_argument(
        "--output", "-o", help="Output file name", type=str, required=True
    )
    parser.add_argument(
        "--hardware",
        help="Hardware used for computation",
        choices=["cpu", "cuda"],
        default="cpu",
    )
    parser.add_argument(
        "--subsetsize",
        help="Streaming option: number of projections processed at a time",
        type=int,
        default=16,
    )
    parser.add_argument(
        "--niterations", "-n", type=int, default=3, help="Number of iterations"
    )
    parser.add_argument(
        "--lambda",
        "-l",
        dest="lambda_",
        type=float,
        default=0.3,
        help="Convergence factor",
    )
    parser.add_argument(
        "--positivity",
        help="Enforces positivity during the reconstruction",
        action="store_true",
    )
    parser.add_argument(
        "--nodisplaced",
        help="Disable the displaced detector filter",
        action="store_true",
    )

    # Ramp filter options
    ramp_group = parser.add_argument_group("Ramp filter")
    ramp_group.add_argument(
        "--pad",
        help="Data padding parameter to correct for truncation",
        type=float,
        default=0.0,
    )
    ramp_group.add_argument(
        "--hann",
        help="Cut frequency for hann window in ]0,1] (0.0 disables it)",
        type=float,
        default=0.0,
    )
    ramp_group.add_argument(
        "--hannY",
        help="Cut frequency for hann window in ]0,1] (0.0 disables it)",
        type=float,
        default=0.0,
    )

    rtk.add_rtkprojectors_group(parser)
    rtk.add_rtkinputprojections_group(parser)
    rtk.add_rtk3Doutputimage_group(parser)
    rtk.add_rtkiterations_group(parser)

    return parser


def process(args_info: argparse.Namespace):
    OutputPixelType = itk.F
    Dimension = 3
    OutputImageType = itk.Image[OutputPixelType, Dimension]

    # Check on hardware parameter
    if not hasattr(itk, "CudaImage") and args_info.hardware == "cuda":
        print("The program has not been compiled with CUDA option.")
        sys.exit(1)

    # Projections reader
    reader = rtk.ProjectionsReader[OutputImageType].New()
    rtk.SetProjectionsReaderFromArgParse(reader, args_info)

    if args_info.verbose:
        print("Reading... ")
    reader.Update()

    # Geometry
    if args_info.verbose:
        print(f"Reading geometry information from {args_info.geometry}...")
    geometry = rtk.read_geometry(args_info.geometry)

    # Create reconstructed image
    constantImageSource = rtk.ConstantImageSource[OutputImageType].New()
    rtk.SetConstantImageSourceFromArgParse(constantImageSource, args_info)

    enforcePositivity = bool(args_info.positivity)

    # Create Iterative FDK filter and connect it
    if args_info.hardware == "cuda":
        ifdk = rtk.CudaIterativeFDKConeBeamReconstructionFilter.New()
        ifdk.SetInput(0, itk.cuda_image_from_image(constantImageSource.GetOutput()))
        ifdk.SetInput(1, itk.cuda_image_from_image(reader.GetOutput()))
    else:
        IFDKCPUType = rtk.IterativeFDKConeBeamReconstructionFilter[
            OutputImageType, OutputImageType, itk.D
        ]
        ifdk = IFDKCPUType.New()
        ifdk.SetInput(0, constantImageSource.GetOutput())
        ifdk.SetInput(1, reader.GetOutput())

    # Common options
    ifdk.SetGeometry(geometry)
    rtk.SetForwardProjectionFromArgParse(args_info, ifdk)
    ifdk.SetNumberOfIterations(args_info.niterations)
    ifdk.SetTruncationCorrection(args_info.pad)
    ifdk.SetHannCutFrequency(args_info.hann)
    ifdk.SetHannCutFrequencyY(args_info.hannY)
    ifdk.SetProjectionSubsetSize(args_info.subsetsize)
    ifdk.SetLambda(args_info.lambda_)
    ifdk.SetEnforcePositivity(enforcePositivity)
    ifdk.SetDisableDisplacedDetectorFilter(args_info.nodisplaced)

    # Iterations reporting
    rtk.SetIterationsReportFromArgParse(args_info, ifdk)

    # Run the filter and write
    if args_info.verbose:
        print("Reconstructing and writing... ")

    # Write
    writer = itk.ImageFileWriter[OutputImageType].New()
    writer.SetFileName(args_info.output)
    writer.SetInput(ifdk.GetOutput())
    writer.Update()


def main(argv=None):
    parser = build_parser()
    args_info = parser.parse_args(argv)
    process(args_info)


if __name__ == "__main__":
    main()
