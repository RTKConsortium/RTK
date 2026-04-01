#!/usr/bin/env python
import argparse
import time
import itk
from itk import RTK as rtk


def build_parser():
    parser = rtk.RTKArgumentParser(
        description="Reconstructs a 3D volume from a sequence of projections with a conjugate gradient technique"
    )

    parser.add_argument(
        "--geometry", "-g", help="XML geometry file name", type=str, required=True
    )
    parser.add_argument(
        "--output", "-o", help="Output file name", type=str, required=True
    )
    parser.add_argument(
        "--projections", help="Projections file name", type=str, required=True
    )
    parser.add_argument(
        "--niterations", "-n", help="Number of iterations", type=int, default=5
    )
    parser.add_argument(
        "--time",
        "-t",
        help="Records elapsed time during the process",
        action="store_true",
    )
    parser.add_argument("--input", "-i", help="Input volume", type=str)
    parser.add_argument(
        "--weights",
        "-w",
        help="Weights file for Weighted Least Squares (WLS)",
        type=str,
    )
    parser.add_argument(
        "--regweights", help="Local regularization weights file", type=str
    )
    parser.add_argument(
        "--gamma", help="Laplacian regularization weight", type=float, default=0.0
    )
    parser.add_argument(
        "--tikhonov", help="Tikhonov regularization weight", type=float, default=0.0
    )
    parser.add_argument(
        "--nocudacg",
        help="Do not perform conjugate gradient calculations on GPU",
        action="store_true",
    )
    parser.add_argument(
        "--mask",
        "-m",
        help="Apply a support binary mask: reconstruction kept null outside the mask",
        type=str,
    )
    parser.add_argument(
        "--nodisplaced",
        help="Disable the displaced detector filter",
        action="store_true",
    )
    parser.add_argument(
        "--targetSDD",
        help="Target sum of squared difference between consecutive iterates, as stopping criterion",
        type=float,
        default=0.0,
    )

    rtk.add_rtk3Doutputimage_group(parser)
    rtk.add_rtkprojectors_group(parser)

    return parser


def process(args_info: argparse.Namespace):
    Dimension = 3
    nMaterials = 3

    DataType = itk.F
    PixelType = itk.Vector[DataType, nMaterials]
    WeightsType = itk.Vector[DataType, nMaterials * nMaterials]

    SingleComponentImageType = itk.Image[DataType, Dimension]
    OutputImageType = itk.Image[PixelType, Dimension]
    WeightsImageType = itk.Image[WeightsType, Dimension]

    # Read projections
    projections = itk.imread(args_info.projections)

    # Geometry
    if args_info.verbose:
        print(f"Reading geometry information from {args_info.geometry}...")
    geometry = rtk.read_geometry(args_info.geometry)

    # Create input: either an existing volume read from a file or a blank image
    if args_info.input is not None:
        input_image = itk.imread(args_info.input)
    else:
        constantImageSource = rtk.ConstantImageSource[OutputImageType].New()
        rtk.SetConstantImageSourceFromArgParse(constantImageSource, args_info)
        constantImageSource.Update()
        input_image = constantImageSource.GetOutput()

    # Read weights if given
    inputWeights = None
    if args_info.weights is not None:
        inputWeights = itk.imread(args_info.weights)

    # Read regularization weights if given
    localRegWeights = None
    if args_info.regweights is not None:
        localRegWeights = itk.imread(args_info.regweights)

    # Read support mask if given
    supportmask = None
    if args_info.mask is not None:
        supportmask = itk.imread(args_info.mask)

    # Set the forward and back projection filters to be used
    conjugategradient = rtk.ConjugateGradientConeBeamReconstructionFilter[
        OutputImageType, SingleComponentImageType, WeightsImageType
    ].New()

    rtk.SetForwardProjectionFromArgParse(args_info, conjugategradient)
    rtk.SetBackProjectionFromArgParse(args_info, conjugategradient)

    conjugategradient.SetInputVolume(input_image)
    conjugategradient.SetInputProjectionStack(projections)
    if inputWeights is not None:
        conjugategradient.SetInputWeights(inputWeights)
    if localRegWeights is not None:
        conjugategradient.SetLocalRegularizationWeights(localRegWeights)

    conjugategradient.SetCudaConjugateGradient(not args_info.nocudacg)

    if supportmask is not None:
        conjugategradient.SetSupportMask(supportmask)

    if args_info.gamma != 0.0:
        conjugategradient.SetGamma(args_info.gamma)
    if args_info.tikhonov != 0.0:
        conjugategradient.SetTikhonov(args_info.tikhonov)

    conjugategradient.SetGeometry(geometry)
    conjugategradient.SetNumberOfIterations(args_info.niterations)
    conjugategradient.SetDisableDisplacedDetectorFilter(args_info.nodisplaced)

    if args_info.time:
        print("Recording elapsed time... ", end="", flush=True)
        start_time = time.time()

    conjugategradient.Update()

    if args_info.time:
        elapsed = time.time() - start_time
        print(f"It took...  {elapsed:.3f} s")

    # Write
    itk.imwrite(conjugategradient.GetOutput(), args_info.output)


def main(argv=None):
    parser = build_parser()
    args_info = parser.parse_args(argv)
    process(args_info)


if __name__ == "__main__":
    main()
