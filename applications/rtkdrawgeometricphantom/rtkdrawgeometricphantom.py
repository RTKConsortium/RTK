#!/usr/bin/env python
import argparse
import sys
import itk
from itk import RTK as rtk
import numpy as np


def build_parser():
    # Argument parsing
    parser = rtk.rtk_argument_parser("Computes a 3D voxelized phantom from a phantom description file.")

    # General options
    parser.add_argument(
        "--verbose", "-v", help="Verbose execution", action="store_true"
    )
    parser.add_argument(
        "--output", "-o", help="Output projections file name", type=str, required=True
    )
    parser.add_argument(
        "--phantomfile",
        help="Parameters of the phantom reference",
        type=str,
        required=True,
    )
    parser.add_argument("--noise", help="Gaussian noise parameter (SD)", type=float)
    parser.add_argument(
        "--phantomscale",
        help="Scaling factor for the phantom dimensions",
        type=rtk.comma_separated_args(float),
        default=[1.0],
    )
    parser.add_argument(
        "--offset",
        help="3D spatial offset of the phantom center",
        type=rtk.comma_separated_args(float),
    )
    parser.add_argument(
        "--rotation",
        help="Rotation matrix for the phantom",
        type=rtk.comma_separated_args(float),
    )

    rtk.add_rtk3Doutputimage_group(parser)

    # Parse the command line arguments
    return parser


def process(args_info: argparse.Namespace):
    # Define output pixel type and dimension
    OutputPixelType = itk.F
    Dimension = 3
    OutputImageType = itk.Image[OutputPixelType, Dimension]

    # Empty volume image
    constant_image_source = rtk.ConstantImageSource[OutputImageType].New()
    rtk.SetConstantImageSourceFromArgParse(constant_image_source, args_info)

    # Offset, scale, rotation
    offset = [0.0] * 3
    if args_info.offset:
        if len(args_info.offset) > 3:
            print("--offset needs up to 3 values", file=sys.stderr)
            sys.exit(1)
        offset = args_info.offset

    scale = [args_info.phantomscale[0]] * Dimension
    if args_info.phantomscale:
        if len(args_info.phantomscale) > 3:
            print("--phantomscale needs up to 3 values", file=sys.stderr)
            sys.exit(1)
        for i in range(len(args_info.phantomscale)):
            scale[i] = args_info.phantomscale[i]

    rot = itk.Matrix[itk.D, Dimension, Dimension]()
    rot.SetIdentity()
    if args_info.rotation:
        if len(args_info.rotation) != 9:
            print("--rotation needs exactly 9 values", file=sys.stderr)
            sys.exit(1)
        itk.matrix_from_array(np.array(args_info.rotation).reshape(3, 3))

    # Reference
    if args_info.verbose:
        print("Creating reference... ", flush=True)

    # DrawGeometricPhantomImageFilter
    dq = rtk.DrawGeometricPhantomImageFilter[OutputImageType, OutputImageType].New()
    dq.SetInput(constant_image_source.GetOutput())
    dq.SetPhantomScale(scale)
    dq.SetOriginOffset(offset)
    dq.SetRotationMatrix(rot)
    dq.SetConfigFile(args_info.phantomfile)

    dq.Update()

    # Add noise
    output = dq.GetOutput()
    if args_info.noise:
        noisy = rtk.AdditiveGaussianNoiseImageFilter[OutputImageType].New()
        noisy.SetInput(output)
        noisy.SetMean(0.0)
        noisy.SetStandardDeviation(args_info.noise)

        noisy.Update()

        output = noisy.GetOutput()

    # Write
    if args_info.verbose:
        print("Writing reference... ", flush=True)

    itk.imwrite(output, args_info.output)


def main(argv=None):
    parser = build_parser()
    args_info = parser.parse_args(argv)
    process(args_info)


if __name__ == "__main__":
    main()
