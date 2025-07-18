#!/usr/bin/env python
import argparse
import sys
import itk
from itk import RTK as rtk


def build_parser():
    # Argument parsing
    parser = rtk.rtk_argument_parser("Computes a 3D voxelized Shepp & Logan phantom with noise [https://www.slaney.org/pct/pct-errata.html]")

    # General options
    parser.add_argument(
        "--verbose", "-v", help="Verbose execution", action="store_true"
    )
    parser.add_argument(
        "--output", "-o", help="Output projections file name", type=str, required=True
    )
    parser.add_argument(
        "--phantomscale",
        help="Scaling factor for the phantom dimensions",
        type=rtk.comma_separated_args(float),
        default=[128],
    )
    parser.add_argument("--noise", help="Gaussian noise parameter (SD)", type=float)
    parser.add_argument(
        "--offset",
        help="3D spatial offset of the phantom center",
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

    # Reference
    if args_info.verbose:
        print("Creating reference... ", flush=True)

    # DrawSheppLoganPhantomImageFilter
    dsl = rtk.DrawSheppLoganFilter[OutputImageType, OutputImageType].New()
    dsl.SetPhantomScale(scale)
    dsl.SetInput(constant_image_source.GetOutput())
    dsl.SetOriginOffset(offset)

    dsl.Update()

    # Add noise
    output = dsl.GetOutput()
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
