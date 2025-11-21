#!/usr/bin/env python
import argparse
import itk
from itk import RTK as rtk


def build_parser():
    parser = rtk.RTKArgumentParser(
        description="Denoises a volume using Daubechies wavelets soft thresholding"
    )

    parser.add_argument(
        "--input", "-i", help="Input file name", type=str, required=True
    )
    parser.add_argument(
        "--output", "-o", help="Output file name", type=str, required=True
    )
    parser.add_argument(
        "--order", help="Order of the Daubechies wavelets", type=int, required=True
    )
    parser.add_argument(
        "--level",
        "-l",
        help="Number of deconstruction levels",
        type=int,
        required=True,
    )
    parser.add_argument(
        "--threshold",
        "-t",
        help="Threshold used in soft thresholding of the wavelets coefficients",
        type=float,
        required=True,
    )

    return parser


def process(args_info: argparse.Namespace):
    OutputPixelType = itk.F
    Dimension = 3
    OutputImageType = itk.Image[OutputPixelType, Dimension]

    # Read the input image
    if args_info.verbose:
        print(f"Reading input from {args_info.input}...")
    # Read as float to match the filter's expected image type
    input_img = itk.imread(args_info.input, OutputPixelType)

    # Create the denoising filter
    wst = rtk.DeconstructSoftThresholdReconstructImageFilter[OutputImageType].New()
    wst.SetInput(input_img)
    wst.SetOrder(args_info.order)
    wst.SetThreshold(args_info.threshold)
    wst.SetNumberOfLevels(args_info.level)

    # Write reconstruction
    if args_info.verbose:
        print(f"Writing output to {args_info.output}...")
    itk.imwrite(wst.GetOutput(), args_info.output)


def main(argv=None):
    parser = build_parser()
    args_info = parser.parse_args(argv)
    process(args_info)


if __name__ == "__main__":
    main()
