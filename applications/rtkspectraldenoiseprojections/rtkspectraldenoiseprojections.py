#!/usr/bin/env python
import argparse
import itk
from itk import RTK as rtk


def build_parser():
    parser = rtk.RTKArgumentParser(
        description="Replaces aberrant pixels by the median in a small neighborhood around them. Pixels are aberrant if the difference between their value and the median is larger that threshold multiplier * the standard deviation in the neighborhood"
    )
    parser.add_argument(
        "--input", "-i", help="Input file name", type=str, required=True
    )
    parser.add_argument(
        "--output", "-o", help="Output file name", type=str, required=True
    )
    parser.add_argument(
        "--multiplier",
        "-m",
        help="Threshold multiplier (actual threshold is obtained by multiplying by standard dev. of neighborhood)",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--radius",
        "-r",
        help="Radius of neighborhood in each direction (actual radius is 2r+1)",
        type=int,
        nargs="+",
    )
    return parser


def process(args_info: argparse.Namespace):
    OutputImageType = itk.VectorImage[itk.F, 3]

    # Reader
    if args_info.verbose:
        print(f"Reading input from {args_info.input}...")
    input_image = itk.imread(args_info.input)

    # Remove aberrant pixels
    median = rtk.ConditionalMedianImageFilter[OutputImageType].New()
    median.SetThresholdMultiplier(args_info.multiplier)
    radius = itk.Size[3]()
    if args_info.radius is None:
        radius.Fill(1)
    else:
        radius.Fill(args_info.radius[0])
        for i in range(len(args_info.radius)):
            radius[i] = args_info.radius[i]
    median.SetRadius(radius)
    median.SetInput(input_image)

    # Write
    if args_info.verbose:
        print(f"Writing output to {args_info.output}...")
    itk.imwrite(median.GetOutput(), args_info.output)


def main(argv=None):
    parser = build_parser()
    args_info = parser.parse_args(argv)
    process(args_info)


if __name__ == "__main__":
    main()
