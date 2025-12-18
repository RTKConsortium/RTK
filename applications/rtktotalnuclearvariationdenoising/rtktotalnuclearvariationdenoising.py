#!/usr/bin/env python
import argparse
import itk
from itk import RTK as rtk


def build_parser():
    parser = rtk.RTKArgumentParser(
        description="Performs total nuclear variation denoising of a 3D + channels image."
    )
    parser.add_argument(
        "--input", "-i", help="Input file name", type=str, required=True
    )
    parser.add_argument(
        "--output", "-o", help="Output file name", type=str, required=True
    )
    parser.add_argument(
        "--gamma",
        "-g",
        help="TV term's weighting parameter",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--niter",
        "-n",
        help="Number of iterations",
        type=int,
        default=5,
    )

    return parser


def process(args_info: argparse.Namespace):
    OutputPixelType = itk.F
    Dimension = 4  # Number of dimensions of the input image
    DimensionsProcessed = 3  # Number of dimensions along which the gradient is computed

    OutputImageType = itk.Image[OutputPixelType, Dimension]
    GradientPixelType = itk.CovariantVector[OutputPixelType, DimensionsProcessed]
    GradientImageType = itk.Image[GradientPixelType, Dimension]

    # Read input
    input_image = itk.imread(args_info.input, pixel_type=OutputPixelType)

    # Apply total nuclear variation denoising
    tv = rtk.TotalNuclearVariationDenoisingBPDQImageFilter[
        OutputImageType,
        GradientImageType,
    ].New()
    tv.SetInput(input_image)
    tv.SetGamma(args_info.gamma)
    tv.SetNumberOfIterations(args_info.niter)

    # Write
    itk.imwrite(tv.GetOutput(), args_info.output)


def main(argv=None):
    parser = build_parser()
    args_info = parser.parse_args(argv)
    process(args_info)


if __name__ == "__main__":
    main()
