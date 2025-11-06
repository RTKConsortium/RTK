#!/usr/bin/env python
import argparse
import itk
from itk import RTK as rtk


def build_parser():
    parser = rtk.RTKArgumentParser(
        description="Performs total variation denoising along the specified dimensions of a 3D image."
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
    Dimension = 3
    OutputImageType = itk.Image[OutputPixelType, Dimension]

    # Read input
    input_img = itk.imread(args_info.input)

    # Compute total variation before denoising
    if args_info.verbose:
        tv = rtk.TotalVariationImageFilter[OutputImageType].New()
        tv.SetInput(input_img)
        tv.Update()
        print(f"TV before denoising = {tv.GetTotalVariation()}")

    if hasattr(itk, "CudaImage"):
        tvdenoise = rtk.CudaTotalVariationDenoisingBPDQImageFilter.New()
        tvdenoise.SetInput(itk.cuda_image_from_image(input_img))
    else:
        GradPixelType = itk.CovariantVector[OutputPixelType, Dimension]
        GradImageType = itk.Image[GradPixelType, Dimension]
        tvdenoise = rtk.TotalVariationDenoisingBPDQImageFilter[
            OutputImageType, GradImageType
        ].New()
        tvdenoise.SetInput(input_img)

    tvdenoise.SetGamma(args_info.gamma)
    tvdenoise.SetNumberOfIterations(args_info.niter)

    # Write
    if args_info.verbose:
        print(f"Writing output to {args_info.output}...")

    writer = itk.ImageFileWriter[OutputImageType].New()
    writer.SetFileName(args_info.output)
    writer.SetInput(tvdenoise.GetOutput())
    writer.Update()

    # Compute total variation after denoising
    if args_info.verbose:
        tv_after = rtk.TotalVariationImageFilter[OutputImageType].New()
        tv_after.SetInput(tvdenoise.GetOutput())
        tv_after.Update()
        print(f"TV after denoising = {tv_after.GetTotalVariation()}")


def main(argv=None):
    parser = build_parser()
    args_info = parser.parse_args(argv)
    process(args_info)


if __name__ == "__main__":
    main()
