#!/usr/bin/env python
import argparse
import sys
import itk
from itk import RTK as rtk


def build_parser():
    parser = rtk.RTKArgumentParser(description="4th order LTI Lag correction")

    parser.add_argument(
        "--output", "-o", help="Output file name", type=str, required=True
    )
    parser.add_argument(
        "--rates",
        "-a",
        help="Lag rates (a0, a1, a2, a3)",
        type=float,
        nargs="+",
        required=True,
    )
    parser.add_argument(
        "--coefficients",
        "-c",
        help="Lag coefficients (b0, b1, b2, b3)",
        type=float,
        nargs="+",
        required=True,
    )

    rtk.add_rtkinputprojections_group(parser)

    return parser


def process(args_info: argparse.Namespace):
    Dimension = 3
    VModelOrder = 4
    OutputImageType = itk.Image[itk.US, Dimension]

    if (
        len(args_info.rates) != VModelOrder
        or len(args_info.coefficients) != VModelOrder
    ):
        print("Expecting 4 lag rates and coefficients values")
        sys.exit(1)

    # Projections reader
    reader = rtk.ProjectionsReader[OutputImageType].New()
    rtk.SetProjectionsReaderFromArgParse(reader, args_info)
    reader.ComputeLineIntegralOff()  # Don't want to preprocess data
    reader.SetFileNames(rtk.GetProjectionsFileNamesFromArgParse(args_info))
    reader.Update()

    a = itk.Vector[itk.F, VModelOrder]()  # Parameter type always float/double
    b = itk.Vector[itk.F, VModelOrder]()
    for i in range(VModelOrder):
        a[i] = args_info.rates[i]
        b[i] = args_info.coefficients[i]

    # Lag correction filter
    if hasattr(itk, "CudaImage"):
        lagfilter = rtk.CudaLagCorrectionImageFilter.New()
        lagfilter.SetInput(itk.cuda_image_from_image(reader.GetOutput()))
    else:
        lagfilter = rtk.LagCorrectionImageFilter[OutputImageType, VModelOrder].New()
        lagfilter.SetInput(reader.GetOutput())

    lagfilter.SetCoefficients(a, b)
    lagfilter.InPlaceOff()
    lagfilter.Update()

    # Streaming filter
    streamer = itk.StreamingImageFilter[OutputImageType, OutputImageType].New()
    streamer.SetInput(lagfilter.GetOutput())
    streamer.SetNumberOfStreamDivisions(100)

    # Save corrected projections
    writer = itk.ImageFileWriter[OutputImageType].New()
    writer.SetFileName(args_info.output)
    writer.SetInput(streamer.GetOutput())
    writer.Update()


def main(argv=None):
    parser = build_parser()
    args_info = parser.parse_args(argv)
    process(args_info)


if __name__ == "__main__":
    main()
