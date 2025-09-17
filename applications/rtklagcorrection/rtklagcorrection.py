#!/usr/bin/env python3
import argparse
import itk
from itk import RTK as rtk


def build_parser():
    parser = rtk.RTKArgumentParser(description="4th order LTI Lag correction")

    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose execution")
    parser.add_argument("--output", "-o", required=True, help="Output file name")
    parser.add_argument("--rates", "-a", nargs="+", type=float, required=True, help="Lag rates (a0 a1 a2 a3)")
    parser.add_argument("--coefficients", "-c", nargs="+", type=float, required=True, help="Lag coefficients (b0 b1 b2 b3)")

    rtk.add_rtkinputprojections_group(parser)

    return parser


def process(args_info: argparse.Namespace):
    Dimension = 3
    VModelOrder = 4

    OutputPixelType = itk.F
    OutputImageType = itk.Image[OutputPixelType, Dimension]
    VectorType = itk.Vector[itk.F, VModelOrder] # Parameter type always float/double

    # Projections reader
    reader = rtk.ProjectionsReader[OutputImageType].New()
    rtk.SetProjectionsReaderFromArgParse(reader, args_info)
    reader.ComputeLineIntegralOff() # Don't want to preprocess data
    reader.SetFileNames(rtk.GetProjectionsFileNamesFromArgParse(args_info))
    reader.Update()


    # Validate coefficients/rates
    if len(args_info.coefficients) != VModelOrder or len(args_info.rates) != VModelOrder:
        print(f"Expecting {VModelOrder} lag rates and coefficients values")
        return 1

    a = VectorType()
    b = VectorType()
    for i in range(VModelOrder):
        a[i] = float(args_info.rates[i])
        b[i] = float(args_info.coefficients[i])

    if hasattr(itk, 'CudaImage'):
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
    itk.imwrite(streamer.GetOutput(), args_info.output)

    if args_info.verbose:
        print('Wrote corrected projections to', args_info.output)

    return 0


def main(argv=None):
    parser = build_parser()
    args_info = parser.parse_args_info(argv)
    return process(args_info)


if __name__ == '__main__':
    main()
