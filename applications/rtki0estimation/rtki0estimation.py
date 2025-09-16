#!/usr/bin/env python3
import argparse
import itk
from itk import RTK as rtk


def build_parser():
    parser = rtk.RTKArgumentParser(description="Estimate I0 from projections")

    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Verbose execution"
    )
    parser.add_argument("--debug", "-d", help="Debug CSV output file name")
    parser.add_argument(
        "--range",
        nargs=3,
        type=int,
        help="Range of projection to analyse min step max",
        metavar=("MIN", "STEP", "MAX"),
    )

    parser.add_argument(
        "--lambda",
        "-l",
        dest="lambda_",
        type=float,
        default=0.8,
        help="RLS estimate coefficient",
    )
    parser.add_argument(
        "--expected",
        "-e",
        dest="expected",
        type=int,
        default=65535,
        help="Expected I0 value",
    )

    rtk.add_rtkinputprojections_group(parser)

    return parser


def process(args_info: argparse.Namespace):
    Dimension = 3
    InputImageType = itk.Image[itk.F, Dimension]
    USImageType = itk.Image[itk.US, Dimension]

    reader = rtk.ProjectionsReader[InputImageType].New()
    rtk.SetProjectionsReaderFromArgParse(reader, args_info)
    reader.SetFileNames(rtk.GetProjectionsFileNamesFromArgParse(args_info))
    reader.UpdateOutputInformation()

    subsetRegion = reader.GetOutput().GetLargestPossibleRegion()

    ExtractFilterType = itk.ExtractImageFilter[InputImageType, InputImageType]
    extract = ExtractFilterType.New()
    extract.InPlaceOff()
    extract.SetDirectionCollapseToSubmatrix()
    extract.SetInput(reader.GetOutput())

    extractSize = list(subsetRegion.GetSize())
    extractSize[2] = 1

    istep = 1
    imin = 1
    imax = subsetRegion.GetSize()[2]
    if args_info.range is not None:
        rmin, rstep, rmax = args_info.range
        if (rmin <= rmax) and (istep <= (rmax - rmin)):
            imin = rmin
            istep = rstep
            imax = min(rmax, imax)

    i0est = rtk.I0EstimationProjectionFilter[USImageType, USImageType, 2].New()
    if args_info.lambda_ is not None:
        i0est.SetLambda(args_info.lambda_)
    if args_info.expected is not None and args_info.expected != 65535:
        i0est.SetExpectedI0(args_info.expected)
    i0est.SaveHistogramsOn()

    I0buffer = []

    CastFilterType = itk.CastImageFilter[InputImageType, USImageType]
    cast = CastFilterType.New()
    cast.SetInput(extract.GetOutput())

    start = list(subsetRegion.GetIndex())
    for i in range(imin, imax, istep):
        start[2] = i
        desiredRegion = itk.ImageRegion[Dimension]()
        desiredRegion.SetIndex(start)
        desiredRegion.SetSize(extractSize)

        extract.SetExtractionRegion(desiredRegion)

        i0est.SetInput(cast.GetOutput())
        i0est.UpdateLargestPossibleRegion()

        I0buffer.append(i0est.GetI0())
        I0buffer.append(i0est.GetI0rls())
        I0buffer.append(i0est.GetI0fwhm())

    if args_info.debug:
        with open(args_info.debug, "w") as f:
            f.write(",".join(str(v) for v in I0buffer))

    return 0


def main(argv=None):
    parser = build_parser()
    args_info = parser.parse_args(argv)
    return process(args_info)


if __name__ == "__main__":
    main()
