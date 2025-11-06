#!/usr/bin/env python
import sys
import argparse
import itk
from itk import RTK as rtk


def build_parser():
    parser = rtk.RTKArgumentParser(
        description="Creates an Amsterdam Shroud image from a sequence of projections [Zijp et al, ICCR, 2004]."
    )

    # General options
    parser.add_argument(
        "--output", "-o", help="Output file name", type=str, required=True
    )
    parser.add_argument(
        "--unsharp", "-u", help="Unsharp mask size", type=int, default=17
    )
    parser.add_argument(
        "--clipbox",
        "-c",
        help="3D clipbox for cropping projections (x1, x2, y1, y2, z1, z2) in mm",
        type=float,
        nargs="+",
    )
    parser.add_argument("--geometry", "-g", help="XML geometry file name", type=str)

    rtk.add_rtkinputprojections_group(parser)

    # Parse the command line arguments
    return parser


def process(args_info: argparse.Namespace):

    if args_info.verbose:
        print("Running Amsterdam Shroud computation...")

    # Define output image type
    OutputPixelType = itk.F
    Dimension = 3
    OutputImageType = itk.Image[OutputPixelType, Dimension]

    # Projections reader
    reader = rtk.ProjectionsReader[OutputImageType].New()
    rtk.SetProjectionsReaderFromArgParse(reader, args_info)

    # Amsterdam Shroud
    shroudFilter = rtk.AmsterdamShroudImageFilter[OutputImageType].New()
    shroudFilter.SetInput(reader.GetOutput())
    shroudFilter.SetUnsharpMaskSize(args_info.unsharp)

    # Corners (if given)
    if args_info.clipbox:
        if len(args_info.clipbox) != 6:
            print(
                "--clipbox requires exactly 6 values, only ",
                args_info.clipbox,
                " given.",
            )
            sys.exit(1)
        if not args_info.geometry:
            print("You must provide the geometry to use --clipbox.")
            sys.exit(1)

        c1 = itk.Point[itk.F, Dimension](args_info.clipbox[0:6:2])
        c2 = itk.Point[itk.F, Dimension](args_info.clipbox[1:6:2])
        shroudFilter.SetCorner1(c1)
        shroudFilter.SetCorner2(c2)

        # Geometry
        if args_info.geometry:
            geometry = rtk.read_geometry(args_info.geometry)
            shroudFilter.SetGeometry(geometry)

    shroudFilter.UpdateOutputInformation()

    # Write output
    writer = itk.ImageFileWriter[itk.Image[itk.D, 2]].New()
    writer.SetFileName(args_info.output)
    writer.SetInput(shroudFilter.GetOutput())
    writer.SetNumberOfStreamDivisions(
        shroudFilter.GetOutput().GetLargestPossibleRegion().GetSize()[1]
    )

    writer.Update()


def main(argv=None):
    parser = build_parser()
    args_info = parser.parse_args(argv)
    process(args_info)


if __name__ == "__main__":
    main()
