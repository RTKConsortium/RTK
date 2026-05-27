#!/usr/bin/env python
import argparse
import itk
from itk import RTK as rtk


def build_parser():
    # Argument parsing
    parser = rtk.RTKArgumentParser(
        description="Creates an RTK geometry file from a sequence of ora.xml files (radART / medPhoton file format)."
    )
    parser.add_argument("--output", "-o", help="Output file name", required=True)
    parser.add_argument(
        "--margin",
        "-m",
        help="Collimation margin (uinf, usup, vinf, vsup)",
        type=float,
        nargs="+",
        default=[0.0],
    )
    parser.add_argument(
        "--optitrack",
        help="OptiTrack object ID (unused by default)",
        type=int,
        default=-1,
    )
    parser.add_argument(
        "--nsort",
        help="Numeric sort for regular expression matches",
        action="store_true",
    )
    parser.add_argument(
        "--submatch",
        help="Index of the submatch that will be used to sort matches",
        type=int,
        default=0,
    )

    projection_group = parser.add_argument_group("Projections")
    projection_group.add_argument(
        "--path", "-p", help="Path containing projections", required=True
    )
    projection_group.add_argument(
        "--regexp",
        "-r",
        help="Regular expression to select projection files in path",
        required=True,
    )
    

    return parser


def process(args: argparse.Namespace):

    margin = itk.Vector[itk.D, 4]()

    margin.Fill(args.margin[0])
    for i in range(min(len(args.margin), 4)):
        margin[i] = float(args.margin[i])

    # Create geometry reader
    fileNames = rtk.GetProjectionsFileNamesFromArgParse(args)

    oraReader = rtk.OraGeometryReader.New()
    oraReader.SetProjectionsFileNames(fileNames)
    oraReader.SetCollimationMargin(margin)
    oraReader.SetOptiTrackObjectID(getattr(args, "optitrack", -1))
    oraReader.UpdateOutputData()

    rtk.write_geometry(oraReader.GetGeometry(), args.output)


def main(argv=None):
    parser = build_parser()
    args_info = parser.parse_args(argv)
    process(args_info)


if __name__ == "__main__":
    main()
