#!/usr/bin/env python
import argparse
import itk
from itk import RTK as rtk


def build_parser():
    # Argument parsing
    parser = argparse.ArgumentParser(
        description="Creates an RTK geometry file from a Varian OBI acquisition."
    )

    parser.add_argument("--verbose", "-v", help="Verbose execution", type=bool)
    parser.add_argument(
        "--xml_file", "-x", help="Varian OBI XML information file on projections"
    )
    parser.add_argument("--output", "-o", help="Output file name", required=True)
    parser.add_argument(
        "--path", "-p", help="Path containing projections", required=True
    )
    parser.add_argument(
        "--regexp", "-r", help="Regular expression to select projection files in path"
    )
    parser.add_argument(
        "--margin",
        "-m",
        help="Collimation margin (uinf, usup, vinf, vsup)",
        type=rtk.comma_separated_args(float),
        default=0.0,
    )

    # Parse the command line arguments
    return parser


def process(args: argparse.Namespace):

    margin = args.margin
    if type(margin) is float:
        margin = [margin]
    for i in range(len(margin), 4, 1):
        margin.append(margin[0])
    print(margin)

    names = itk.RegularExpressionSeriesFileNames.New()
    names.SetDirectory(args.path)
    names.SetRegularExpression(args.regexp)

    reader = rtk.OraGeometryReader.New()
    reader.SetProjectionsFileNames(names.GetFileNames())
    reader.SetCollimationMargin(margin)
    reader.UpdateOutputData()

    rtk.write_geometry(reader.GetGeometry(), args.output)


def main(argv=None):
    parser = build_parser()
    args_info = parser.parse_args(argv)
    process(args_info)


if __name__ == "__main__":
    main()
