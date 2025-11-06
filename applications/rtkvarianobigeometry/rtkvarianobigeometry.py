#!/usr/bin/env python
import argparse
import itk
from itk import RTK as rtk


def build_parser():
    # Argument parsing
    parser = rtk.RTKArgumentParser(
        description="Creates an RTK geometry file from a Varian OBI acquisition."
    )

    parser.add_argument(
        "--xml_file",
        "-x",
        help="Varian OBI XML information file on projections",
        required=True,
    )
    parser.add_argument("--output", "-o", help="Output file name", required=True)
    parser.add_argument(
        "--path", "-p", help="Path containing projections", required=True
    )
    parser.add_argument(
        "--regexp",
        "-r",
        help="Regular expression to select projection files in path",
        required=True,
    )

    # Parse the command line arguments
    return parser


def process(args: argparse.Namespace):

    names = itk.RegularExpressionSeriesFileNames.New()
    names.SetDirectory(args.path)
    names.SetRegularExpression(args.regexp)

    reader = rtk.VarianObiGeometryReader.New()
    reader.SetXMLFileName(args.xml_file)
    reader.SetProjectionsFileNames(names.GetFileNames())
    reader.UpdateOutputData()

    rtk.write_geometry(reader.GetGeometry(), args.output)


def main(argv=None):
    parser = build_parser()
    args_info = parser.parse_args(argv)
    process(args_info)


if __name__ == "__main__":
    main()
