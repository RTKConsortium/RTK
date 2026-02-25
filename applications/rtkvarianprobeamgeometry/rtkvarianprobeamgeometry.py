#!/usr/bin/env python
import argparse
from itk import RTK as rtk


def build_parser():
    parser = rtk.RTKArgumentParser(
        description=("Creates an RTK geometry file from a Varian ProBeam acquisition.")
    )

    parser.add_argument(
        "--xml_file",
        "-x",
        help="Varian ProBeam XML information file on projections",
        required=True,
    )
    parser.add_argument(
        "--output",
        "-o",
        help="Output file name",
        required=True,
    )
    rtk.add_rtkinputprojections_group(parser)

    return parser


def process(args_info: argparse.Namespace):
    # Create geometry reader
    if args_info.verbose:
        print(f"Reading Varian ProBeam XML: {args_info.xml_file}")

    reader = rtk.VarianProBeamGeometryReader.New()
    reader.SetXMLFileName(args_info.xml_file)

    fileNames = rtk.GetProjectionsFileNamesFromArgParse(args_info)
    reader.SetProjectionsFileNames(fileNames)

    reader.UpdateOutputData()

    # Write geometry
    if args_info.verbose:
        print(f"Writing geometry to: {args_info.output}")
    rtk.write_geometry(reader.GetGeometry(), args_info.output)


def main(argv=None):
    parser = build_parser()
    args_info = parser.parse_args(argv)
    return process(args_info)


if __name__ == "__main__":
    main()
