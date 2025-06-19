#!/usr/bin/env python
import argparse
from itk import RTK as rtk


def build_parser():
    # Argument parsing
    parser = rtk.RTKArgumentParser(
        description="Creates an RTK geometry file from an Elekta Synergy acquisition."
    )

    parser.add_argument("--verbose", "-v", help="Verbose execution", type=bool)
    parser.add_argument(
        "--xml", "-x", help="XML file name (starting with XVI5)", required=True
    )
    parser.add_argument("--output", "-o", help="Output file name", required=True)

    return parser


def process(args_info: argparse.Namespace):

    reader = rtk.ElektaXVI5GeometryXMLFileReader.New()
    reader.SetFilename(args_info.xml)
    reader.GenerateOutputInformation()

    rtk.write_geometry(reader.GetGeometry(), args_info.output)


def main(argv=None):
    parser = build_parser()
    args_info = parser.parse_args(argv)
    process(args_info)


if __name__ == "__main__":
    main()
