#!/usr/bin/env python
import argparse
from itk import RTK as rtk


def build_parser():
    parser = rtk.RTKArgumentParser(
        description="Creates an RTK geometry file from an acquisition exported on the XRad system."
    )

    parser.add_argument(
        "--input", "-i", help="Input sinogram header file", type=str, required=True
    )
    parser.add_argument(
        "--output", "-o", help="Output file name", type=str, required=True
    )

    return parser


def process(args_info: argparse.Namespace):
    # Create geometry reader
    if args_info.verbose:
        print(f"Reading XRad header: {args_info.input}")
    reader = rtk.XRadGeometryReader.New()
    reader.SetImageFileName(args_info.input)
    reader.UpdateOutputData()

    # Write
    if args_info.verbose:
        print(f"Writing geometry to: {args_info.output}")
    rtk.write_geometry(reader.GetGeometry(), args_info.output)


def main(argv=None):
    parser = build_parser()
    args_info = parser.parse_args(argv)
    return process(args_info)


if __name__ == "__main__":
    main()
