#!/usr/bin/env python
import argparse
import itk
from itk import RTK as rtk


def build_parser():
    parser = rtk.RTKArgumentParser(
        description="Creates an RTK geometry file from an iMagX acquisition."
    )

    parser.add_argument("--calibration", "-c", help="iMagX Calibration file")
    parser.add_argument("--room_setup", "-s", help="iMagX room setup file")
    parser.add_argument("--output", "-o", required=True, help="Output file name")
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Verbose execution"
    )

    # Projections group (path, regexp, nsort, submatch)
    rtk.add_rtkinputprojections_group(parser)

    return parser


def process(args_info: argparse.Namespace):
    Dimension = 3

    # Create geometry reader
    imagxReader = rtk.ImagXGeometryReader[itk.Image[itk.F, Dimension]].New()
    fileNames = rtk.GetProjectionsFileNamesFromArgParse(args_info)
    imagxReader.SetProjectionsFileNames(fileNames)

    if getattr(args_info, "calibration", None):
        imagxReader.SetCalibrationXMLFileName(args_info.calibration)
    if getattr(args_info, "room_setup", None):
        imagxReader.SetRoomXMLFileName(args_info.room_setup)

    imagxReader.UpdateOutputData()
    # Write
    rtk.WriteGeometry(imagxReader.GetGeometry(), args_info.output)

    return 0


def main(argv=None):
    parser = build_parser()
    args_info = parser.parse_args(argv)
    return process(args_info)


if __name__ == "__main__":
    main()
