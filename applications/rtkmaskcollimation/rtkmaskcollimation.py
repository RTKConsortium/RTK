#!/usr/bin/env python3
import argparse
import itk
from itk import RTK as rtk


def build_parser():
    parser = rtk.RTKArgumentParser(
        description="Masks out the collimator from the projections"
    )

    parser.add_argument(
        "--geometry", "-g", required=True, help="XML geometry file name"
    )
    parser.add_argument(
        "--output", "-o", required=True, help="Output projections file name"
    )

    rtk.add_rtkinputprojections_group(parser)

    return parser


def process(args_info: argparse.Namespace):
    Dimension = 3
    OutputImageType = itk.Image[itk.F, Dimension]

    # Geometry
    if args_info.verbose:
        print(f"Reading geometry information from {args_info.geometry}...")
    geometry = rtk.read_geometry(args_info.geometry)

    # Projections reader
    reader = rtk.ProjectionsReader[OutputImageType].New()
    rtk.SetProjectionsReaderFromArgParse(reader, args_info)

    # Create projection image filter
    ofm = rtk.MaskCollimationImageFilter[OutputImageType, OutputImageType].New()
    ofm.SetInput(reader.GetOutput())
    ofm.SetGeometry(geometry)

    # Write
    writer = itk.ImageFileWriter[OutputImageType].New()
    writer.SetFileName(args_info.output)
    writer.SetInput(ofm.GetOutput())
    if args_info.verbose:
        print("Projecting and writing...")
    writer.UpdateOutputInformation()
    writer.SetNumberOfStreamDivisions(
        reader.GetOutput().GetLargestPossibleRegion().GetSize()[2]
    )
    writer.Update()

    return 0


def main(argv=None):
    parser = build_parser()
    args_info = parser.parse_args(argv)
    return process(args_info)


if __name__ == "__main__":
    main()
