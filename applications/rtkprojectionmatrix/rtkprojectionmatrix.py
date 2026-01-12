#!/usr/bin/env python
import argparse
import sys
import itk
from itk import RTK as rtk


def build_parser():
    parser = rtk.RTKArgumentParser(
        description="Saves the sparse system matrix of rtk::JosephBackProjectionImageFilter in a file. "
        "Only works in 2D."
    )

    # General options
    parser.add_argument(
        "--geometry",
        "-g",
        help="XML geometry file name",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--output",
        "-o",
        help="Output file name for the sparse matrix",
        type=str,
        required=True,
    )

    # RTK specific groups
    rtk.add_rtkinputprojections_group(parser)
    rtk.add_rtk3Doutputimage_group(parser)

    return parser


def process(args_info: argparse.Namespace):
    OutputPixelType = itk.F
    Dimension = 3
    OutputImageType = itk.Image[OutputPixelType, Dimension]

    reader = rtk.ProjectionsReader[OutputImageType].New()
    rtk.SetProjectionsReaderFromArgParse(reader, args_info)

    if args_info.verbose:
        print("Reading projections...")
    reader.Update()

    # Create back projection image filter
    if reader.GetOutput().GetLargestPossibleRegion().GetSize()[1] != 1:
        print(
            "This tool has been designed for 2D, i.e., with one row in the sinogram only."
        )
        sys.exit(1)

    direction = reader.GetOutput().GetDirection()
    if abs(direction[0, 0]) != 1.0 or abs(direction[1, 1]) != 1.0:
        print("Projections with non-diagonal Direction is not handled.")
        sys.exit(1)

    if args_info.verbose:
        print(f"Reading geometry information from {args_info.geometry}...")
    geometry = rtk.ReadGeometry(args_info.geometry)

    constant_image_source = rtk.ConstantImageSource[OutputImageType].New()
    rtk.SetConstantImageSourceFromArgParse(constant_image_source, args_info)
    constant_image_source.Update()

    if constant_image_source.GetOutput().GetLargestPossibleRegion().GetSize()[1] != 3:
        print(
            "This tool has been designed for 2D with Joseph project. "
            "Joseph requires at least 2 slices in the y direction for bilinear interpolation. "
            "To have one slice exactly in front of the row, use 3 slices in the volume "
            "with the central slice in front of the single projection row."
        )
        sys.exit(1)

    direction = constant_image_source.GetOutput().GetDirection()
    if not direction.GetVnlMatrix().is_identity():
        print("Volume with non-identity Direction is not handled.")
        sys.exit(1)

    if reader.GetOutput().GetLargestPossibleRegion().GetSize()[2] != len(
        geometry.GetGantryAngles()
    ):
        print("Number of projections in the geometry and in the stack do not match.")
        sys.exit(1)

    if args_info.verbose:
        print("Backprojecting volume and recording matrix values...")

    backProjection = rtk.JosephBackProjectionImageFilter[
        OutputImageType,
        OutputImageType,
    ].New()
    backProjection.SetInput(constant_image_source.GetOutput())
    backProjection.SetInput(1, reader.GetOutput())
    backProjection.SetGeometry(geometry)
    backProjection.Update()

    if args_info.verbose:
        print("Writing matrix to disk...")

    matlab_matrix = rtk.MatlabSparseMatrix[OutputImageType].New()
    matlab_matrix.Save(args_info.output)


def main(argv=None):
    parser = build_parser()
    args_info = parser.parse_args(argv)
    process(args_info)


if __name__ == "__main__":
    main()
