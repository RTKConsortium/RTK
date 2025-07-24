import argparse
import sys
import itk
from itk import RTK as rtk
import numpy as np


def build_parser():
    # Argument parsing
    parser = argparse.ArgumentParser(
        description="Computes projections through a 3D phantom described by a file, according to a geometry"
    )
    # General options
    parser.add_argument(
        "--verbose", "-v", help="Verbose execution", action="store_true"
    )
    parser.add_argument(
        "--geometry", "-g", help="XML geometry file name", type=str, required=True
    )
    parser.add_argument(
        "--output", "-o", help="Output projections file name", type=str, required=True
    )
    parser.add_argument(
        "--phantomfile",
        help="Configuration parameters for the phantom",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--phantomscale",
        help="Scaling factor for the phantom dimensions",
        type=rtk.comma_separated_args(float),
        default=[1.0],
    )
    parser.add_argument(
        "--offset",
        help="3D spatial offset of the phantom center",
        type=rtk.comma_separated_args(float),
    )
    parser.add_argument(
        "--rotation",
        help="Rotation matrix for the phantom",
        type=rtk.comma_separated_args(float),
    )

    rtk.add_rtk3Doutputimage_group(parser)

    # Parse the command line arguments
    return parser


def process(args_info: argparse.Namespace):

    OutputPixelType = itk.F
    Dimension = 3
    OutputImageType = itk.Image[OutputPixelType, Dimension]

    # Geometry
    if args_info.verbose:
        print(f"Reading geometry information from {args_info.geometry}...")
    geometry = rtk.read_geometry(args_info.geometry)

    # Create a stack of empty projection images
    constantImageSource = rtk.ConstantImageSource[OutputImageType].New()
    rtk.SetConstantImageSourceFromArgParse(constantImageSource, args_info)

    # Adjust size according to geometry
    sizeOutput = constantImageSource.GetSize()
    sizeOutput[2] = len(geometry.GetGantryAngles())
    constantImageSource.SetSize(sizeOutput)

    # Create SheppLogan Phantom
    offset = [0] * Dimension
    if args_info.offset is not None:
        if len(args_info.offset) > 3:
            print("--offset needs up to 3 values")
            sys.exit(1)
        for i in range(len(args_info.offset)):
            offset[i] = args_info.offset[i]

    scale = [args_info.phantomscale[0]] * 3
    if len(args_info.phantomscale) > 3:
        print("--phantomscale needs up to 3 values")
        sys.exit(1)
    for i in range(len(args_info.phantomscale)):
        scale[i] = args_info.phantomscale[i]

    rot = itk.Matrix[itk.D, Dimension, Dimension]()
    rot.SetIdentity()

    if args_info.rotation is not None:
        if len(args_info.rotation) != 9:
            print("--rotation needs exactly 9 values")
            sys.exit(1)
        itk.matrix_from_array(np.array(args_info.rotation).reshape(3, 3))

    ppc = rtk.ProjectGeometricPhantomImageFilter[OutputImageType, OutputImageType].New()
    ppc.SetInput(constantImageSource.GetOutput())
    ppc.SetGeometry(geometry)
    ppc.SetPhantomScale(scale)
    ppc.SetOriginOffset(offset)
    ppc.SetRotationMatrix(rot)
    ppc.SetConfigFile(args_info.phantomfile)
    ppc.Update()

    # Write
    if args_info.verbose:
        print("Projecting and writing... ")

    itk.imwrite(ppc.GetOutput(), args_info.output)


def main(argv=None):
    parser = build_parser()
    args_info = parser.parse_args(argv)
    process(args_info)


if __name__ == "__main__":
    main()
