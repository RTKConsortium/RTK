import argparse
import itk
from itk import RTK as rtk


def build_parser():
    # Argument parsing
    parser = rtk.RTKArgumentParser(
        description="Computes projections through a 3D Shepp & Logan phantom, according to a geometry"
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
        "--phantomscale",
        help="Scaling factor for the phantom dimensions",
        type=float,
        nargs="+",
        default=[128],
    )
    parser.add_argument("--noise", help="Gaussian noise parameter (SD)", type=float)
    parser.add_argument(
        "--offset",
        help="3D spatial offset of the phantom center",
        type=float,
        nargs="+",
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
        print(f"Reading geometry information from {args_info.geometry}")
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
        for i in range((min(len(args_info.offset), Dimension))):
            offset[i] = args_info.offset[i]

    scale = [args_info.phantomscale[0]] * 3
    for i in range(min(len(args_info.phantomscale), Dimension)):
        scale[i] = args_info.phantomscale[i]

    slp = rtk.SheppLoganPhantomFilter[OutputImageType, OutputImageType].New()
    slp.SetInput(constantImageSource.GetOutput())
    slp.SetGeometry(geometry)
    slp.SetPhantomScale(scale)
    slp.SetOriginOffset(offset)
    slp.Update()

    output = slp.GetOutput()

    # Add noise
    if args_info.noise:
        noisy = rtk.AdditiveGaussianNoiseImageFilter[
            OutputImageType, OutputImageType
        ].New()
        noisy.SetInput(slp.GetOutput())
        noisy.SetMean(0.0)
        noisy.SetStandardDeviation(args_info.noise)
        noisy.Update()
        output = noisy.GetOutput()

    # Write
    if args_info.verbose:
        print("Projecting and writing... ")

    itk.imwrite(output, args_info.output)


def main(argv=None):
    parser = build_parser()
    args_info = parser.parse_args(argv)
    process(args_info)


if __name__ == "__main__":
    main()
