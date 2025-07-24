import argparse
import itk
from itk import RTK as rtk


def build_parser():
    parser = argparse.ArgumentParser(
        description="Reads raw projection images, converts them to attenuation and stacks them into a single output image file"
    )
    parser.add_argument(
        "--verbose", "-v", help="Verbose execution", action="store_true"
    )
    parser.add_argument(
        "--output", "-o", help="Output file name", type=str, required=True
    )
    rtk.add_rtkinputprojections_group(parser)

    # Parse the command line arguments
    return parser


def process(args_info: argparse.Namespace):

    OutputPixelType = itk.F
    Dimension = 3
    OutputImageType = itk.Image[OutputPixelType, Dimension]

    # Projections reader
    reader = rtk.ProjectionsReader[OutputImageType].New()
    rtk.SetProjectionsReaderFromArgParse(reader, args_info)
    if args_info.verbose:
        print(f"Reading projections...")

    # Write
    writer = itk.ImageFileWriter[OutputImageType].New()
    writer.SetFileName(args_info.output)
    writer.SetInput(reader.GetOutput())
    if args_info.verbose:
        print(f"Writing output to: {args_info.output}")
    writer.Update()


def main(argv=None):
    parser = build_parser()
    args_info = parser.parse_args(argv)
    process(args_info)


if __name__ == "__main__":
    main()
