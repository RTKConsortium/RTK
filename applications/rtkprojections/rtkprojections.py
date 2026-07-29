import argparse
import itk
from itk import RTK as rtk


def build_parser():
    parser = rtk.RTKArgumentParser(
        description="Reads raw projection images, converts them to attenuation and stacks them into a single output image file"
    )

    parser.add_argument(
        "--output", "-o", help="Output file name", type=str, required=True
    )
    rtk.add_rtkinputprojections_group(parser)
    rtk.add_rtknoise_group(parser)

    # Parse the command line arguments
    return parser


def process(args_info: argparse.Namespace):

    OutputPixelType = itk.F
    Dimension = 3
    fileNames = rtk.GetProjectionsFileNamesFromArgParse(args_info)

    inputIsVectorImage = False
    if args_info.component is None and fileNames:
        imageio = itk.ImageIOFactory.CreateImageIO(
            fileNames[0], itk.CommonEnums.IOFileMode_ReadMode
        )
        if imageio is not None:
            imageio.SetFileName(fileNames[0])
            imageio.ReadImageInformation()
            inputIsVectorImage = (
                imageio.GetPixelType() == itk.CommonEnums.IOPixel_VECTOR
            )

    if inputIsVectorImage:
        if args_info.poisson is not None or args_info.gaussian is not None:
            raise RuntimeError("Noise addition is not supported for vector projections")

        OutputImageType = itk.VectorImage[OutputPixelType, Dimension]
        reader = rtk.ProjectionsReader[OutputImageType].New()
        rtk.SetProjectionsReaderFromArgParse(reader, args_info)
        if args_info.verbose:
            print(f"Reading projections...")

        writer = itk.ImageFileWriter[OutputImageType].New()
        writer.SetFileName(args_info.output)
        writer.SetInput(reader.GetOutput())
        if args_info.verbose:
            print(f"Writing output to: {args_info.output}")
        writer.Update()
    else:
        OutputImageType = itk.Image[OutputPixelType, Dimension]

        # Projections reader
        reader = rtk.ProjectionsReader[OutputImageType].New()
        rtk.SetProjectionsReaderFromArgParse(reader, args_info)
        if args_info.verbose:
            print(f"Reading projections...")

        # Add noise
        output = rtk.AddNoiseFromArgParse(reader.GetOutput(), args_info)

        # Write
        writer = itk.ImageFileWriter[OutputImageType].New()
        writer.SetFileName(args_info.output)
        writer.SetInput(output)
        if args_info.verbose:
            print(f"Writing output to: {args_info.output}")
        writer.Update()


def main(argv=None):
    parser = build_parser()
    args_info = parser.parse_args(argv)
    process(args_info)


if __name__ == "__main__":
    main()
