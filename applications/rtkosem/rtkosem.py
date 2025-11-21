#!/usr/bin/env python3
import argparse
import itk
from itk import RTK as rtk


def build_parser():
    parser = rtk.RTKArgumentParser(
        description="Reconstructs a 3D volume from a sequence of projections with Ordered subset expectation maximization."
    )

    parser.add_argument(
        "--geometry", "-g", help="XML geometry file name", type=str, required=True
    )
    parser.add_argument(
        "--output", "-o", help="Output file name", type=str, required=True
    )
    parser.add_argument(
        "--niterations", "-n", help="Number of iterations", type=int, default=5
    )
    parser.add_argument("--input", "-i", help="Input volume", type=str)
    parser.add_argument(
        "--nprojpersubset",
        help="Number of projections processed between each update of the reconstructed volume (several for OSEM, all for MLEM)",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--betaregularization",
        help="Hyperparameter for the regularization",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--nostorenormalizationimages",
        help="Do not store the normalization images during the reconstruction",
        action="store_true",
    )

    # RTK specific groups
    rtk.add_rtkinputprojections_group(parser)
    rtk.add_rtkprojectors_group(parser)
    rtk.add_rtk3Doutputimage_group(parser)

    return parser


def process(args_info: argparse.Namespace):
    # Define pixel type and dimension
    OutputPixelType = itk.F
    Dimension = 3
    InputImageType = itk.Image[OutputPixelType, Dimension]

    # Projections reader
    reader = rtk.ProjectionsReader[InputImageType].New()
    rtk.SetProjectionsReaderFromArgParse(reader, args_info)
    if args_info.verbose:
        print("Reading projections...")
    reader.Update()
    projections = reader.GetOutput()

    # Geometry
    if args_info.verbose:
        print(f"Reading geometry information from {args_info.geometry}...")
    geometry = rtk.read_geometry(args_info.geometry)

    # Create input: either an existing volume read from a file or a blank image
    if args_info.input:
        if args_info.verbose:
            print(f"Reading input volume from {args_info.input}...")
        inputFilter = itk.imread(args_info.input)
    else:
        # Create new empty volume
        constant = rtk.ConstantImageSource[InputImageType].New()
        rtk.SetConstantImageSourceFromArgParse(constant, args_info)
        constant.SetConstant(1.0)
        constant.Update()
        inputFilter = constant.GetOutput()

    # OSEM reconstruction filter
    if hasattr(itk, "CudaImage"):
        CudaImageType = itk.CudaImage[OutputPixelType, Dimension]
        osem = rtk.OSEMConeBeamReconstructionFilter[CudaImageType, CudaImageType].New()
        osem.SetInput(itk.cuda_image_from_image(inputFilter))
        osem.SetInput(1, itk.cuda_image_from_image(projections))
    else:
        osem = rtk.OSEMConeBeamReconstructionFilter[
            InputImageType, InputImageType
        ].New()
        osem.SetInput(inputFilter)
        osem.SetInput(1, projections)

    osem.SetGeometry(geometry)

    # Set the forward and back projection filters
    rtk.SetForwardProjectionFromArgParse(args_info, osem)
    rtk.SetBackProjectionFromArgParse(args_info, osem)

    if args_info.betaregularization is not None:
        osem.SetBetaRegularization(float(args_info.betaregularization))

    osem.SetNumberOfIterations(args_info.niterations)
    osem.SetNumberOfProjectionsPerSubset(args_info.nprojpersubset)
    osem.SetStoreNormalizationImages(not args_info.nostorenormalizationimages)

    # Report iterations
    progressCommand = rtk.PercentageProgressCommand(osem)
    osem.AddObserver(
        itk.ProgressEvent(), progressCommand.callback
    )  # Register the callback for progress
    osem.AddObserver(itk.EndEvent(), progressCommand.End)  # Register end notification

    # Write
    if args_info.verbose:
        print("Reconstructing and writing...")
    WriterType = itk.ImageFileWriter[InputImageType].New()
    WriterType.SetFileName(args_info.output)
    WriterType.SetInput(osem.GetOutput())
    WriterType.Update()


def main(argv=None):
    parser = build_parser()
    args_info = parser.parse_args(argv)
    process(args_info)


if __name__ == "__main__":
    main()
