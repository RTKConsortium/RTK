#!/usr/bin/env python
import argparse
import itk
from itk import RTK as rtk


def build_parser():
    # argument parsing
    parser = argparse.ArgumentParser(
        description="Reconstructs a 3D volume from a sequence of projections with a conjugate gradient technique",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--verbose", "-v", help="Verbose execution", action="store_true"
    )
    parser.add_argument("--geometry", "-g", help="Geometry file name", required=True)
    parser.add_argument("--output", "-o", help="Output file name", required=True)
    parser.add_argument(
        "--niterations", "-n", type=int, default=5, help="Number of iterations"
    )
    parser.add_argument("--input", "-i", help="Input volume")
    parser.add_argument(
        "--weights", "-w", help="Weights file for Weighted Least Squares (WLS)"
    )
    parser.add_argument("--gamma", help="Laplacian regularization weight")
    parser.add_argument("--tikhonov", help="Tikhonov regularization weight")
    parser.add_argument(
        "--nocudacg",
        help="Do not perform conjugate gradient calculations on GPU",
        action="store_true",
    )
    parser.add_argument(
        "--mask",
        "-m",
        help="Apply a support binary mask: reconstruction kept null outside the mask)",
    )
    parser.add_argument(
        "--nodisplaced",
        help="Disable the displaced detector filter",
        action="store_true",
    )

    rtk.add_rtkinputprojections_group(parser)
    rtk.add_rtk3Doutputimage_group(parser)
    rtk.add_rtkprojectors_group(parser)
    rtk.add_rtkiterations_group(parser)

    # Parse the command line arguments
    return parser

def process(args_info: argparse.Namespace):

    OutputPixelType = itk.F
    Dimension = 3

    OutputImageType = itk.Image[OutputPixelType, Dimension]

    # Projections reader
    reader = rtk.ProjectionsReader[OutputImageType].New()
    rtk.SetProjectionsReaderFromArgParse(reader, args_info)

    # Geometry
    if args_info.verbose:
        print(f"Reading geometry information from {args_info.geometry}")
    geometry = rtk.read_geometry(args_info.geometry)

    # Create input: either an existing volume read from a file or a blank image
    if args_info.input is not None:
        # Read an existing image to initialize the volume
        InputReaderType = itk.ImageFileReader[OutputImageType]
        inputReader = InputReaderType.New()
        inputReader.SetFileName(args_info.input)
        inputFilter = inputReader
    else:
        # Create new empty volume
        ConstantImageSourceType = rtk.ConstantImageSource[OutputImageType]
        constantImageSource = ConstantImageSourceType.New()
        rtk.SetConstantImageSourceFromArgParse(constantImageSource, args_info)
        inputFilter = constantImageSource

    # Read weights if given, otherwise default to weights all equal to one
    if args_info.weights is not None:
        WeightsReaderType = itk.ImageFileReader[OutputImageType]
        weightsReader = WeightsReaderType.New()
        weightsReader.SetFileName(args_info.weights)
        weightsSource = weightsReader
    else:
        ConstantWeightsSourceType = rtk.ConstantImageSource[OutputImageType]
        constantWeightsSource = ConstantWeightsSourceType.New()

        # Set the weights to be like the projections
        reader.UpdateOutputInformation()

        constantWeightsSource.SetInformationFromImage(reader.GetOutput())
        constantWeightsSource.SetConstant(1.0)
        weightsSource = constantWeightsSource

    # Read Support Mask if given
    if args_info.mask is not None:
        supportmask = itk.imread(args_info.mask)

    # Set the forward and back projection filters to be used
    if hasattr(itk, "CudaImage"):
        OutputCudaImageType = itk.CudaImage[OutputPixelType, Dimension]
        ConjugateGradientFilterType = rtk.ConjugateGradientConeBeamReconstructionFilter[
            OutputCudaImageType
        ]
        conjugategradient = ConjugateGradientFilterType.New()
        conjugategradient.SetInput(itk.cuda_image_from_image(inputFilter.GetOutput()))
        conjugategradient.SetInput(1, itk.cuda_image_from_image(reader.GetOutput()))
        conjugategradient.SetInput(
            2, itk.cuda_image_from_image(weightsSource.GetOutput())
        )
        conjugategradient.SetCudaConjugateGradient(not args_info.nocudacg)
        if args_info.mask is not None:
            conjugategradient.SetSupportMask(itk.cuda_image_from_image(supportmask))
    else:
        ConjugateGradientFilterType = rtk.ConjugateGradientConeBeamReconstructionFilter[
            OutputImageType
        ]
        conjugategradient = ConjugateGradientFilterType.New()
        conjugategradient.SetInput(inputFilter.GetOutput())
        conjugategradient.SetInput(1, reader.GetOutput())
        conjugategradient.SetInput(2, weightsSource.GetOutput())
        if args_info.mask is not None:
            conjugategradient.SetSupportMask(supportmask)

    rtk.SetForwardProjectionFromArgParse(args_info, conjugategradient)
    rtk.SetBackProjectionFromArgParse(args_info, conjugategradient)
    rtk.SetIterationsReportFromArgParse(args_info, conjugategradient)

    if args_info.gamma is not None:
        conjugategradient.SetGamma(args_info.gamma)
    if args_info.tikhonov is not None:
        conjugategradient.SetTikhonov(args_info.tikhonov)

    conjugategradient.SetGeometry(geometry)
    conjugategradient.SetNumberOfIterations(args_info.niterations)
    conjugategradient.SetDisableDisplacedDetectorFilter(args_info.nodisplaced)

    rtk.SetIterationsReportFromArgParse(args_info, conjugategradient)

    conjugategradient.Update()

    # Write
    writer = itk.ImageFileWriter[OutputImageType].New()
    writer.SetFileName(args_info.output)
    writer.SetInput(conjugategradient.GetOutput())
    writer.Update()

def main(argv=None):
    parser = build_parser()
    args_info = parser.parse_args(argv)
    process(args_info)


if __name__ == "__main__":
    main()
