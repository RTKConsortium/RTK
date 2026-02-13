#!/usr/bin/env python
import argparse
import itk
from itk import RTK as rtk


def build_parser():
    parser = rtk.RTKArgumentParser(
        description="Alternates between conjugate gradient reconstruction and regularization"
    )

    parser.add_argument(
        "--geometry", "-g", help="XML geometry file name", type=str, required=True
    )
    parser.add_argument(
        "--output", "-o", help="Output file name", type=str, required=True
    )
    parser.add_argument(
        "--niter",
        "-n",
        help="Number of iterations",
        type=int,
        default=5,
    )

    parser.add_argument("--input", "-i", help="Input volume", type=str)
    parser.add_argument(
        "--weights",
        "-w",
        help="Weights file for Weighted Least Squares (WLS)",
        type=str,
    )

    parser.add_argument(
        "--gammalaplacian",
        help="Laplacian regularization weight",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--tikhonov",
        help="Tikhonov regularization weight",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--cgiter",
        help="Number of conjugate gradient nested iterations",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--nocudacg",
        help="Do not perform conjugate gradient calculations on GPU",
        action="store_true",
    )
    parser.add_argument(
        "--mask",
        "-m",
        help="Apply a support binary mask: reconstruction kept null outside the mask)",
        type=str,
    )
    parser.add_argument(
        "--nodisplaced",
        help="Disable the displaced detector filter",
        action="store_true",
    )

    reg = parser.add_argument_group("Regularization")
    reg.add_argument(
        "--nopositivity",
        help="Do not enforce positivity",
        action="store_true",
    )
    reg.add_argument(
        "--tviter",
        help="Total variation regularization: number of iterations",
        type=int,
        default=10,
    )
    reg.add_argument(
        "--gammatv",
        help="Total variation spatial regularization parameter. The larger, the smoother",
        type=float,
    )
    reg.add_argument(
        "--threshold",
        help="Daubechies wavelets spatial regularization: soft threshold",
        type=float,
    )
    reg.add_argument(
        "--order",
        help="Daubechies wavelets spatial regularization: order of the wavelets",
        type=int,
        default=5,
    )
    reg.add_argument(
        "--levels",
        help="Daubechies wavelets spatial regularization: number of decomposition levels",
        type=int,
        default=3,
    )
    reg.add_argument(
        "--soft",
        help="Soft threshold for image domain sparsity enforcement",
        type=float,
    )

    rtk.add_rtkinputprojections_group(parser)
    rtk.add_rtk3Doutputimage_group(parser)
    rtk.add_rtkprojectors_group(parser)
    rtk.add_rtkiterations_group(parser)

    return parser


def process(args_info: argparse.Namespace) -> None:
    OutputPixelType = itk.F
    Dimension = 3
    OutputImageType = itk.Image[OutputPixelType, Dimension]

    # Projections reader
    reader = rtk.ProjectionsReader[OutputImageType].New()
    rtk.SetProjectionsReaderFromArgParse(reader, args_info)

    # Geometry
    if args_info.verbose:
        print(f"Reading geometry information from {args_info.geometry}...")
    geometry = rtk.read_geometry(args_info.geometry)

    # Input volume: file or constant image
    if args_info.input:
        inputFilter = itk.ImageFileReader[OutputImageType].New()
        inputFilter.SetFileName(args_info.input)
    else:
        inputFilter = rtk.ConstantImageSource[OutputImageType].New()
        rtk.SetConstantImageSourceFromArgParse(inputFilter, args_info)

    inputFilter.Update()

    # Weights: file or constant 1 image matching projections
    if args_info.weights:
        weightsSource = itk.ImageFileReader[OutputImageType].New()
        weightsSource.SetFileName(args_info.weights)
    else:
        weightsSource = rtk.ConstantImageSource[OutputImageType].New()
        reader.UpdateOutputInformation()
        weightsSource.SetInformationFromImage(reader.GetOutput())
        weightsSource.SetConstant(1.0)

    # Support mask
    if args_info.mask:
        supportmask = itk.imread(args_info.mask)

    # Set the forward and back projection filters to be used
    if hasattr(itk, "CudaImage") and not args_info.nocudacg:
        CudaImageType = itk.CudaImage[OutputPixelType, Dimension]
        regularizedConjugateGradient = (
            rtk.RegularizedConjugateGradientConeBeamReconstructionFilter[
                CudaImageType
            ].New()
        )
        regularizedConjugateGradient.SetInputVolume(
            itk.cuda_image_from_image(inputFilter.GetOutput())
        )
        regularizedConjugateGradient.SetInputProjectionStack(
            itk.cuda_image_from_image(reader.GetOutput())
        )
        regularizedConjugateGradient.SetInputWeights(
            itk.cuda_image_from_image(weightsSource.GetOutput())
        )
        if args_info.mask:
            regularizedConjugateGradient.SetSupportMask(
                itk.cuda_image_from_image(supportmask)
            )

    else:
        regularizedConjugateGradient = (
            rtk.RegularizedConjugateGradientConeBeamReconstructionFilter[
                OutputImageType
            ].New()
        )

        # Inputs
        regularizedConjugateGradient.SetInputVolume(inputFilter.GetOutput())
        regularizedConjugateGradient.SetInputProjectionStack(reader.GetOutput())
        regularizedConjugateGradient.SetInputWeights(weightsSource.GetOutput())
        if args_info.mask:
            regularizedConjugateGradient.SetSupportMask(supportmask)

    rtk.SetForwardProjectionFromArgParse(args_info, regularizedConjugateGradient)
    rtk.SetBackProjectionFromArgParse(args_info, regularizedConjugateGradient)

    regularizedConjugateGradient.SetGeometry(geometry)
    regularizedConjugateGradient.SetMainLoop_iterations(args_info.niter)
    regularizedConjugateGradient.SetCG_iterations(args_info.cgiter)
    regularizedConjugateGradient.SetCudaConjugateGradient(not args_info.nocudacg)
    regularizedConjugateGradient.SetDisableDisplacedDetectorFilter(
        args_info.nodisplaced
    )
    regularizedConjugateGradient.SetPerformPositivity(not args_info.nopositivity)
    regularizedConjugateGradient.SetGamma(args_info.gammalaplacian)
    regularizedConjugateGradient.SetTikhonov(args_info.tikhonov)

    # TV
    if args_info.gammatv is not None:
        regularizedConjugateGradient.SetGammaTV(args_info.gammatv)
        regularizedConjugateGradient.SetTV_iterations(args_info.tviter)
        regularizedConjugateGradient.SetPerformTVSpatialDenoising(True)
    else:
        regularizedConjugateGradient.SetPerformTVSpatialDenoising(False)

    # Wavelets
    if args_info.threshold is not None:
        regularizedConjugateGradient.SetSoftThresholdWavelets(args_info.threshold)
        regularizedConjugateGradient.SetOrder(args_info.order)
        regularizedConjugateGradient.SetNumberOfLevels(args_info.levels)
        regularizedConjugateGradient.SetPerformWaveletsSpatialDenoising(True)
    else:
        regularizedConjugateGradient.SetPerformWaveletsSpatialDenoising(False)

    # Image-domain soft threshold
    if args_info.soft is not None:
        regularizedConjugateGradient.SetSoftThresholdOnImage(args_info.soft)
        regularizedConjugateGradient.SetPerformSoftThresholdOnImage(True)
    else:
        regularizedConjugateGradient.SetPerformSoftThresholdOnImage(False)

    # Iteration reporting
    rtk.SetIterationsReportFromArgParse(args_info, regularizedConjugateGradient)

    if args_info.verbose:
        print("Reconstructing...")

    regularizedConjugateGradient.Update()

    if args_info.verbose:
        print(f"Writing output to {args_info.output}...")

    # Write
    writer = itk.ImageFileWriter[OutputImageType].New()
    writer.SetFileName(args_info.output)
    writer.SetInput(regularizedConjugateGradient.GetOutput())
    writer.Update()


def main(argv=None):
    parser = build_parser()
    args_info = parser.parse_args(argv)
    process(args_info)


if __name__ == "__main__":
    main()
