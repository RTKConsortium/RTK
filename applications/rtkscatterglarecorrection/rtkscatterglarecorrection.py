#!/usr/bin/env python
import argparse
import sys
import itk
from itk import RTK as rtk


def build_parser():
    parser = rtk.RTKArgumentParser(
        description="Reads projection images and correct them for scatter glare"
    )

    parser.add_argument("--output", "-o", help="Output filename", type=str)
    parser.add_argument(
        "--bufferSize",
        help="Number of projections computed at the same time",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--difference",
        "-d",
        help="Output the difference between input and corrected images",
        action="store_true",
    )

    algorithm = parser.add_argument_group("Algorithm parameters")
    algorithm.add_argument(
        "--coefficients",
        "-c",
        help="Deconvolution kernel coefficients",
        type=float,
        nargs="+",
        required=True,
    )

    rtk.add_rtkinputprojections_group(parser)

    return parser


def process(args_info: argparse.Namespace):
    Dimension = 3
    InputImageType = itk.Image[itk.F, Dimension]

    if len(args_info.coefficients) != 2:
        print("--coefficients requires exactly 2 coefficients")
        sys.exit(1)

    reader = rtk.ProjectionsReader[InputImageType].New()
    reader.SetFileNames(rtk.GetProjectionsFileNamesFromArgParse(args_info))
    reader.ComputeLineIntegralOff()
    reader.UpdateOutputInformation()

    # Input projection parameters
    sizeInput = list(reader.GetOutput().GetLargestPossibleRegion().GetSize())
    Nproj = int(sizeInput[2])

    if hasattr(itk, "CudaImage"):
        SFilter = rtk.CudaScatterGlareCorrectionImageFilter.New()
        constantSource = rtk.ConstantImageSource[itk.CudaImage[itk.F, Dimension]].New()
    else:
        SFilter = rtk.ScatterGlareCorrectionImageFilter[
            InputImageType, InputImageType, itk.F
        ].New()
        constantSource = rtk.ConstantImageSource[InputImageType].New()

    SFilter.SetTruncationCorrection(0.0)
    SFilter.SetCoefficients(args_info.coefficients)

    paste = itk.PasteImageFilter[InputImageType].New()
    paste.SetSourceImage(SFilter.GetOutput())
    paste.SetDestinationImage(constantSource.GetOutput())

    if args_info.verbose:
        print("Starting processing")
    projid = 0
    first = True
    while projid < Nproj:
        curBufferSize = min(args_info.bufferSize, Nproj - projid)

        sliceRegionA = reader.GetOutput().GetLargestPossibleRegion()
        sizeA = list(sliceRegionA.GetSize())
        indexA = list(sliceRegionA.GetIndex())
        desiredRegionA = itk.ImageRegion[Dimension]()
        desiredRegionA.SetSize([sizeA[0], sizeA[1], curBufferSize])
        desiredRegionA.SetIndex([indexA[0], indexA[1], projid])

        extract = itk.ExtractImageFilter[InputImageType, InputImageType].New()
        extract.SetDirectionCollapseToIdentity()
        extract.SetExtractionRegion(desiredRegionA)
        extract.SetInput(reader.GetOutput())
        extract.Update()

        image = extract.GetOutput()
        image.DisconnectPipeline()

        if hasattr(itk, "CudaImage"):
            SFilter.SetInput(itk.cuda_image_from_image(image))
        else:
            SFilter.SetInput(image)
        SFilter.GetOutput().SetRequestedRegion(image.GetRequestedRegion())
        SFilter.Update()

        procImage = SFilter.GetOutput()
        procImage.DisconnectPipeline()

        if args_info.difference:
            subtractFilter = itk.SubtractImageFilter[
                InputImageType, InputImageType, InputImageType
            ].New()
            subtractFilter.SetInput1(image)
            subtractFilter.SetInput2(procImage)
            subtractFilter.Update()
            outImage = subtractFilter.GetOutput()
            outImage.DisconnectPipeline()
        else:
            outImage = procImage

        current_idx = list(outImage.GetLargestPossibleRegion().GetIndex())
        current_idx[2] = projid

        if first:
            sizeInput_local = list(outImage.GetLargestPossibleRegion().GetSize())
            sizeInput_local[2] = Nproj

            spacingInput = outImage.GetSpacing()
            originInput = outImage.GetOrigin()

            imageDirection = itk.Matrix[itk.D, Dimension, Dimension]()
            imageDirection.SetIdentity()

            # Initialization of the output volume
            constantSource.SetOrigin(originInput)
            constantSource.SetSpacing(spacingInput)
            constantSource.SetDirection(imageDirection)
            constantSource.SetSize(sizeInput_local)
            constantSource.SetConstant(0.0)
            first = False
        else:
            paste.SetDestinationImage(paste.GetOutput())

        paste.SetSourceImage(outImage)
        paste.SetSourceRegion(outImage.GetLargestPossibleRegion())
        paste.SetDestinationIndex(current_idx)
        paste.Update()

        projid += curBufferSize

    if args_info.output:
        itk.imwrite(paste.GetOutput(), args_info.output)


def main(argv=None):
    parser = build_parser()
    args_info = parser.parse_args(argv)
    process(args_info)


if __name__ == "__main__":
    main()
