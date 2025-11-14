#!/usr/bin/env python
import argparse
import sys
import math
import os
import itk
from itk import RTK as rtk


def build_parser():
    parser = rtk.RTKArgumentParser(description="Polynomial gain correction projections")

    parser.add_argument(
        "--output", "-o", help="Output file name", type=str, required=True
    )
    parser.add_argument(
        "--calibDir",
        "-c",
        help="Directory containing the calibration files",
        type=str,
        required=True,
    )
    parser.add_argument("--Gain", help="Gain maps filename", type=str, required=True)
    parser.add_argument("--Dark", help="Offset map filename", type=str, required=True)
    parser.add_argument(
        "--K", help="Normalization coefficient", type=float, default=1.0
    )
    parser.add_argument(
        "--bufferSize",
        help="Number of projections computed at the same time",
        type=int,
        default=4,
    )

    rtk.add_rtkinputprojections_group(parser)
    rtk.add_rtk3Doutputimage_group(parser)

    return parser


def process(args_info: argparse.Namespace):
    Dimension = 3

    InputImageType = itk.Image[itk.US, Dimension]
    OutputImageType = itk.Image[itk.F, Dimension]

    reader = rtk.ProjectionsReader[InputImageType].New()
    rtk.SetProjectionsReaderFromArgParse(reader, args_info)
    reader.ComputeLineIntegralOff()  # Don't want to preprocess data
    reader.SetFileNames(rtk.GetProjectionsFileNamesFromArgParse(args_info))
    reader.UpdateOutputInformation()

    # Input projection parameters
    Nprojections = reader.GetOutput().GetLargestPossibleRegion().GetSize()[2]
    if not Nprojections:
        print("No DR found to process!")
        sys.exit(1)

    # Load dark image
    darkFile = os.path.join(args_info.calibDir, args_info.Dark)
    readerDark = itk.ImageFileReader[InputImageType].New()
    readerDark.SetFileName(darkFile)
    readerDark.Update()
    darkImage = readerDark.GetOutput()
    darkImage.DisconnectPipeline()

    # Get gain image
    gainFile = os.path.join(args_info.calibDir, args_info.Gain)
    readerGain = itk.ImageFileReader[OutputImageType].New()
    readerGain.SetFileName(gainFile)
    readerGain.Update()
    gainImage = readerGain.GetOutput()
    gainImage.DisconnectPipeline()

    if hasattr(itk, "CudaImage"):
        gainfilter = rtk.CudaPolynomialGainCorrectionImageFilter.New()
        gainfilter.SetDarkImage(itk.cuda_image_from_image(darkImage))
        gainfilter.SetGainCoefficients(itk.cuda_image_from_image(gainImage))
    else:
        gainfilter = rtk.PolynomialGainCorrectionImageFilter[
            InputImageType, OutputImageType
        ].New()
        gainfilter.SetDarkImage(darkImage)
        gainfilter.SetGainCoefficients(gainImage)

    gainfilter.SetK(args_info.K)

    # Create empty volume for storing processed images
    constantImageSource = rtk.ConstantImageSource[OutputImageType].New()
    pasteFilter = itk.PasteImageFilter[OutputImageType].New()
    pasteFilter.SetDestinationImage(constantImageSource.GetOutput())

    bufferSize = args_info.bufferSize
    Nbuffers = int(math.ceil(float(Nprojections) / float(bufferSize)))

    first = True
    for bid in range(Nbuffers):
        bufferIdx = bid * bufferSize
        currentBufferSize = min(Nprojections - bufferIdx, bufferSize)

        print(
            f"Processing buffer no {bid} starting at image {bufferIdx} of size {currentBufferSize}"
        )

        sliceRegion = reader.GetOutput().GetLargestPossibleRegion()
        desiredRegion = itk.ImageRegion[Dimension]()
        size = list(sliceRegion.GetSize())
        index = list(sliceRegion.GetIndex())
        size[2] = currentBufferSize
        index[2] = bufferIdx
        desiredRegion.SetSize(size)
        desiredRegion.SetIndex(index)

        extract = itk.ExtractImageFilter[InputImageType, InputImageType].New()
        extract.SetDirectionCollapseToIdentity()
        extract.SetExtractionRegion(desiredRegion)
        extract.SetInput(reader.GetOutput())
        extract.Update()

        buffer = extract.GetOutput()
        buffer.DisconnectPipeline()
        if hasattr(itk, "CudaImage"):
            gainfilter.SetInput(itk.cuda_image_from_image(buffer))
        else:
            gainfilter.SetInput(buffer)

        gainfilter.Update()

        if first:
            # Initialization of the output volume
            sizeInput = list(
                gainfilter.GetOutput().GetLargestPossibleRegion().GetSize()
            )
            sizeInput[2] = Nprojections
            spacingInput = gainfilter.GetOutput().GetSpacing()
            originInput = gainfilter.GetOutput().GetOrigin()

            imageDirection = itk.Matrix[itk.D, Dimension, Dimension]()
            imageDirection.SetIdentity()

            constantImageSource.SetOrigin(originInput)
            constantImageSource.SetSpacing(spacingInput)
            constantImageSource.SetDirection(imageDirection)
            constantImageSource.SetSize(sizeInput)
            constantImageSource.SetConstant(0.0)

        procBuffer = gainfilter.GetOutput()
        procBuffer.DisconnectPipeline()

        current_idx = list(procBuffer.GetLargestPossibleRegion().GetIndex())
        current_idx[2] = bufferIdx

        if first:
            first = False
        else:
            pasteFilter.SetDestinationImage(pasteFilter.GetOutput())

        pasteFilter.SetSourceImage(procBuffer)
        pasteFilter.SetSourceRegion(procBuffer.GetLargestPossibleRegion())
        pasteFilter.SetDestinationIndex(current_idx)
        pasteFilter.Update()

    itk.imwrite(pasteFilter.GetOutput(), args_info.output)


def main(argv=None):
    parser = build_parser()
    args_info = parser.parse_args(argv)
    process(args_info)


if __name__ == "__main__":
    main()
