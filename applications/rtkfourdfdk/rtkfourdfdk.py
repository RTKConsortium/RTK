#!/usr/bin/env python
import argparse
import sys
import itk
from itk import RTK as rtk


def build_parser():
    parser = rtk.RTKArgumentParser(
        description="Reconstructs a 3D + time (4D) sequence using FDK with one projection per respiratory cycle in each frame."
    )
    parser.add_argument(
        "--geometry", "-g", help="XML geometry file name", type=str, required=True
    )
    parser.add_argument(
        "--output", "-o", help="Output file name", type=str, required=True
    )
    parser.add_argument(
        "--signal",
        help="File containing the phase of each projection",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--hardware",
        help="Hardware used for computation",
        choices=["cpu", "cuda"],
        default="cpu",
    )
    parser.add_argument(
        "--divisions",
        "-d",
        help="Streaming option: number of stream divisions of the 3D reconstruction per frame",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--subsetsize",
        help="Streaming option: number of projections processed at a time",
        type=int,
        default=16,
    )
    parser.add_argument(
        "--nodisplaced",
        help="Disable the displaced detector filter",
        action="store_true",
    )
    # Ramp filter
    ramp = parser.add_argument_group("Ramp filter")
    ramp.add_argument(
        "--pad",
        help="Data padding parameter to correct for truncation",
        type=float,
        default=0.0,
    )
    ramp.add_argument(
        "--hann",
        help="Cut frequency for Hann window in ]0,1] (0.0 disables it)",
        type=float,
        default=0.0,
    )
    ramp.add_argument(
        "--hannY",
        help="Cut frequency for Hann window in ]0,1] (0.0 disables it)",
        type=float,
        default=0.0,
    )

    # RTK common groups
    rtk.add_rtkinputprojections_group(parser)
    rtk.add_rtk4Doutputimage_group(parser)
    return parser


def process(args_info: argparse.Namespace):
    OutputPixelType = itk.F
    OutputImageType = itk.Image[OutputPixelType, 3]

    # Check on hardware parameter
    if args_info.hardware == "cuda" and not hasattr(itk, "CudaImage"):
        print("The program has not been compiled with cuda option")
        sys.exit(1)

    # Projections reader
    reader = rtk.ProjectionsReader[OutputImageType].New()
    rtk.SetProjectionsReaderFromArgParse(reader, args_info)

    # Geometry
    if args_info.verbose:
        print(f"Reading geometry information from {args_info.geometry}...")
    geometry = rtk.read_geometry(args_info.geometry)

    # Part specific to 4D
    selector = rtk.SelectOneProjectionPerCycleImageFilter[OutputImageType].New()
    selector.SetInput(reader.GetOutput())
    selector.SetInputGeometry(geometry)
    selector.SetSignalFilename(args_info.signal)

    # Displaced detector weighting
    if args_info.hardware == "cuda":
        ddf = rtk.CudaDisplacedDetectorImageFilter.New()
        ddf.SetInput(itk.cuda_image_from_image(selector.GetOutput()))
    else:
        ddf = rtk.DisplacedDetectorForOffsetFieldOfViewImageFilter[
            OutputImageType
        ].New()
        ddf.SetInput(selector.GetOutput())
    ddf.SetGeometry(selector.GetOutputGeometry())
    ddf.SetDisable(args_info.nodisplaced)

    # Short scan image filter
    if args_info.hardware == "cuda":
        pssf = rtk.CudaParkerShortScanImageFilter.New()
    else:
        pssf = rtk.ParkerShortScanImageFilter[OutputImageType].New()
    pssf.SetInput(ddf.GetOutput())
    pssf.SetGeometry(selector.GetOutputGeometry())
    pssf.InPlaceOff()

    # Create one frame of the reconstructed image
    constantImageSource = rtk.ConstantImageSource[OutputImageType].New()
    rtk.SetConstantImageSourceFromArgParse(constantImageSource, args_info)
    constantImageSource.Update()

    # FDK reconstruction filtering
    if args_info.hardware == "cuda":
        feldkamp = rtk.CudaFDKConeBeamReconstructionFilter.New()
        feldkamp.SetInput(0, itk.cuda_image_from_image(constantImageSource.GetOutput()))
        feldkamp.SetInput(1, pssf.GetOutput())
    else:
        feldkamp = rtk.FDKConeBeamReconstructionFilter[OutputImageType].New()
        feldkamp.SetInput(0, constantImageSource.GetOutput())
        feldkamp.SetInput(1, pssf.GetOutput())
    feldkamp.SetGeometry(selector.GetOutputGeometry())
    feldkamp.GetRampFilter().SetTruncationCorrection(args_info.pad)
    feldkamp.GetRampFilter().SetHannCutFrequency(args_info.hann)
    feldkamp.GetRampFilter().SetHannCutFrequencyY(args_info.hannY)
    feldkamp.SetProjectionSubsetSize(args_info.subsetsize)

    # Streaming depending on streaming capability of writer
    streamerBP = itk.StreamingImageFilter[OutputImageType, OutputImageType].New()
    streamerBP.SetInput(feldkamp.GetOutput())
    streamerBP.SetNumberOfStreamDivisions(args_info.divisions)
    # Prefer streaming splits along Z for cache friendliness (like rtkfdk)
    splitter = itk.ImageRegionSplitterDirection.New()
    splitter.SetDirection(2)
    streamerBP.SetRegionSplitter(splitter)
    # Initialize meta-information once
    streamerBP.UpdateOutputInformation()

    # Create empty 4D image
    fourDConstantImageSource = rtk.ConstantImageSource[
        itk.Image[OutputPixelType, 4]
    ].New()
    rtk.SetConstantImageSourceFromArgParse(fourDConstantImageSource, args_info)
    fourDInputSize = fourDConstantImageSource.GetSize()

    fourDInputSize[3] = args_info.frames
    fourDConstantImageSource.SetSize(fourDInputSize)
    fourDConstantImageSource.Update()

    # Go over each frame, reconstruct 3D frame and paste with NumPy views into the 4D image
    volOut = fourDConstantImageSource.GetOutput()
    arr4d = itk.GetArrayViewFromImage(volOut)
    num_frames = args_info.frames
    for f in range(num_frames):
        if args_info.verbose:
            print(f"Reconstructing frame #{f}...")
        selector.SetPhase(f / float(num_frames))
        streamerBP.UpdateLargestPossibleRegion()

        arr3d = itk.GetArrayViewFromImage(streamerBP.GetOutput())
        arr4d[f] = arr3d

    # Write
    if args_info.verbose:
        print(f"Writing output to {args_info.output}...")
    writer = itk.ImageFileWriter[itk.Image[OutputPixelType, 4]].New()
    writer.SetFileName(args_info.output)
    writer.SetInput(volOut)
    writer.SetNumberOfStreamDivisions(args_info.divisions)
    writer.Update()


def main(argv=None):
    parser = build_parser()
    args_info = parser.parse_args(argv)
    process(args_info)


if __name__ == "__main__":
    main()
