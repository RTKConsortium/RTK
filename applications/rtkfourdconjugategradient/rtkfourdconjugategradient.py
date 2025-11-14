#!/usr/bin/env python
import argparse
import itk
from itk import RTK as rtk
import numpy as np


def build_parser():
    parser = rtk.RTKArgumentParser(
        description="Reconstructs a 3D + time sequence of volumes from a projection stack and a respiratory/cardiac signal, with a conjugate gradient technique."
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
    parser.add_argument(
        "--cudacg",
        help="Perform conjugate gradient calculations on GPU",
        action="store_true",
    )
    parser.add_argument("--input", "-i", help="Input volume", type=str)
    parser.add_argument(
        "--nodisplaced",
        help="Disable the displaced detector filter",
        action="store_true",
    )
    parser.add_argument(
        "--signal",
        help="File containing the phase of each projection",
        type=str,
        required=True,
    )
    rtk.add_rtkinputprojections_group(parser)
    rtk.add_rtk4Doutputimage_group(parser)
    return parser


def process(args_info: argparse.Namespace):
    OutputPixelType = itk.F
    VolumeSeriesType = itk.Image[OutputPixelType, 4]
    ProjectionStackType = itk.Image[OutputPixelType, 3]

    # Projections reader
    reader = rtk.ProjectionsReader[ProjectionStackType].New()
    rtk.SetProjectionsReaderFromArgParse(reader, args_info)
    reader.UpdateLargestPossibleRegion()

    # Geometry
    if args_info.verbose:
        print(f"Reading geometry information from {args_info.geometry}...")
    geometry = rtk.read_geometry(args_info.geometry)

    # Create input: either an existing volume read from a file or a blank image
    if args_info.input:
        # Read an existing image to initialize the volume
        inputReader = itk.ImageFileReader[VolumeSeriesType].New()
        inputReader.SetFileName(args_info.input)
        inputFilter = inputReader
    else:
        constantImageSource = rtk.ConstantImageSource[VolumeSeriesType].New()
        rtk.SetConstantImageSourceFromArgParse(constantImageSource, args_info)
        # Set 4th dimension (frames)
        size = constantImageSource.GetSize()
        size[3] = int(args_info.frames)
        constantImageSource.SetSize(size)
        inputFilter = constantImageSource

    inputFilter.Update()
    inputFilter.ReleaseDataFlagOn()

    # Reorder projections
    signal = np.loadtxt(args_info.signal).tolist()
    reorder = rtk.ReorderProjectionsImageFilter[
        ProjectionStackType, ProjectionStackType
    ].New()
    reorder.SetInput(reader.GetOutput())
    reorder.SetInputGeometry(geometry)
    reorder.SetInputSignal(signal)
    reorder.Update()
    # Release the memory holding the stack of original projections
    reader.GetOutput().ReleaseData()

    # Interpolation weights
    signalToInterpolationWeights = rtk.SignalToInterpolationWeights.New()
    signalToInterpolationWeights.SetSignal(reorder.GetOutputSignal())
    signalToInterpolationWeights.SetNumberOfReconstructedFrames(
        inputFilter.GetOutput().GetLargestPossibleRegion().GetSize()[3]
    )
    signalToInterpolationWeights.Update()

    if args_info.cudacg and not hasattr(itk, "CudaImage"):
        raise RuntimeError(
            "CUDA conjugate gradient requested but RTK was not built with CUDA support."
        )

    if args_info.cudacg:
        CudaVolumeSeriesType = itk.CudaImage[OutputPixelType, 4]
        CudaProjectionStackType = itk.CudaImage[OutputPixelType, 3]
        # Conjugate gradient filter
        conjugategradient = rtk.FourDConjugateGradientConeBeamReconstructionFilter[
            CudaVolumeSeriesType, CudaProjectionStackType
        ].New()
        conjugategradient.SetInputVolumeSeries(
            itk.cuda_image_from_image(inputFilter.GetOutput())
        )
        conjugategradient.SetWeights(
            itk.cuda_image_from_image(signalToInterpolationWeights.GetOutput())
        )
        conjugategradient.SetInputProjectionStack(
            itk.cuda_image_from_image(reorder.GetOutput())
        )
    else:
        # Conjugate gradient filter
        conjugategradient = rtk.FourDConjugateGradientConeBeamReconstructionFilter[
            VolumeSeriesType, ProjectionStackType
        ].New()
        conjugategradient.SetInputVolumeSeries(inputFilter.GetOutput())
        conjugategradient.SetWeights(signalToInterpolationWeights.GetOutput())
        conjugategradient.SetInputProjectionStack(reorder.GetOutput())

    rtk.SetForwardProjectionFromArgParse(args_info, conjugategradient)
    rtk.SetBackProjectionFromArgParse(args_info, conjugategradient)
    conjugategradient.SetNumberOfIterations(args_info.niterations)
    conjugategradient.SetCudaConjugateGradient(args_info.cudacg)
    conjugategradient.SetDisableDisplacedDetectorFilter(args_info.nodisplaced)
    conjugategradient.SetGeometry(reorder.GetOutputGeometry())
    conjugategradient.SetSignal(reorder.GetOutputSignal())

    rtk.SetIterationsReportFromArgParse(args_info, conjugategradient)

    if args_info.verbose:
        print("Running 4D conjugate gradient reconstruction...")
    conjugategradient.Update()

    if args_info.verbose:
        print(f"Writing output to {args_info.output}...")
    writer = itk.ImageFileWriter[VolumeSeriesType].New()
    writer.SetFileName(args_info.output)
    writer.SetInput(conjugategradient.GetOutput())
    writer.Update()


def main(argv=None):
    parser = build_parser()
    args_info = parser.parse_args(argv)
    process(args_info)


if __name__ == "__main__":
    main()
