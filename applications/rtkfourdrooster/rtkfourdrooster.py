#!/usr/bin/env python
import argparse
import sys
import itk
from itk import RTK as rtk


def build_parser():
    parser = rtk.RTKArgumentParser(
        description=(
            "Reconstructs a 3D + time sequence of volumes from a projection stack "
            "and a respiratory/cardiac signal, applying TV regularization in space "
            "and time, and restricting motion to a region of interest."
        )
    )

    parser.add_argument(
        "--geometry", "-g", help="XML geometry file name", type=str, required=True
    )
    parser.add_argument("--input", "-i", help="Input volume", type=str)
    parser.add_argument(
        "--output", "-o", help="Output file name", type=str, required=True
    )
    parser.add_argument(
        "--niter",
        "-n",
        help="Number of main loop iterations",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--cgiter",
        help="Number of conjugate gradient nested iterations",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--cudacg",
        help="Perform conjugate gradient calculations on GPU",
        action="store_true",
    )
    parser.add_argument(
        "--cudadvfinterpolation",
        help="Perform DVF interpolation calculations on GPU",
        action="store_true",
    )
    parser.add_argument(
        "--nodisplaced",
        help="Disable the displaced detector filter",
        action="store_true",
    )

    # Phase gating
    parser.add_argument(
        "--signal",
        help="File containing the phase of each projection",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--shift",
        help="Phase shift applied on the DVFs to simulate phase estimation errors",
        type=float,
        default=0.0,
    )

    # Regularization
    parser.add_argument(
        "--nopositivity",
        help="Do not enforce positivity",
        action="store_true",
    )
    parser.add_argument(
        "--motionmask",
        help="Motion mask file: binary image with ones where movement can occur and zeros elsewhere",
        type=str,
    )
    parser.add_argument(
        "--tviter",
        help="Total variation (spatial, temporal and nuclear) regularization: number of iterations",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--gamma_space",
        help="Total variation spatial regularization parameter. The larger, the smoother",
        type=float,
    )
    parser.add_argument(
        "--threshold",
        help="Daubechies wavelets spatial regularization: soft threshold",
        type=float,
    )
    parser.add_argument(
        "--order",
        help="Daubechies wavelets spatial regularization: order of the wavelets",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--levels",
        help="Daubechies wavelets spatial regularization: number of decomposition levels",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--gamma_time",
        help="Total variation temporal regularization parameter. The larger, the smoother",
        type=float,
    )
    parser.add_argument(
        "--lambda_time",
        help="Temporal gradient's L0 norm regularization parameter. The larger, the stronger",
        type=float,
    )
    parser.add_argument(
        "--l0iter",
        help="Temporal gradient's L0 norm regularization: number of iterations",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--gamma_tnv",
        help="Total nuclear variation regularization parameter. The larger, the smoother",
        type=float,
    )

    # Motion-compensation
    parser.add_argument(
        "--dvf",
        help="Input 4D DVF",
        type=str,
    )
    parser.add_argument(
        "--idvf",
        help=(
            "Input 4D inverse DVF. Inverse transform computed by conjugate gradient if not provided"
        ),
        type=str,
    )
    parser.add_argument(
        "--nn",
        help="Nearest neighbor interpolation (default is trilinear)",
        action="store_true",
    )

    # RTK common groups (projections input)
    rtk.add_rtkinputprojections_group(parser)
    rtk.add_rtkprojectors_group(parser)
    rtk.add_rtk4Doutputimage_group(parser)
    rtk.add_rtkiterations_group(parser)

    return parser


def process(args_info: argparse.Namespace):
    OutputPixelType = itk.F
    DVFVectorType = itk.CovariantVector[OutputPixelType, 3]
    VolumeSeriesType = itk.Image[OutputPixelType, 4]
    ProjectionStackType = itk.Image[OutputPixelType, 3]
    DVFSequenceImageType = itk.Image[
        DVFVectorType, VolumeSeriesType.GetImageDimension()
    ]

    VolumeType = ProjectionStackType

    # Check CUDA-related options vs. build capabilities
    if (args_info.cudacg or args_info.cudadvfinterpolation) and not hasattr(
        itk, "CudaImage"
    ):
        print(
            "Error: CUDA options (--cudacg, --cudadvfinterpolation) were requested but ITK/RTK was "
            "not built with CUDA support.",
        )
        sys.exit(1)

    # Projections reader
    reader = rtk.ProjectionsReader[ProjectionStackType].New()
    rtk.SetProjectionsReaderFromArgParse(reader, args_info)

    # Geometry
    if args_info.verbose:
        print(f"Reading geometry information from {args_info.geometry}...")
    geometry = rtk.read_geometry(args_info.geometry)

    # Create input: either an existing volume read from a file or a blank image
    if args_info.input:
        # Read existing 4D volume
        inputFilter = itk.ImageFileReader[VolumeSeriesType].New()
        inputFilter.SetFileName(args_info.input)
    else:
        # Constant 4D volume
        constantImageSource = rtk.ConstantImageSource[VolumeSeriesType].New()
        rtk.SetConstantImageSourceFromArgParse(constantImageSource, args_info)
        # Set 4th dimension (frames)
        size = constantImageSource.GetSize()
        size[3] = args_info.frames
        constantImageSource.SetSize(size)
        inputFilter = constantImageSource

    inputFilter.Update()
    inputFilter.ReleaseDataFlagOn()

    # Re-order geometry and projections
    # In the new order, projections with identical phases are packed together
    signal = rtk.read_signal_file(args_info.signal)
    reorder = rtk.ReorderProjectionsImageFilter[
        ProjectionStackType, ProjectionStackType
    ].New()
    reorder.SetInput(reader.GetOutput())
    reorder.SetInputGeometry(geometry)
    reorder.SetInputSignal(signal)
    reorder.Update()

    # Release the memory holding the stack of original projections
    reader.GetOutput().ReleaseData()

    # Compute the interpolation weights
    signalToInterpolationWeights = rtk.SignalToInterpolationWeights.New()
    signalToInterpolationWeights.SetSignal(reorder.GetOutputSignal())
    nb_frames = inputFilter.GetOutput().GetLargestPossibleRegion().GetSize()[3]
    signalToInterpolationWeights.SetNumberOfReconstructedFrames(nb_frames)
    signalToInterpolationWeights.Update()

    # Create the 4DROOSTER filter, connect the basic inputs, and set the basic parameters
    # Also set the forward and back projection filters to be used
    if hasattr(itk, "CudaImage"):
        CudaVolumeSeriesType = itk.CudaImage[OutputPixelType, 4]
        CudaProjectionStackType = itk.CudaImage[OutputPixelType, 3]
        rooster = rtk.FourDROOSTERConeBeamReconstructionFilter[
            CudaVolumeSeriesType, CudaProjectionStackType
        ].New()
        rooster.SetInputVolumeSeries(itk.cuda_image_from_image(inputFilter.GetOutput()))
        rooster.SetInputProjectionStack(itk.cuda_image_from_image(reorder.GetOutput()))
    else:
        rooster = rtk.FourDROOSTERConeBeamReconstructionFilter[
            VolumeSeriesType, ProjectionStackType
        ].New()
        rooster.SetInputVolumeSeries(inputFilter.GetOutput())
        rooster.SetInputProjectionStack(reorder.GetOutput())

    rooster.SetCG_iterations(args_info.cgiter)
    rooster.SetMainLoop_iterations(args_info.niter)
    rooster.SetPhaseShift(args_info.shift)
    rooster.SetCudaConjugateGradient(args_info.cudacg)
    rooster.SetUseCudaCyclicDeformation(args_info.cudadvfinterpolation)
    rooster.SetDisableDisplacedDetectorFilter(args_info.nodisplaced)
    rooster.SetGeometry(reorder.GetOutputGeometry())
    rooster.SetWeights(signalToInterpolationWeights.GetOutput())
    rooster.SetSignal(reorder.GetOutputSignal())

    rtk.SetIterationsReportFromArgParse(args_info, rooster)

    # For each optional regularization step, set whether or not
    # it should be performed, and provide the necessary inputs

    # Positivity
    rooster.SetPerformPositivity(not args_info.nopositivity)

    rtk.SetForwardProjectionFromArgParse(args_info, rooster)
    rtk.SetBackProjectionFromArgParse(args_info, rooster)
    # Motion mask
    if args_info.motionmask:
        reader = itk.ImageFileReader[VolumeType].New()
        reader.SetFileName(args_info.motionmask)
        if hasattr(itk, "CudaImage"):
            motionMask = itk.cuda_image_from_image(reader.GetOutput())
        else:
            motionMask = reader.GetOutput()
        rooster.SetMotionMask(motionMask)
        rooster.SetPerformMotionMask(True)
    else:
        rooster.SetPerformMotionMask(False)

    # Spatial TV
    if args_info.gamma_space is not None:
        rooster.SetGammaTVSpace(args_info.gamma_space)
        rooster.SetTV_iterations(args_info.tviter)
        rooster.SetPerformTVSpatialDenoising(True)
    else:
        rooster.SetPerformTVSpatialDenoising(False)

    # Spatial wavelets
    if args_info.threshold is not None:
        rooster.SetSoftThresholdWavelets(args_info.threshold)
        rooster.SetOrder(args_info.order)
        rooster.SetNumberOfLevels(args_info.levels)
        rooster.SetPerformWaveletsSpatialDenoising(True)
    else:
        rooster.SetPerformWaveletsSpatialDenoising(False)

    # Temporal TV
    if args_info.gamma_time is not None:
        rooster.SetGammaTVTime(args_info.gamma_time)
        rooster.SetTV_iterations(args_info.tviter)
        rooster.SetPerformTVTemporalDenoising(True)
    else:
        rooster.SetPerformTVTemporalDenoising(False)

    # Temporal L0
    if args_info.lambda_time is not None:
        rooster.SetLambdaL0Time(args_info.lambda_time)
        rooster.SetL0_iterations(args_info.l0iter)
        rooster.SetPerformL0TemporalDenoising(True)
    else:
        rooster.SetPerformL0TemporalDenoising(False)

    # Total nuclear variation
    if args_info.gamma_tnv is not None:
        rooster.SetGammaTNV(args_info.gamma_tnv)
        rooster.SetTV_iterations(args_info.tviter)
        rooster.SetPerformTNVDenoising(True)
    else:
        rooster.SetPerformTNVDenoising(False)

    # Warping
    if args_info.dvf:
        rooster.SetPerformWarping(True)

        if args_info.nn:
            rooster.SetUseNearestNeighborInterpolationInWarping(True)

        reader = itk.ImageFileReader[DVFSequenceImageType].New()
        reader.SetFileName(args_info.dvf)
        dvf = reader.GetOutput()
        if hasattr(itk, "CudaImage"):
            dvf = itk.cuda_image_from_image(dvf)
        rooster.SetDisplacementField(dvf)

        if args_info.idvf:
            rooster.SetComputeInverseWarpingByConjugateGradient(False)
            reader.SetFileName(args_info.idvf)
            idvf = reader.GetOutput()
            if hasattr(itk, "CudaImage"):
                idvf = itk.cuda_image_from_image(idvf)
            rooster.SetInverseDisplacementField(idvf)
    else:
        rooster.SetPerformWarping(False)

    rooster.Update()

    # Write
    if args_info.verbose:
        print(f"Writing output to {args_info.output}...")
    writer = itk.ImageFileWriter[itk.Image[OutputPixelType, 4]].New()
    writer.SetFileName(args_info.output)
    writer.SetInput(rooster.GetOutput())
    writer.Update()


def main(argv=None):
    parser = build_parser()
    args_info = parser.parse_args(argv)
    process(args_info)


if __name__ == "__main__":
    main()
