#!/usr/bin/env python
import argparse
import itk
from itk import RTK as rtk


def build_parser():
    parser = rtk.RTKArgumentParser(
        description=(
            "Reconstructs a 3D + material vector volume from a vector projection stack,"
            " alternating between conjugate gradient optimization and regularization."
        )
    )

    parser.add_argument(
        "--geometry", "-g", help="XML geometry file name", required=True
    )
    parser.add_argument("--output", "-o", help="Output file name", required=True)
    parser.add_argument(
        "--niter", "-n", help="Number of main loop iterations", type=int, default=5
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
    parser.add_argument("--input", "-i", help="Input volume (materials) file")
    parser.add_argument(
        "--projection", "-p", help="Vector projections file", required=True
    )
    parser.add_argument(
        "--nodisplaced",
        help="Disable the displaced detector filter",
        action="store_true",
    )

    # Regularization
    parser.add_argument(
        "--nopositivity", help="Do not enforce positivity", action="store_true"
    )
    parser.add_argument("--tviter", help="TV iterations", type=int, default=10)
    parser.add_argument(
        "--gamma_space", help="Spatial TV regularization parameter", type=float
    )
    parser.add_argument("--threshold", help="Wavelets soft threshold", type=float)
    parser.add_argument("--order", help="Wavelets order", type=int, default=5)
    parser.add_argument("--levels", help="Wavelets levels", type=int, default=3)
    parser.add_argument(
        "--gamma_time", help="Temporal TV regularization parameter", type=float
    )
    parser.add_argument(
        "--lambda_time", help="Temporal L0 regularization parameter", type=float
    )
    parser.add_argument("--l0iter", help="L0 iterations", type=int, default=5)
    parser.add_argument("--gamma_tnv", help="TNV regularization parameter", type=float)

    # Projector choices
    rtk.add_rtkprojectors_group(parser)
    rtk.add_rtk3Doutputimage_group(parser)
    return parser


def process(args_info: argparse.Namespace):
    PixelValueType = itk.F
    Dimension = 3

    DecomposedProjectionType = itk.VectorImage[PixelValueType, Dimension]
    MaterialsVolumeType = itk.VectorImage[PixelValueType, Dimension]
    VolumeSeriesType = itk.Image[PixelValueType, Dimension + 1]
    ProjectionStackType = itk.Image[PixelValueType, Dimension]

    # Projections reader
    if args_info.verbose:
        print(f"Reading decomposed projections from {args_info.projection}...")
    proj_reader = itk.ImageFileReader[DecomposedProjectionType].New()
    proj_reader.SetFileName(args_info.projection)
    proj_reader.Update()
    decomposedProjection = proj_reader.GetOutput()

    NumberOfMaterials = decomposedProjection.GetVectorLength()

    # Geometry
    if args_info.verbose:
        print(f"Reading geometry from {args_info.geometry}...")
    geometry = rtk.read_geometry(args_info.geometry)

    # Create 4D input. Fill it either with an existing materials volume
    # read from a file or a blank image
    vecVol2VolSeries = rtk.VectorImageToImageFilter[
        MaterialsVolumeType, VolumeSeriesType
    ].New()

    if args_info.input is not None:
        if args_info.like is not None:
            print("WARNING: --like ignored since --input was given")
        if args_info.verbose:
            print(f"Reading input volume {args_info.input}...")
        reference = itk.imread(args_info.input)
        vecVol2VolSeries.SetInput(reference)
        vecVol2VolSeries.Update()
        input = vecVol2VolSeries.GetOutput()
    elif args_info.like is not None:
        if args_info.verbose:
            print(f"Reading reference volume {args_info.like} to infer geometry...")
        reference = itk.imread(args_info.like)
        vecVol2VolSeries.SetInput(reference)
        vecVol2VolSeries.UpdateOutputInformation()
        constantImageSource = rtk.ConstantImageSource[VolumeSeriesType].New()
        constantImageSource.SetInformationFromImage(vecVol2VolSeries.GetOutput())
        constantImageSource.Update()
        input = constantImageSource.GetOutput()
    else:
        # Create new empty volume
        constantImageSource = rtk.ConstantImageSource[VolumeSeriesType].New()

        imageSize = itk.Size[4]()
        imageSize.Fill(args_info.size[0])
        for i in range(min(len(args_info.size), Dimension)):
            imageSize[i] = args_info.size[i]
        imageSize[Dimension] = NumberOfMaterials

        imageSpacing = itk.Vector[itk.D, 4]()
        imageSpacing.Fill(args_info.spacing[0])
        for i in range(min(len(args_info.spacing), Dimension)):
            imageSpacing[i] = args_info.spacing[i]
        imageSpacing[Dimension] = 1.0

        imageOrigin = itk.Point[itk.D, 4]()
        for i in range(Dimension):
            imageOrigin[i] = imageSpacing[i] * (imageSize[i] - 1) * -0.5
        if args_info.origin is not None:
            for i in range(min(len(args_info.origin), Dimension)):
                imageOrigin[i] = args_info.origin[i]
        imageOrigin[Dimension] = 0.0

        imageDirection = itk.Matrix[itk.D, 4, 4]()
        imageDirection.SetIdentity()
        if args_info.direction is not None:
            for i in range(Dimension):
                for j in range(Dimension):
                    imageDirection[i][j] = args_info.direction[i * Dimension + j]

        constantImageSource.SetOrigin(imageOrigin)
        constantImageSource.SetSpacing(imageSpacing)
        constantImageSource.SetDirection(imageDirection)
        constantImageSource.SetSize(imageSize)
        constantImageSource.SetConstant(0.0)
        constantImageSource.Update()
        input = constantImageSource.GetOutput()

    # Duplicate geometry and transform the N M-vector projections into N*M scalar projections
    # Each material will occupy one frame of the 4D reconstruction, therefore all projections
    # of one material need to have the same phase.
    # Note : the 4D CG filter is optimized when projections with identical phases are packed together

    # Geometry
    initialNumberOfProjections = (
        decomposedProjection.GetLargestPossibleRegion().GetSize()[Dimension - 1]
    )
    for material in range(1, NumberOfMaterials):
        for proj in range(initialNumberOfProjections):
            geometry.AddProjectionInRadians(
                geometry.GetSourceToIsocenterDistances()[proj],
                geometry.GetSourceToDetectorDistances()[proj],
                geometry.GetGantryAngles()[proj],
                geometry.GetProjectionOffsetsX()[proj],
                geometry.GetProjectionOffsetsY()[proj],
                geometry.GetOutOfPlaneAngles()[proj],
                geometry.GetInPlaneAngles()[proj],
                geometry.GetSourceOffsetsX()[proj],
                geometry.GetSourceOffsetsY()[proj],
            )
            geometry.SetCollimationOfLastProjection(
                geometry.GetCollimationUInf()[proj],
                geometry.GetCollimationUSup()[proj],
                geometry.GetCollimationVInf()[proj],
                geometry.GetCollimationVSup()[proj],
            )

    # Signal
    fakeSignal = []
    for material in range(NumberOfMaterials):
        v = round(material / NumberOfMaterials * 1000.0) / 1000.0
        for proj in range(initialNumberOfProjections):
            fakeSignal.append(v)

    # Projections
    vproj2proj = rtk.VectorImageToImageFilter[
        DecomposedProjectionType, ProjectionStackType
    ].New()
    vproj2proj.SetInput(decomposedProjection)
    vproj2proj.Update()

    # Release the memory holding the stack of original projections
    decomposedProjection.ReleaseData()

    # Compute the interpolation weights
    signalToInterpolationWeights = rtk.SignalToInterpolationWeights.New()
    signalToInterpolationWeights.SetSignal(fakeSignal)
    signalToInterpolationWeights.SetNumberOfReconstructedFrames(NumberOfMaterials)
    signalToInterpolationWeights.Update()

    # Set the forward and back projection filters to be used
    # Instantiate ROOSTER with CUDA image types if available, otherwise CPU types
    if hasattr(itk, "CudaImage"):
        cudaVolumeSeriesType = itk.CudaImage[PixelValueType, Dimension + 1]
        cudaProjectionStackType = itk.CudaImage[PixelValueType, Dimension]
        rooster = rtk.FourDROOSTERConeBeamReconstructionFilter[
            cudaVolumeSeriesType, cudaProjectionStackType
        ].New()
        rooster.SetInputVolumeSeries(itk.cuda_image_from_image(input))
        rooster.SetInputProjectionStack(
            itk.cuda_image_from_image(vproj2proj.GetOutput())
        )
    else:
        rooster = rtk.FourDROOSTERConeBeamReconstructionFilter[
            VolumeSeriesType, ProjectionStackType
        ].New()
        rooster.SetInputVolumeSeries(input)
        rooster.SetInputProjectionStack(vproj2proj.GetOutput())

    # Configure projectors from args
    rtk.SetForwardProjectionFromArgParse(args_info, rooster)
    rtk.SetBackProjectionFromArgParse(args_info, rooster)

    rooster.SetCG_iterations(args_info.cgiter)
    rooster.SetMainLoop_iterations(args_info.niter)
    rooster.SetCudaConjugateGradient(args_info.cudacg)
    rooster.SetDisableDisplacedDetectorFilter(args_info.nodisplaced)

    rooster.SetGeometry(geometry)
    rooster.SetWeights(signalToInterpolationWeights.GetOutput())
    rooster.SetSignal(fakeSignal)

    # For each optional regularization step, set whether or not
    # it should be performed, and provide the necessary inputs

    # Positivity
    rooster.SetPerformPositivity(not args_info.nopositivity)

    # No motion mask is used, since there is no motion
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

    if args_info.verbose:
        print("Running ROOSTER reconstruction...")
    rooster.Update()

    # Convert to result to a vector image
    volSeries2VecVol = rtk.ImageToVectorImageFilter[
        VolumeSeriesType, MaterialsVolumeType
    ].New()
    volSeries2VecVol.SetInput(rooster.GetOutput())
    volSeries2VecVol.Update()

    # Write
    if args_info.verbose:
        print(f"Writing output to {args_info.output}...")
    writer = itk.ImageFileWriter[MaterialsVolumeType].New()
    writer.SetFileName(args_info.output)
    writer.SetInput(volSeries2VecVol.GetOutput())
    writer.Update()


def main(argv=None):
    parser = build_parser()
    args_info = parser.parse_args(argv)
    process(args_info)


if __name__ == "__main__":
    main()
