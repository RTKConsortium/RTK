#!/usr/bin/env python
import argparse
import itk
import numpy as np
from itk import RTK as rtk


def build_parser():
    parser = rtk.RTKArgumentParser(
        description=(
            "Computes expected photon counts from incident spectrum, material "
            "attenuations, detector response and material-decomposed projections"
        )
    )
    parser.add_argument(
        "--output",
        "-o",
        help="Output file name (photon counts)",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--input", "-i", help="Material-decomposed projections", type=str, required=True
    )
    parser.add_argument(
        "--detector", "-d", help="Detector response file", type=str, required=True
    )
    parser.add_argument(
        "--incident", help="Incident spectrum file", type=str, required=True
    )
    parser.add_argument(
        "--attenuations",
        "-a",
        help="Material attenuations file",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--thresholds",
        "-t",
        help="Lower threshold of bins, expressed in pulse height",
        type=int,
        nargs="+",
        required=True,
    )
    parser.add_argument(
        "--cramer_rao", help="Output Cramer-Rao lower bound file", type=str
    )
    parser.add_argument(
        "--variances", help="Output variances of photon counts", type=str
    )
    return parser


def process(args_info: argparse.Namespace):
    PixelValueType = itk.F
    Dimension = 3

    DecomposedProjectionType = itk.VectorImage[PixelValueType, Dimension]
    MeasuredProjectionsType = itk.VectorImage[PixelValueType, Dimension]
    IncidentSpectrumImageType = itk.Image[PixelValueType, Dimension]
    DetectorResponseImageType = itk.Image[PixelValueType, Dimension - 1]
    MaterialAttenuationsImageType = itk.Image[PixelValueType, Dimension - 1]

    # Read all inputs
    if args_info.verbose:
        print(f"Reading decomposed projections from {args_info.input}...")
    decomposedProjectionReader = itk.ImageFileReader[DecomposedProjectionType].New()
    decomposedProjectionReader.SetFileName(args_info.input)
    decomposedProjectionReader.Update()
    decomposedProjection = decomposedProjectionReader.GetOutput()

    if args_info.verbose:
        print(f"Reading incident spectrum from {args_info.incident}...")
    incidentSpectrumReader = itk.ImageFileReader[IncidentSpectrumImageType].New()
    incidentSpectrumReader.SetFileName(args_info.incident)
    incidentSpectrumReader.Update()
    incidentSpectrum = incidentSpectrumReader.GetOutput()

    if args_info.verbose:
        print(f"Reading detector response from {args_info.detector}...")
    detectorResponseReader = itk.ImageFileReader[DetectorResponseImageType].New()
    detectorResponseReader.SetFileName(args_info.detector)
    detectorResponseReader.Update()
    detectorResponse = detectorResponseReader.GetOutput()

    if args_info.verbose:
        print(f"Reading material attenuations from {args_info.attenuations}...")
    materialAttenuationsReader = itk.ImageFileReader[
        MaterialAttenuationsImageType
    ].New()
    materialAttenuationsReader.SetFileName(args_info.attenuations)
    materialAttenuationsReader.Update()
    materialAttenuations = materialAttenuationsReader.GetOutput()

    # Get parameters from the images
    NumberOfMaterials = materialAttenuations.GetLargestPossibleRegion().GetSize()[0]
    NumberOfSpectralBins = len(args_info.thresholds)
    MaximumEnergy = incidentSpectrum.GetLargestPossibleRegion().GetSize()[0]

    # Generate a set of zero-filled photon count projections
    measuredProjections = MeasuredProjectionsType.New()
    measuredProjections.CopyInformation(decomposedProjection)
    measuredProjections.SetVectorLength(NumberOfSpectralBins)
    measuredProjections.Allocate()

    # Read the thresholds on command line
    thresholds = itk.VariableLengthVector[itk.D]()
    thresholds.SetSize(NumberOfSpectralBins + 1)
    for i in range(NumberOfSpectralBins):
        thresholds[i] = args_info.thresholds[i]

    # Add the maximum pulse height at the end
    MaximumPulseHeight = detectorResponse.GetLargestPossibleRegion().GetSize()[1]
    thresholds[NumberOfSpectralBins] = MaximumPulseHeight

    # Check that the inputs have the expected size
    idx = itk.Index[3]()
    idx.Fill(0)
    if decomposedProjection.GetPixel(idx).Size() != NumberOfMaterials:
        raise RuntimeError(
            f"Decomposed projections vector size {decomposedProjection.GetPixel(idx).Size()} != {NumberOfMaterials}"
        )

    if measuredProjections.GetPixel(idx).Size() != NumberOfSpectralBins:
        raise RuntimeError(
            f"Spectral projections vector size {measuredProjections.GetPixel(idx).Size()} != {NumberOfSpectralBins}"
        )

    if detectorResponse.GetLargestPossibleRegion().GetSize()[0] != MaximumEnergy:
        raise RuntimeError(
            f"Detector response energies {detectorResponse.GetLargestPossibleRegion().GetSize()[0]} != {MaximumEnergy}"
        )

    # Create and set the filter
    forward = rtk.SpectralForwardModelImageFilter[
        DecomposedProjectionType,
        MeasuredProjectionsType,
        IncidentSpectrumImageType,
        DetectorResponseImageType,
        MaterialAttenuationsImageType,
    ].New()
    forward.SetInputDecomposedProjections(decomposedProjection)
    forward.SetInputMeasuredProjections(measuredProjections)
    forward.SetInputIncidentSpectrum(incidentSpectrum)
    forward.SetDetectorResponse(detectorResponse)
    forward.SetMaterialAttenuations(materialAttenuations)
    forward.SetThresholds(thresholds)
    if args_info.cramer_rao:
        forward.SetComputeCramerRaoLowerBound(True)
    if args_info.variances:
        forward.SetComputeVariances(True)

    if args_info.verbose:
        print("Running spectral forward model...")
    forward.Update()

    # Inspect output and convert if necessary so we write a 3D VectorImage
    out = forward.GetOutput()
    if args_info.verbose:
        print("Forward output type:", type(out))
        print("ImageDimension:", out.GetImageDimension())
        print("ComponentsPerPixel:", out.GetNumberOfComponentsPerPixel())
        print("Size:", tuple(out.GetLargestPossibleRegion().GetSize()))

    # If filter produced a 4D scalar image (dim=4, comps=1), convert it to a 3D VectorImage
    if out.GetImageDimension() == 4 and out.GetNumberOfComponentsPerPixel() == 1:
        if args_info.verbose:
            print(
                "Converting 4D scalar image to 3D VectorImage (components->vector)..."
            )
        arr4 = itk.array_view_from_image(out)
        # Move the first axis (extra spatial dim) to the last axis to get (Z,Y,X,Components)
        arr_vec = np.moveaxis(arr4, 0, -1)
        arr_vec = arr_vec.astype(np.float32)
        vec_img = itk.image_from_array(arr_vec, is_vector=True)
        # copy spatial metadata from original (origin, spacing, direction may need adjustment)
        vec_img.SetOrigin(out.GetOrigin()[:3])
        vec_img.SetSpacing(out.GetSpacing()[:3])
        try:
            vec_img.SetDirection(out.GetDirection()[0:3, 0:3])
        except Exception:
            # some ITK builds may not support slicing direction; ignore if unavailable
            pass
        writer_input = vec_img
    else:
        writer_input = out

    if args_info.verbose:
        print(f"Writing output photon counts to {args_info.output}...")
    WriterType = itk.ImageFileWriter[MeasuredProjectionsType]
    writer = WriterType.New()
    writer.SetFileName(args_info.output)
    writer.SetInput(writer_input)
    writer.SetImageIO(itk.MetaImageIO.New())
    writer.Update()

    # If requested, write the Cramer-Rao lower bound
    if args_info.cramer_rao:
        if args_info.verbose:
            print(f"Writing Cramer-Rao lower bound to {args_info.cramer_rao}...")
        # Cramer-Rao output has same type as measured projections
        cramer_writer = WriterType.New()
        cramer_writer.SetFileName(args_info.cramer_rao)
        cramer_writer.SetInput(forward.GetOutputCramerRaoLowerBound())
        cramer_writer.SetImageIO(itk.MetaImageIO.New())
        cramer_writer.Update()

    # If requested, write the variance
    if args_info.variances:
        if args_info.verbose:
            print(f"Writing variances to {args_info.variances}...")
        var_writer = WriterType.New()
        var_writer.SetFileName(args_info.variances)
        var_writer.SetInput(forward.GetOutputVariances())
        var_writer.SetImageIO(itk.MetaImageIO.New())
        var_writer.Update()


def main(argv=None):
    parser = build_parser()
    args_info = parser.parse_args(argv)
    process(args_info)


if __name__ == "__main__":
    main()
