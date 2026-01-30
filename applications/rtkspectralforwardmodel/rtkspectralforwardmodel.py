#!/usr/bin/env python
import argparse
import itk
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
    decomposedProjection = itk.imread(args_info.input)

    if args_info.verbose:
        print(f"Reading incident spectrum from {args_info.incident}...")
    incidentSpectrum = itk.imread(args_info.incident)

    if args_info.verbose:
        print(f"Reading detector response from {args_info.detector}...")
    detectorResponse = itk.imread(args_info.detector)

    if args_info.verbose:
        print(f"Reading material attenuations from {args_info.attenuations}...")
    materialAttenuations = itk.imread(args_info.attenuations)

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
    thresholds = [int(t) for t in args_info.thresholds]
    MaximumPulseHeight = detectorResponse.GetLargestPossibleRegion().GetSize()[1]
    # Add the maximum pulse height at the end
    thresholds.append(MaximumPulseHeight)

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
        DecomposedProjectionType, MeasuredProjectionsType, IncidentSpectrumImageType
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

    # Write output
    if args_info.verbose:
        print(f"Writing output photon counts to {args_info.output}...")
    itk.imwrite(forward.GetOutput(), args_info.output)

    # If requested, write the Cramer-Rao lower bound
    if args_info.cramer_rao:
        if args_info.verbose:
            print(f"Writing Cramer-Rao lower bound to {args_info.cramer_rao}...")
        itk.imwrite(forward.GetOutputCramerRaoLowerBound(), args_info.cramer_rao)

    # If requested, write the variance
    if args_info.variances:
        if args_info.verbose:
            print(f"Writing variances to {args_info.variances}...")
        itk.imwrite(forward.GetOutputVariances(), args_info.variances)


def main(argv=None):
    parser = build_parser()
    args_info = parser.parse_args(argv)
    process(args_info)


if __name__ == "__main__":
    main()
