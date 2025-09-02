#!/usr/bin/env python
import argparse
import sys
import itk
from itk import RTK as rtk


def build_parser():
    parser = rtk.RTKArgumentParser(
        description="Computes expected photon counts from incident spectrum, material attenuations, detector response and material-decomposed projections"
    )

    # General options
    parser.add_argument(
        "--verbose", "-v", help="Verbose execution", action="store_true"
    )
    parser.add_argument(
        "--output", "-o", help="Output file name (high and low energy projections)", type=str, required=True
    )
    parser.add_argument(
        "--input", "-i", help="Material-decomposed projections", type=str, required=True
    )
    parser.add_argument(
        "--high", help="Incident spectrum image, high energy", type=str, required=True
    )
    parser.add_argument(
        "--low", help="Incident spectrum image, low energy", type=str, required=True
    )
    parser.add_argument(
        "--detector", "-d", help="Detector response file", type=str
    )
    parser.add_argument(
        "--attenuations", "-a", help="Material attenuations file", type=str, required=True
    )
    parser.add_argument(
        "--variances", help="Output variances of measured energies, file name", type=str
    )

    return parser


def process(args_info: argparse.Namespace):
    # Define pixel type and dimension
    PixelValueType = itk.F
    Dimension = 3

    DecomposedProjectionType = itk.VectorImage[PixelValueType, Dimension]
    DualEnergyProjectionsType = itk.VectorImage[PixelValueType, Dimension]
    IncidentSpectrumImageType = itk.Image[PixelValueType, Dimension]
    DetectorResponseImageType = itk.Image[PixelValueType, Dimension - 1]
    MaterialAttenuationsImageType = itk.Image[PixelValueType, Dimension - 1]

    # Read all inputs
    if args_info.verbose:
        print("Reading material-decomposed projections...")
    decomposed_projection_reader = itk.ImageFileReader[DecomposedProjectionType].New()
    decomposed_projection_reader.SetFileName(args_info.input)
    decomposed_projection_reader.Update()

    if args_info.verbose:
        print("Reading incident spectrum (high energy)...")
    incident_spectrum_reader_high_energy = itk.ImageFileReader[IncidentSpectrumImageType].New()
    incident_spectrum_reader_high_energy.SetFileName(args_info.high)
    incident_spectrum_reader_high_energy.Update()

    if args_info.verbose:
        print("Reading incident spectrum (low energy)...")
    incident_spectrum_reader_low_energy = itk.ImageFileReader[IncidentSpectrumImageType].New()
    incident_spectrum_reader_low_energy.SetFileName(args_info.low)
    incident_spectrum_reader_low_energy.Update()

    if args_info.verbose:
        print("Reading material attenuations...")
    material_attenuations_reader = itk.ImageFileReader[MaterialAttenuationsImageType].New()
    material_attenuations_reader.SetFileName(args_info.attenuations)
    material_attenuations_reader.Update()

    # Get parameters from the images
    maximum_energy = incident_spectrum_reader_high_energy.GetOutput().GetLargestPossibleRegion().GetSize(0)

    # If the detector response is given by the user, use it. Otherwise, assume it is included in the
    # incident spectrum, and fill the response with ones
    if args_info.detector:
        if args_info.verbose:
            print("Reading detector response...")
        detector_response_reader = itk.ImageFileReader[DetectorResponseImageType].New()
        detector_response_reader.SetFileName(args_info.detector)
        detector_response_reader.Update()
        detector_image = detector_response_reader.GetOutput()
    else:
        if args_info.verbose:
            print("No detector response provided, using default (ones)...")
        detector_source = rtk.ConstantImageSource[DetectorResponseImageType].New()
        detector_source.SetSize([1, maximum_energy])
        detector_source.SetConstant(1.0)
        detector_source.Update()
        detector_image = detector_source.GetOutput()

    # Generate a set of zero-filled intensity projections
    dual_energy_projections = DualEnergyProjectionsType.New()
    dual_energy_projections.CopyInformation(decomposed_projection_reader.GetOutput())
    dual_energy_projections.SetVectorLength(2)
    dual_energy_projections.Allocate()

    # Check that the inputs have the expected size
    if decomposed_projection_reader.GetOutput().GetVectorLength() != 2:
        print(f"Error: Decomposed projections image has vector length {decomposed_projection_reader.GetOutput().GetVectorLength()}, should be 2")
        sys.exit(1)

    if material_attenuations_reader.GetOutput().GetLargestPossibleRegion().GetSize()[1] != maximum_energy:
        print(f"Error: Material attenuations image has {material_attenuations_reader.GetOutput().GetLargestPossibleRegion().GetSize()[1]} energies, should have {maximum_energy}")
        sys.exit(1)

    # Create and set the filter
    if args_info.verbose:
        print("Setting up spectral forward model filter...")
    forward = rtk.SpectralForwardModelImageFilter[DecomposedProjectionType, DualEnergyProjectionsType].New()
    forward.SetInputDecomposedProjections(decomposed_projection_reader.GetOutput())
    forward.SetInputMeasuredProjections(dual_energy_projections)
    forward.SetInputIncidentSpectrum(incident_spectrum_reader_high_energy.GetOutput())
    forward.SetInputSecondIncidentSpectrum(incident_spectrum_reader_low_energy.GetOutput())
    forward.SetDetectorResponse(detector_image)
    forward.SetMaterialAttenuations(material_attenuations_reader.GetOutput())
    forward.SetIsSpectralCT(False)
    forward.SetComputeVariances(args_info.variances is not None)

    forward.Update()

    # Write output
    if args_info.verbose:
        print(f"Writing output to {args_info.output}...")
    writer = itk.ImageFileWriter[DualEnergyProjectionsType].New()
    writer.SetInput(forward.GetOutput())
    writer.SetFileName(args_info.output)
    writer.Update()

    # If requested, write the variances
    if args_info.variances:
        if args_info.verbose:
            print(f"Writing variances to {args_info.variances}...")

        itk.imwrite(forward.GetOutputVariances(), args_info.variances)

    if args_info.verbose:
        print("Processing completed successfully.")


def main(argv=None):
    parser = build_parser()
    args_info = parser.parse_args(argv)
    process(args_info)


if __name__ == "__main__":
    main()
