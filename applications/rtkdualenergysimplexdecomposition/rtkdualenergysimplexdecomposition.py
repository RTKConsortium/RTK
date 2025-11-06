#!/usr/bin/env python
import argparse
import sys
import itk
from itk import RTK as rtk


def build_parser():
    parser = rtk.RTKArgumentParser(
        description="Decomposes dual energy projections into materials"
    )

    # General options
    parser.add_argument(
        "--input",
        "-i",
        help="Initial solution for material decomposition, file name",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--output",
        "-o",
        help="Output file name (decomposed projections)",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--dual",
        help="Projections to be decomposed, VectorImage file name",
        type=str,
        required=True,
    )
    parser.add_argument("--detector", "-d", help="Detector response file", type=str)
    parser.add_argument(
        "--high", help="High energy incident spectrum file", type=str, required=True
    )
    parser.add_argument(
        "--low", help="Low energy incident spectrum file", type=str, required=True
    )
    parser.add_argument(
        "--attenuations",
        "-a",
        help="Material attenuations file",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--niterations", "-n", help="Number of iterations", type=int, default=300
    )
    parser.add_argument(
        "--weightsmap",
        "-w",
        help="File name for the output weights map (inverse noise variance)",
        type=str,
    )
    parser.add_argument(
        "--restarts",
        "-r",
        help="Allow random restarts during optimization",
        action="store_true",
    )
    parser.add_argument(
        "--guess",
        "-g",
        help="Ignore values in input and initialize the simplex with a simple heuristic instead",
        action="store_true",
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
        print("Reading initial solution for material decomposition...")
    decomposed_projection_reader = itk.ImageFileReader[DecomposedProjectionType].New()
    decomposed_projection_reader.SetFileName(args_info.input)
    decomposed_projection_reader.Update()

    if args_info.verbose:
        print("Reading dual energy projections...")
    dual_energy_projection_reader = itk.ImageFileReader[DualEnergyProjectionsType].New()
    dual_energy_projection_reader.SetFileName(args_info.dual)
    dual_energy_projection_reader.Update()

    if args_info.verbose:
        print("Reading incident spectrum (high energy)...")
    incident_spectrum_reader_high_energy = itk.ImageFileReader[
        IncidentSpectrumImageType
    ].New()
    incident_spectrum_reader_high_energy.SetFileName(args_info.high)
    incident_spectrum_reader_high_energy.Update()

    if args_info.verbose:
        print("Reading incident spectrum (low energy)...")
    incident_spectrum_reader_low_energy = itk.ImageFileReader[
        IncidentSpectrumImageType
    ].New()
    incident_spectrum_reader_low_energy.SetFileName(args_info.low)
    incident_spectrum_reader_low_energy.Update()

    if args_info.verbose:
        print("Reading material attenuations...")
    material_attenuations_reader = itk.ImageFileReader[
        MaterialAttenuationsImageType
    ].New()
    material_attenuations_reader.SetFileName(args_info.attenuations)
    material_attenuations_reader.Update()

    # Get parameters from the images
    maximum_energy = (
        incident_spectrum_reader_high_energy.GetOutput()
        .GetLargestPossibleRegion()
        .GetSize(0)
    )

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

    # Validate detector response size
    if detector_image.GetLargestPossibleRegion().GetSize()[1] != maximum_energy:
        print(
            f"Error: Detector response image has {detector_image.GetLargestPossibleRegion().GetSize()[1]} energies, should have {maximum_energy}"
        )
        sys.exit(1)

    # Create and set the filter
    if args_info.verbose:
        print("Setting up simplex spectral projections decomposition filter...")
    simplex = rtk.SimplexSpectralProjectionsDecompositionImageFilter[
        DecomposedProjectionType,
        DualEnergyProjectionsType,
        IncidentSpectrumImageType,
        DetectorResponseImageType,
        MaterialAttenuationsImageType,
    ].New()

    simplex.SetInputDecomposedProjections(decomposed_projection_reader.GetOutput())
    simplex.SetGuessInitialization(args_info.guess)
    simplex.SetInputMeasuredProjections(dual_energy_projection_reader.GetOutput())
    simplex.SetInputIncidentSpectrum(incident_spectrum_reader_high_energy.GetOutput())
    simplex.SetInputSecondIncidentSpectrum(
        incident_spectrum_reader_low_energy.GetOutput()
    )
    simplex.SetDetectorResponse(detector_image)
    simplex.SetMaterialAttenuations(material_attenuations_reader.GetOutput())
    simplex.SetNumberOfIterations(args_info.niterations)
    simplex.SetOptimizeWithRestarts(args_info.restarts)
    simplex.SetIsSpectralCT(False)

    simplex.Update()

    # Write output
    if args_info.verbose:
        print(f"Writing decomposed projections to {args_info.output}...")
    writer = itk.ImageFileWriter[DecomposedProjectionType].New()
    writer.SetInput(simplex.GetOutput(0))
    writer.SetFileName(args_info.output)
    writer.Update()

    # If requested, write the weights map
    if args_info.weightsmap:
        if args_info.verbose:
            print(f"Writing weights map to {args_info.weightsmap}...")

        # Note: The weights map output might need to be accessed differently
        # depending on the actual filter implementation
        weights_writer = itk.ImageFileWriter[DecomposedProjectionType].New()
        weights_writer.SetInput(simplex.GetOutputWeights())
        weights_writer.SetFileName(args_info.weightsmap)
        weights_writer.Update()

    if args_info.verbose:
        print("Decomposition completed successfully.")


def main(argv=None):
    parser = build_parser()
    args_info = parser.parse_args(argv)
    process(args_info)


if __name__ == "__main__":
    main()
