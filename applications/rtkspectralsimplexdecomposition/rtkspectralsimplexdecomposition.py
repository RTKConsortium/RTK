#!/usr/bin/env python
import argparse
import itk
from itk import RTK as rtk


def build_parser():
    parser = rtk.RTKArgumentParser(
        description="Decomposes spectral projections into materials"
    )
    parser.add_argument(
        "--output",
        "-o",
        required=True,
        help="Output file name (decomposed projections)",
    )
    parser.add_argument(
        "--input", "-i", help="Decomposed projections for initialization file name"
    )
    parser.add_argument(
        "--spectral", "-s", required=True, help="Spectral projections to be decomposed"
    )
    parser.add_argument(
        "--detector", "-d", required=True, help="Detector response file"
    )
    parser.add_argument("--incident", required=True, help="Incident spectrum file")
    parser.add_argument(
        "--attenuations", "-a", required=True, help="Material attenuations file"
    )
    parser.add_argument(
        "--niterations", "-n", type=int, default=300, help="Number of iterations"
    )
    parser.add_argument(
        "--thresholds",
        "-t",
        type=float,
        nargs="+",
        required=True,
        help="Lower threshold of bins, expressed in pulse height",
    )
    parser.add_argument(
        "--weightsmap",
        "-w",
        help="File name for the output weights map (inverse noise variance)",
    )
    parser.add_argument(
        "--restarts",
        "-r",
        action="store_true",
        help="Allow random restarts during optimization",
    )
    parser.add_argument(
        "--fischer", "-f", help="File name for the Fischer information matrix"
    )
    parser.add_argument(
        "--log",
        "-l",
        action="store_true",
        help="Log transform each bin, and concatenate the projections with the decomposed ones",
    )
    parser.add_argument(
        "--guess",
        "-g",
        action="store_true",
        help="Ignore values in input and initialize the simplex with a simple heuristic instead",
    )
    return parser


def process(args_info: argparse.Namespace):
    PixelValueType = itk.F
    Dimension = 3

    DecomposedProjectionType = itk.VectorImage[PixelValueType, Dimension]
    SpectralProjectionsType = itk.VectorImage[PixelValueType, Dimension]
    IncidentSpectrumImageType = itk.Image[PixelValueType, Dimension]
    DetectorResponseImageType = itk.Image[PixelValueType, Dimension - 1]
    MaterialAttenuationsImageType = itk.Image[PixelValueType, Dimension - 1]

    # Read inputs
    # Readers
    if args_info.verbose:
        print(f"Reading decomposed projections from {args_info.input}...")
    if args_info.input:
        reader = itk.ImageFileReader[DecomposedProjectionType].New()
        reader.SetFileName(args_info.input)
        reader.Update()
        decomposedProjection = reader.GetOutput()
    else:
        decomposedProjection = None

    if args_info.verbose:
        print(f"Reading spectral projections from {args_info.spectral}...")
    reader = itk.ImageFileReader[SpectralProjectionsType].New()
    reader.SetFileName(args_info.spectral)
    reader.Update()
    spectralProjection = reader.GetOutput()

    if args_info.verbose:
        print(f"Reading incident spectrum from {args_info.incident}...")
    reader = itk.ImageFileReader[IncidentSpectrumImageType].New()
    reader.SetFileName(args_info.incident)
    reader.Update()
    incidentSpectrum = reader.GetOutput()

    if args_info.verbose:
        print(f"Reading detector response from {args_info.detector}...")
    reader = itk.ImageFileReader[DetectorResponseImageType].New()
    reader.SetFileName(args_info.detector)
    reader.Update()
    detectorResponse = reader.GetOutput()

    if args_info.verbose:
        print(f"Reading material attenuations from {args_info.attenuations}...")
    reader = itk.ImageFileReader[MaterialAttenuationsImageType].New()
    reader.SetFileName(args_info.attenuations)
    reader.Update()
    materialAttenuations = reader.GetOutput()

    # Parameters
    NumberOfMaterials = int(
        materialAttenuations.GetLargestPossibleRegion().GetSize()[0]
    )
    NumberOfSpectralBins = int(spectralProjection.GetVectorLength())
    MaximumEnergy = int(incidentSpectrum.GetLargestPossibleRegion().GetSize()[0])

    # Thresholds
    if len(args_info.thresholds) != NumberOfSpectralBins:
        raise RuntimeError(
            f"Number of thresholds {len(args_info.thresholds)} does not match the number of bins {NumberOfSpectralBins}"
        )
    thresholds = itk.VariableLengthVector[itk.D]()
    thresholds.SetSize(NumberOfSpectralBins + 1)
    for i in range(NumberOfSpectralBins):
        thresholds[i] = float(args_info.thresholds[i])
    MaximumPulseHeight = int(detectorResponse.GetLargestPossibleRegion().GetSize()[1])
    thresholds[NumberOfSpectralBins] = MaximumPulseHeight

    # Sanity checks
    idx = itk.Index[3]()
    idx.Fill(0)
    if decomposedProjection is not None:
        if decomposedProjection.GetPixel(idx).Size() != NumberOfMaterials:
            raise RuntimeError(
                f"Decomposed projections vector size {decomposedProjection.GetPixel(idx).Size()} != {NumberOfMaterials}"
            )

    if spectralProjection.GetPixel(idx).Size() != NumberOfSpectralBins:
        raise RuntimeError(
            f"Spectral projections vector size {spectralProjection.GetPixel(idx).Size()} != {NumberOfSpectralBins}"
        )

    if detectorResponse.GetLargestPossibleRegion().GetSize()[0] != MaximumEnergy:
        raise RuntimeError(
            f"Detector response energies {detectorResponse.GetLargestPossibleRegion().GetSize()[0]} != {MaximumEnergy}"
        )

    # Create and set the filter
    simplex = rtk.SimplexSpectralProjectionsDecompositionImageFilter[
        DecomposedProjectionType,
        SpectralProjectionsType,
        IncidentSpectrumImageType,
        DetectorResponseImageType,
        MaterialAttenuationsImageType,
    ].New()
    if decomposedProjection is not None:
        simplex.SetInputDecomposedProjections(decomposedProjection)
    simplex.SetGuessInitialization(args_info.guess)
    simplex.SetInputMeasuredProjections(spectralProjection)
    simplex.SetInputIncidentSpectrum(incidentSpectrum)
    simplex.SetDetectorResponse(detectorResponse)
    simplex.SetMaterialAttenuations(materialAttenuations)
    simplex.SetThresholds(thresholds)
    simplex.SetNumberOfIterations(args_info.niterations)
    simplex.SetOptimizeWithRestarts(args_info.restarts)
    simplex.SetLogTransformEachBin(args_info.log)

    # Note: The simplex filter is set to perform several searches for each pixel,
    # with different initializations, and keep the best one (SetOptimizeWithRestart(true)).
    # While it may yield better results, these initializations are partially random,
    # which makes the output non-reproducible.
    # The default behavior, used for example in the tests, is not to use this feature
    # (SetOptimizeWithRestart(false)), which makes the output reproducible.

    if args_info.weightsmap:
        simplex.SetOutputInverseCramerRaoLowerBound(True)
    if args_info.fischer:
        simplex.SetOutputFischerMatrix(True)

    if args_info.verbose:
        print("Running simplex decomposition...")
    simplex.Update()

    itk.imwrite(simplex.GetOutput(0), args_info.output)
    if args_info.weightsmap:
        itk.imwrite(simplex.GetOutput(1), args_info.weightsmap)
    if args_info.fischer:
        itk.imwrite(simplex.GetOutput(2), args_info.fischer)


def main(argv=None):
    parser = build_parser()
    args_info = parser.parse_args(argv)
    process(args_info)


if __name__ == "__main__":
    main()
