#!/usr/bin/env python
import argparse
import itk
from itk import RTK as rtk


def write_signal_to_text_file(signal_image, filename):

    # Convert ITK image to numpy array
    signal_array = itk.array_from_image(signal_image)

    # Write to text file
    with open(filename, 'w') as f:
        for value in signal_array.flatten():
            f.write(f"{value}\n")


def build_parser():
    parser = rtk.RTKArgumentParser(
        description="Extracts the phase from a signal."
    )

    # General options
    parser.add_argument(
        "--verbose", "-v", help="Verbose execution", action="store_true"
    )
    parser.add_argument(
        "--input", "-i", help="Input signal", type=str, required=True
    )
    parser.add_argument(
        "--output", "-o", help="Output file name for the Hilbert phase signal", type=str, required=True
    )
    parser.add_argument(
        "--movavg", help="Moving average size applied before phase extraction", type=int, default=1
    )
    parser.add_argument(
        "--unsharp", help="Unsharp mask size applied before phase extraction", type=int, default=55
    )
    parser.add_argument(
        "--model",
        help="Phase model",
        choices=["LOCAL_PHASE", "LINEAR_BETWEEN_MINIMA", "LINEAR_BETWEEN_MAXIMA"],
        default="LINEAR_BETWEEN_MINIMA"
    )

    return parser


def process(args_info: argparse.Namespace):
    # Define pixel type and dimension
    PixelType = itk.D  # double
    Dimension = 1
    ImageType = itk.Image[PixelType, Dimension]

    # Read input signal
    if args_info.verbose:
        print(f"Reading input signal from {args_info.input}...")

    reader = itk.ImageFileReader[ImageType].New()
    reader.SetFileName(args_info.input)
    reader.Update()
    signal = reader.GetOutput()

    # Process phase signal if required
    PhaseFilter = rtk.ExtractPhaseImageFilter[ImageType]
    phase = PhaseFilter.New()
    phase.SetInput(signal)
    phase.SetMovingAverageSize(args_info.movavg)
    phase.SetUnsharpMaskSize(args_info.unsharp)

    # Map string to enum integer value
    model_values = {
        "LOCAL_PHASE": 0,
        "LINEAR_BETWEEN_MINIMA": 1,
        "LINEAR_BETWEEN_MAXIMA": 2
    }
    phase.SetModel(model_values[args_info.model])

    # Write output phase signal
    if args_info.verbose:
        print(f"Writing phase signal to {args_info.output}...")

    write_signal_to_text_file(phase.GetOutput(), args_info.output)

    if args_info.verbose:
        print("Phase extraction completed successfully.")


def main(argv=None):
    parser = build_parser()
    args_info = parser.parse_args(argv)
    process(args_info)


if __name__ == "__main__":
    main()
