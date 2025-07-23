#!/usr/bin/env python
import argparse
from cmath import phase
import sys
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
    parser = argparse.ArgumentParser(
        description="Extracts the breathing signal from a shroud image.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # General options
    parser.add_argument(
        "--verbose", "-v", help="Verbose execution", action="store_true"
    )
    parser.add_argument(
        "--input", "-i", help="Input shroud image file name", type=str, required=True
    )
    parser.add_argument(
        "--amplitude", "-a", help="Maximum breathing amplitude explored in mm", type=float
    )
    parser.add_argument(
        "--output", "-o", help="Output file name", type=str, required=True
    )
    parser.add_argument(
        "--method", "-m",
        help="Method to use",
        choices=["Reg1D", "DynamicProgramming"],
        default="Reg1D"
    )

    # Phase extraction options
    phase_group = parser.add_argument_group("Phase extraction")
    phase_group.add_argument(
        "--phase", "-p", help="Output file name for the Hilbert phase signal", type=str
    )
    phase_group.add_argument(
        "--movavg", help="Moving average size applied before phase extraction", type=int, default=1
    )
    phase_group.add_argument(
        "--unsharp", help="Unsharp mask size applied before phase extraction", type=int, default=55
    )
    phase_group.add_argument(
        "--model",
        help="Phase model",
        type=str,
        choices=["LOCAL_PHASE", "LINEAR_BETWEEN_MINIMA", "LINEAR_BETWEEN_MAXIMA"],
        default="LINEAR_BETWEEN_MINIMA"
    )

    return parser


def process(args_info: argparse.Namespace):
    InputPixelType = itk.D
    OutputPixelType = itk.D
    Dimension = 2

    OutputImageType = itk.Image[OutputPixelType, Dimension - 1]

    # Read input image
    if args_info.verbose:
        print(f"Reading input shroud image from {args_info.input}...")

    input_image = itk.imread(args_info.input, InputPixelType)

    # Extract shroud signal
    if args_info.verbose:
        print(f"Extracting shroud signal using method: {args_info.method}")

    if args_info.method == "DynamicProgramming":
        if args_info.amplitude is None:
            print("Error: You must supply a maximum amplitude to look for when using DynamicProgramming method.")
            sys.exit(1)

        shroud_filter = rtk.DPExtractShroudSignalImageFilter[InputPixelType, OutputPixelType].New()
        shroud_filter.SetInput(input_image)
        shroud_filter.SetAmplitude(args_info.amplitude)

        shroud_filter.Update()
        shroud_signal = shroud_filter.GetOutput()

    elif args_info.method == "Reg1D":
        shroud_filter = rtk.Reg1DExtractShroudSignalImageFilter[InputPixelType, OutputPixelType].New()
        shroud_filter.SetInput(input_image)

        shroud_filter.Update()
        shroud_signal = shroud_filter.GetOutput()

    else:
        print(f"Error: The specified method '{args_info.method}' does not exist.")
        sys.exit(1)

    # Write output signal
    if args_info.verbose:
        print(f"Writing shroud signal to {args_info.output}...")

    write_signal_to_text_file(shroud_signal, args_info.output)

    # Process phase signal if requested
    if args_info.phase:
        phase = rtk.ExtractPhaseImageFilter[OutputImageType].New()
        phase.SetInput(shroud_signal)
        phase.SetMovingAverageSize(args_info.movavg)
        phase.SetUnsharpMaskSize(args_info.unsharp)
        # Map string to enum integer value
        model_values = {
            "LOCAL_PHASE": 0,
            "LINEAR_BETWEEN_MINIMA": 1,
            "LINEAR_BETWEEN_MAXIMA": 2
        }
        phase.SetModel(model_values[args_info.model])

        phase.Update()

        if args_info.verbose:
            print(f"Writing phase signal to {args_info.phase}...")

        write_signal_to_text_file(phase.GetOutput(), args_info.phase)

    if args_info.verbose:
        print("Processing completed successfully.")


def main(argv=None):
    parser = build_parser()
    args_info = parser.parse_args(argv)
    process(args_info)


if __name__ == "__main__":
    main()
