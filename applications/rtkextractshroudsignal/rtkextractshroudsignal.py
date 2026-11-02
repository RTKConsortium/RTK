#!/usr/bin/env python
import sys
import argparse

import itk
from itk import RTK as rtk


def write_signal_to_text_file(signal_image, filename):
    # Convert ITK image to NumPy array
    signal_array = itk.array_from_image(signal_image)

    # Write signal to text file
    with open(filename, "w") as output_file:
        for value in signal_array.flatten():
            output_file.write(f"{value}\n")


def build_parser():
    parser = rtk.RTKArgumentParser(
        description="Extracts the breathing signal from a shroud image."
    )

    # General options
    parser.add_argument(
        "--input", "-i", help="Input shroud image file name", type=str, required=True
    )
    parser.add_argument(
        "--amplitude",
        "-a",
        help="Maximum breathing amplitude explored in mm",
        type=float,
    )
    parser.add_argument(
        "--output", "-o", help="Output file name", type=str, required=True
    )
    parser.add_argument(
        "--method",
        "-m",
        help="Method to use (Reg1D or DynamicProgramming)",
        choices=("Reg1D", "DynamicProgramming"),
        default="Reg1D",
    )

    # Phase extraction
    parser.add_argument(
        "--phase", "-p", help="Output file name for the Hilbert phase signal", type=str
    )
    parser.add_argument(
        "--movavg",
        help="Moving average size applied before phase extraction",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--unsharp",
        help="Unsharp mask size applied before phase extraction",
        type=int,
        default=55,
    )
    parser.add_argument(
        "--model",
        help="Phase model",
        choices=["LOCAL_PHASE", "LINEAR_BETWEEN_MINIMA", "LINEAR_BETWEEN_MAXIMA"],
        default="LINEAR_BETWEEN_MINIMA",
    )
    return parser


def process(args_info: argparse.Namespace):
    if args_info.method == "DynamicProgramming" and args_info.amplitude is None:
        print("You must supply a maximum amplitude to look for.")
        sys.exit(1)

    if args_info.verbose:
        print(f"Reading input shroud image from {args_info.input}...")

    # Define input and output image types
    PixelType = itk.D
    Dimension = 2
    InputImageType = itk.Image[PixelType, Dimension]
    OutputImageType = itk.Image[PixelType, Dimension - 1]

    # Read input shroud image
    reader = itk.ImageFileReader[InputImageType].New()
    reader.SetFileName(args_info.input)
    reader.Update()
    inputImage = reader.GetOutput()

    # Extract shroud signal
    if args_info.method == "DynamicProgramming":
        ShroudFilter = rtk.DPExtractShroudSignalImageFilter[PixelType, PixelType]
        shroudFilter = ShroudFilter.New()
        shroudFilter.SetAmplitude(args_info.amplitude)
    elif args_info.method == "Reg1D":
        ShroudFilter = rtk.Reg1DExtractShroudSignalImageFilter[PixelType, PixelType]
        shroudFilter = ShroudFilter.New()
    else:
        print("The specified method does not exist.")
        sys.exit(1)

    shroudFilter.SetInput(inputImage)
    shroudFilter.Update()
    shroudSignal = shroudFilter.GetOutput()

    if args_info.verbose:
        print(f"Writing shroud signal to {args_info.output}...")
    write_signal_to_text_file(shroudSignal, args_info.output)

    if args_info.phase:
        if args_info.verbose:
            print(f"Extracting phase signal to {args_info.phase}...")

        PhaseFilter = rtk.ExtractPhaseImageFilter[OutputImageType]
        phase = PhaseFilter.New()
        phase.SetInput(shroudSignal)
        phase.SetMovingAverageSize(args_info.movavg)
        phase.SetUnsharpMaskSize(args_info.unsharp)
        model_values = {
            "LOCAL_PHASE": 0,
            "LINEAR_BETWEEN_MINIMA": 1,
            "LINEAR_BETWEEN_MAXIMA": 2,
        }
        phase.SetModel(model_values[args_info.model])
        phase.Update()
        write_signal_to_text_file(phase.GetOutput(), args_info.phase)

    if args_info.verbose:
        print("Shroud signal extraction completed successfully.")


def main(argv=None):
    parser = build_parser()
    args_info = parser.parse_args(argv)
    process(args_info)


if __name__ == "__main__":
    main()
