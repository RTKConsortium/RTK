#!/usr/bin/env python
import argparse
import itk
from itk import RTK as rtk
import numpy as np


def build_parser():
    parser = rtk.RTKArgumentParser(
        description="Generates an RGB image showing the phase peaks on top of the shroud"
    )

    parser.add_argument(
        "--input", "-i", help="Input shroud image file name", type=str, required=True
    )
    parser.add_argument(
        "--output",
        "-o",
        help="Output RGB image file name (.png, .jpg, ...)",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--signal",
        help="File containing the phase of each projection",
        type=str,
        required=True,
    )

    return parser


def process(args_info: argparse.Namespace):
    if args_info.verbose:
        print(f"Reading input image: {args_info.input}")

    Dimension = 2
    InputImageType = itk.Image[itk.D, Dimension]
    OutputImageType = itk.Image[itk.RGBPixel[itk.UC], Dimension]

    # Read input image
    readImage = itk.imread(args_info.input, itk.D)

    # Read signal file
    signal = np.loadtxt(args_info.signal, delimiter=";", usecols=0)

    # Locate the minima in the signal
    minima = np.zeros(signal.shape, dtype=bool)
    if signal.size > 1:
        minima[1:] = signal[1:] < signal[:-1]

    # Compute min and max of shroud to scale output
    in_array = itk.array_from_image(readImage)
    min_val = float(in_array.min())
    max_val = float(in_array.max())

    # Create output RGB image
    RGBout = OutputImageType.New()
    RGBout.SetRegions(readImage.GetLargestPossibleRegion())
    RGBout.SetSpacing(readImage.GetSpacing())
    RGBout.SetOrigin(readImage.GetOrigin())
    RGBout.SetDirection(readImage.GetDirection())
    RGBout.Allocate()

    # Fill the output via a writable numpy view (shape: Y x X x 3)
    rgb_view = itk.array_view_from_image(RGBout)

    # Scale input to [0, 255] and copy to all channels
    if max_val == min_val:
        scaled = np.full(in_array.shape, 255, dtype=np.uint8)
    else:
        scaled = np.clip(
            np.floor((in_array - min_val) * 255.0 / (max_val - min_val)), 0, 255
        ).astype(np.uint8)

    rgb_view[:, :, 0] = scaled
    rgb_view[:, :, 1] = scaled
    rgb_view[:, :, 2] = scaled

    # If it is a minimum row, draw a red pixel
    rgb_view[minima, :, 0] = 255
    rgb_view[minima, :, 1] = 0
    rgb_view[minima, :, 2] = 0

    itk.imwrite(RGBout, args_info.output)


def main(argv=None):
    parser = build_parser()
    args_info = parser.parse_args(argv)
    process(args_info)


if __name__ == "__main__":
    main()
