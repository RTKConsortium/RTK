#!/usr/bin/env python
import argparse
import sys
import math
import itk
from itk import RTK as rtk
import numpy as np


def build_parser():
    parser = rtk.RTKArgumentParser(
        description="Generates an RGB image showing the phase peaks on top of the shroud"
    )

    parser.add_argument("--verbose", "-v", help="Verbose execution", action="store_true")
    parser.add_argument("--input", "-i", help="Input shroud image file name", type=str, required=True)
    parser.add_argument("--output", "-o", help="Output RGB image file name (.png, .jpg, ...)", type=str, required=True)
    parser.add_argument("--signal", help="File containing the phase of each projection", type=str, required=True)

    return parser


def process(args_info: argparse.Namespace):
    if args_info.verbose:
        print(f"Reading input image: {args_info.input}")

    # Read
    readImage = itk.imread(args_info.input)

    # Read signal file
    signalReader = itk.CSVArray2DFileReader[itk.D].New()
    signalReader.SetFileName(args_info.signal)
    signalReader.SetFieldDelimiterCharacter(';')
    signalReader.HasRowHeadersOff()
    signalReader.HasColumnHeadersOff()
    signalReader.Update()

    # Extract the first column as a numpy vector
    signal = np.asarray(signalReader.GetArray2DDataObject().GetColumn(0))

    # Locate minima: minima[i] is true if signal[i] < signal[i-1]; first element false
    minima = np.zeros(signal.shape, dtype=bool)
    if signal.size > 1:
        minima[1:] = signal[1:] < signal[:-1]

    # Pythonic / NumPy-based implementation (fast and concise)
    # Get input as a NumPy array (shape: [rows, cols])
    in_array = itk.GetArrayFromImage(readImage)

    # Compute min/max and scaled grayscale [0..255]
    min_val = float(np.min(in_array))
    max_val = float(np.max(in_array))
    if max_val == min_val:
        scaled = np.clip(np.floor(in_array).astype(np.int32), 0, 255).astype(np.uint8)
    else:
        scaled = np.clip(np.floor((in_array - min_val) * 255.0 / (max_val - min_val)), 0, 255).astype(np.uint8)

    # Build RGB array (rows, cols, 3)
    rgb_np = np.stack([scaled, scaled, scaled], axis=-1)

    # Map minima (signal) to image rows. The C++ code indexes minima by Y (row)
    img_rows = in_array.shape[0]
    sig_len = minima.shape[0]
    if sig_len == img_rows:
        minima_rows = minima
    else:
        rows = np.linspace(0, sig_len - 1, img_rows).round().astype(int)
        minima_rows = minima[rows]

    # Paint rows where minima_rows is True in red (R=255, G=B=0)
    if minima_rows.any():
        rgb_np[minima_rows, :, :] = np.array([255, 0, 0], dtype=np.uint8)

    # Convert NumPy array to an ITK image view and copy spatial metadata
    itk_image = itk.image_view_from_array(rgb_np)
    itk_image.SetSpacing(readImage.GetSpacing())
    itk_image.SetOrigin(readImage.GetOrigin())
    itk_image.SetDirection(readImage.GetDirection())

    itk.imwrite(itk_image, args_info.output)


def main(argv=None):
    parser = build_parser()
    args_info = parser.parse_args(argv)
    process(args_info)


if __name__ == "__main__":
    main()
