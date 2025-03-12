#!/usr/bin/env python
import argparse
import sys
import itk
from itk import RTK as rtk
import numpy as np

def MSE(reference, reconstruction):
    """Calculate Mean Squared Error (MSE) between two images."""
    # Convert images to numpy arrays for easier manipulation
    ref_array = itk.GetArrayFromImage(reference)
    rec_array = itk.GetArrayFromImage(reconstruction)

    # Calculate MSE
    mse = np.mean((ref_array - rec_array) ** 2)
    return mse

def main():
    # Argument parsing
    parser = argparse.ArgumentParser(
        description='Checks the MSE of a reconstructed image against a reference.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--verbose', '-v', help='Verbose execution', action='store_true')
    parser.add_argument('--config', help='Config file', type=str)
    parser.add_argument('--reference', '-i', help='Reference volume', type=rtk.comma_separated_args(str), required=True)
    parser.add_argument('--reconstruction', '-j', help='Reconstructed volume', type=rtk.comma_separated_args(str), required=True)
    parser.add_argument('--threshold', '-t', help='MSE threshold', type=rtk.comma_separated_args(float), required=True)

    args = parser.parse_args()

    # Maximum number of comparisons to perform
    n_max = max(len(args.reference), len(args.reconstruction), len(args.threshold))

    for i in range(n_max):
        reference_index = min(len(args.reference) - 1, i)
        reconstruction_index = min(len(args.reconstruction) - 1, i)
        threshold_index = min(len(args.threshold) - 1, i)

        reference = itk.imread(args.reference[reference_index])
        reconstruction = itk.imread(args.reconstruction[reconstruction_index])

        mse = MSE(reference, reconstruction)

        if mse > args.threshold[threshold_index]:
            print(f"Error comparing {args.reference[reference_index]} and {args.reconstruction[reconstruction_index]}:")
            print(f"MSE {mse} above given threshold {args.threshold[threshold_index]}")
            return sys.exit(1)

        if args.verbose:
            print(f"Compared {args.reference[reference_index]} and {args.reconstruction[reconstruction_index]}: MSE = {mse}")

if __name__ == '__main__':
    sys.exit(main())
