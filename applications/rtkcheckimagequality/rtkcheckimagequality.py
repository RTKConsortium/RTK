import argparse
import sys
import itk
from itk import RTK as rtk

def build_parser():
    parser = argparse.ArgumentParser(
        description="Checks the MSE of a reconstructed image against a reference.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--reference",
        "-i",
        required=True,
        type=rtk.comma_separated_args(str),
        help="Reference volume(s)",
    )
    parser.add_argument(
        "--reconstruction",
        "-j",
        required=True,
        type=rtk.comma_separated_args(str),
        help="Reconstructed volume(s)",
    )
    parser.add_argument(
        "--threshold",
        "-t",
        required=True,
        type=rtk.comma_separated_args(float),
        help="MSE threshold(s)",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose output"
    )
    return parser

def mse(reference, reconstruction):
    arr_ref = itk.array_view_from_image(reference)
    arr_rec = itk.array_view_from_image(reconstruction)
    return float(((arr_ref - arr_rec) ** 2).sum())

def process(args_info: argparse.Namespace):
    # Maximum number of comparisons to perform (depends on the number of inputs)
    n_max = max(
        len(args_info.reference),
        len(args_info.reconstruction),
        len(args_info.threshold),
    )

    for i in range(n_max):
        reference_index = min(len(args_info.reference) - 1, i)
        reconstruction_index = min(len(args_info.reconstruction) - 1, i)
        threshold_index = min(len(args_info.threshold) - 1, i)

        if args_info.verbose:
            print(f"Reading reference image: {args_info.reference[reference_index]}")
        reference = itk.imread(args_info.reference[reference_index])

        if args_info.verbose:
            print(f"Reading reconstruction image: {args_info.reconstruction[reconstruction_index]}")
        reconstruction = itk.imread(args_info.reconstruction[reconstruction_index])

        mse_val = mse(reference, reconstruction)
        if args_info.verbose:
            print(f"MSE: {mse_val} (threshold: {args_info.threshold[threshold_index]})")

        if mse_val > args_info.threshold[threshold_index]:
            print(
                f"Error comparing {args_info.reference[reference_index]} and {args_info.reconstruction[reconstruction_index]}:\n"
                f"MSE {mse_val} above given threshold {args_info.threshold[threshold_index]}",
                file=sys.stderr,
            )
            sys.exit(1)
    if args_info.verbose:
        print("All comparisons passed.")

def main(argv=None):
    parser = build_parser()
    args_info = parser.parse_args(argv)
    process(args_info)

if __name__ == "__main__":
    main()
