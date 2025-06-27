import argparse
import sys
from itk import RTK as rtk

def build_parser():
    parser = argparse.ArgumentParser(
        description="Read Bioscan geometry and write RTK geometry XML file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--output", "-o", required=True, help="Output geometry XML file"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose output"
    )
    rtk.add_rtkinputprojections_group(parser)
    return parser

def process(args_info: argparse.Namespace):
    # Create geometry reader
    bioscanReader = rtk.BioscanGeometryReader.New()
    bioscanReader.SetProjectionsFileNames(rtk.GetProjectionsFileNamesFromArgParse(args_info))
    if args_info.verbose:
        print("Reading Bioscan geometry...")
    bioscanReader.UpdateOutputData()

    # Write geometry
    if args_info.verbose:
        print(f"Writing geometry to {args_info.output}...")
    rtk.write_geometry(bioscanReader.GetGeometry(), args_info.output)
    if args_info.verbose:
        print("Done.")

def main(argv=None):
    parser = build_parser()
    args_info = parser.parse_args(argv)
    process(args_info)

if __name__ == "__main__":
    main()