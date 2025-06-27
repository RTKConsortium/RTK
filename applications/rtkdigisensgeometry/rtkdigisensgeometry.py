import argparse
from itk import RTK as rtk

def build_parser():
    parser = argparse.ArgumentParser(
        description="Read Digisens geometry XML and write RTK geometry XML file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--xml_file", "-x", required=True, help="Input Digisens XML calibration file"
    )
    parser.add_argument(
        "--output", "-o", required=True, help="Output geometry XML file"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose output"
    )
    return parser

def process(args_info: argparse.Namespace):
    # Create geometry reader
    reader = rtk.DigisensGeometryReader.New()
    reader.SetXMLFileName(args_info.xml_file)
    if args_info.verbose:
        print(f"Reading Digisens XML: {args_info.xml_file}")
    reader.UpdateOutputData()

    # Write geometry
    if args_info.verbose:
        print(f"Writing geometry to {args_info.output}")
    rtk.write_geometry(reader.GetGeometry(), args_info.output)
    if args_info.verbose:
        print("Done.")

def main(argv=None):
    parser = build_parser()
    args_info = parser.parse_args(argv)
    process(args_info)

if __name__ == "__main__":
    main()