#!/usr/bin/env python
import argparse
import sys
import itk
from itk import RTK as rtk


def main():
    # Argument parsing
    parser = argparse.ArgumentParser(
        description="Creates an RTK geometry file from an Elekta Synergy acquisition."
    )

    parser.add_argument("--xml", "-x", help="XML file name (starting with XVI5)")
    parser.add_argument("--output", "-o", help="Output file name")

    args = parser.parse_args()

    if args.xml is None or args.output is None:
        parser.print_help()
        sys.exit()

    reader = rtk.ElektaXVI5GeometryXMLFileReader.New()
    reader.SetFilename(args.xml)
    reader.GenerateOutputInformation()

    rtk.write_geometry(reader.GetGeometry(), args.output)


if __name__ == "__main__":
    main()
