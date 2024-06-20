#!/usr/bin/env python
import argparse
import sys
import itk
from itk import RTK as rtk

def main():
  # Argument parsing
  parser = argparse.ArgumentParser(description=
    "Creates an RTK geometry file from an Elekta Synergy acquisition.")

  parser.add_argument('--verbose', '-v', help='Verbose execution', type=bool)
  parser.add_argument('--xml', '-x', help='XML file name (starting with XVI5)')
  parser.add_argument('--output', '-o', help='Output file name')

  args = parser.parse_args()

  if args.xml is None or args.output is None:
    parser.print_help()
    sys.exit()

  reader = rtk.ElektaXVI5GeometryXMLFileReader.New()
  reader.SetFilename(args.xml)
  reader.GenerateOutputInformation()

  xmlWriter = rtk.ThreeDCircularProjectionGeometryXMLFileWriter.New()
  xmlWriter.SetFilename(args.output)
  xmlWriter.SetObject(reader.GetGeometry())
  xmlWriter.WriteFile()

if __name__ == '__main__':
  main()
