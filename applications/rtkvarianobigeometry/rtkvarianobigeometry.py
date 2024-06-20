#!/usr/bin/env python
import argparse
import sys
import itk
from itk import RTK as rtk

def main():
  # Argument parsing
  parser = argparse.ArgumentParser(description=
    "Creates an RTK geometry file from a Varian OBI acquisition.")

  parser.add_argument('--verbose', '-v', help='Verbose execution', type=bool)
  parser.add_argument('--xml_file', '-x', help='Varian OBI XML information file on projections')
  parser.add_argument('--output', '-o', help='Output file name')
  parser.add_argument('--path', '-p', help='Path containing projections', required=True)
  parser.add_argument('--regexp', '-r', help='Regular expression to select projection files in path')

  args = parser.parse_args()

  if args.xml_file is None or args.output is None:
    parser.print_help()
    sys.exit()

  names = itk.RegularExpressionSeriesFileNames.New()
  names.SetDirectory(args.path);
  names.SetRegularExpression(args.regexp);

  reader = rtk.VarianObiGeometryReader.New()
  reader.SetXMLFileName(args.xml_file)
  reader.SetProjectionsFileNames(names.GetFileNames())
  reader.UpdateOutputData()

  xmlWriter = rtk.ThreeDCircularProjectionGeometryXMLFileWriter.New()
  xmlWriter.SetFilename(args.output)
  xmlWriter.SetObject(reader.GetGeometry())
  xmlWriter.WriteFile()

if __name__ == '__main__':
  main()
