#!/usr/bin/env python
import argparse
import sys
import itk
from itk import RTK as rtk

def main():
  # Argument parsing
  parser = argparse.ArgumentParser(
      description='Creates an RTK geometry file from a sequence of x-ray projections of a Bioscan NanoSPECT/CT scanner.',
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)

  parser.add_argument('--verbose', '-v', help='Verbose execution', action='store_true')
  parser.add_argument('--config', help='Config file', type=str)
  parser.add_argument('--output', '-o', help='Output file name', required=True)

  parser.add_argument('--path', '-p', help='Path containing projections', required=True, type=str)
  parser.add_argument('--regexp', '-r', help='Regular expression to select projection files in path', required=True, type=str)
  parser.add_argument('--nsort', help='Numeric sort for regular expression matches', action='store_true')
  parser.add_argument('--submatch', help='Index of the submatch that will be used to sort matches', type=int, default=0)

  rtk.add_rtkinputprojections_group(parser)

  args_info = parser.parse_args()

  # Create geometry reader
  bioscan_reader = rtk.BioscanGeometryReader.New()
  bioscan_reader.SetProjectionsFileNames(rtk.GetProjectionsFileNamesFromArgParse(args_info))

  bioscan_reader.UpdateOutputData()

  # Write
  geometry = bioscan_reader.GetGeometry()
  rtk.WriteGeometry(geometry, args_info.output)

  if args_info.verbose:
      print(f"Geometry file written to {args_info.output}.")

if __name__ == '__main__':
    main()
