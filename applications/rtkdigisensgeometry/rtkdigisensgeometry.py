#!/usr/bin/env python
import argparse
import sys
import itk
from itk import RTK as rtk

def main():
  # Argument parsing
  parser = argparse.ArgumentParser(
      description='Creates an RTK geometry file from a Digisens geometry calibration.',
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)

  parser.add_argument('--verbose', '-v', help='Verbose execution', action='store_true')
  parser.add_argument('--config', help='Config file', type=str)
  parser.add_argument('--xml_file', '-x', help='Digisens XML information file', type=str, required=True)
  parser.add_argument('--output', '-o', help="Output file name", type=str, required=True)

  # Parse the command line arguments
  args_info = parser.parse_args()

  reader = rtk.DigisensGeometryReader.New()
  reader.SetXMLFileName(args_info.xml_file)
  reader.UpdateOutputData()

  rtk.WriteGeometry(reader.GetGeometry(), args_info.output)

if __name__ == '__main__':
  main()