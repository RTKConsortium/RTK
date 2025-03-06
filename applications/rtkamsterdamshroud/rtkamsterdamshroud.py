#!/usr/bin/env python
import sys
import argparse
import itk
from itk import RTK as rtk

def main():
  parser = argparse.ArgumentParser(
      description="Creates an Amsterdam Shroud image from a sequence of projections [Zijp et al, ICCR, 2004].",
      formatter_class=argparse.ArgumentDefaultsHelpFormatter
  )

  # Command-line options (translated from GGO)
  parser.add_argument("--verbose", "-v", help="Verbose execution", action="store_true")
  parser.add_argument("--config", help="Config file (not implemented in this script)", type=str)
  parser.add_argument("--output", "-o", help="Output file name", type=str, required=True)
  parser.add_argument("--unsharp", "-u", help="Unsharp mask size", type=int, default=17)
  parser.add_argument("--clipbox","-c",help="3D clipbox for cropping projections (x1, x2, y1, y2, z1, z2) in mm",
      type=rtk.comma_separated_args_info(float))
  parser.add_argument("--geometry", "-g", help="XML geometry file name", type=str)
  parser.add_argument("--input", "-i", help="Input projections file", type=str, required=True)

  rtk.add_rtkinputprojections_group(parser)

  args_info = parser.parse_args()

  if args_info.verbose:
      print("Running Amsterdam Shroud computation...")

  # Define output image type
  OutputPixelType = itk.D
  Dimension = 3
  OutputImageType = itk.Image[OutputPixelType, Dimension]

  # Projections reader
  reader = rtk.ProjectionsReader[OutputImageType].New()
  rtk.SetProjectionsReaderFromArgParse(reader, args_info)

  # Amsterdam Shroud
  shroudFilter = rtk.AmsterdamShroudImageFilter[OutputImageType].New()
  shroudFilter.SetInput(reader.GetOutput())
  shroudFilter.SetUnsharpMaskSize(args_info.unsharp)

  # Corners (if given)
  if args_info.clipbox:
      if len(args_info.clipbox) != 6:
          print("--clipbox requires exactly 6 values, only ",args_info.clipbox," given.")
          sys.exit(1)
      if not args_info.geometry:
          print("You must provide the geometry to use --clipbox.")
          sys.exit(1)

      c1 = itk.Point[itk.D, 3]()
      c2 = itk.Point[itk.D, 3]()
      for i in range(3):
          c1[i] = args_info.clipbox[i * 2]
          c2[i] = args_info.clipbox[i * 2 + 1]
      shroudFilter.SetCorner1(c1)
      shroudFilter.SetCorner2(c2)

      # Geometry
      if args_info.geometry:
          geometry = rtk.ReadGeometry(args_info.geometry)
          shroudFilter.SetGeometry(geometry)

  shroudFilter.UpdateOutputInformation()

  # Write output
  writer = itk.ImageFileWriter[shroudFilter.GetOutput().GetType()].New()
  writer.SetFileName(args_info.output)
  writer.SetInput(shroudFilter.GetOutput())
  writer.SetNumberOfStreamDivisions(shroudFilter.GetOutput().GetLargestPossibleRegion().GetSize(1))

  writer.Update()

  if args_info.verbose:
      print("Amsterdam Shroud computation completed successfully.")

if __name__ == "__main__":
    main()
