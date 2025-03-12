#!/usr/bin/env python
import argparse
import sys
import itk
from itk import RTK as rtk

def main():
  # Argument parsing
  parser = argparse.ArgumentParser(
      description="Projects a 3D voxelized phantom onto a stack of 2D projections.",
      formatter_class=argparse.ArgumentDefaultsHelpFormatter
  )
  parser.add_argument('--verbose', '-v', help="Verbose execution", action='store_true')
  parser.add_argument('--config', help="Config file", type=str)
  parser.add_argument('--geometry', '-g', help="XML geometry file name", type=str, required=True)
  parser.add_argument('--output', '-o', help="Output projections file name", type=str, required=True)
  parser.add_argument('--phantomfile', help="Configuration parameters for the phantom", type=str, required=True)
  parser.add_argument('--phantomscale', help="Scaling factor for the phantom dimensions", type=rtk.comma_separated_args(float), default=[1.0])
  parser.add_argument('--offset', help="3D spatial offset of the phantom center", type=rtk.comma_separated_args(float))
  parser.add_argument('--forbild', '-f', help="Interpret phantomfile as Forbild file", action='store_true')
  parser.add_argument('--rotation', help="Rotation matrix for the phantom", type=rtk.comma_separated_args(float))

  rtk.add_rtk3Doutputimage_group(parser)

  # Parse the command line arguments
  args_info = parser.parse_args()

  # Define output pixel type and dimension
  OutputPixelType = itk.F
  Dimension = 3
  OutputImageType = itk.Image[OutputPixelType, Dimension]

  # Read geometry file
  if args_info.verbose:
      print(f"Reading geometry information from {args_info.geometry}...")

  geometry = rtk.ReadGeometry(args_info.geometry)

  # Create a stack of empty projection images
  constant_image_source = rtk.ConstantImageSource[OutputImageType].New()
  rtk.SetConstantImageSourceFromArgParse(constant_image_source, args_info)

  # Adjust size according to geometry
  size_output = list(constant_image_source.GetSize())
  size_output[2] = len(geometry.GetGantryAngles())  # Number of projections
  constant_image_source.SetSize(size_output)

  # Offset, scale, rotation
  offset = [0.0, 0.0, 0.0]
  if args_info.offset:
      if len(args_info.offset) > 3:
          print("--offset needs up to 3 values", file=sys.stderr)
          sys.exit(1)
      offset[:len(args_info.offset)] = args_info.offset

  scale = [args_info.phantomscale[0]] * Dimension
  if len(args_info.phantomscale) > 3:
      print("--phantomscale needs up to 3 values", file=sys.stderr)
      sys.exit(1)
  for i in range(min(len(args_info.phantomscale), Dimension)):
      scale[i] = args_info.phantomscale[i]

  rot = itk.Matrix[itk.F, Dimension, Dimension]()
  rot.SetIdentity()
  if args_info.rotation:
      if len(args_info.rotation) != 9:
          print("--rotation needs exactly 9 values", file=sys.stderr)
          sys.exit(1)
      for i in range(Dimension):
          for j in range(Dimension):
              rot[i][j] = args_info.rotation[i * 3 + j]

  # Set up the projection of the phantom
  ppc = rtk.ProjectGeometricPhantomImageFilter[OutputImageType, OutputImageType].New()
  ppc.SetInput(constant_image_source.GetOutput())
  ppc.SetGeometry(geometry)
  ppc.SetPhantomScale(scale)
  ppc.SetOriginOffset(offset)
  ppc.SetRotationMatrix(rot)
  ppc.SetConfigFile(args_info.phantomfile)
  ppc.SetIsForbildConfigFile(args_info.forbild)

  ppc.Update()

  # Write the output projections
  if args_info.verbose:
      print("Projecting and writing...")

  itk.imwrite(ppc.GetOutput(), args_info.output)

if __name__ == '__main__':
    main()