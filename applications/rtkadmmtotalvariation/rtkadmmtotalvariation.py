#!/usr/bin/env python
import argparse
import sys
import itk
from itk import RTK as rtk

def main():
  # Argument parsing
  parser = argparse.ArgumentParser(
      description="Perform ADMM total variation reconstruction on cone-beam projections.",
      formatter_class=argparse.ArgumentDefaultsHelpFormatter
  )

  # General options
  parser.add_argument("-v", "--verbose", help="Verbose execution", action="store_true")
  parser.add_argument("--config", help="Config file", type=str)
  parser.add_argument("-g", "--geometry", help="XML geometry file name", type=str, required=True)
  parser.add_argument("-o", "--output", help="Output file name", type=str, required=True)
  parser.add_argument("-n", "--niterations", help="Number of iterations", type=int, default=1)
  parser.add_argument("--alpha", help="Regularization parameter", type=float, default=0.1)
  parser.add_argument("--beta", help="Augmented Lagrangian constraint multiplier", type=float, default=1.0)
  parser.add_argument("--CGiter", help="Number of nested iterations of conjugate gradient", type=int, default=5)
  parser.add_argument("-i", "--input", help="Input volume", type=str)
  parser.add_argument("--nodisplaced", help="Disable the displaced detector filter", action="store_true")

  # Phase gating options
  parser.add_argument("--phases", help="File containing the phase of each projection", type=str)
  parser.add_argument("-c", "--windowcenter", help="Target reconstruction phase", type=float, default=0.0)
  parser.add_argument("-w", "--windowwidth", help="Tolerance around the target phase", type=float, default=1.0)
  parser.add_argument("-s", "--windowshape", help="Shape of the gating window", type=str, choices=["Rectangular", "Triangular"], default="Rectangular")


  # RTK specific groups
  rtk.add_rtkprojectors_group(parser)
  rtk.add_rtkiterations_group(parser)

  # Parse arguments
  args_info = parser.parse_args()

  # Define output pixel type and dimension
  OutputPixelType = itk.F
  Dimension = 3
  OutputImageType = itk.Image[OutputPixelType, Dimension]
  if hasattr(itk, 'CudaImage'):
      OutputCudaImageType = itk.CudaImage[OutputPixelType, Dimension]
      GradientOutputImageType = itk.CudaImage[itk.CovariantVector[OutputPixelType, Dimension], Dimension]
  else:
      GradientOutputImageType = itk.Image[itk.CovariantVector[OutputPixelType, Dimension], Dimension]

  # Projections reader
  projectionsReader = rtk.ProjectionsReader[OutputImageType].New()
  rtk.SetProjectionsReaderFromArgParse(projectionsReader, args_info)

  # Geometry
  if args_info.verbose:
      print(f"Reading geometry from {args_info.geometry}...")

  geometry = rtk.ReadGeometry(args_info.geometry)

  # Phase gating weights reader
  phaseGating = rtk.PhaseGatingImageFilter[OutputImageType].New()
  if args_info.phases:
      phaseGating.SetPhasesFileName(args_info.phases)
      phaseGating.SetGatingWindowWidth(args_info.windowwidth)
      phaseGating.SetGatingWindowCenter(args_info.windowcenter)
      phaseGating.SetGatingWindowShape(args_info.windowshape)
      phaseGating.SetInputProjectionStack(projectionsReader.GetOutput())
      phaseGating.SetInputGeometry(geometry)

      phaseGating.Update()

  # Create input: either an existing volume read from a file or a blank image
  if args_info.input:
      # Read an existing image to initialize the volume
      inputReader = itk.ImageFileReader[OutputImageType].New()
      inputReader.SetFileName(args_info.input)
      inputFilter = inputReader
  else:
      # Create new empty volume
      constantImageSource = rtk.ConstantImageSource[OutputImageType].New()
      inputFilter = constantImageSource

  if hasattr(itk, 'CudaImage'):
      admmFilter = rtk.ADMMTotalVariationConeBeamReconstructionFilter[OutputCudaImageType, GradientOutputImageType].New()
  else:
      admmFilter = rtk.ADMMTotalVariationConeBeamReconstructionFilter[OutputImageType, GradientOutputImageType].New()

  # Set the forward and back projection filters to be used inside admmFilter
  rtk.SetForwardProjectionFromArgParse(args_info, admmFilter)
  rtk.SetBackProjectionFromArgParse(args_info, admmFilter)

  # Set all four numerical parameters
  admmFilter.SetCG_iterations(args_info.CGiter)
  admmFilter.SetAL_iterations(args_info.niterations)
  admmFilter.SetAlpha(args_info.alpha)
  admmFilter.SetBeta(args_info.beta)
  admmFilter.SetDisableDisplacedDetectorFilter(args_info.nodisplaced)

  # Set the inputs of the ADMM filter
  admmFilter.SetInput(0, rtk.CudaImageFromImage(inputFilter.GetOutput()))
  if args_info.phases:
      admmFilter.SetInput(1, rtk.CudaImageFromImage(phaseGating.GetOutput()))
      admmFilter.SetGeometry(phaseGating.GetOutputGeometry())
      admmFilter.SetGatingWeights(phaseGating.GetGatingWeightsOnSelectedProjections())
  else:
      admmFilter.SetInput(1, rtk.CudaImageFromImage(projectionsReader.GetOutput()))
      admmFilter.SetGeometry(geometry)

  rtk.SetIterationsReportFromArgParse(args_info, admmFilter)

  admmFilter.Update()

  itk.imwrite(admmFilter.GetOutput(), args_info.output)

if __name__ == '__main__':
    main()
