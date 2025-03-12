#!/usr/bin/env python
import argparse
import sys
import itk
import math
from itk import RTK as rtk

def main():
  # Argument parsing
  parser = argparse.ArgumentParser(
      description="Reconstructs a 3D volume from a sequence of projections [Feldkamp, David, Kress, 1984].",
      formatter_class=argparse.ArgumentDefaultsHelpFormatter
  )

  # General options
  parser.add_argument('--verbose', '-v', help="Verbose execution", action='store_true')
  parser.add_argument('--config', help="Config file", type=str)
  parser.add_argument('--geometry', '-g', help="XML geometry file name", type=str, required=True)
  parser.add_argument('--output', '-o', help="Output file name", type=str, required=True)
  parser.add_argument('--hardware', help="Hardware used for computation", choices=['cpu', 'cuda'], default="cpu")
  parser.add_argument('--lowmem', '-l', help="Load only one projection per thread in memory", action='store_true')
  parser.add_argument('--divisions', '-d', help="Streaming option: number of stream divisions of the CT", type=int, default=1)
  parser.add_argument('--subsetsize', help="Streaming option: number of projections processed at a time", type=int, default=16)
  parser.add_argument('--nodisplaced', help="Disable the displaced detector filter", action='store_true')
  parser.add_argument('--short', help="Minimum angular gap to detect a short scan (converted to radians)", type=float, default=20.0)

  # Ramp filter options
  ramp_group = parser.add_argument_group("Ramp filter")
  ramp_group.add_argument('--pad', help="Data padding parameter to correct for truncation", type=float, default=0.0)
  ramp_group.add_argument('--hann', help="Cut frequency for Hann window in ]0,1] (0.0 disables it)", type=float, default=0.0)
  ramp_group.add_argument('--hannY', help="Cut frequency for Hann window in ]0,1] (0.0 disables it)", type=float, default=0.0)

  # Motion compensation options
  motion_group = parser.add_argument_group(
      "Motion compensation described in [Rit et al, TMI, 2009] and [Rit et al, Med Phys, 2009]"
  )
  motion_group.add_argument('--signal', help="Signal file name", type=str)
  motion_group.add_argument('--dvf', help="Input 4D DVF", type=str)

  # RTK specific groups
  rtk.add_rtkinputprojections_group(parser)
  rtk.add_rtk3Doutputimage_group(parser)

  # Parse the command line arguments
  args_info = parser.parse_args()

  # Convert `short` from degrees to radians
  args_info.short = args_info.short * math.pi / 180.0

  # Define output pixel type and dimension
  OutputPixelType = itk.F
  Dimension = 3
  OutputImageType = itk.Image[OutputPixelType, Dimension]

  # Projections reader
  reader = rtk.ProjectionsReader[OutputImageType].New()
  rtk.SetProjectionsReaderFromArgParse(reader, args_info)

  if not args_info.lowmem:
      if args_info.verbose:
          print("Reading...")
      reader.Update()

  if not hasattr(itk, 'CudaImage') and args_info.hardware == "cuda":
    print("The program has not been compiled with CUDA option.")
    sys.exit(1)

  # Geometry
  if args_info.verbose:
      print(f'Reading geometry information from {args_info.geometry}')

  geometry = rtk.ReadGeometry(args_info.geometry)

  # Check on hardware parameter
  # CUDA classes are non-templated, so they do not require a type specification.
  if args_info.hardware == "cuda":
      ddf = rtk.CudaDisplacedDetectorImageFilter.New()
      pssf = rtk.CudaParkerShortScanImageFilter.New()
  else:
      ddf = rtk.DisplacedDetectorForOffsetFieldOfViewImageFilter[OutputImageType].New()
      pssf = rtk.ParkerShortScanImageFilter[OutputImageType].New()

  # Displaced detector weighting
  ddf.SetInput(rtk.GetCudaImageFromImage(reader.GetOutput()))
  ddf.SetGeometry(geometry)
  ddf.SetDisable(args_info.nodisplaced)

  # Short scan image filter
  pssf.SetInput(ddf.GetOutput())
  pssf.SetGeometry(geometry)
  pssf.InPlaceOff()
  pssf.SetAngularGapThreshold(args_info.short)

  # Create reconstructed image
  constantImageSource = rtk.ConstantImageSource[OutputImageType].New()
  rtk.SetConstantImageSourceFromArgParse(constantImageSource, args_info)

  # Motion-compensated objects for the compensation of a cyclic deformation.
  # Although these will only be used if the command line options for motion
  # compensation are set, we still create the object beforehand to avoid auto
  # destruction.
  DVFPixelType = itk.Vector[itk.F, 3]
  DVFImageType = itk.Image[DVFPixelType, 3]
  DVFImageSequenceType = itk.Image[DVFPixelType, 4]

  # Create the deformation filter and reader for DVF
  deformation = rtk.CyclicDeformationImageFilter[DVFImageSequenceType, DVFImageType].New()
  dvfReader = itk.ImageFileReader[DVFImageSequenceType].New()
  deformation.SetInput(dvfReader.GetOutput())

  # Set up the back projection filter for motion compensation
  bp = rtk.FDKWarpBackProjectionImageFilter[OutputImageType, OutputImageType, type(deformation)].New()
  bp.SetDeformation(deformation)
  bp.SetGeometry(geometry)

  # FDK reconstruction filtering
  if (args_info.hardware == "cpu"):
      feldkamp = rtk.FDKConeBeamReconstructionFilter[OutputImageType].New()
  else :
      feldkamp = rtk.CudaFDKConeBeamReconstructionFilter.New()

  # Progress reporting
  progressCommand = rtk.PercentageProgressCommand(feldkamp)
  if args_info.verbose:
      print("Registering progress observer...")
      feldkamp.AddObserver(itk.ProgressEvent(), progressCommand.callback)  # Register the callback for progress
      feldkamp.AddObserver(itk.EndEvent(), progressCommand.End)  # Register end notification

  # Set inputs and options for the FDK filter
  feldkamp.SetInput(0, rtk.GetCudaImageFromImage(constantImageSource.GetOutput()))
  feldkamp.SetInput(1, pssf.GetOutput())
  feldkamp.SetGeometry(geometry)
  feldkamp.GetRampFilter().SetTruncationCorrection(args_info.pad)
  feldkamp.GetRampFilter().SetHannCutFrequency(args_info.hann)
  feldkamp.GetRampFilter().SetHannCutFrequencyY(args_info.hannY)
  feldkamp.SetProjectionSubsetSize(args_info.subsetsize)


  # Motion compensated CBCT settings
  if args_info.signal and args_info.dvf:
      if args_info.hardware == "cuda":
          print("Motion compensation is not supported in CUDA. Aborting")
          sys.exit(1)  # Exit if CUDA is selected with motion compensation
      dvfReader.SetFileName(args_info.dvf)
      deformation.SetSignalFilename(args_info.signal)
      feldkamp.SetBackProjectionFilter(bp)

  # Streaming depending on streaming capability of writer
  streamerBP = itk.StreamingImageFilter[OutputImageType, OutputImageType].New()
  streamerBP.SetInput(feldkamp.GetOutput())
  streamerBP.SetNumberOfStreamDivisions(args_info.divisions)

  # Create a splitter to control how the image region is divided during streaming
  splitter = itk.ImageRegionSplitterDirection.New()
  splitter.SetDirection(2)
  streamerBP.SetRegionSplitter(splitter)

  # Write
  if args_info.verbose:
      print("Reconstructing and writing...")

  itk.imwrite(streamerBP.GetOutput(), args_info.output)

if __name__ == '__main__':
    main()