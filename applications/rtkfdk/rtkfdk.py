#!/usr/bin/env python
import argparse
import sys
import math
import itk
from itk import RTK as rtk
import shlex


def build_parser():
    parser = argparse.ArgumentParser(
        description="Reconstructs a 3D volume from a sequence of projections [Feldkamp, David, Kress, 1984].",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # General options
    parser.add_argument(
        "--verbose", "-v", help="Verbose execution", action="store_true"
    )
    parser.add_argument(
        "--geometry", "-g", help="XML geometry file name", type=str, required=True
    )
    parser.add_argument(
        "--output", "-o", help="Output file name", type=str, required=True
    )
    parser.add_argument(
        "--hardware",
        help="Hardware used for computation",
        choices=["cpu", "cuda"],
        default="cpu",
    )
    parser.add_argument(
        "--lowmem",
        "-l",
        help="Load only one projection per thread in memory",
        action="store_true",
    )
    parser.add_argument(
        "--divisions",
        "-d",
        help="Streaming option: number of stream divisions of the CT",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--subsetsize",
        help="Streaming option: number of projections processed at a time",
        type=int,
        default=16,
    )
    parser.add_argument(
        "--nodisplaced",
        help="Disable the displaced detector filter",
        action="store_true",
    )
    parser.add_argument(
        "--short",
        help="Minimum angular gap to detect a short scan (converted to radians)",
        type=float,
        default=20.0,
    )

    # Ramp filter options
    ramp_group = parser.add_argument_group("Ramp filter")
    ramp_group.add_argument(
        "--pad",
        help="Data padding parameter to correct for truncation",
        type=float,
        default=0.0,
    )
    ramp_group.add_argument(
        "--hann",
        help="Cut frequency for Hann window in ]0,1] (0.0 disables it)",
        type=float,
        default=0.0,
    )
    ramp_group.add_argument(
        "--hannY",
        help="Cut frequency for Hann window in ]0,1] (0.0 disables it)",
        type=float,
        default=0.0,
    )

    # Motion compensation options
    motion_group = parser.add_argument_group(
        "Motion compensation described in [Rit et al, TMI, 2009] and [Rit et al, Med Phys, 2009]"
    )
    motion_group.add_argument("--signal", help="Signal file name", type=str)
    motion_group.add_argument("--dvf", help="Input 4D DVF", type=str)

    # RTK specific groups
    rtk.add_rtkinputprojections_group(parser)
    rtk.add_rtk3Doutputimage_group(parser)

    # Parse the command line arguments
    return parser


def process(args: argparse.Namespace):
    # Define pixel type and dimension
    OutputPixelType = itk.F
    Dimension = 3
    OutputImageType = itk.Image[OutputPixelType, Dimension]

    # Projections reader
    reader = rtk.ProjectionsReader[OutputImageType].New()
    rtk.SetProjectionsReaderFromArgParse(reader, args)

    if not args.lowmem:
        if args.verbose:
            print("Reading projections...")
        reader.Update()

    # Geometry
    if args.verbose:
        print(f"Reading geometry information from {args.geometry}...")
    geometry = rtk.read_geometry(args.geometry)

    # Check on hardware parameter
    if not hasattr(itk, "CudaImage") and args.hardware == "cuda":
        print("The program has not been compiled with CUDA option.")
        sys.exit(1)

    # CUDA classes are non-templated, so they do not require a type specification.
    if args.hardware == "cuda":
        ddf = rtk.CudaDisplacedDetectorImageFilter.New()
        ddf.SetInput(itk.cuda_image_from_image(reader.GetOutput()))
        pssf = rtk.CudaParkerShortScanImageFilter.New()
    else:
        ddf = rtk.DisplacedDetectorForOffsetFieldOfViewImageFilter[
            OutputImageType
        ].New()
        ddf.SetInput(reader.GetOutput())
        pssf = rtk.ParkerShortScanImageFilter[OutputImageType].New()

    # Displaced detector weighting
    ddf.SetGeometry(geometry)
    ddf.SetDisable(args.nodisplaced)

    # Short scan image filter
    pssf.SetInput(ddf.GetOutput())
    pssf.SetGeometry(geometry)
    pssf.InPlaceOff()
    pssf.SetAngularGapThreshold(args.short * math.pi / 180.0)

    # Create reconstructed image
    constantImageSource = rtk.ConstantImageSource[OutputImageType].New()
    rtk.SetConstantImageSourceFromArgParse(constantImageSource, args)

    # Motion-compensated objects for the compensation of a cyclic deformation.
    # Although these will only be used if the command line options for motion
    # compensation are set, we still create the object beforehand to avoid auto
    # destruction.
    DVFPixelType = itk.Vector[itk.F, 3]
    DVFImageType = itk.Image[DVFPixelType, 3]
    DVFImageSequenceType = itk.Image[DVFPixelType, 4]

    # Create the deformation filter and reader for DVF
    deformation = rtk.CyclicDeformationImageFilter[
        DVFImageSequenceType, DVFImageType
    ].New()
    dvfReader = itk.ImageFileReader[DVFImageSequenceType].New()
    deformation.SetInput(dvfReader.GetOutput())

    # Set up the back projection filter for motion compensation
    bp = rtk.FDKWarpBackProjectionImageFilter[
        OutputImageType, OutputImageType, type(deformation)
    ].New()
    bp.SetDeformation(deformation)
    bp.SetGeometry(geometry)

    # FDK reconstruction filtering
    if args.hardware == "cuda":
        feldkamp = rtk.CudaFDKConeBeamReconstructionFilter.New()
        feldkamp.SetInput(0, itk.cuda_image_from_image(constantImageSource.GetOutput()))
    else:
        feldkamp = rtk.FDKConeBeamReconstructionFilter[OutputImageType].New()
        feldkamp.SetInput(0, constantImageSource.GetOutput())

    # Set inputs and options for the FDK filter
    feldkamp.SetInput(1, pssf.GetOutput())
    feldkamp.SetGeometry(geometry)
    feldkamp.GetRampFilter().SetTruncationCorrection(args.pad)
    feldkamp.GetRampFilter().SetHannCutFrequency(args.hann)
    feldkamp.GetRampFilter().SetHannCutFrequencyY(args.hannY)
    feldkamp.SetProjectionSubsetSize(args.subsetsize)

    # Progress reporting
    if args.verbose:
        progressCommand = rtk.PercentageProgressCommand(feldkamp)
        feldkamp.AddObserver(itk.ProgressEvent(), progressCommand.callback)
        feldkamp.AddObserver(itk.EndEvent(), progressCommand.End)

    # Motion compensated CBCT settings
    if args.signal and args.dvf:
        if args.hardware == "cuda":
            print("Motion compensation is not supported in CUDA. Aborting")
            sys.exit(1)  # Exit if CUDA is selected with motion compensation
        dvfReader.SetFileName(args.dvf)
        deformation.SetSignalFilename(args.signal)
        feldkamp.SetBackProjectionFilter(bp)

    # Streaming depending on streaming capability of writer
    streamerBP = itk.StreamingImageFilter[OutputImageType, OutputImageType].New()
    streamerBP.SetInput(feldkamp.GetOutput())
    streamerBP.SetNumberOfStreamDivisions(args.divisions)

    # Create a splitter to control how the image region is divided during streaming
    splitter = itk.ImageRegionSplitterDirection.New()
    splitter.SetDirection(2)
    streamerBP.SetRegionSplitter(splitter)

    if args.verbose:
        print(f"Reconstructing and writing...")

    # Write
    itk.imwrite(streamerBP.GetOutput(), args.output)


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    process(args)


def rtkfdk(*args, **kwargs):
    """
    Unified RTKFDK entry point.

    Usage:
      • Shell-style: rtkfdk("-p . -r proj.mha -g geom.xml -o out.mha")
      • Python API:  rtkfdk(path=".", regexp="proj.mha", geometry="geom.xml", output="out.mha")
    """
    # Shell-style style
    if len(args) == 1 and isinstance(args[0], str) and not kwargs:
        argv = shlex.split(args[0])
        return main(argv)

    # Python API style
    if len(args) == 0:
        parser = build_parser()
        parsed = rtk.parse_kwargs(parser, func_name="rtkfdk", **kwargs)
        return process(parsed)


rtk.patch_signature(rtkfdk, build_parser())

if __name__ == "__main__":
    main()
