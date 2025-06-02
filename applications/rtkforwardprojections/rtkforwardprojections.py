#!/usr/bin/env python
import sys
import argparse
import itk
from itk import RTK as rtk


def build_parser():
    parser = argparse.ArgumentParser(
        description="Projects a volume according to a geometry file."
    )
    parser.add_argument(
        "--verbose", "-v", help="Verbose execution", action="store_true"
    )
    parser.add_argument(
        "--geometry", "-g", help="XML geometry file name", type=str, required=True
    )
    parser.add_argument(
        "--input", "-i", help="Input volume file name", type=str, required=True
    )
    parser.add_argument(
        "--output", "-o", help="Output projections file name", type=str, required=True
    )
    parser.add_argument(
        "--step",
        "-s",
        help="Step size along ray (for CudaRayCast only)",
        type=float,
        default=1,
    )
    parser.add_argument(
        "--lowmem",
        "-l",
        help="Compute only one projection at a time",
        action="store_true",
    )

    # Projectors
    rtkprojectors_group = parser.add_argument_group("Projectors")
    rtkprojectors_group.add_argument(
        "--fp",
        "-f",
        help="Forward projection method",
        choices=[
            "Joseph",
            "JosephAttenuated",
            "CudaRayCast",
            "Zeng",
            "MIP",
            "CudaWrapRayCast",
        ],
        default="Joseph",
    )
    rtkprojectors_group.add_argument(
        "--attenuationmap",
        help="Attenuation map relative to the volume to perfom the attenuation correction",
        type=str,
    )
    rtkprojectors_group.add_argument(
        "--sigmazero",
        help="PSF value at a distance of 0 meter of the detector",
        type=float,
    )
    rtkprojectors_group.add_argument(
        "--alphapsf", help="Slope of the PSF against the detector distance", type=float
    )
    rtkprojectors_group.add_argument(
        "--inferiorclipimage",
        help="Value of the inferior clip of the ray for each pixel of the projections (only with Joseph-based projector)",
        type=str,
    )
    rtkprojectors_group.add_argument(
        "--superiorclipimage",
        help="Value of the superior clip of the ray for each pixel of the projections (only with Joseph-based projector)",
        type=str,
    )

    # Motion compensation options
    warp_forwardprojection_group = parser.add_argument_group(
        "Motion compensation described in [Rit et al, TMI, 2009]"
    )
    warp_forwardprojection_group.add_argument(
        "--signal", help="Signal file name", type=str
    )
    warp_forwardprojection_group.add_argument("--dvf", help="Input 4D DVF", type=str)

    rtk.add_rtk3Doutputimage_group(parser)

    return parser


def process(args_info):
    OutputPixelType = itk.F
    Dimension = 3
    OutputImageType = itk.Image[OutputPixelType, Dimension]
    OutputCudaImageType = itk.CudaImage[OutputPixelType, Dimension]

    # Geometry
    if args_info.verbose:
        print(
            f"Reading geometry information from {args_info.geometry}...",
            end="",
            flush=True,
        )
    geometry = rtk.read_geometry(args_info.geometry)
    if args_info.verbose:
        print(" done.")

    # Create a stack of empty projection images
    ConstantImageSourceType = rtk.ConstantImageSource[OutputImageType]
    constantImageSource = ConstantImageSourceType.New()
    rtk.SetConstantImageSourceFromArgParse(constantImageSource, args_info)

    # Adjust size according to geometry
    sizeOutput = list(constantImageSource.GetSize())
    sizeOutput[2] = len(geometry.GetGantryAngles())
    constantImageSource.SetSize(sizeOutput)

    # Input reader
    if args_info.verbose:
        print(f"Reading input volume {args_info.input}...")
    inputVolume = itk.imread(args_info.input)

    attenuation_map = None
    if args_info.attenuationmap:
        if args_info.verbose:
            print(f"Reading attenuation map {args_info.attenuationmap}...")
        attenuation_map = itk.imread(args_info.attenuationmap)

    inferiorClipImage = None
    superiorClipImage = None
    if args_info.inferiorclipimage:
        if args_info.verbose:
            print(f"Reading inferior clip image {args_info.inferiorclipimage}...")
        inferiorClipImage = itk.imread(args_info.inferiorclipimage)
    if args_info.superiorclipimage:
        if args_info.verbose:
            print(f"Reading superior clip image {args_info.superiorclipimage}...")
        superiorClipImage = itk.imread(args_info.superiorclipimage)

    # Create forward projection image filter
    if args_info.verbose:
        print("Projecting volume...")

    # Select the forward projector
    if args_info.fp == "Joseph":
        forwardProjection = rtk.JosephForwardProjectionImageFilter[
            OutputImageType, OutputImageType
        ].New()
    elif args_info.fp == "JosephAttenuated":
        forwardProjection = rtk.JosephForwardAttenuatedProjectionImageFilter[
            OutputImageType, OutputImageType
        ].New()
        if attenuation_map is None:
            print(
                "JosephAttenuatedForwardProjection requires an attenuation map. "
                "Please provide it using --attenuationmap."
            )
            sys.exit(1)
    elif args_info.fp == "Zeng":
        forwardProjection = rtk.ZengForwardProjectionImageFilter[
            OutputImageType, OutputImageType
        ].New()
    elif args_info.fp == "MIP":
        forwardProjection = rtk.MaximumIntensityProjectionImageFilter[
            OutputImageType, OutputImageType
        ].New()
    elif args_info.fp == "CudaRayCast":
        if hasattr(itk, "CudaImage"):
            forwardProjection = rtk.CudaForwardProjectionImageFilter[
                OutputCudaImageType
            ].New()
            forwardProjection.SetStepSize(args_info.step)
        else:
            print("The program has not been compiled with cuda option")
            sys.exit(1)
    elif args_info.fp == "CudaWrapRayCast":
        if hasattr(itk, "CudaImage"):
            if not args_info.dvf or not args_info.signal:
                print("CudaWrapRayCast requires --dvf and --signal arguments.")
                sys.exit(1)
            forwardProjection = rtk.CudaWarpForwardProjectionImageFilter[
                OutputCudaImageType
            ].New()
            forwardProjection.SetStepSize(args_info.step)

            dvf_image = itk.imread(args_info.dvf)
            forwardProjection.SetDisplacementField(dvf_image)
            forwardProjection.SetSignalFilename(args_info.signal)
        else:
            print("The program has not been compiled with cuda option")
            sys.exit(1)

    if args_info.fp in ["CudaWrapRayCast", "CudaRayCast"]:
        forwardProjection.SetInput(
            itk.cuda_image_from_image(constantImageSource.GetOutput())
        )
        forwardProjection.SetInput(1, itk.cuda_image_from_image(inputVolume))
        if attenuation_map:
            forwardProjection.SetInput(2, itk.cuda_image_from_image(attenuation_map))
    else:
        forwardProjection.SetInput(constantImageSource.GetOutput())
        forwardProjection.SetInput(1, inputVolume)
        if attenuation_map:
            forwardProjection.SetInput(2, attenuation_map)

    if inferiorClipImage is not None:
        if args_info.fp == "Joseph":
            forwardProjection.SetInferiorClipImage(inferiorClipImage)
        elif args_info.fp == "JosephAttenuated":
            forwardProjection.SetInferiorClipImage(inferiorClipImage)
        elif args_info.fp == "MIP":
            forwardProjection.SetInferiorClipImage(inferiorClipImage)
    if superiorClipImage is not None:
        if args_info.fp == "Joseph":
            forwardProjection.SetSuperiorClipImage(superiorClipImage)
        elif args_info.fp == "JosephAttenuated":
            forwardProjection.SetSuperiorClipImage(superiorClipImage)
        elif args_info.fp == "MIP":
            forwardProjection.SetSuperiorClipImage(superiorClipImage)
    if args_info.sigmazero and args_info.fp == "Zeng":
        forwardProjection.SetSigmaZero(args_info.sigmazero)
    if args_info.alphapsf and args_info.fp == "Zeng":
        forwardProjection.SetAlpha(args_info.alphapsf)

    forwardProjection.SetGeometry(geometry)

    if not args_info.lowmem:
        forwardProjection.Update()

    # Write
    if args_info.verbose:
        print("Writing...")

    writter = itk.ImageFileWriter[OutputImageType].New()
    writter.SetFileName(args_info.output)
    writter.SetInput(forwardProjection.GetOutput())
    writter.Update()

    if args_info.verbose:
        print("Processing completed successfully.")


def main(argv=None):
    parser = build_parser()
    args_info = parser.parse_args(argv)
    process(args_info)


if __name__ == "__main__":
    main()
