#!/usr/bin/env python
import sys
import argparse
import itk
from itk import RTK as rtk


def build_parser():
    parser = argparse.ArgumentParser(
        description="Backprojects a volume according to a geometry file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # General arguments
    parser.add_argument(
        "--verbose", "-v", help="Verbose execution", action="store_true"
    )
    parser.add_argument(
        "--geometry", "-g", help="XML geometry file name", type=str, required=True
    )
    parser.add_argument(
        "--output", "-o", help="Output volume file name", type=str, required=True
    )

    # Projectors group
    rtkprojectors_group = parser.add_argument_group("Projectors")
    rtkprojectors_group.add_argument(
        "--bp",
        help="Backprojection method",
        type=str,
        choices=[
            "VoxelBasedBackProjection",
            "FDKBackProjection",
            "FDKWarpBackProjection",
            "Joseph",
            "JosephAttenuated",
            "Zeng",
            "CudaFDKBackProjection",
            "CudaBackProjection",
            "CudaRayCast",
        ],
        default="VoxelBasedBackProjection",
    )

    rtkprojectors_group.add_argument(
        "--attenuationmap", help="Attenuation map for attenuation correction", type=str
    )
    rtkprojectors_group.add_argument(
        "--sigmazero",
        help="PSF value at a distance of 0 meter of the detector",
        type=float,
    )
    rtkprojectors_group.add_argument(
        "--alphapsf", help="Slope of the PSF against the detector distance", type=float
    )

    # Warped backprojections group
    Warped_backprojection_group = parser.add_argument_group(
        "Warped backprojections group"
    )
    Warped_backprojection_group.add_argument(
        "--signal", help="Signal file name", type=str
    )
    Warped_backprojection_group.add_argument("--dvf", help="Input 4D DVF", type=str)

    rtk.add_rtk3Doutputimage_group(parser)
    rtk.add_rtkinputprojections_group(parser)

    # Parse the command line arguments
    return parser


def process(args_info: argparse.Namespace):
    # Define output pixel type and dimension
    OutputPixelType = itk.F
    Dimension = 3
    OutputImageType = itk.Image[OutputPixelType, Dimension]
    if hasattr(itk, "CudaImage"):
        OutputCudaImageType = itk.CudaImage[OutputPixelType, Dimension]

    # Geometry
    if args_info.verbose:
        print(f"Reading geometry from {args_info.geometry}...")
    geometry = rtk.read_geometry(args_info.geometry)
    if args_info.verbose:
        print(f"done.")

    # Create empty volume
    constantImageSource = rtk.ConstantImageSource[OutputImageType].New()
    rtk.SetConstantImageSourceFromArgParse(constantImageSource, args_info)

    # Projections reader
    reader = rtk.ProjectionsReader[itk.Image[itk.F, 3]].New()
    rtk.SetProjectionsReaderFromArgParse(reader, args_info)
    reader.Update()

    attenuation_map = None
    if args_info.attenuationmap:
        if args_info.verbose:
            print(f"Reading attenuation map from {args_info.attenuationmap}...")
        # Read an existing image to initialize the attenuation map
        attenuation_map = itk.imread(args_info.attenuationmap)

    # In case warp backprojection is used, we create a deformation
    DVFPixelType = itk.Vector[itk.F, 3]
    DVFImageSequenceType = itk.Image[DVFPixelType, 4]
    DVFImageType = itk.Image[DVFPixelType, 3]
    DeformationType = rtk.CyclicDeformationImageFilter[
        DVFImageSequenceType, DVFImageType
    ]

    bp = None
    if args_info.bp == "VoxelBasedBackProjection":
        bp = rtk.BackProjectionImageFilter[OutputImageType, OutputImageType].New()
    elif args_info.bp == "FDKBackProjection":
        bp = rtk.FDKBackProjectionImageFilter[OutputImageType, OutputImageType].New()
    elif args_info.bp == "FDKWarpBackProjection":
        if not args_info.dvf or not args_info.signal:
            raise ValueError(
                "FDKWarpBackProjection requires input 4D deformation vector field and signal file names"
            )
        deformation = rtk.CyclicDeformationImageFilter[
            DVFImageSequenceType, DVFImageType
        ].New()
        deformation.SetInput(itk.imread(args_info.dvf))
        bp = rtk.FDKWarpBackProjectionImageFilter[
            OutputImageType, OutputImageType, DeformationType
        ].New()
        deformation.SetSignalFilename(args_info.signal)
        bp.SetDeformation(deformation)
    elif args_info.bp == "Joseph":
        bp = rtk.JosephBackProjectionImageFilter[OutputImageType, OutputImageType].New()
    elif args_info.bp == "JosephAttenuated":
        bp = rtk.JosephBackAttenuatedProjectionImageFilter[
            OutputImageType, OutputImageType
        ].New()
    elif args_info.bp == "Zeng":
        bp = rtk.ZengBackProjectionImageFilter[OutputImageType, OutputImageType].New()
        if args_info.sigmazero:
            bp.SetSigmaZero(args_info.sigmazero)
        if args_info.alphapsf:
            bp.SetAlpha(args_info.alphapsf)

    elif args_info.bp == "CudaFDKBackProjection":
        if hasattr(itk, "CudaImage"):
            bp = rtk.CudaFDKBackProjectionImageFilter.New()
        else:
            print("The program has not been compiled with cuda option")
            sys.exit(1)
    elif args_info.bp == "CudaBackProjection":
        if hasattr(itk, "CudaImage"):
            bp = rtk.CudaBackProjectionImageFilter[OutputCudaImageType].New()
        else:
            print("The program has not been compiled with cuda option")
            sys.exit(1)
    elif args_info.bp == "CudaRayCast":
        if hasattr(itk, "CudaImage"):
            bp = rtk.CudaRayCastBackProjectionImageFilter.New()
        else:
            print("The program has not been compiled with cuda option")
            sys.exit(1)

    if args_info.bp in ["CudaFDKBackProjection", "CudaBackProjection", "CudaRayCast"]:
        bp.SetInput(itk.cuda_image_from_image(constantImageSource.GetOutput()))
        bp.SetInput(1, itk.cuda_image_from_image(reader.GetOutput()))
        if attenuation_map:
            bp.SetInput(2, itk.cuda_image_from_image(attenuation_map))
    else:
        bp.SetInput(constantImageSource.GetOutput())
        bp.SetInput(1, reader.GetOutput())
        if attenuation_map:
            bp.SetInput(2, attenuation_map)

    bp.SetGeometry(geometry)

    bp.Update()

    # Write
    if args_info.verbose:
        print(f"Writing output to {args_info.output}...")

    writter = itk.ImageFileWriter[OutputImageType].New()
    writter.SetFileName(args_info.output)
    writter.SetInput(bp.GetOutput())
    writter.Update()

    if args_info.verbose:
        print("Processing completed successfully.")


def main(argv=None):
    parser = build_parser()
    args_info = parser.parse_args(argv)
    process(args_info)


if __name__ == "__main__":
    main()
