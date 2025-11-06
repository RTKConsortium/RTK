#!/usr/bin/env python
import argparse
import itk
from itk import RTK as rtk


def build_parser():
    parser = rtk.RTKArgumentParser(
        description="Computes the field of view of a reconstruction."
    )

    parser.add_argument(
        "--geometry", "-g", help="XML geometry file name", type=str, required=True
    )
    parser.add_argument(
        "--output", "-o", help="Output projections file name", type=str, required=True
    )
    parser.add_argument(
        "--reconstruction", help="Reconstruction file unmasked", type=str, required=True
    )
    parser.add_argument(
        "--mask",
        "-m",
        help="Output a binary mask instead of a masked image",
        action="store_true",
    )
    parser.add_argument(
        "--displaced",
        "-d",
        help="Assume that a displaced detector has been used",
        action="store_true",
    )
    parser.add_argument(
        "--bp",
        "-b",
        help="Slow alternative for non cylindrical FOVs: backproject projections filled with ones and threshold result.",
        action="store_true",
    )
    parser.add_argument(
        "--hardware",
        help="Hardware used for computation (with --bp only)",
        choices=["cpu", "cuda"],
        default="cpu",
    )
    rtk.add_rtkinputprojections_group(parser)
    return parser


def process(args_info: argparse.Namespace):
    OutputPixelType = itk.F
    Dimension = 3
    OutputImageType = itk.Image[OutputPixelType, Dimension]
    use_cuda = args_info.hardware == "cuda"
    if use_cuda:
        if not hasattr(itk, "CudaImage"):
            raise RuntimeError("The program has not been compiled with CUDA option.")

    # Projections reader
    reader = rtk.ProjectionsReader[OutputImageType].New()
    rtk.SetProjectionsReaderFromArgParse(reader, args_info)

    # Geometry
    if args_info.verbose:
        print(f"Reading geometry information from {args_info.geometry}...")
    geometry = rtk.read_geometry(args_info.geometry)

    # Reconstruction reader
    if args_info.verbose:
        print(f"Reading reconstruction from {args_info.reconstruction}...")
    unmasked_reconstruction = itk.imread(args_info.reconstruction)

    if not args_info.bp:
        # FOV filter
        fieldofview = rtk.FieldOfViewImageFilter[OutputImageType, OutputImageType].New()
        fieldofview.SetMask(args_info.mask)
        fieldofview.SetInput(0, unmasked_reconstruction)
        fieldofview.SetProjectionsStack(reader.GetOutput())
        fieldofview.SetGeometry(geometry)
        fieldofview.SetDisplacedDetector(args_info.displaced)
        fieldofview.Update()
        if args_info.verbose:
            print(f"Writing output to {args_info.output}...")
        itk.imwrite(fieldofview.GetOutput(), args_info.output)
    else:
        if args_info.displaced:
            raise RuntimeError("Options --displaced and --bp are not compatible (yet).")
        MaskImgType = itk.image[itk.US, Dimension]
        reader.UpdateOutputInformation()
        ones = rtk.ConstantImageSource[MaskImgType].New()
        ones.SetConstant(1)
        ones.SetInformationFromImage(reader.GetOutput())
        zeroVol = rtk.ConstantImageSource[MaskImgType].New()
        zeroVol.SetConstant(0.0)
        zeroVol.SetInformationFromImage(unmasked_reconstruction)

        if use_cuda:
            CudaMaskImgType = itk.CudaImage[OutputPixelType, Dimension]
            bp = rtk.CudaBackProjectionImageFilter[CudaMaskImgType].New()
            bp.SetInput(itk.cuda_image_from_image(zeroVol.GetOutput()))
            bp.SetInput(1, itk.cuda_image_from_image(ones.GetOutput()))
        else:
            bp = rtk.BackProjectionImageFilter[MaskImgType, MaskImgType].New()
            bp.SetInput(zeroVol.GetOutput())
            bp.SetInput(1, ones.GetOutput())
        bp.SetGeometry(geometry)

        thresh = itk.ThresholdImageFilter[MaskImgType].New()
        thresh.SetInput(bp.GetOutput())
        thresh.ThresholdBelow(len(geometry.GetGantryAngles()) - 1)
        thresh.SetOutsideValue(0.0)

        if args_info.mask:
            div = itk.DivideImageFilter[MaskImgType, MaskImgType, MaskImgType].New()
            div.SetInput(thresh.GetOutput())
            div.SetConstant2(len(geometry.GetGantryAngles()))
            if args_info.verbose:
                print(f"Writing mask output to {args_info.output}...")
            itk.imwrite(div.GetOutput(), args_info.output)
        else:
            raise NotImplementedError(
                "Option --bp without --mask is not implemented (yet)."
            )


def main(argv=None):
    parser = build_parser()
    args_info = parser.parse_args(argv)
    process(args_info)


if __name__ == "__main__":
    main()
