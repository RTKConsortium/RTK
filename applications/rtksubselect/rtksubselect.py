#!/usr/bin/env python
import argparse
import itk
from itk import RTK as rtk


def build_parser():
    # argument parsing
    parser = rtk.RTKArgumentParser(
        description="Subselect projections from a stack and write updated geometry."
    )

    parser.add_argument(
        "--geometry", "-g", help="XML geometry file name", type=str, required=True
    )
    parser.add_argument(
        "--out_geometry", help="Output geometry file name", type=str, required=True
    )
    parser.add_argument(
        "--out_proj", help="Output projections stack file name", type=str, required=True
    )

    parser.add_argument(
        "--first", "-f", help="First projection index", type=int, default=0
    )
    parser.add_argument("--last", "-l", help="Last projection index", type=int)
    parser.add_argument(
        "--step", "-s", help="Step between projections", type=int, default=1
    )
    parser.add_argument(
        "--list",
        help="List of projection indices to keep (0-based)",
        type=int,
        nargs="+",
    )
    # RTK specific groups
    rtk.add_rtkinputprojections_group(parser)

    # Parse the command line arguments
    return parser


def process(args_info: argparse.Namespace):
    OutputPixelType = itk.F
    Dimension = 3
    OutputImageType = itk.Image[OutputPixelType, Dimension]

    # Projections reader
    reader = rtk.ProjectionsReader[OutputImageType].New()
    rtk.SetProjectionsReaderFromArgParse(reader, args_info)

    # Geometry
    if args_info.verbose:
        print(f"Reading geometry information from {args_info.geometry}...")
    geometry = rtk.read_geometry(args_info.geometry)

    # Compute the indices of the selected projections
    indices = []
    n = len(geometry.GetGantryAngles())
    if args_info.last:
        n = min(args_info.last, n)
    if args_info.list:
        for i in args_info.list:
            indices.append(i)
    else:
        start = args_info.first
        step = args_info.step
        for noProj in range(start, n, step):
            indices.append(noProj)

    # Output RTK geometry object
    outputGeometry = rtk.ThreeDCircularProjectionGeometry.New()

    # Output projections object
    source = rtk.ConstantImageSource[OutputImageType].New()
    source.SetInformationFromImage(reader.GetOutput())
    outputSize = itk.size(reader.GetOutput())
    outputSize_list = [s for s in outputSize]
    outputSize_list[Dimension - 1] = len(indices)
    source.SetSize(outputSize_list)
    source.SetConstant(0)
    source.Update()

    # Fill in the outputGeometry and the output projections
    paste = itk.PasteImageFilter[OutputImageType].New()
    paste.SetSourceImage(reader.GetOutput())
    paste.SetDestinationImage(source.GetOutput())

    for i, src_idx in enumerate(indices):
        # If it is not the first projection, we need to use the output of the paste filter as input
        if i:
            pimg = paste.GetOutput()
            pimg.DisconnectPipeline()
            paste.SetDestinationImage(pimg)

        sourceRegion = reader.GetOutput().GetLargestPossibleRegion()
        sourceRegion.SetIndex(Dimension - 1, src_idx)
        sourceRegion.SetSize(Dimension - 1, 1)
        paste.SetSourceRegion(sourceRegion)

        destinationIndex = reader.GetOutput().GetLargestPossibleRegion().GetIndex()
        destinationIndex.SetElement(Dimension - 1, i)
        paste.SetDestinationIndex(destinationIndex)

        paste.Update()

        # Fill in the output geometry object
        outputGeometry.SetRadiusCylindricalDetector(
            geometry.GetRadiusCylindricalDetector()
        )
        outputGeometry.AddProjectionInRadians(
            geometry.GetSourceToIsocenterDistances()[src_idx],
            geometry.GetSourceToDetectorDistances()[src_idx],
            geometry.GetGantryAngles()[src_idx],
            geometry.GetProjectionOffsetsX()[src_idx],
            geometry.GetProjectionOffsetsY()[src_idx],
            geometry.GetOutOfPlaneAngles()[src_idx],
            geometry.GetInPlaneAngles()[src_idx],
            geometry.GetSourceOffsetsX()[src_idx],
            geometry.GetSourceOffsetsY()[src_idx],
        )
        outputGeometry.SetCollimationOfLastProjection(
            geometry.GetCollimationUInf()[src_idx],
            geometry.GetCollimationUSup()[src_idx],
            geometry.GetCollimationVInf()[src_idx],
            geometry.GetCollimationVSup()[src_idx],
        )

    # Geometry writer
    if args_info.verbose:
        print(f"Writing geometry information in {args_info.out_geometry}...")
    rtk.write_geometry(outputGeometry, args_info.out_geometry)

    # Write
    if args_info.verbose:
        print(f"Writing selected projections to {args_info.out_proj}...")
    itk.imwrite(paste.GetOutput(), args_info.out_proj)


def main(argv=None):
    parser = build_parser()
    args_info = parser.parse_args(argv)
    process(args_info)


if __name__ == "__main__":
    main()
