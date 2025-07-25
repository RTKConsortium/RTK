#!/usr/bin/env python
import argparse
from itk import RTK as rtk


def build_parser():
    # Argument parsing
    parser = argparse.ArgumentParser(
        description="Creates an RTK geometry file from simulated/regular trajectory. See https://docs.openrtk.org/en/latest/documentation/docs/Geometry.html for more information."
    )

    parser.add_argument(
        "--verbose", "-v", type=bool, default=False, help="Verbose execution"
    )
    parser.add_argument(
        "--nproj", "-n", type=int, help="Number of projections", required=True
    )
    parser.add_argument("--output", "-o", help="Output file name", required=True)
    parser.add_argument(
        "--first_angle", "-f", type=float, default=0, help="First angle in degrees"
    )
    parser.add_argument(
        "--arc",
        "-a",
        type=float,
        default=360,
        help="Angular arc covevered by the acquisition in degrees",
    )
    parser.add_argument(
        "--sdd", type=float, default=1536, help="Source to detector distance (mm)"
    )
    parser.add_argument(
        "--sid", type=float, default=1000, help="Source to isocenter distance (mm)"
    )
    parser.add_argument(
        "--proj_iso_x",
        type=float,
        default=0,
        help="X coordinate of detector point (0,0) mm in rotated coordinate system",
    )
    parser.add_argument(
        "--proj_iso_y",
        type=float,
        default=0,
        help="Y coordinate of detector point (0,0) mm in rotated coordinate system",
    )
    parser.add_argument(
        "--source_x",
        type=float,
        default=0,
        help="X coordinate of source in rotated coordinate system",
    )
    parser.add_argument(
        "--source_y",
        type=float,
        default=0,
        help="Y coordinate of source in rotated coordinate system",
    )
    parser.add_argument("--out_angle", type=float, default=0, help="Out of plane angle")
    parser.add_argument("--in_angle", type=float, default=0, help="In plane angle")
    parser.add_argument(
        "--rad_cyl",
        type=float,
        default=0,
        help="Radius cylinder of cylindrical detector",
    )

    # Parse the command line arguments
    return parser


def process(args: argparse.Namespace):
    # Simulated Geometry
    GeometryType = rtk.ThreeDCircularProjectionGeometry
    geometry = GeometryType.New()

    for noProj in range(0, args.nproj):
        angle = args.first_angle + noProj * args.arc / args.nproj
        geometry.AddProjection(
            args.sid,
            args.sdd,
            angle,
            args.proj_iso_x,
            args.proj_iso_y,
            args.out_angle,
            args.in_angle,
            args.source_x,
            args.source_y,
        )

    geometry.SetRadiusCylindricalDetector(args.rad_cyl)

    rtk.write_geometry(geometry, args.output)


def main(argv=None):
    parser = build_parser()
    args_info = parser.parse_args(argv)
    process(args_info)


if __name__ == "__main__":
    main()
