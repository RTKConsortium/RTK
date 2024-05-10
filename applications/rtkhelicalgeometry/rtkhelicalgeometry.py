#!/usr/bin/env python
import argparse
import sys
from itk import RTK as rtk

if __name__ == '__main__':
  # Argument parsing
  parser = argparse.ArgumentParser(description=
    "Creates an RTK helical geometry file from regular helical trajectory. See http://www.openrtk.org/Doxygen/DocGeo3D.html for more information.")


  parser.add_argument('--nproj', '-n', type=int, help='Number of projections')
  parser.add_argument('--output', '-o', help='Output file name')
  parser.add_argument('--verbose', '-v', type=bool, default=False, help='Verbose execution')
  parser.add_argument('--config', '-c', help='Config file')
  parser.add_argument('--first_angle', '-f', type=float, help='First angle in degrees')
  parser.add_argument('--first_sy', '-y', type=float, help='First vertical position (default = centered around 0)')
  parser.add_argument('--arc', '-a', type=float, default=360, help='Angular arc covevered by the acquisition in degrees')
  parser.add_argument('--pitch', '-p', type=float, default=200, help='Helix pitch (vertical displacement in one full (2pi) rotation')
  parser.add_argument('--sdd', type=float, default=1536, help='Source to detector distance (mm)')
  parser.add_argument('--sid', type=float, default=1000, help='Source to isocenter distance (mm)')
  parser.add_argument('--rad_cyl', type=float, default=0, help='Radius cylinder of cylindrical detector')

  args = parser.parse_args()

  if args.nproj is None or args.output is None :
    parser.print_help()
    sys.exit()

  # Simulated Geometry
  GeometryType = rtk.ThreeDCircularProjectionGeometry
  geometry = GeometryType.New()

  for noProj in range(0, args.nproj):

    # Compute the angles
    angular_gap = args.arc/args.nproj
    if args.first_angle is None :
      first_angle = -0.5*angular_gap*(args.nproj-1)
    else :
      first_angle = args.first_angle

    angle = first_angle + noProj * angular_gap

    # Compute vertical positions
    vertical_coverage = args.arc/360.0*args.pitch
    vertical_gap = vertical_coverage/args.nproj
    if args.first_sy is None :
      first_sy = -0.5*vertical_gap*(args.nproj-1)
    else :
      first_sy = args.first_sy

    sy = first_sy + noProj * vertical_gap

    geometry.AddProjection(args.sid,
                           args.sdd,
                           angle,
                           0.,
                           sy,
                           0.,
                           0.,
                           0.,
                           sy)

  geometry.SetRadiusCylindricalDetector(args.rad_cyl)

  writer = rtk.ThreeDCircularProjectionGeometryXMLFileWriter.New()
  writer.SetFilename(args.output)
  writer.SetObject(geometry)
  writer.WriteFile()
