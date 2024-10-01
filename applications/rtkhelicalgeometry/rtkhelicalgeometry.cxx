/*=========================================================================
 *
 *  Copyright RTK Consortium
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *         http://www.apache.org/licenses/LICENSE-2.0.txt
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 *=========================================================================*/

#include "rtkhelicalgeometry_ggo.h"
#include "rtkGgoFunctions.h"

#include "rtkThreeDHelicalProjectionGeometryXMLFileWriter.h"

int
main(int argc, char * argv[])
{
  GGO(rtkhelicalgeometry, args_info);

  // RTK geometry object
  using GeometryType = rtk::ThreeDHelicalProjectionGeometry;
  GeometryType::Pointer geometry = GeometryType::New();

  // Projection matrices
  for (int noProj = 0; noProj < args_info.nproj_arg; noProj++)
  {
    // Compute the angles
    double angular_gap = args_info.arc_arg / args_info.nproj_arg;
    double first_angle = 0.;
    if (!args_info.first_angle_given)
    {
      first_angle = -0.5 * angular_gap * (args_info.nproj_arg - 1);
    }
    else
      first_angle = args_info.first_angle_arg;

    double angle = first_angle + noProj * angular_gap;


    // Compute the vertical displacement
    double vertical_coverage = args_info.arc_arg / 360 * args_info.pitch_arg;
    double vertical_gap = vertical_coverage / args_info.nproj_arg;
    double first_sy = 0.;
    if (!args_info.first_sy_given)
    {
      first_sy = -0.5 * vertical_gap * (args_info.nproj_arg - 1);
    }
    else
    {
      first_sy = args_info.first_sy_arg;
    }

    double sy = first_sy + noProj * vertical_gap;

    geometry->AddProjection(args_info.sid_arg, args_info.sdd_arg, angle, 0, sy, 0, 0, 0, sy);
  }

  // Set cylindrical detector radius
  if (args_info.rad_cyl_given)
    geometry->SetRadiusCylindricalDetector(args_info.rad_cyl_arg);


  // Write
  rtk::ThreeDHelicalProjectionGeometryXMLFileWriter::Pointer xmlWriter =
    rtk::ThreeDHelicalProjectionGeometryXMLFileWriter::New();
  xmlWriter->SetFilename(args_info.output_arg);
  xmlWriter->SetObject(&(*geometry));
  TRY_AND_EXIT_ON_ITK_EXCEPTION(xmlWriter->WriteFile())

  return EXIT_SUCCESS;
}
