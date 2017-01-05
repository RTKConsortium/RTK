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

#include "rtksimulatedgeometry_ggo.h"
#include "rtkGgoFunctions.h"

#include "rtkThreeDCircularProjectionGeometryXMLFile.h"

int main(int argc, char * argv[])
{
  GGO(rtksimulatedgeometry, args_info);

  // RTK geometry object
  typedef rtk::ThreeDCircularProjectionGeometry GeometryType;
  GeometryType::Pointer geometry = GeometryType::New();

  // Projection matrices
  for(int noProj=0; noProj<args_info.nproj_arg; noProj++)
    {
    double angle = args_info.first_angle_arg + noProj * args_info.arc_arg / args_info.nproj_arg;
    geometry->AddProjection(args_info.sid_arg,
                            args_info.sdd_arg,
                            angle,
                            args_info.proj_iso_x_arg,
                            args_info.proj_iso_y_arg,
                            args_info.out_angle_arg,
                            args_info.in_angle_arg,
                            args_info.source_x_arg,
                            args_info.source_y_arg);
    }

  // Set cylindrical detector radius
  if (args_info.rad_cyl_given)
    geometry->SetRadiusCylindricalDetector(args_info.rad_cyl_arg);

  // Write
  rtk::ThreeDCircularProjectionGeometryXMLFileWriter::Pointer xmlWriter =
    rtk::ThreeDCircularProjectionGeometryXMLFileWriter::New();
  xmlWriter->SetFilename(args_info.output_arg);
  xmlWriter->SetObject(&(*geometry) );
  TRY_AND_EXIT_ON_ITK_EXCEPTION( xmlWriter->WriteFile() )

  return EXIT_SUCCESS;
}
