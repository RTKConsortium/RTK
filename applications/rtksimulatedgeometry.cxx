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
                            args_info.proj_iso_y_arg);
    }

  // Write
  rtk::ThreeDCircularProjectionGeometryXMLFileWriter::Pointer xmlWriter =
    rtk::ThreeDCircularProjectionGeometryXMLFileWriter::New();
  xmlWriter->SetFilename(args_info.output_arg);
  xmlWriter->SetObject(&(*geometry) );
  TRY_AND_EXIT_ON_ITK_EXCEPTION( xmlWriter->WriteFile() )

  return EXIT_SUCCESS;
}
