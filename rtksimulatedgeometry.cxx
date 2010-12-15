#include "rtksimulatedgeometry_ggo.h"
#include "rtkGgoFunctions.h"

#include "itkThreeDCircularProjectionGeometryXMLFile.h"

#include <itkTimeProbe.h>


int main(int argc, char * argv[])
{
  GGO(rtksimulatedgeometry, args_info);

  // RTK geometry object
  typedef itk::ThreeDCircularProjectionGeometry GeometryType;
  GeometryType::Pointer geometry = GeometryType::New();

  // Global parameters
  geometry->SetSourceToDetectorDistance(args_info.sdd_arg);
  geometry->SetSourceToIsocenterDistance(args_info.sid_arg);

  // Projection matrices
  for(int noProj=0; noProj<args_info.nproj_arg; noProj++)
  {
    double angle = args_info.first_angle_arg + noProj * args_info.arc_arg / args_info.nproj_arg;
    geometry->AddProjection(angle, args_info.proj_iso_x_arg, args_info.proj_iso_y_arg);
  }

  // Write
  itk::ThreeDCircularProjectionGeometryXMLFileWriter::Pointer xmlWriter = itk::ThreeDCircularProjectionGeometryXMLFileWriter::New();
  xmlWriter->SetFilename(args_info.output_arg);
  xmlWriter->SetObject(&(*geometry));
  xmlWriter->WriteFile();

  return EXIT_SUCCESS;
}
