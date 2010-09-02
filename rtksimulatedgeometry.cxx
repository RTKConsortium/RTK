#include "rtkThreeDCircularGeometryXMLFile.h"
#include "rtksimulatedgeometry_ggo.h"

#include <itkTimeProbe.h>


int main(int argc, char * argv[])
{
  GGO(rtksimulatedgeometry, args_info);

  // RTK geometry object
  typedef rtk::ThreeDCircularGeometry GeometryType;
  GeometryType::Pointer geometry = GeometryType::New();

  // Global parameters
  geometry->SetSourceToDetectorDistance(args_info.sdd_arg);
  geometry->SetSourceToIsocenterDistance(args_info.sid_arg);

  // Projection matrices
  for(int noProj=0; noProj<args_info.nproj_arg; noProj++)
  {
    double angle = args_info.first_angle_arg + noProj * args_info.arc_arg / args_info.nproj_arg;
    geometry->AddProjection(angle, 0.0, 0.0);
  }

  // Write
  rtk::ThreeDCircularGeometryXMLFileWriter::Pointer xmlWriter = rtk::ThreeDCircularGeometryXMLFileWriter::New();
  xmlWriter->SetFilename(args_info.output_arg);
  xmlWriter->SetObject(&(*geometry));
  xmlWriter->WriteFile();

  return EXIT_SUCCESS;
}
