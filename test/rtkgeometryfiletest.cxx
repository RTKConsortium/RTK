#include "rtkTestConfiguration.h"
#include "rtkThreeDCircularProjectionGeometryXMLFile.h"
#include "rtkMacro.h"
#include <itksys/SystemTools.hxx>

using GeometryType = rtk::ThreeDCircularProjectionGeometry;

void
WriteReadAndCheck(GeometryType * geometry)
{
  const char   fileName[] = "rtkgeometryfiletest.out";
  const double epsilon = 1e-13;

  rtk::ThreeDCircularProjectionGeometryXMLFileWriter::Pointer xmlWriter =
    rtk::ThreeDCircularProjectionGeometryXMLFileWriter::New();
  xmlWriter->SetFilename(fileName);
  xmlWriter->SetObject(geometry);
  TRY_AND_EXIT_ON_ITK_EXCEPTION(xmlWriter->WriteFile())

  rtk::ThreeDCircularProjectionGeometryXMLFileReader::Pointer xmlReader;
  xmlReader = rtk::ThreeDCircularProjectionGeometryXMLFileReader::New();
  xmlReader->SetFilename(fileName);
  TRY_AND_EXIT_ON_ITK_EXCEPTION(xmlReader->GenerateOutputInformation())

  itksys::SystemTools::RemoveFile(fileName);

  GeometryType * geoRead = xmlReader->GetOutputObject();
  for (unsigned int i = 0; i < geometry->GetGantryAngles().size(); i++)
  {
#define CHECK_GEOMETRY_PARAMETER(paramName)                                                   \
  {                                                                                           \
    double val1 = geoRead->Get##paramName()[i];                                               \
    double val2 = geometry->Get##paramName()[i];                                              \
    if (itk::Math::abs(val1 - val2) > epsilon)                                                \
    {                                                                                         \
      std::cerr << #paramName " differ [" << val1 << "] read from written file vs. [" << val2 \
                << "] for the reference, difference=[" << val1 - val2 << "]." << std::endl;   \
      exit(1);                                                                                \
    }                                                                                         \
  }

    CHECK_GEOMETRY_PARAMETER(GantryAngles);
    CHECK_GEOMETRY_PARAMETER(OutOfPlaneAngles);
    CHECK_GEOMETRY_PARAMETER(InPlaneAngles);
    CHECK_GEOMETRY_PARAMETER(SourceToIsocenterDistances);
    CHECK_GEOMETRY_PARAMETER(SourceOffsetsX);
    CHECK_GEOMETRY_PARAMETER(SourceOffsetsY);
    CHECK_GEOMETRY_PARAMETER(SourceToDetectorDistances);
    CHECK_GEOMETRY_PARAMETER(ProjectionOffsetsX);
    CHECK_GEOMETRY_PARAMETER(ProjectionOffsetsY);
  }
}

/**
 * \file rtkgeometryfiletest.cxx
 *
 * \brief Functional tests for classes managing RTK geometry data
 *
 * This test creates different RTK geometries and compares the result to
 * to the expected one, read from a baseline .txt file in the RTK format.
 *
 * \author Simon Rit
 */

int
main(int, char **)
{
  // Create a geometry object with 1 projection
  auto geometry = GeometryType::New();
  geometry->AddProjection(615., 548., 36., 1.3, 1.57, 15.4, 13.48, 5.42, 7.56);
  WriteReadAndCheck(geometry);

  // Create a geometry object with 5 projections with similar geometry parameters
  geometry = GeometryType::New();
  for (int i = 0; i < 5; i++)
    geometry->AddProjection(615., 548., 36., 1.3, 1.57, 15.4, 13.48, 5.42, 7.56);
  WriteReadAndCheck(geometry);

  // Create a geometry object with 5 projections with different geometry parameters
  geometry = GeometryType::New();
  geometry->AddProjection(615., 15., -8., 1.3, 1.57, 15.4, 13.48, 5.42, 7.56);
  geometry->AddProjection(354., 873., 3218., 354.4, 6587.1, .483, 384.4, 6874.4, 384.4);
  geometry->AddProjection(1869., 6987., 684., 681., -45.1, 6548.1, 6547., .214, .217);
  geometry->AddProjection(1532., 3218., 98732., -184.5, 548.1, -659.4, 123.4, 87.4, -15476.);
  geometry->AddProjection(578., 68., 9879., -38.4, 2158.4, -158.4, -43.3, 3218.4, 325.4);
  WriteReadAndCheck(geometry);

  std::cout << "\n\nTest PASSED! " << std::endl;
  return EXIT_SUCCESS;
}
