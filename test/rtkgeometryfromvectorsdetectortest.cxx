#include "rtkThreeDCircularProjectionGeometry.h"

#include <itkEuler3DTransform.h>
#include <itkMersenneTwisterRandomVariateGenerator.h>
#include <itkMath.h>

#include <vector>

/**
 * \file rtkgeometryfromvectorsdetectortest.cxx
 *
 * \brief Test geometry defined by source position, detector position and orientation
 *
 * This test checks ThreeDCircularProjectionGeometry::AddProjection with the
 * source position, the detector position and the detector row and column
 * vectors. For many source and detector positions and orientations, the
 * projections of a few markers computed with the projection matrix of the
 * geometry are compared to the intersection of the source-to-marker line with
 * the detector plane.
 *
 * \author Phil Steininger
 */

using GeometryType = rtk::ThreeDCircularProjectionGeometry;
using PointType = GeometryType::PointType;
using VectorType = GeometryType::VectorType;

int
rtkgeometryfromvectorsdetectortest(int, char *[])
{
  using RandomType = itk::Statistics::MersenneTwisterRandomVariateGenerator;

  // Random markers around the isocenter
  auto generator = RandomType::New();
  generator->SetSeed(123456);
  std::vector<PointType> markers(5);
  for (auto & marker : markers)
    for (unsigned int d = 0; d < 3; d++)
      marker[d] = generator->GetUniformVariate(-50., 50.);

  // Reference geometry
  PointType sourcePosition;
  sourcePosition[0] = 0.;
  sourcePosition[1] = 0.;
  sourcePosition[2] = 1000.;
  PointType detectorPosition;
  detectorPosition[0] = -(1024. * 0.4) / 2. + 0.4 / 2.;
  detectorPosition[1] = -(1024. * 0.4) / 2. + 0.4 / 2.;
  detectorPosition[2] = -536.;
  VectorType detectorRowVector{};
  detectorRowVector[0] = 1.;
  VectorType detectorColumnVector{};
  detectorColumnVector[1] = 1.;

  // Critical angles and random angles in [-180, 180] degrees
  std::vector<double> gantryAngles{ -180., -90., -0., 0., 90., 180., -150., -120., -60., -30., 150., 120., 60., 30. };
  std::vector<double> outOfPlaneAngles = gantryAngles;
  std::vector<double> inPlaneAngles = gantryAngles;
  generator->SetSeed(123456);
  for (unsigned int i = 0; i < 10; i++)
  {
    inPlaneAngles.push_back(generator->GetUniformVariate(-180., 180.));
    gantryAngles.push_back(generator->GetUniformVariate(-180., 180.));
    outOfPlaneAngles.push_back(generator->GetUniformVariate(-180., 180.));
  }

  constexpr double epsilon = 1e-3; // mm
  unsigned int     failures = 0;
  auto             euler = itk::Euler3DTransform<double>::New();
  for (const double gantryAngle : gantryAngles)
    for (const double outOfPlaneAngle : outOfPlaneAngles)
      for (const double inPlaneAngle : inPlaneAngles)
      {
        // Rotate the device around the isocenter with all three angles
        euler->SetRotation(itk::Math::pi_over_180 * outOfPlaneAngle,
                           itk::Math::pi_over_180 * gantryAngle,
                           itk::Math::pi_over_180 * inPlaneAngle);
        PointType  source = euler->TransformPoint(sourcePosition);
        PointType  detector = euler->TransformPoint(detectorPosition);
        VectorType row = euler->TransformVector(detectorRowVector);
        VectorType column = euler->TransformVector(detectorColumnVector);

        // Random offsets of the source and the detector
        for (unsigned int d = 0; d < 3; d++)
        {
          source[d] += generator->GetUniformVariate(-100., 100.);
          detector[d] += generator->GetUniformVariate(-100., 100.);
        }

        auto geometry = GeometryType::New();
        if (!geometry->AddProjection(source, detector, row, column))
        {
          std::cerr << "ERROR: AddProjection failed for out-of-plane angle " << outOfPlaneAngle << ", gantry angle "
                    << gantryAngle << " and in-plane angle " << inPlaneAngle << std::endl;
          return EXIT_FAILURE;
        }
        const GeometryType::MatrixType matrix = geometry->GetMatrices().back();

        const VectorType normal = itk::CrossProduct(row, column);
        for (const PointType & marker : markers)
        {
          // Reference: intersection of the source-to-marker line with the detector plane
          const VectorType sourceToMarker = marker - source;
          const double     t = (normal * (detector - source)) / (normal * sourceToMarker);
          const VectorType detectorToIntersection = (source + t * sourceToMarker) - detector;
          const double     refU = detectorToIntersection * row;
          const double     refV = detectorToIntersection * column;

          // Projection with the geometry matrix
          GeometryType::HomogeneousVectorType hv;
          for (unsigned int d = 0; d < 3; d++)
            hv[d] = marker[d];
          hv[3] = 1.;
          const auto   proj = matrix * hv;
          const double u = proj[0] / proj[2];
          const double v = proj[1] / proj[2];

          if (!(std::abs(refU - u) <= epsilon && std::abs(refV - v) <= epsilon))
          {
            if (failures++ < 10)
              std::cerr << "ERROR: out-of-plane angle " << outOfPlaneAngle << ", gantry angle " << gantryAngle
                        << ", in-plane angle " << inPlaneAngle << ": expected (" << refU << ", " << refV
                        << ") but got (" << u << ", " << v << ")" << std::endl;
          }
        }
      }

  if (failures)
  {
    std::cerr << "Test FAILED with " << failures << " mismatching marker projections" << std::endl;
    return EXIT_FAILURE;
  }
  std::cout << "Test PASSED! " << std::endl;
  return EXIT_SUCCESS;
}
