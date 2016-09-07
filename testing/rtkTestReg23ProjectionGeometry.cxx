//
//std
#include <stdlib.h>
#include <iostream>
#include <vector>
#include <math.h>
//ITK
#include <itkMersenneTwisterRandomVariateGenerator.h>
#include <itkEuler3DTransform.h>
#include <itkVector.h>
#include <itkPoint.h>
//RTK/ORA
#include "rtkReg23ProjectionGeometry.h"


//
bool Verbose = false; // spam the console
#define VERBOSE(x) \
if (Verbose) std::cout x;


/** Print command line parameter help. **/
void PrintHelp(char *binaryName)
{
  std::cerr << "\n\n***** TEST: ora::Reg23ProjectionGeometry *****\n\n";

  if (binaryName)
    std::cerr << binaryName;
  else
    std::cerr << "<executable>";
  std::cerr << " [options] \n\n";

  std::cerr << "\n available options:\n";
  std::cerr << "   -h or --help ... display this help and exit application\n";
  std::cerr << "   -v or --verbose ... verbose messages on std::cout\n";
  std::cerr << "\n\n***************************************************\n\n";
}

typedef rtk::Reg23ProjectionGeometry GeometryType;

/**
 * @brief Intersect detector plane with line between source and a specified
 * marker.
 * @param S source position
 * @param marker phantom marker
 * @param n normal of detector plane
 * @param detectorOrigin detector position
 * @param x returned intersection point in WCS
 * @return TRUE if successful intersection was detected
 */
void IntersectPlaneWithLine(const GeometryType::PointType &S,
                            const GeometryType::PointType &marker,
                            const GeometryType::VectorType &n,
                            const GeometryType::PointType &detectorOrigin,
                            GeometryType::VectorType &x)
{
  double num, den;
  double marker1[3];

  marker1[0] = marker[0] - S[0];
  marker1[1] = marker[1] - S[1];
  marker1[2] = marker[2] - S[2];

  num = (n[0] * detectorOrigin[0] + n[1] * detectorOrigin[1] + n[2] * detectorOrigin[2])
      - (n[0] * S[0] + n[1] * S[1] + n[2] * S[2]);
  den = n[0] * marker1[0] + n[1] * marker1[1] + n[2] * marker1[2];

  double t = num / den;
  x[0] = S[0] + t * marker1[0];
  x[1] = S[1] + t * marker1[1];
  x[2] = S[2] + t * marker1[2];
}


/** @brief Unit test to check geometrical correctness of
 * ora::Reg23ProjectionGeometry utility class.
 * @author phil steininger
 * @version 1.0
 * @param argc run program with "-h" option in order to get more info
 * @param argv run program with "-h" option in order to get more info
 * @return EXIT_SUCCESS if the test approves correctness of the utility class
 */
int main(int argc, char *argv[])
{
  // parse cmdln args:
  std::string sarg;
  for (int i = 1; i < argc; i++)
  {
    sarg = std::string(argv[i]);
    if (sarg == "-h" || sarg == "--help")
    {
      PrintHelp(argc > 0 ? argv[0] : ITK_NULLPTR);
      return EXIT_SUCCESS;
    }
    if (sarg == "-v" || sarg == "--verbose")
    {
      Verbose = true;
      continue;
    }
  }

  bool ok = true;
  bool lok = true;

  VERBOSE(<< "\n\nStart testing ora::Reg23ProjectionGeometry\n\n")

  typedef itk::Statistics::MersenneTwisterRandomVariateGenerator RandomType;
  typedef itk::Euler3DTransform<double> EulerType;
  typedef itk::Point<double, 2> Point2DType;

  GeometryType::PointType sourcePosition;
  GeometryType::PointType detectorPosition;
  GeometryType::VectorType detectorRowDirection;
  GeometryType::VectorType detectorColumnDirection;
  std::vector<GeometryType::PointType> phantomMarkers;

  VERBOSE(<< "  * Generating data sets ... ")
  lok = true;
  {
    // determine a few random phantom markers around isocenter
    const int NUM_PHANTOM_MARKERS = 5;
    RandomType::Pointer generator = RandomType::New();
    generator->Initialize(123456);
    GeometryType::PointType p;
    for (int i = 0; i < NUM_PHANTOM_MARKERS; i++)
    {
      for (int d = 0; d < 3; d++)
        p[d] = generator->GetUniformVariate(-50, 50);
      phantomMarkers.push_back(p);
    }
    // setup a basic REG23-like geometry imitating basically XVI at 0 deg w.r.t
    // IEC CS (normally encapsulated in REG23's ora::ProjectionGeometry):
    sourcePosition[0] = 0.;
    sourcePosition[1] = 0.;
    sourcePosition[2] = 1000.;
    detectorPosition[0] = -(1024. * 0.4) / 2. + 0.4 / 2.;
    detectorPosition[1] = -(1024. * 0.4) / 2. + 0.4 / 2.;
    detectorPosition[2] = -536.;
    detectorRowDirection[0] = 1.;
    detectorRowDirection[1] = 0.;
    detectorRowDirection[2] = 0.;
    detectorColumnDirection[0] = 0.;
    detectorColumnDirection[1] = 1.;
    detectorColumnDirection[2] = 0.;
  }
  ok = ok && lok;
  VERBOSE(<< (lok ? "OK" : "FAILURE") << "\n")


  VERBOSE(<< "  * Swirl imaging device, compute reference projections and "
          << "configure RTK projection list ... ")
  lok = true;
  std::vector<Point2DType> referenceMarkerProjections;
  std::vector<Point2DType> rtkMarkerProjections;
  std::vector<GeometryType::PointType> anglesList;
  GeometryType::PointType aaa;
  GeometryType::Pointer rtkProjectionsList = GeometryType::New();
  {
    const double degreesToRadians = atan(1.0) / 45.;

    RandomType::Pointer generator = RandomType::New();
    generator->Initialize(123456);

    GeometryType::PointType locSourcePosition;
    GeometryType::PointType locDetectorPosition;
    GeometryType::VectorType locDetectorPositionVec;
    GeometryType::VectorType locDetectorRowDirection;
    GeometryType::VectorType locDetectorColumnDirection;
    GeometryType::VectorType n;
    GeometryType::VectorType X;
    Point2DType p2d;
    GeometryType::MatrixType rtkMatrix;
    GeometryType::HomogeneousVectorType hv;
    GeometryType::VectorType tmp;

    EulerType::Pointer eu = EulerType::New();
    std::vector<double> gantryAngles;
    std::vector<double> outOfPlaneAngles;
    std::vector<double> inPlaneAngles;
    std::vector<double> criticalAngles;

    // create list of "critical" angles in general:
    criticalAngles.push_back(-180);
    criticalAngles.push_back(-90);
    criticalAngles.push_back(-0);
    criticalAngles.push_back(0);
    criticalAngles.push_back(90);
    criticalAngles.push_back(180);
    // further potentially interesting angles:
    criticalAngles.push_back(-150);
    criticalAngles.push_back(-120);
    criticalAngles.push_back(-60);
    criticalAngles.push_back(-30);
    criticalAngles.push_back(150);
    criticalAngles.push_back(120);
    criticalAngles.push_back(60);
    criticalAngles.push_back(30);
    inPlaneAngles.insert(inPlaneAngles.end(), criticalAngles.begin(), criticalAngles.end());
    gantryAngles.insert(gantryAngles.end(), criticalAngles.begin(), criticalAngles.end());
    outOfPlaneAngles.insert(outOfPlaneAngles.end(), criticalAngles.begin(), criticalAngles.end());
    // randomly sample further angles in [-180;+180] deg range:
    const int NUM_RANDOM_ANGLES = 100;
    for (int u = 0; u < NUM_RANDOM_ANGLES; u++)
    {
      inPlaneAngles.push_back(generator->GetUniformVariate(-180, 180));
      gantryAngles.push_back(generator->GetUniformVariate(-180, 180));
      outOfPlaneAngles.push_back(generator->GetUniformVariate(-180, 180));
    }

    for (std::size_t ga = 0; ga < gantryAngles.size(); ++ga)
    {
      for (std::size_t oa = 0; oa < outOfPlaneAngles.size(); ++oa)
      {
        for (std::size_t ia = 0; ia < inPlaneAngles.size(); ++ia)
        {
          rtkProjectionsList->Clear(); // reduce required memory

          // "swirl" the device around (not only circular trajectory, spherical):
          eu->SetRotation(outOfPlaneAngles[oa] * degreesToRadians,
                          gantryAngles[ga] * degreesToRadians,
                          inPlaneAngles[ia] * degreesToRadians);
          locSourcePosition = eu->TransformPoint(sourcePosition);
          locDetectorPosition = eu->TransformPoint(detectorPosition);
          locDetectorRowDirection = eu->TransformVector(detectorRowDirection);
          locDetectorColumnDirection = eu->TransformVector(detectorColumnDirection);

          // offset detector and source randomly to make it more interesting:
          for (int d = 0; d < 3; d++)
          {
            locSourcePosition[d] += generator->GetUniformVariate(-100, 100);
            locDetectorPosition[d] += generator->GetUniformVariate(-100, 100);
            locDetectorPositionVec[d] = locDetectorPosition[d]; // helper
          }

          // compute analytically the relative 2D projection position related to the
          // detector position:
          for (std::size_t k = 0; k < phantomMarkers.size(); k++)
          {
            n = itk::CrossProduct(locDetectorRowDirection, locDetectorColumnDirection);
            IntersectPlaneWithLine(locSourcePosition, phantomMarkers[k], n,
                                   locDetectorPosition, X);
            p2d[0] = X * locDetectorRowDirection
                - locDetectorPositionVec * locDetectorRowDirection;
            p2d[1] = X * locDetectorColumnDirection
                - locDetectorPositionVec * locDetectorColumnDirection;

            referenceMarkerProjections.push_back(p2d);
          }

          // add to RTK projection list using the new method:
          if (!rtkProjectionsList->AddReg23Projection(locSourcePosition,
                                                      locDetectorPosition,
                                                      locDetectorRowDirection,
                                                      locDetectorColumnDirection))
          {
            lok = false;
          }

          // compute RTK projection for each marker:
          rtkMatrix = rtkProjectionsList->GetMatrices()[rtkProjectionsList->GetMatrices().size() - 1];
          for (std::size_t k = 0; k < phantomMarkers.size(); k++)
          {
            hv[0] = phantomMarkers[k][0];
            hv[1] = phantomMarkers[k][1];
            hv[2] = phantomMarkers[k][2];
            hv[3] = 1.;
            tmp = rtkMatrix * hv;
            p2d[0] = tmp[0] / tmp[2]; // perspective divide
            p2d[1] = tmp[1] / tmp[2];

            rtkMarkerProjections.push_back(p2d);

            // for possible error output later on:
            aaa[0] = outOfPlaneAngles[oa];
            aaa[1] = gantryAngles[ga];
            aaa[2] = inPlaneAngles[ia];
            anglesList.push_back(aaa);
          }
        }
      }
    }
  }
  ok = ok && lok;
  VERBOSE(<< (lok ? "OK" : "FAILURE") << "\n")


  VERBOSE(<< "  * Check whether RTK projections match reference projections ... ")
  lok = true;
  {
    const double EPSILON = 1e-3; // im unit is mm -> correct in um-range
    std::size_t fails = 0;
    if (referenceMarkerProjections.size() == rtkMarkerProjections.size())
    {
      for (std::size_t i = 0; i < referenceMarkerProjections.size(); ++i)
      {
        if (fabs(referenceMarkerProjections[i][0] - rtkMarkerProjections[i][0]) > EPSILON ||
            fabs(referenceMarkerProjections[i][1] - rtkMarkerProjections[i][1]) > EPSILON)
        {
          VERBOSE(<< "\nAngle-combination out-of-plane=" << anglesList[i][0]
                  << ", gantry=" << anglesList[i][1]
                  << ", in-plane=" << anglesList[i][2] << " failed:\n")
          VERBOSE(<< "  " << referenceMarkerProjections[i] << " vs. "
                  << rtkMarkerProjections[i] << "\n")
          lok = false;
          fails++;
        }
      }
      VERBOSE(<< " [ok: " << (rtkMarkerProjections.size() - fails) << ", fails: " << fails << "] ")
    }
    else
    {
      lok = false;
    }
  }
  ok = ok && lok;
  VERBOSE(<< (lok ? "OK" : "FAILURE") << "\n")


  VERBOSE(<< "Test result: ")
  if (ok)
  {
    VERBOSE(<< "OK\n\n")
    return EXIT_SUCCESS;
  }
  else
  {
    VERBOSE(<< "FAILURE\n\n")
    return EXIT_FAILURE;
  }
}
