// RTK
#include "rtkThreeDCircularProjectionGeometry.h"


int
main(int, char **)
{
  rtk::ThreeDCircularProjectionGeometry::Pointer geometry1, geometry2;
  geometry1 = rtk::ThreeDCircularProjectionGeometry::New();
  geometry2 = rtk::ThreeDCircularProjectionGeometry::New();

  const double sid = 1000.;
  const double sdd = 0.; // Parallel geometry
  for (double oa = -60.; oa < 65.; oa += 60.)
  {
    for (double ia = -360.; ia < 360.; ia += 60.)
    {
      for (double ga = -360.; ga < 360.; ga += 45.)
      {
        for (double px = -50.; px < 55.; px += 25.)
        {
          for (double py = -50.; py < 55.; py += 25.)
          {
            geometry1->AddProjection(sid, sdd, ga, px, py, oa, ia);
            geometry2->AddProjection(geometry1->GetMatrices().back());

            double g1 = geometry1->GetGantryAngles().back();
            double g2 = geometry2->GetGantryAngles().back();
            double d = g1 - g2;
            d = itk::Math::abs(geometry1->ConvertAngleBetweenMinusAndPlusPIRadians(d));
            if (d > 0.01)
            {
              std::cerr << "ERROR: GantryAngles 1 is " << g1 << " and GantryAngles 2 is " << g2;
              return EXIT_FAILURE;
            }

            double ia1 = geometry1->GetInPlaneAngles().back();
            double ia2 = geometry2->GetInPlaneAngles().back();
            d = ia1 - ia2;
            d = itk::Math::abs(geometry1->ConvertAngleBetweenMinusAndPlusPIRadians(d));
            if (d > 0.01)
            {
              std::cerr << "ERROR: InPlaneAngles 1 is " << ia1 << " and InPlaneAngles 2 is " << ia2;
              return EXIT_FAILURE;
            }

            double oa1 = geometry1->GetOutOfPlaneAngles().back();
            double oa2 = geometry2->GetOutOfPlaneAngles().back();
            d = oa1 - oa2;
            d = itk::Math::abs(geometry1->ConvertAngleBetweenMinusAndPlusPIRadians(d));
            if (d > 0.01)
            {
              std::cerr << "ERROR: OutOfPlaneAngle 1 is " << oa1 << " and OutOfPlaneAngle 2 is " << oa2;
              return EXIT_FAILURE;
            }

            double py1 = geometry1->GetProjectionOffsetsY().back();
            double py2 = geometry2->GetProjectionOffsetsY().back();
            d = itk::Math::abs(py1 - py2);
            if (d > 0.01)
            {
              std::cerr << "ERROR: ProjectionOffsetsY 1 is " << py1 << " and ProjectionOffsetsY 2 is " << py2;
              return EXIT_FAILURE;
            }

            double px1 = geometry1->GetProjectionOffsetsX().back();
            double px2 = geometry2->GetProjectionOffsetsX().back();
            d = itk::Math::abs(px1 - px2);
            if (d > 0.01)
            {
              std::cerr << "ERROR: ProjectionOffsetsX 1 is " << px1 << " and ProjectionOffsetsX 2 is " << px2;
              return EXIT_FAILURE;
            }

            double sid1 = geometry1->GetSourceToIsocenterDistances().back();
            double sid2 = geometry2->GetSourceToIsocenterDistances().back();
            d = itk::Math::abs(sid1 - sid2);
            if (d > 0.01)
            {
              std::cerr << "ERROR: GetSourceToIsocenterDistances 1 is " << sid1
                        << " and GetSourceToIsocenterDistances 2 is " << sid2;
              return EXIT_FAILURE;
            }

            double sdd1 = geometry1->GetSourceToDetectorDistances().back();
            double sdd2 = geometry2->GetSourceToDetectorDistances().back();
            d = itk::Math::abs(sdd1 - sdd2);
            if (d > 0.01)
            {
              std::cerr << "ERROR: GetSourceToDetectorDistances 1 is " << sdd1
                        << " and GetSourceToDetectorDistances 2 is " << sdd2;
              return EXIT_FAILURE;
            }
          }
        }
      }
    }
  }
  std::cout << "Test PASSED! " << std::endl;
  return EXIT_SUCCESS;
}
