//RTK
#include "rtkThreeDCircularProjectionGeometry.h"


int main(int , char **)
{
  rtk::ThreeDCircularProjectionGeometry::Pointer geometry1, geometry2;
  geometry1 = rtk::ThreeDCircularProjectionGeometry::New();
  geometry2 = rtk::ThreeDCircularProjectionGeometry::New();

  for(double oa=-60.; oa<65.; oa+=60.)
  for(double ia=-60.; ia<65.; ia+=60.)
  for(double ga=0.; ga<360.; ga+=45.)
  for(double sx=-50.; sx<55.; sx+=25.)
  for(double sy=-50.; sy<55.; sy+=25.)
  for(double px=-50.; px<55.; px+=25.)
  for(double py=-50.; py<55.; py+=25.)
  for(double sdd=200.; sdd<405.; sdd+=200.)
  for(double sid=100.; sid<200.; sid+=200.)
    {
    geometry1->AddProjection(sid, sdd, ga, px, py, oa, ia, sx, sy);
    geometry2->AddProjection(geometry1->GetMatrices().back());

    double d = geometry1->GetGantryAngles().back()-geometry2->GetGantryAngles().back();
    d = std::abs( geometry1->ConvertAngleBetweenMinusAndPlusPIRadians( d ) );
    if( d > 0.01 )
      {
      std::cerr << "ERROR: GantryAngles 1 is "
                << geometry1->GetGantryAngles().back()
                << " and GantryAngles 2 is "
                << geometry2->GetGantryAngles().back();
      return EXIT_FAILURE;
      }

    d = geometry1->GetInPlaneAngles().back()-geometry2->GetInPlaneAngles().back();
    d = std::abs( geometry1->ConvertAngleBetweenMinusAndPlusPIRadians( d ) );
    if( d > 0.01 )
      {
      std::cerr << "ERROR: InPlaneAngles 1 is "
                << geometry1->GetInPlaneAngles().back()
                << " and InPlaneAngles 2 is "
                << geometry2->GetInPlaneAngles().back();
      return EXIT_FAILURE;
      }

    d = geometry1->GetOutOfPlaneAngles().back()-geometry2->GetOutOfPlaneAngles().back();
    d = std::abs( geometry1->ConvertAngleBetweenMinusAndPlusPIRadians( d ) );
    if( d > 0.01 )
      {
      std::cerr << "ERROR: OutOfPlaneAngle 1 is "
                << geometry1->GetOutOfPlaneAngles().back()
                << " and OutOfPlaneAngle 2 is "
                << geometry2->GetOutOfPlaneAngles().back();
      return EXIT_FAILURE;
      }

    if( std::abs(geometry1->GetProjectionOffsetsY().back()-geometry2->GetProjectionOffsetsY().back()) > 0.01 )
      {
      std::cerr << "ERROR: ProjectionOffsetsY 1 is "
                << geometry1->GetProjectionOffsetsY().back()
                << " and ProjectionOffsetsY 2 is "
                << geometry2->GetProjectionOffsetsY().back();
      return EXIT_FAILURE;
      }

    if( std::abs(geometry1->GetProjectionOffsetsX().back()-geometry2->GetProjectionOffsetsX().back()) > 0.01 )
      {
      std::cerr << "ERROR: ProjectionOffsetsX 1 is "
                << geometry1->GetProjectionOffsetsX().back()
                << " and ProjectionOffsetsX 2 is "
                << geometry2->GetProjectionOffsetsX().back();
      return EXIT_FAILURE;
      }

    if( std::abs(geometry1->GetSourceToIsocenterDistances().back()-geometry2->GetSourceToIsocenterDistances().back()) > 0.01 )
      {
      std::cerr << "ERROR: GetSourceToIsocenterDistances 1 is "
                << geometry1->GetSourceToIsocenterDistances().back()
                << " and GetSourceToIsocenterDistances 2 is "
                << geometry2->GetSourceToIsocenterDistances().back();
      return EXIT_FAILURE;
      }

    if( std::abs(geometry1->GetSourceToDetectorDistances().back()-geometry2->GetSourceToDetectorDistances().back()) > 0.01 )
      {
      std::cerr << "ERROR: GetSourceToDetectorDistances 1 is "
                << geometry1->GetSourceToDetectorDistances().back()
                << " and GetSourceToDetectorDistances 2 is "
                << geometry2->GetSourceToDetectorDistances().back();
      return EXIT_FAILURE;
      }

    if( std::abs(geometry1->GetSourceOffsetsY().back()-geometry2->GetSourceOffsetsY().back()) > 0.01 )
      {
      std::cerr << "ERROR: SourceOffsetsY 1 is "
                << geometry1->GetSourceOffsetsY().back()
                << " and SourceOffsetsY 2 is "
                << geometry2->GetSourceOffsetsY().back();
      return EXIT_FAILURE;
      }

    if( std::abs(geometry1->GetSourceOffsetsX().back()-geometry2->GetSourceOffsetsX().back()) > 0.01 )
      {
      std::cerr << "ERROR: SourceOffsetsX 1 is "
                << geometry1->GetSourceOffsetsX().back()
                << " and SourceOffsetsX 2 is "
                << geometry2->GetSourceOffsetsX().back();
      return EXIT_FAILURE;
      }

    }
  std::cout << "Test PASSED! " << std::endl;
  return EXIT_SUCCESS;
}
