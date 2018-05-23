//RTK
#include "rtkThreeDCircularProjectionGeometry.h"
#include "rtkTest.h"

int main(int , char **)
{
  rtk::ThreeDCircularProjectionGeometry::Pointer geometry1, geometry2;
  geometry1 = rtk::ThreeDCircularProjectionGeometry::New();

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
    geometry1->SetCollimationOfLastProjection(px*0.3, px*0.1, py*0.4, py*0.2);
    }
  geometry1->SetRadiusCylindricalDetector(300.);
  geometry2 = geometry1->Clone();
  CheckGeometries(geometry1, geometry2);
  std::cout << "Test PASSED! " << std::endl;
  return EXIT_SUCCESS;
}
