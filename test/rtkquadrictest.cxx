#include "rtkQuadricShape.h"

/**
 * \file rtkquadrictest.cxx
 *
 * \brief Unit test for quadric shape
 *
 * This file contains specfic test for the quadric shape.
 *
 * \author Simon Rit
 */

int
main(int, char **)
{
  // Test https://github.com/RTKConsortium/RTK/issues/819#issuecomment-3337586635
  // The following is [Cylinder_y: x=0 y=0 z=0 r=0.1  l=100 rho=1]
  auto cylinder = rtk::QuadricShape::New();
  cylinder->SetDensity(1.);
  cylinder->SetA(100. - 1.42109e-14);
  cylinder->SetC(100. - 1.42109e-14);
  cylinder->SetJ(-1.);
  rtk::QuadricShape::ScalarType nearDist, farDist;
  auto                          rayOrigin = itk::MakePoint(52.43581919641932, -0.5, 998.6243011589495);
  auto                          rayDirection = itk::MakeVector(-0.05233595624294384, 0., -0.9986295347545739);
  TRY_AND_EXIT_ON_ITK_EXCEPTION(cylinder->IsIntersectedByRay(rayOrigin, rayDirection, nearDist, farDist));
  return EXIT_SUCCESS;
}
