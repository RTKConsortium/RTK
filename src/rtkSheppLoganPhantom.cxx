/*=========================================================================
 *
 *  Copyright RTK Consortium
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *         https://www.apache.org/licenses/LICENSE-2.0.txt
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 *=========================================================================*/

#include "rtkSheppLoganPhantom.h"

namespace rtk
{
SheppLoganPhantom ::SheppLoganPhantom()
{
  SetEllipsoid(0.69, 0.90, 0.92, 0., 0., 0., 0., 2.);
  SetEllipsoid(0.6624, 0.880, 0.874, 0., 0., 0., 0., -0.98);
  SetEllipsoid(0.41, 0.21, 0.16, -0.22, -0.25, 0., 108., -0.02);
  SetEllipsoid(0.31, 0.22, 0.11, 0.22, -0.25, 0., 72., -0.02);
  SetEllipsoid(0.21, 0.50, 0.25, 0., -0.25, 0.35, 0., 0.02);
  SetEllipsoid(0.046, 0.046, 0.046, 0., -0.25, 0.10, 0., 0.02);
  SetEllipsoid(0.046, 0.02, 0.023, -0.08, -0.25, -0.65, 0., 0.01);
  SetEllipsoid(0.046, 0.02, 0.023, 0.06, -0.25, -0.65, 90., 0.01);
  SetEllipsoid(0.056, 0.1, 0.04, 0.06, 0.625, -0.105, 90., 0.02);
  SetEllipsoid(0.056, 0.1, 0.056, 0., 0.625, 0.1, 0., -0.02);
}

void
SheppLoganPhantom ::SetEllipsoid(ScalarType spax,
                                 ScalarType spay,
                                 ScalarType spaz,
                                 ScalarType centerx,
                                 ScalarType centery,
                                 ScalarType centerz,
                                 ScalarType angle,
                                 ScalarType density)
{
  auto semiprincipalaxis = itk::MakeVector(spax, spay, spaz);
  auto center = itk::MakeVector(centerx, centery, centerz);
  auto q = QuadricShape::New();
  q->SetEllipsoid(center, semiprincipalaxis, angle);
  q->SetDensity(density);
  this->AddConvexShape(q.GetPointer());
}

} // namespace rtk
