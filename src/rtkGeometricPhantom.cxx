/*=========================================================================
 *
 *  Copyright RTK Consortium
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *         http://www.apache.org/licenses/LICENSE-2.0.txt
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 *=========================================================================*/

#include "rtkGeometricPhantom.h"

namespace rtk
{
void
GeometricPhantom ::Rescale(const VectorType & r)
{
  for (auto & convexShape : m_ConvexShapes)
  {
    convexShape->Rescale(r);
  }
}

void
GeometricPhantom ::Translate(const VectorType & t)
{
  for (auto & convexShape : m_ConvexShapes)
  {
    convexShape->Translate(t);
  }
}

void
GeometricPhantom ::Rotate(const RotationMatrixType & r)
{
  for (auto & convexShape : m_ConvexShapes)
  {
    convexShape->Rotate(r);
  }
}

void
GeometricPhantom ::AddConvexShape(const ConvexShape *co)
{
  m_ConvexShapes.push_back( co->Clone() );
  for (size_t i = 0; i < m_PlaneDirections.size(); i++)
    m_ConvexShapes.back()->AddClipPlane(m_PlaneDirections[i], m_PlanePositions[i]);
}

void
GeometricPhantom ::AddClipPlane(const VectorType & dir, const ScalarType & pos)
{
  for (size_t i = 0; i < m_PlanePositions.size(); i++)
  {
    if (dir == m_PlaneDirections[i] && pos == m_PlanePositions[i])
      return;
  }
  m_PlaneDirections.push_back(dir);
  m_PlanePositions.push_back(pos);
  for (auto & convexShape : m_ConvexShapes)
    convexShape->AddClipPlane(dir, pos);
}

} // namespace rtk
