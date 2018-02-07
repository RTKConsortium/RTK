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
GeometricPhantom
::Rescale(const VectorType &r)
{
  for(size_t i=0; i<m_ConvexShapes.size(); i++)
    {
    m_ConvexShapes[i]->Rescale(r);
    }
}

void
GeometricPhantom
::Translate(const VectorType &t)
{
  for(size_t i=0; i<m_ConvexShapes.size(); i++)
    {
    m_ConvexShapes[i]->Translate(t);
    }
}

void
GeometricPhantom
::Rotate(const RotationMatrixType &r)
{
  for(size_t i=0; i<m_ConvexShapes.size(); i++)
    {
    m_ConvexShapes[i]->Rotate(r);
    }
}

void
GeometricPhantom
::AddConvexShape(const ConvexShapePointer &co)
{
  m_ConvexShapes.push_back(co);
}

}
