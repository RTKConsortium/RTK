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

#ifndef __rtkDrawQuadricImageFilter_cxx
#define __rtkDrawQuadricImageFilter_cxx

#include "rtkDrawQuadricImageFilter.h"

namespace rtk
{

rtk::DrawQuadricSpatialObject::DrawQuadricSpatialObject()
{

  this->m_SqpFunctor = EQPFunctionType::New();
  m_Angle = 0;
  m_Axis.Fill(90.);
  m_Center.Fill(0.);
  this->UpdateParameters();
}

void rtk::DrawQuadricSpatialObject::UpdateParameters()
{
  m_SqpFunctor->Translate(m_Axis);
  m_SqpFunctor->Rotate(m_Angle, m_Center);
}

bool rtk::DrawQuadricSpatialObject::IsInside(const rtk::DrawQuadricSpatialObject::PointType& point) const
{

  double QuadricEllip = m_SqpFunctor->GetA()*point[0]*point[0]   +
                        m_SqpFunctor->GetB()*point[1]*point[1]   +
                        m_SqpFunctor->GetC()*point[2]*point[2]   +
                        m_SqpFunctor->GetD()*point[0]*point[1]   +
                        m_SqpFunctor->GetE()*point[0]*point[2]   +
                        m_SqpFunctor->GetF()*point[1]*point[2]   +
                        m_SqpFunctor->GetG()*point[0] + m_SqpFunctor->GetH()*point[1] +
                        m_SqpFunctor->GetI()*point[2] + m_SqpFunctor->GetJ();
 if(QuadricEllip<0)
    return true;
 return false;
}

}// end namespace rtk

#endif
