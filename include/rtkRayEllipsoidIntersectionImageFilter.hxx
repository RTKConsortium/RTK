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

#ifndef rtkRayEllipsoidIntersectionImageFilter_hxx
#define rtkRayEllipsoidIntersectionImageFilter_hxx

#include <itkImageRegionConstIterator.h>
#include <itkImageRegionIteratorWithIndex.h>

#include "rtkHomogeneousMatrix.h"

namespace rtk
{

template <class TInputImage, class TOutputImage>
RayEllipsoidIntersectionImageFilter<TInputImage, TOutputImage>
::RayEllipsoidIntersectionImageFilter():m_EQPFunctor( EQPFunctionType::New() )
{
  m_Axis.Fill(0.);
  m_Center.Fill(0.);
  m_Angle = 0.;
  m_Figure = "Ellipsoid";
}

template <class TInputImage, class TOutputImage>
void RayEllipsoidIntersectionImageFilter<TInputImage, TOutputImage>::BeforeThreadedGenerateData()
{
  Superclass::BeforeThreadedGenerateData();

  typename EQPFunctionType::VectorType semiprincipalaxis;
  typename EQPFunctionType::VectorType center;
  semiprincipalaxis[0] = m_Axis[0];
  semiprincipalaxis[1] = m_Axis[1];
  semiprincipalaxis[2] = m_Axis[2];
  center[0] = m_Center[0];
  center[1] = m_Center[1];
  center[2] = m_Center[2];

  //Set type of Figure
  m_EQPFunctor->SetFigure(m_Figure);
  //Translate from regular expression to quadric
  m_EQPFunctor->Translate(semiprincipalaxis);
  //Applies rotation and translation if necessary
  m_EQPFunctor->Rotate(m_Angle, center);
  //Setting parameters in order to compute the projections
  this->GetRQIFunctor()->SetA( m_EQPFunctor->GetA() );
  this->GetRQIFunctor()->SetB( m_EQPFunctor->GetB() );
  this->GetRQIFunctor()->SetC( m_EQPFunctor->GetC() );
  this->GetRQIFunctor()->SetD( m_EQPFunctor->GetD() );
  this->GetRQIFunctor()->SetE( m_EQPFunctor->GetE() );
  this->GetRQIFunctor()->SetF( m_EQPFunctor->GetF() );
  this->GetRQIFunctor()->SetG( m_EQPFunctor->GetG() );
  this->GetRQIFunctor()->SetH( m_EQPFunctor->GetH() );
  this->GetRQIFunctor()->SetI( m_EQPFunctor->GetI() );
  this->GetRQIFunctor()->SetJ( m_EQPFunctor->GetJ() );
}
}// end namespace rtk

#endif
