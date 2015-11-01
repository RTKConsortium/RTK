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

#ifndef __rtkDrawCylinderImageFilter_txx
#define __rtkDrawCylinderImageFilter_txx

#include <iostream>
#include <itkImageRegionConstIterator.h>
#include <itkImageRegionIterator.h>

#include "rtkDrawCylinderImageFilter.h"

namespace rtk
{

template <class TInputImage, class TOutputImage, typename TFunction>
DrawCylinderImageFilter<TInputImage, TOutputImage, TFunction>
::DrawCylinderImageFilter()
{  
  
  this->m_spatialObject.m_Axis.Fill(90.);
  this->m_spatialObject.m_Center.Fill(0.);  
  this->m_spatialObject.m_Angle = 0.;
  this->m_spatialObject.sqpFunctor = EQPFunctionType::New();
  this->m_spatialObject.sqpFunctor->SetFigure("Cylinder");
}

}// end namespace rtk

#endif
