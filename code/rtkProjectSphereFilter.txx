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

#ifndef __rtkProjectSphereFilter_txx
#define __rtkProjectSphereFilter_txx

#include <iostream>
#include <fstream>
#include <sstream>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

#include <itkImageRegionConstIterator.h>
#include <itkImageRegionIteratorWithIndex.h>
#include "rtkHomogeneousMatrix.h"

namespace rtk
{
template <class TInputImage, class TOutputImage>
ProjectSphereFilter<TInputImage, TOutputImage>
::ProjectSphereFilter():
 m_SphereScale(128.),
 m_PhantomOriginOffsetX(0.)
{
}

template< class TInputImage, class TOutputImage >
void ProjectSphereFilter< TInputImage, TOutputImage >::GenerateData()
{
  REIType::Pointer rei;

  rei = REIType::New();
  rei->SetSemiPrincipalAxisX(0.69*m_SphereScale);
  rei->SetSemiPrincipalAxisZ(0.92*m_SphereScale);
  rei->SetSemiPrincipalAxisY(0.90*m_SphereScale);
  rei->SetCenterX(0.);
  rei->SetCenterZ(0.);
  rei->SetCenterY(0.);
  rei->SetRotationAngle(0.);
  rei->SetMultiplicativeConstant(1.);

  rei->SetNumberOfThreads(this->GetNumberOfThreads());

  rei->SetInput( this->GetInput() );
  rei->SetGeometry( this->GetGeometry() );

  //Update
  rei->Update();
  this->GraftOutput( rei->GetOutput() );
}
} // end namespace rtk

#endif
