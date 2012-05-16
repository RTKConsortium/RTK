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

#ifndef __rtkSheppLoganPhantomFilter_txx
#define __rtkSheppLoganPhantomFilter_txx

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
SheppLoganPhantomFilter<TInputImage, TOutputImage>
::SheppLoganPhantomFilter():
 m_SQPFunctor( SQPFunctionType::New() ),
 m_PhantomScale(128.),
 m_PhantomOriginOffsetX(0.)
{
}
template< class TInputImage, class TOutputImage >
void SheppLoganPhantomFilter< TInputImage, TOutputImage >::GenerateData()
{
  m_SQPFunctor = SQPFunctionType::New();
  std::vector< REIType::Pointer > rei(10);
  unsigned int NumberOfFig = 10;

  for ( unsigned int j = 0; j < NumberOfFig; j++ )
    {
      rei[j] = REIType::New();
    }
  rei[0]->SetSemiPrincipalAxisX(0.69*m_PhantomScale);
  rei[0]->SetSemiPrincipalAxisY(0.92*m_PhantomScale);
  rei[0]->SetSemiPrincipalAxisZ(0.90*m_PhantomScale);
  rei[0]->SetCenterX(m_PhantomOriginOffsetX);
  rei[0]->SetCenterY(0.);
  rei[0]->SetCenterZ(0.);
  rei[0]->SetRotationAngle(0.);
  rei[0]->SetMultiplicativeConstant(2.);

  rei[1]->SetSemiPrincipalAxisX(0.6624*m_PhantomScale);
  rei[1]->SetSemiPrincipalAxisY(0.874*m_PhantomScale);
  rei[1]->SetSemiPrincipalAxisZ(0.880*m_PhantomScale);
  rei[1]->SetCenterX(m_PhantomOriginOffsetX);
  rei[1]->SetCenterY(0.);
  rei[1]->SetCenterZ(0.);
  rei[1]->SetRotationAngle(0.);
  rei[1]->SetMultiplicativeConstant(-0.98);

  rei[2]->SetSemiPrincipalAxisX(0.41*m_PhantomScale);
  rei[2]->SetSemiPrincipalAxisY(0.16*m_PhantomScale);
  rei[2]->SetSemiPrincipalAxisZ(0.21*m_PhantomScale);
  rei[2]->SetCenterX(-0.22*m_PhantomScale + m_PhantomOriginOffsetX);
  rei[2]->SetCenterY(0.);
  rei[2]->SetCenterZ(-0.25*m_PhantomScale);
  rei[2]->SetRotationAngle(108.);
  rei[2]->SetMultiplicativeConstant(-0.02);

  rei[3]->SetSemiPrincipalAxisX(0.31*m_PhantomScale);
  rei[3]->SetSemiPrincipalAxisY(0.11*m_PhantomScale);
  rei[3]->SetSemiPrincipalAxisZ(0.22*m_PhantomScale);
  rei[3]->SetCenterX(0.22*m_PhantomScale + m_PhantomOriginOffsetX);
  rei[3]->SetCenterY(0.);
  rei[3]->SetCenterZ(-0.25*m_PhantomScale);
  rei[3]->SetRotationAngle(72.);
  rei[3]->SetMultiplicativeConstant(-0.02);

  rei[4]->SetSemiPrincipalAxisX(0.21*m_PhantomScale);
  rei[4]->SetSemiPrincipalAxisY(0.25*m_PhantomScale);
  rei[4]->SetSemiPrincipalAxisZ(0.50*m_PhantomScale);
  rei[4]->SetCenterX(m_PhantomOriginOffsetX);
  rei[4]->SetCenterY(-0.35*m_PhantomScale);
  rei[4]->SetCenterZ(-0.25*m_PhantomScale);
  rei[4]->SetRotationAngle(0.);
  rei[4]->SetMultiplicativeConstant(0.02);

  rei[5]->SetSemiPrincipalAxisX(0.046*m_PhantomScale);
  rei[5]->SetSemiPrincipalAxisY(0.046*m_PhantomScale);
  rei[5]->SetSemiPrincipalAxisZ(0.046*m_PhantomScale);
  rei[5]->SetCenterX(m_PhantomOriginOffsetX);
  rei[5]->SetCenterY(-0.10*m_PhantomScale);
  rei[5]->SetCenterZ(-0.25*m_PhantomScale);
  rei[5]->SetRotationAngle(0.);
  rei[5]->SetMultiplicativeConstant(0.02);

  rei[6]->SetSemiPrincipalAxisX(0.046*m_PhantomScale);
  rei[6]->SetSemiPrincipalAxisY(0.023*m_PhantomScale);
  rei[6]->SetSemiPrincipalAxisZ(0.020*m_PhantomScale);
  rei[6]->SetCenterX(-0.08*m_PhantomScale + m_PhantomOriginOffsetX);
  rei[6]->SetCenterY(0.650*m_PhantomScale);
  rei[6]->SetCenterZ(-0.250*m_PhantomScale);
  rei[6]->SetRotationAngle(0.);
  rei[6]->SetMultiplicativeConstant(0.01);

  rei[7]->SetSemiPrincipalAxisX(0.046*m_PhantomScale);
  rei[7]->SetSemiPrincipalAxisY(0.023*m_PhantomScale);
  rei[7]->SetSemiPrincipalAxisZ(0.020*m_PhantomScale);
  rei[7]->SetCenterX(0.06*m_PhantomScale + m_PhantomOriginOffsetX);
  rei[7]->SetCenterY(0.65*m_PhantomScale);
  rei[7]->SetCenterZ(-0.25*m_PhantomScale);
  rei[7]->SetRotationAngle(90.);
  rei[7]->SetMultiplicativeConstant(0.01);

  rei[8]->SetSemiPrincipalAxisX(0.056*m_PhantomScale);
  rei[8]->SetSemiPrincipalAxisY(0.040*m_PhantomScale);
  rei[8]->SetSemiPrincipalAxisZ(0.010*m_PhantomScale);
  rei[8]->SetCenterX(0.060*m_PhantomScale + m_PhantomOriginOffsetX);
  rei[8]->SetCenterY(0.105*m_PhantomScale);
  rei[8]->SetCenterZ(0.625*m_PhantomScale);
  rei[8]->SetRotationAngle(90.);
  rei[8]->SetMultiplicativeConstant(0.02);

  rei[9]->SetSemiPrincipalAxisX(0.056*m_PhantomScale);
  rei[9]->SetSemiPrincipalAxisY(0.056*m_PhantomScale);
  rei[9]->SetSemiPrincipalAxisZ(0.100*m_PhantomScale);
  rei[9]->SetCenterX(m_PhantomOriginOffsetX);
  rei[9]->SetCenterY(-0.100*m_PhantomScale);
  rei[9]->SetCenterZ(0.625*m_PhantomScale);
  rei[9]->SetRotationAngle(0.);
  rei[9]->SetMultiplicativeConstant(-0.02);


  for ( unsigned int i = 0; i < NumberOfFig; i++ )
    {
    if ( i == ( NumberOfFig - 1 ) ) //last case
      {
      if(i==0) //just one ellipsoid
        rei[i]->SetInput( rei[i]->GetOutput() );
      else
        rei[i]->SetInput( rei[i-1]->GetOutput() );
      rei[i]->SetGeometry( this->GetGeometry() );
      }

    if (i>0) //other cases
      {
      rei[i]->SetInput( rei[i-1]->GetOutput() );
      rei[i]->SetGeometry( this->GetGeometry() );
      }

    else //first case
      {
      rei[i]->SetInput( this->GetInput() );
      rei[i]->SetGeometry( this->GetGeometry() );
      }
    }
  //Update
  rei[NumberOfFig - 1]->Update();
  this->GraftOutput( rei[NumberOfFig - 1]->GetOutput() );
}
} // end namespace rtk

#endif
