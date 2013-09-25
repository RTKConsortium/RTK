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
 m_PhantomScale(128.),
 m_PhantomOriginOffsetX(0.)
{
}

template< class TInputImage, class TOutputImage >
void SheppLoganPhantomFilter< TInputImage, TOutputImage >::GenerateData()
{
  std::vector< typename REIType::Pointer > rei(10);
  unsigned int NumberOfFig = 10;

  for ( unsigned int j = 0; j < NumberOfFig; j++ )
    {
      rei[j] = REIType::New();
    }
  typename REIType::VectorType semiprincipalaxis, center;
  semiprincipalaxis[0] = 0.69*m_PhantomScale;
  semiprincipalaxis[1] = 0.90*m_PhantomScale;
  semiprincipalaxis[2] = 0.92*m_PhantomScale;
//  semiprincipalaxis[3] = -1.;
  center[0] = m_PhantomOriginOffsetX;
  center[1] = 0.;
  center[2] = 0.;
  rei[0]->SetAxis(semiprincipalaxis);
  rei[0]->SetCenter(center);
  rei[0]->SetAngle(0.);
  rei[0]->SetDensity(2.);
  rei[0]->SetFigure("Ellipsoid");

  semiprincipalaxis[0] = 0.6624*m_PhantomScale;
  semiprincipalaxis[1] = 0.880*m_PhantomScale;
  semiprincipalaxis[2] = 0.874*m_PhantomScale;
  center[0] = m_PhantomOriginOffsetX;
  center[1] = 0.;
  center[2] = 0.;
  rei[1]->SetAxis(semiprincipalaxis);
  rei[1]->SetCenter(center);
  rei[1]->SetAngle(0.);
  rei[1]->SetDensity(-0.98);
  rei[1]->SetFigure("Ellipsoid");

  semiprincipalaxis[0] = 0.41*m_PhantomScale;
  semiprincipalaxis[1] = 0.21*m_PhantomScale;
  semiprincipalaxis[2] = 0.16*m_PhantomScale;
  center[0] = -0.22*m_PhantomScale + m_PhantomOriginOffsetX;
  center[1] = -0.25*m_PhantomScale;
  center[2] = 0.;
  rei[2]->SetAxis(semiprincipalaxis);
  rei[2]->SetCenter(center);
  rei[2]->SetAngle(108.);
  rei[2]->SetDensity(-0.02);
  rei[2]->SetFigure("Ellipsoid");

  semiprincipalaxis[0] = 0.31*m_PhantomScale;
  semiprincipalaxis[1] = 0.22*m_PhantomScale;
  semiprincipalaxis[2] = 0.11*m_PhantomScale;
  center[0] = 0.22*m_PhantomScale + m_PhantomOriginOffsetX;
  center[1] = -0.25*m_PhantomScale;
  center[2] = 0.;
  rei[3]->SetAxis(semiprincipalaxis);
  rei[3]->SetCenter(center);
  rei[3]->SetAngle(72.);
  rei[3]->SetDensity(-0.02);
  rei[3]->SetFigure("Ellipsoid");

  semiprincipalaxis[0] = 0.21*m_PhantomScale;
  semiprincipalaxis[1] = 0.50*m_PhantomScale;
  semiprincipalaxis[2] = 0.25*m_PhantomScale;
  center[0] = m_PhantomOriginOffsetX;
  center[1] = -0.25*m_PhantomScale;
  center[2] = 0.35*m_PhantomScale;
  rei[4]->SetAxis(semiprincipalaxis);
  rei[4]->SetCenter(center);
  rei[4]->SetAngle(0.);
  rei[4]->SetDensity(0.02);
  rei[4]->SetFigure("Ellipsoid");

  semiprincipalaxis[0] = 0.046*m_PhantomScale;
  semiprincipalaxis[1] = 0.046*m_PhantomScale;
  semiprincipalaxis[2] = 0.046*m_PhantomScale;
  center[0] = m_PhantomOriginOffsetX;
  center[1] = -0.25*m_PhantomScale;
  center[2] = 0.10*m_PhantomScale;
  rei[5]->SetAxis(semiprincipalaxis);
  rei[5]->SetCenter(center);
  rei[5]->SetAngle(0.);
  rei[5]->SetDensity(0.02);
  rei[5]->SetFigure("Ellipsoid");

  semiprincipalaxis[0] = 0.046*m_PhantomScale;
  semiprincipalaxis[1] = 0.020*m_PhantomScale;
  semiprincipalaxis[2] = 0.023*m_PhantomScale;
  center[0] = -0.08*m_PhantomScale + m_PhantomOriginOffsetX;
  center[1] = -0.250*m_PhantomScale;
  center[2] = -0.650*m_PhantomScale;
  rei[6]->SetAxis(semiprincipalaxis);
  rei[6]->SetCenter(center);
  rei[6]->SetAngle(0.);
  rei[6]->SetDensity(0.01);
  rei[6]->SetFigure("Ellipsoid");

  semiprincipalaxis[0] = 0.046*m_PhantomScale;
  semiprincipalaxis[1] = 0.020*m_PhantomScale;
  semiprincipalaxis[2] = 0.023*m_PhantomScale;
  center[0] = 0.06*m_PhantomScale + m_PhantomOriginOffsetX;
  center[1] = -0.25*m_PhantomScale;
  center[2] = -0.65*m_PhantomScale;
  rei[7]->SetAxis(semiprincipalaxis);
  rei[7]->SetCenter(center);
  rei[7]->SetAngle(90.);
  rei[7]->SetDensity(0.01);
  rei[7]->SetFigure("Ellipsoid");

  semiprincipalaxis[0] = 0.056*m_PhantomScale;
  semiprincipalaxis[1] = 0.010*m_PhantomScale;
  semiprincipalaxis[2] = 0.040*m_PhantomScale;
  center[0] = 0.060*m_PhantomScale + m_PhantomOriginOffsetX;
  center[1] = 0.625*m_PhantomScale;
  center[2] = -0.105*m_PhantomScale;
  rei[8]->SetAxis(semiprincipalaxis);
  rei[8]->SetCenter(center);
  rei[8]->SetAngle(90.);
  rei[8]->SetDensity(0.02);
  rei[8]->SetFigure("Ellipsoid");

  semiprincipalaxis[0] = 0.056*m_PhantomScale;
  semiprincipalaxis[1] = 0.100*m_PhantomScale;
  semiprincipalaxis[2] = 0.056*m_PhantomScale;
  center[0] = m_PhantomOriginOffsetX;
  center[1] = 0.625*m_PhantomScale;
  center[2] = 0.100*m_PhantomScale;
  rei[9]->SetAxis(semiprincipalaxis);
  rei[9]->SetCenter(center);
  rei[9]->SetAngle(0.);
  rei[9]->SetDensity(-0.02);
  rei[9]->SetFigure("Ellipsoid");


  for ( unsigned int i = 0; i < NumberOfFig; i++ )
    {
    rei[i]->SetNumberOfThreads(this->GetNumberOfThreads());
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
