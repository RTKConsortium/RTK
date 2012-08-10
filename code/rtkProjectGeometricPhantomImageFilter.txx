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

#ifndef __rtkProjectGeometricPhantomImageFilter_txx
#define __rtkProjectGeometricPhantomImageFilter_txx

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
template< class TInputImage, class TOutputImage >
void ProjectGeometricPhantomImageFilter< TInputImage, TOutputImage >::GenerateData()
{
  CFRType::Pointer cfr = CFRType::New();
  cfr->Config(m_ConfigFile);
  m_Fig = cfr->GetFig();

  std::vector< REIType::Pointer > rei( m_Fig.size() );
  for ( unsigned int i = 0; i < m_Fig.size(); i++ )
    {
    rei[i] = REIType::New();
    //Set GrayScale value, axes, center...
    rei[i]->SetMultiplicativeConstant(m_Fig[i][7]);
    rei[i]->SetSemiPrincipalAxisX(m_Fig[i][0]);
    rei[i]->SetSemiPrincipalAxisY(m_Fig[i][1]);
    rei[i]->SetSemiPrincipalAxisZ(m_Fig[i][2]);

    rei[i]->SetCenterX(m_Fig[i][3]);
    rei[i]->SetCenterY(m_Fig[i][4]);
    rei[i]->SetCenterZ(m_Fig[i][5]);

    rei[i]->SetRotationAngle(m_Fig[i][6]);

    if ( i == ( m_Fig.size() - 1 ) ) //last case
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
  rei[ m_Fig.size() - 1]->Update();
  this->GraftOutput( rei[m_Fig.size()-1]->GetOutput() );
}
} // end namespace rtk

#endif
