#ifndef __itkSheppLoganPhantomFilter_txx
#define __itkSheppLoganPhantomFilter_txx

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

namespace itk
{
template< class TInputImage, class TOutputImage >
void SheppLoganPhantomFilter< TInputImage, TOutputImage >::GenerateData()
{
  m_SQPFunctor = SQPFunctionType::New();
  m_SQPFunctor->Config(m_ConfigFile);
  m_Fig = m_SQPFunctor->GetFig();

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
} // end namespace itk

#endif
