#ifndef __itkRayEllipsoidIntersectionImageFilter_txx
#define __itkRayEllipsoidIntersectionImageFilter_txx

#include <itkImageRegionConstIterator.h>
#include <itkImageRegionIteratorWithIndex.h>

#include "rtkHomogeneousMatrix.h"

namespace itk
{

template <class TInputImage, class TOutputImage>
void RayEllipsoidIntersectionImageFilter<TInputImage, TOutputImage>::BeforeThreadedGenerateData()
{
  this->Translate();
}

template <class TInputImage, class TOutputImage>
void RayEllipsoidIntersectionImageFilter<TInputImage, TOutputImage>::Translate()
{
  this->GetRQIFunctor()->SetA( 1/pow(m_SemiPrincipalAxisX,2.0) );
  this->GetRQIFunctor()->SetB( 1/pow(m_SemiPrincipalAxisY,2.0) );
  this->GetRQIFunctor()->SetC( 1/pow(m_SemiPrincipalAxisZ,2.0) );
  this->GetRQIFunctor()->SetD( 0. );
  this->GetRQIFunctor()->SetE( 0. );
  this->GetRQIFunctor()->SetF( 0. );
  this->GetRQIFunctor()->SetG( 2*(-1*m_CenterEllipsoidX_0)/pow(m_SemiPrincipalAxisX,2.0) );
  this->GetRQIFunctor()->SetH( 2*(-1*m_CenterEllipsoidY_0)/pow(m_SemiPrincipalAxisY,2.0) );
  this->GetRQIFunctor()->SetI( 2*(-1*m_CenterEllipsoidZ_0)/pow(m_SemiPrincipalAxisZ,2.0) );
  this->GetRQIFunctor()->SetJ( -1+pow(m_CenterEllipsoidX_0/m_SemiPrincipalAxisX,2.0)+pow(m_CenterEllipsoidY_0/m_SemiPrincipalAxisY,2.0)+pow(m_CenterEllipsoidZ_0/m_SemiPrincipalAxisZ,2.0) );
  //this->GetRQIFunctor()->SetJ( -1 );
}

}// end namespace itk

#endif
