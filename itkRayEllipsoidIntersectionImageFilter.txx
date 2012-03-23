#ifndef __itkRayEllipsoidIntersectionImageFilter_txx
#define __itkRayEllipsoidIntersectionImageFilter_txx

#include <itkImageRegionConstIterator.h>
#include <itkImageRegionIteratorWithIndex.h>

#include "rtkHomogeneousMatrix.h"

namespace itk
{

template <class TInputImage, class TOutputImage>
RayEllipsoidIntersectionImageFilter<TInputImage, TOutputImage>
::RayEllipsoidIntersectionImageFilter():
 m_SemiPrincipalAxisX(0.),
 m_SemiPrincipalAxisY(0.),
 m_SemiPrincipalAxisZ(0.),
 m_CenterX(0.),
 m_CenterY(0.),
 m_CenterZ(0.),
 m_RotationAngle(0.)
{
}

template <class TInputImage, class TOutputImage>
void RayEllipsoidIntersectionImageFilter<TInputImage, TOutputImage>::BeforeThreadedGenerateData()
{
  this->Translate();
  if (m_RotationAngle==0.)
  {
    this->Rotate();
  }
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
  this->GetRQIFunctor()->SetG( 2*(-1*m_CenterX)/pow(m_SemiPrincipalAxisX,2.0) );
  this->GetRQIFunctor()->SetH( 2*(-1*m_CenterY)/pow(m_SemiPrincipalAxisY,2.0) );
  this->GetRQIFunctor()->SetI( 2*(-1*m_CenterZ)/pow(m_SemiPrincipalAxisZ,2.0) );
  this->GetRQIFunctor()->SetJ( -1+pow(m_CenterX/m_SemiPrincipalAxisX,2.0)+pow(m_CenterY/m_SemiPrincipalAxisY,2.0)+pow(m_CenterZ/m_SemiPrincipalAxisZ,2.0) );
}

template <class TInputImage, class TOutputImage>
void RayEllipsoidIntersectionImageFilter<TInputImage, TOutputImage>::Rotate()
{
  this->GetRQIFunctor()->SetA( this->GetRQIFunctor()->GetA()*pow(cos(m_RotationAngle*(Math::pi/180)), 2.0) + this->GetRQIFunctor()->GetB()*pow(sin(m_RotationAngle*(Math::pi/180)),2.0) );
  this->GetRQIFunctor()->SetB( this->GetRQIFunctor()->GetA()*pow(sin(m_RotationAngle*(Math::pi/180)), 2.0) + this->GetRQIFunctor()->GetB()*pow(cos(m_RotationAngle*(Math::pi/180)),2.0) );
  this->GetRQIFunctor()->SetC( this->GetRQIFunctor()->GetC() );
  this->GetRQIFunctor()->SetD( 2*cos(m_RotationAngle*(Math::pi/180))*sin(m_RotationAngle*(Math::pi/180))*(this->GetRQIFunctor()->GetB() - this->GetRQIFunctor()->GetA()));
  this->GetRQIFunctor()->SetE( 0. );
  this->GetRQIFunctor()->SetF( 0. );
  this->GetRQIFunctor()->SetG( this->GetRQIFunctor()->GetG()*cos(m_RotationAngle*(Math::pi/180)) + this->GetRQIFunctor()->GetH()*sin(m_RotationAngle*(Math::pi/180)) );
  this->GetRQIFunctor()->SetH( this->GetRQIFunctor()->GetG()*(-1)*sin(m_RotationAngle*(Math::pi/180)) + this->GetRQIFunctor()->GetH()*cos(m_RotationAngle*(Math::pi/180)) );
  this->GetRQIFunctor()->SetI( this->GetRQIFunctor()->GetI() );
  this->GetRQIFunctor()->SetJ( this->GetRQIFunctor()->GetJ() );
}

}// end namespace itk

#endif
