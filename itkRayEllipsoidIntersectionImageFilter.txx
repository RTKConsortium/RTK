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
 m_RotationAngle(0.),
 m_SQPFunctor( SQPFunctionType::New() )
{
}

template <class TInputImage, class TOutputImage>
void RayEllipsoidIntersectionImageFilter<TInputImage, TOutputImage>::BeforeThreadedGenerateData()
{
  typename SQPFunctionType::VectorType semiprincipalaxis;
  typename SQPFunctionType::VectorType center;
  semiprincipalaxis.push_back(m_SemiPrincipalAxisX);
  semiprincipalaxis.push_back(m_SemiPrincipalAxisY);
  semiprincipalaxis.push_back(m_SemiPrincipalAxisZ);
  center.push_back(m_CenterX);
  center.push_back(m_CenterY);
  center.push_back(m_CenterZ);

  //Translate from regular expression to quadric
  m_SQPFunctor->Translate(semiprincipalaxis);
  //Applies rotation and translation if necessary
  m_SQPFunctor->Rotate(m_RotationAngle, center);
  //Setting parameters in order to compute the projections
  this->GetRQIFunctor()->SetA( m_SQPFunctor->GetA() );
  this->GetRQIFunctor()->SetB( m_SQPFunctor->GetB() );
  this->GetRQIFunctor()->SetC( m_SQPFunctor->GetC() );
  this->GetRQIFunctor()->SetD( m_SQPFunctor->GetD() );
  this->GetRQIFunctor()->SetE( m_SQPFunctor->GetE() );
  this->GetRQIFunctor()->SetF( m_SQPFunctor->GetF() );
  this->GetRQIFunctor()->SetG( m_SQPFunctor->GetG() );
  this->GetRQIFunctor()->SetH( m_SQPFunctor->GetH() );
  this->GetRQIFunctor()->SetI( m_SQPFunctor->GetI() );
  this->GetRQIFunctor()->SetJ( m_SQPFunctor->GetJ() );
}
}// end namespace itk

#endif
