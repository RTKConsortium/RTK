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
  this->Translate();//Translate from regular expression to quadric
  this->Rotate();//Applies rotation and translation if necessary
}

template <class TInputImage, class TOutputImage>
void RayEllipsoidIntersectionImageFilter<TInputImage, TOutputImage>::Translate()
{
  //Regular Ellipsoid (No rotation, No Translation)
  this->GetRQIFunctor()->SetA( 1/pow(m_SemiPrincipalAxisX,2.0) );
  this->GetRQIFunctor()->SetB( 1/pow(m_SemiPrincipalAxisY,2.0) );
  this->GetRQIFunctor()->SetC( 1/pow(m_SemiPrincipalAxisZ,2.0) );
  this->GetRQIFunctor()->SetD( 0. );
  this->GetRQIFunctor()->SetE( 0. );
  this->GetRQIFunctor()->SetF( 0. );
  this->GetRQIFunctor()->SetG( 0. );//2*(-1*0)/pow(m_SemiPrincipalAxisX,2.0) );
  this->GetRQIFunctor()->SetH( 0. );//2*(-1*0)/pow(m_SemiPrincipalAxisY,2.0) );
  this->GetRQIFunctor()->SetI( 0. );//2*(-1*0)/pow(m_SemiPrincipalAxisZ,2.0) );
  this->GetRQIFunctor()->SetJ( -1 );//+pow(0/m_SemiPrincipalAxisX,2.0)+pow(0/m_SemiPrincipalAxisY,2.0)+pow(0/m_SemiPrincipalAxisZ,2.0) );
}

template <class TInputImage, class TOutputImage>
void RayEllipsoidIntersectionImageFilter<TInputImage, TOutputImage>::Rotate()
{
  //Temporary Quadric Parameters
  double TempA = this->GetRQIFunctor()->GetA();
  double TempB = this->GetRQIFunctor()->GetB();
  double TempC = this->GetRQIFunctor()->GetC();
  double TempD = this->GetRQIFunctor()->GetD();
  double TempE = this->GetRQIFunctor()->GetE();
  double TempF = this->GetRQIFunctor()->GetF();
  double TempG = this->GetRQIFunctor()->GetG();
  double TempH = this->GetRQIFunctor()->GetH();
  double TempI = this->GetRQIFunctor()->GetI();
  double TempJ = this->GetRQIFunctor()->GetJ();

  //Applying Rotation
  this->GetRQIFunctor()->SetA( TempA*pow(cos(m_RotationAngle*(Math::pi/180)), 2.0) + TempB*pow(sin(m_RotationAngle*(Math::pi/180)),2.0) );
  this->GetRQIFunctor()->SetB( TempA*pow(sin(m_RotationAngle*(Math::pi/180)), 2.0) + TempB*pow(cos(m_RotationAngle*(Math::pi/180)),2.0) );
  this->GetRQIFunctor()->SetC( TempC );
  this->GetRQIFunctor()->SetD( 2*cos(m_RotationAngle*(Math::pi/180))*sin(m_RotationAngle*(Math::pi/180))*(TempB - TempA));
  this->GetRQIFunctor()->SetE( 0. );
  this->GetRQIFunctor()->SetF( 0. );
  this->GetRQIFunctor()->SetG( TempG*cos(m_RotationAngle*(Math::pi/180)) + TempH*sin(m_RotationAngle*(Math::pi/180)) );
  this->GetRQIFunctor()->SetH( TempG*(-1)*sin(m_RotationAngle*(Math::pi/180)) + TempH*cos(m_RotationAngle*(Math::pi/180)) );
  this->GetRQIFunctor()->SetI( TempI );
  this->GetRQIFunctor()->SetJ( TempJ );

  //Saving Quadric Parameters for Translation
  TempA = this->GetRQIFunctor()->GetA();
  TempB = this->GetRQIFunctor()->GetB();
  TempC = this->GetRQIFunctor()->GetC();
  TempD = this->GetRQIFunctor()->GetD();
  TempE = this->GetRQIFunctor()->GetE();
  TempF = this->GetRQIFunctor()->GetF();
  TempG = this->GetRQIFunctor()->GetG();
  TempH = this->GetRQIFunctor()->GetH();
  TempI = this->GetRQIFunctor()->GetI();
  TempJ = this->GetRQIFunctor()->GetJ();

  //Translation Parameters
  double TransG = -2*TempA*m_CenterX - TempD*m_CenterY - TempE*m_CenterZ;
  double TransH = -2*TempB*m_CenterY - TempD*m_CenterX - TempF*m_CenterZ;
  double TransI = -2*TempC*m_CenterZ - TempE*m_CenterX - TempF*m_CenterY;
  double TransJ = TempA*pow(m_CenterX,2.0) + TempB*pow(m_CenterY,2.0) + TempC*pow(m_CenterZ,2.0) + TempD*m_CenterX*m_CenterY + TempE*m_CenterX*m_CenterZ + TempF*m_CenterY*m_CenterZ - TempG*m_CenterX - TempH*m_CenterY - TempI*m_CenterZ;

  //Applying Translation
  this->GetRQIFunctor()->SetG( this->GetRQIFunctor()->GetG() + TransG );
  this->GetRQIFunctor()->SetH( this->GetRQIFunctor()->GetH() + TransH );
  this->GetRQIFunctor()->SetI( this->GetRQIFunctor()->GetI() + TransI );
  this->GetRQIFunctor()->SetJ( this->GetRQIFunctor()->GetJ() + TransJ );
}

}// end namespace itk

#endif
