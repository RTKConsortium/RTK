#include "rtkHomogeneousMatrix.h"

#include <itkCenteredEuler3DTransform.h>
#include <itkEuler2DTransform.h>

//--------------------------------------------------------------------
itk::Matrix< double, 3, 3 >
Get2DRigidTransformationHomogeneousMatrix( double angleX, double transX, double transY )
{
  const double degreesToRadians = vcl_atan(1.0) / 45.0;

  typedef itk::Euler2DTransform<double> TwoDTransformType;
  TwoDTransformType::Pointer xfm2D = TwoDTransformType::New();
  xfm2D->SetIdentity();
  xfm2D->SetRotation( angleX*degreesToRadians );

  itk::Matrix< double, 3, 3 > matrix;
  matrix.Fill(0.0);
  for(int i=0; i<2; i++)
    for(int j=0; j<2; j++)
      matrix[i][j] = xfm2D->GetMatrix()[i][j];

  matrix[0][2] = transX;
  matrix[1][2] = transY;
  matrix[2][2] = 1.0;

  return matrix;
}

//--------------------------------------------------------------------
itk::Matrix< double, 4, 4 >
Get3DRigidTransformationHomogeneousMatrix( double angleX, double angleY, double angleZ, double transX, double transY, double transZ )
{
  const double degreesToRadians = vcl_atan(1.0) / 45.0;

  typedef itk::CenteredEuler3DTransform<double> ThreeDTransformType;
  ThreeDTransformType::Pointer xfm3D = ThreeDTransformType::New();
  xfm3D->SetIdentity();
  xfm3D->SetRotation( angleX*degreesToRadians, angleY*degreesToRadians, angleZ*degreesToRadians );

  itk::Matrix< double, 4, 4 > matrix;
  matrix.Fill(0.0);
  for(int i=0; i<3; i++)
    for(int j=0; j<3; j++)
      matrix[i][j] = xfm3D->GetMatrix()[i][j];

  matrix[0][3] = transX;
  matrix[1][3] = transY;
  matrix[2][3] = transZ;
  matrix[3][3] = 1.0;

  return matrix;
}
