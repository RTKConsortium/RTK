#ifndef RTKHOMOGENEOUSMATRIX_H
#define RTKHOMOGENEOUSMATRIX_H

#include <itkMatrix.h>

//--------------------------------------------------------------------
itk::Matrix< double, 3, 3 >
Get2DRigidTransformationHomogeneousMatrix( double angleX, double transX, double transY );

//--------------------------------------------------------------------
itk::Matrix< double, 4, 4 >
Get3DRigidTransformationHomogeneousMatrix( double angleX, double angleY, double angleZ, double transX, double transY, double transZ );

//--------------------------------------------------------------------
template< unsigned int TDimension >
itk::Matrix< double, TDimension, TDimension+1 >
GetProjectionMagnificationMatrix( double sdd, double sid )
{
  itk::Matrix< double, TDimension, TDimension+1 > matrix;
  matrix.Fill(0.0);
  for(unsigned int i=0; i<TDimension-1; i++)
    matrix[i][i] = sdd;
  matrix[TDimension-1][TDimension-1] = 1.0;
  matrix[TDimension-1][TDimension  ] = sid;
  return matrix;
}

#endif // RTKHOMOGENEOUSMATRIX_H
