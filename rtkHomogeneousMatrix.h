#ifndef RTKHOMOGENEOUSMATRIX_H
#define RTKHOMOGENEOUSMATRIX_H

#include <itkMatrix.h>
#include <itkImage.h>

//--------------------------------------------------------------------
itk::Matrix< double, 3, 3 >
Get2DScalingHomogeneousMatrix( double scalingX, double scalingY );

//--------------------------------------------------------------------
itk::Matrix< double, 3, 3 >
Get2DRigidTransformationHomogeneousMatrix( double angleX, double transX, double transY );

//--------------------------------------------------------------------
itk::Matrix< double, 4, 4 >
Get3DTranslationHomogeneousMatrix( double transX, double transY, double transZ );

//--------------------------------------------------------------------
itk::Matrix< double, 4, 4 >
Get3DRigidTransformationHomogeneousMatrix( double angleX, double angleY, double angleZ,
                                           double transX, double transY, double transZ );

//--------------------------------------------------------------------
itk::Matrix< double, 4, 4 >
Get3DRotationHomogeneousMatrix( itk::Vector<double, 3> axis, double angle );

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

//--------------------------------------------------------------------
/** Get IndexToPhysicalPoint matrix from an image (no accessor provided by ITK) */
//template <class TPixel, unsigned int VImageDimension>
template <class TImageType>
itk::Matrix<double, TImageType::ImageDimension + 1, TImageType::ImageDimension + 1>
GetIndexToPhysicalPointMatrix(const TImageType *image)
{
  const unsigned int Dimension = TImageType::ImageDimension;

  itk::Matrix<double, Dimension + 1, Dimension + 1> matrix;
  matrix.Fill(0.0);

  itk::Index<Dimension>                                 index;
  itk::Point<typename TImageType::PixelType, Dimension> point;

  for(unsigned int j=0; j<Dimension; j++)
    {
    index.Fill(0);
    index[j] = 1;
    image->TransformIndexToPhysicalPoint(index,point);
    for(unsigned int i=0; i<Dimension; i++)
      matrix[i][j] = point[i]-image->GetOrigin()[i];
    }
  for(unsigned int i=0; i<Dimension; i++)
    matrix[i][Dimension] = image->GetOrigin()[i];
  matrix[Dimension][Dimension] = 1.0;

  return matrix;
}

//--------------------------------------------------------------------
/** Get PhysicalPointToIndex matrix from an image (no accessor provided by ITK) */
//template <class TPixel, unsigned int VImageDimension>
template <class TImageType>
itk::Matrix<double, TImageType::ImageDimension + 1, TImageType::ImageDimension + 1>
GetPhysicalPointToIndexMatrix(const TImageType *image)
{
  typedef itk::Matrix<double, TImageType::ImageDimension + 1, TImageType::ImageDimension + 1> MatrixType;
  return MatrixType(GetIndexToPhysicalPointMatrix<TImageType>(image).GetInverse() );
}

#endif // RTKHOMOGENEOUSMATRIX_H
