#ifndef RTKHOMOGENEOUSMATRIX_H
#define RTKHOMOGENEOUSMATRIX_H

#include <itkMatrix.h>
#include <itkImage.h>

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
