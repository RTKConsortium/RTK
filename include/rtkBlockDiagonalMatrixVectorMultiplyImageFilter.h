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
#ifndef rtkBlockDiagonalMatrixVectorMultiplyImageFilter_h
#define rtkBlockDiagonalMatrixVectorMultiplyImageFilter_h

#include "itkBinaryFunctorImageFilter.h"

namespace rtk
{
namespace Functor
{
/**
 * \class MatrixVectorMult
 * \brief Functor for voxelwise matrix-vector multiplication
 */
template< typename TVector, typename TMatrixAsVector >
class MatrixVectorMult
{
public:
  MatrixVectorMult() {}
  ~MatrixVectorMult() {}
  bool operator!=(const MatrixVectorMult &) const
  {
    return false;
  }

  bool operator==(const MatrixVectorMult & other) const
  {
    return !( *this != other );
  }

  // Input 1 is the vector image, Input 2 is the matrix image
  inline TVector operator()(const TVector & A, const TMatrixAsVector & B) const
  {
  vnl_matrix<typename TMatrixAsVector::ComponentType> mat = vnl_matrix<typename TMatrixAsVector::ComponentType>(TVector::Dimension, TVector::Dimension);
  mat.copy_in(B.GetVnlVector().data_block());
  itk::Matrix<typename TMatrixAsVector::ComponentType, TVector::Dimension, TVector::Dimension> matrix = itk::Matrix<typename TMatrixAsVector::ComponentType, TVector::Dimension, TVector::Dimension>(mat);
  return(matrix * A);
  }
};
}

/** \class rtkBlockDiagonalMatrixVectorMultiplyImageFilter
 * \brief
 *
 */
template< typename TImageOfVectors, typename TImageOfMatricesAsVectors>
class ITK_EXPORT BlockDiagonalMatrixVectorMultiplyImageFilter:
  public
  itk::BinaryFunctorImageFilter< TImageOfVectors, TImageOfMatricesAsVectors, TImageOfVectors,
                              Functor::MatrixVectorMult<
                              typename TImageOfVectors::PixelType,
                              typename TImageOfMatricesAsVectors::PixelType>   >
{
public:
  /** Standard class typedefs. */
  typedef BlockDiagonalMatrixVectorMultiplyImageFilter                                        Self;
  typedef itk::BinaryFunctorImageFilter< TImageOfVectors, TImageOfMatricesAsVectors, TImageOfVectors,
                                         Functor::MatrixVectorMult<
                                            typename TImageOfVectors::PixelType,
                                            typename TImageOfMatricesAsVectors::PixelType> >  Superclass;
  typedef itk::SmartPointer< Self >                                                           Pointer;
  typedef itk::SmartPointer< const Self >                                                     ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Runtime information support. */
  itkTypeMacro(BlockDiagonalMatrixVectorMultiplyImageFilter,
               itk::BinaryFunctorImageFilter);

protected:
  BlockDiagonalMatrixVectorMultiplyImageFilter() {}
  virtual ~BlockDiagonalMatrixVectorMultiplyImageFilter() {}
  void VerifyInputInformation(){}

private:
  ITK_DISALLOW_COPY_AND_ASSIGN(BlockDiagonalMatrixVectorMultiplyImageFilter);
};
} // end namespace rtk

#endif
