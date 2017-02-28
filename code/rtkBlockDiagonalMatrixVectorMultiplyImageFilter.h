/*=========================================================================
 *
 *  Copyright Insight Software Consortium
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
template< typename TInput1, typename TInput2 = TInput1, typename TOutput = TInput1 >
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
  inline TOutput operator()(const TInput1 & A, const TInput2 & B) const
  {
  vnl_matrix<typename TInput2::ComponentType> mat = vnl_matrix<typename TInput2::ComponentType>(A.GetSize(), A.GetSize());
  mat.copy_in(B.GetDataPointer());
  vnl_vector<typename TInput1::ComponentType> vect(A.GetDataPointer(),A.GetSize());
  vnl_vector<typename TInput1::ComponentType> vnl_result = mat * vect;

  TOutput result;
  result.SetSize(A.GetSize());
  result.SetData(vnl_result.data_block(),A.GetSize(),false);

  return static_cast< TOutput >( result );
  }
};
}
/** \class rtkBlockDiagonalMatrixVectorMultiplyImageFilter
 * \brief Pixel-wise multiplication of two images.
 *
 * This class is templated over the types of the two
 * input images and the type of the output image.
 * Numeric conversions (castings) are done by the C++ defaults.
 *
 */
template< typename TInputImage1, typename TInputImage2 = TInputImage1, typename TOutputImage = TInputImage1 >
class ITK_EXPORT BlockDiagonalMatrixVectorMultiplyImageFilter:
  public
  itk::BinaryFunctorImageFilter< TInputImage1, TInputImage2, TOutputImage,
                            Functor::MatrixVectorMult<
                              typename TInputImage1::PixelType,
                              typename TInputImage2::PixelType,
                              typename TOutputImage::PixelType >   >
{
public:
  /** Standard class typedefs. */
  typedef BlockDiagonalMatrixVectorMultiplyImageFilter Self;
  typedef itk::BinaryFunctorImageFilter< TInputImage1, TInputImage2, TOutputImage,
                                    Functor::MatrixVectorMult<
                                      typename TInputImage1::PixelType,
                                      typename TInputImage2::PixelType,
                                      typename TOutputImage::PixelType >
                                    >                                 Superclass;
  typedef itk::SmartPointer< Self >       Pointer;
  typedef itk::SmartPointer< const Self > ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Runtime information support. */
  itkTypeMacro(BlockDiagonalMatrixVectorMultiplyImageFilter,
               itk::BinaryFunctorImageFilter);


protected:
  BlockDiagonalMatrixVectorMultiplyImageFilter() {}
  virtual ~BlockDiagonalMatrixVectorMultiplyImageFilter() {}

private:
  ITK_DISALLOW_COPY_AND_ASSIGN(BlockDiagonalMatrixVectorMultiplyImageFilter);
};
} // end namespace rtk

#endif
