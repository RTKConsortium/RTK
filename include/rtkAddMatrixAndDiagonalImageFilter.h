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
#ifndef rtkAddMatrixAndDiagonalImageFilter_h
#define rtkAddMatrixAndDiagonalImageFilter_h

#include "itkImageToImageFilter.h"
#include "rtkMacro.h"

namespace rtk
{
/** \class AddMatrixAndDiagonalImageFilter
 * \brief For each vector-valued pixel, adds a vector to the diagonal of a matrix
 *
 * This filter takes in input an image of vectors of length n (input 1)
 * and an image of vectors of length n*n (input 2). The vectors in input 2
 * are used as n*n matrices, and those in input 1 are assumed to be a compact
 * representation of diagonal matrices of size n*n (thus with only n non-null
 * values).
 *
 * \author Cyril Mory
 *
 * \ingroup RTK
 *
 */
template <class TDiagonal,
          class TMatrix = itk::Image<itk::Vector<typename TDiagonal::PixelType::ValueType,
                                                 TDiagonal::PixelType::Dimension * TDiagonal::PixelType::Dimension>,
                                     TDiagonal::ImageDimension>>
class AddMatrixAndDiagonalImageFilter : public itk::ImageToImageFilter<TMatrix, TMatrix>
{
public:
  ITK_DISALLOW_COPY_AND_ASSIGN(AddMatrixAndDiagonalImageFilter);

  /** Standard class type alias. */
  using Self = AddMatrixAndDiagonalImageFilter;
  using Superclass = itk::ImageToImageFilter<TMatrix, TMatrix>;
  using Pointer = itk::SmartPointer<Self>;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(AddMatrixAndDiagonalImageFilter, itk::ImageToImageFilter);

  /** Convenient parameters extracted from template types */
  static constexpr unsigned int nChannels = TDiagonal::PixelType::Dimension;

  /** Convenient type alias */
  using dataType = typename TDiagonal::PixelType::ValueType;

  /** Set methods for all inputs, since they have different types */
  void
  SetInputDiagonal(const TDiagonal * gradient);
  void
  SetInputMatrix(const TMatrix * hessian);

protected:
  AddMatrixAndDiagonalImageFilter();
  ~AddMatrixAndDiagonalImageFilter() override = default;

  void
  GenerateInputRequestedRegion() override;

  /** Does the real work. */
  void
  DynamicThreadedGenerateData(const typename TDiagonal::RegionType & outputRegionForThread) override;

  /** Getters for the inputs */
  typename TDiagonal::ConstPointer
  GetInputDiagonal();
  typename TMatrix::ConstPointer
  GetInputMatrix();
};
} // namespace rtk


#ifndef ITK_MANUAL_INSTANTIATION
#  include "rtkAddMatrixAndDiagonalImageFilter.hxx"
#endif

#endif
