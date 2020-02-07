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

#include "itkImageToImageFilter.h"
#include "rtkMacro.h"

namespace rtk
{
/** \class BlockDiagonalMatrixVectorMultiplyImageFilter
 * \brief Multiplies matrix by vector
 *
 * \author Cyril Mory
 *
 * \ingroup RTK
 *
 */
template <class TVectorImage,
          class TMatrixImage =
            itk::Image<itk::Vector<typename TVectorImage::PixelType::ValueType,
                                   TVectorImage::PixelType::Dimension * TVectorImage::PixelType::Dimension>,
                       TVectorImage::ImageDimension>>
class BlockDiagonalMatrixVectorMultiplyImageFilter : public itk::ImageToImageFilter<TVectorImage, TVectorImage>
{
public:
  ITK_DISALLOW_COPY_AND_ASSIGN(BlockDiagonalMatrixVectorMultiplyImageFilter);

  /** Standard class type alias. */
  using Self = BlockDiagonalMatrixVectorMultiplyImageFilter;
  using Superclass = itk::ImageToImageFilter<TVectorImage, TVectorImage>;
  using Pointer = itk::SmartPointer<Self>;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(BlockDiagonalMatrixVectorMultiplyImageFilter, itk::ImageToImageFilter);

  /** Convenient parameters extracted from template types */
  static constexpr unsigned int nChannels = TVectorImage::PixelType::Dimension;

  /** Convenient type alias */
  using dataType = typename TVectorImage::PixelType::ValueType;

  /** Set methods for all inputs, since they have different types */
  void
  SetInput1(const TVectorImage * vector);
  void
  SetInput2(const TMatrixImage * matrix);

protected:
  BlockDiagonalMatrixVectorMultiplyImageFilter();
  ~BlockDiagonalMatrixVectorMultiplyImageFilter() override = default;

  void
  GenerateInputRequestedRegion() override;

  /** Does the real work. */
  void
  DynamicThreadedGenerateData(const typename TVectorImage::RegionType & outputRegionForThread) override;

  /** Getters for the inputs */
  typename TVectorImage::ConstPointer
  GetInput1();
  typename TMatrixImage::ConstPointer
  GetInput2();
};
} // namespace rtk


#ifndef ITK_MANUAL_INSTANTIATION
#  include "rtkBlockDiagonalMatrixVectorMultiplyImageFilter.hxx"
#endif

#endif
