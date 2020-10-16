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

#ifndef rtkForwardWarpImageFilter_h
#define rtkForwardWarpImageFilter_h

#include <itkWarpImageFilter.h>

#include "rtkMacro.h"

namespace rtk
{

/** \class ForwardWarpImageFilter
 * \brief Warps an image using splat instead of interpolation
 *
 * Deforms an image using a Displacement Vector Field. Adjoint operator
 * of the itkWarpImageFilter
 *
 * \test rtkfourdroostertest
 *
 * \author Cyril Mory
 *
 * \ingroup RTK
 */
template <class TInputImage,
          class TOutputImage = TInputImage,
          class TDVF = itk::Image<itk::CovariantVector<typename TInputImage::PixelType, TInputImage::ImageDimension>,
                                  TInputImage::ImageDimension>>
class ForwardWarpImageFilter : public itk::WarpImageFilter<TInputImage, TOutputImage, TDVF>
{
public:
#if ITK_VERSION_MAJOR == 5 && ITK_VERSION_MINOR == 1
  ITK_DISALLOW_COPY_AND_ASSIGN(ForwardWarpImageFilter);
#else
  ITK_DISALLOW_COPY_AND_MOVE(ForwardWarpImageFilter);
#endif

  /** Standard class type alias. */
  using Self = ForwardWarpImageFilter;
  using Superclass = itk::WarpImageFilter<TInputImage, TOutputImage, TDVF>;
  using Pointer = itk::SmartPointer<Self>;
  using ConstPointer = itk::SmartPointer<const Self>;

  /** Displacement type */
  using DisplacementFieldType = TDVF;
  using DisplacementFieldPointer = typename DisplacementFieldType::Pointer;
  using DisplacementFieldConstPointer = typename DisplacementFieldType::ConstPointer;
  using DisplacementType = typename DisplacementFieldType::PixelType;

  /** Point type */
  using CoordRepType = double;
  using PointType = itk::Point<CoordRepType, itkGetStaticConstMacro(ImageDimension)>;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(ForwardWarpImageFilter, Superclass);

protected:
  ForwardWarpImageFilter();
  ~ForwardWarpImageFilter() override = default;

  void
  GenerateData() override;

  // Redefine stuff that is private in the Superclass
  void
                                   Protected_EvaluateDisplacementAtPhysicalPoint(const PointType & point, DisplacementType & output);
  bool                             m_Protected_DefFieldSizeSame;
  typename TOutputImage::IndexType m_Protected_StartIndex;
  typename TOutputImage::IndexType m_Protected_EndIndex;
};

} // end namespace rtk

#ifndef rtk_MANUAL_INSTANTIATION
#  include "rtkForwardWarpImageFilter.hxx"
#endif

#endif
