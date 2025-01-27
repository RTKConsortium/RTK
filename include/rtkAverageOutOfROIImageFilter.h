/*=========================================================================
 *
 *  Copyright RTK Consortium
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *         https://www.apache.org/licenses/LICENSE-2.0.txt
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 *=========================================================================*/
#ifndef rtkAverageOutOfROIImageFilter_h
#define rtkAverageOutOfROIImageFilter_h

#include "itkInPlaceImageFilter.h"

#include <itkImageRegionSplitterDirection.h>

#include "rtkMacro.h"

namespace rtk
{
/** \class AverageOutOfROIImageFilter
 * \brief Averages along the last dimension if the pixel is outside ROI
 *
 * This filter takes in input a n-D image and an (n-1)D binary image
 * representing a region of interest (1 inside the ROI, 0 outside).
 * The filter walks through the ROI image, and :
 * - if it contains 0, pixels in the n-D image a replaced with their
 * average along the last dimension
 * - if it contains 1, nothing happens
 *
 * This filter is used in rtk4DROOSTERConeBeamReconstructionFilter in
 * order to average along time between phases, everywhere except where
 * movement is expected to occur.
 *
 * \test rtkfourdroostertest.cxx
 *
 * \author Cyril Mory
 *
 * \ingroup RTK
 *
 */
template <class TInputImage, class TROI = itk::Image<typename TInputImage::PixelType, TInputImage::ImageDimension - 1>>

class ITK_TEMPLATE_EXPORT AverageOutOfROIImageFilter : public itk::InPlaceImageFilter<TInputImage, TInputImage>
{
public:
  ITK_DISALLOW_COPY_AND_MOVE(AverageOutOfROIImageFilter);

  /** Standard class type alias. */
  using Self = AverageOutOfROIImageFilter;
  using Superclass = itk::ImageToImageFilter<TInputImage, TInputImage>;
  using Pointer = itk::SmartPointer<Self>;
  using LowerDimImage = itk::Image<typename TInputImage::PixelType, TInputImage::ImageDimension - 1>;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkOverrideGetNameOfClassMacro(AverageOutOfROIImageFilter);

  /** The image containing the weights applied to the temporal components */
  void
  SetROI(const TROI * Map);

protected:
  AverageOutOfROIImageFilter();
  ~AverageOutOfROIImageFilter() override = default;

  typename TROI::Pointer
  GetROI();

  void
  GenerateOutputInformation() override;
  void
  GenerateInputRequestedRegion() override;

  /** Does the real work. */
  void
  ThreadedGenerateData(const typename TInputImage::RegionType & outputRegionForThread,
                       itk::ThreadIdType                        itkNotUsed(threadId)) override;

  /** Splits the OutputRequestedRegion along the first direction, not the last */
  const itk::ImageRegionSplitterBase *
                                             GetImageRegionSplitter() const override;
  itk::ImageRegionSplitterDirection::Pointer m_Splitter;
};
} // namespace rtk


#ifndef ITK_MANUAL_INSTANTIATION
#  include "rtkAverageOutOfROIImageFilter.hxx"
#endif

#endif
