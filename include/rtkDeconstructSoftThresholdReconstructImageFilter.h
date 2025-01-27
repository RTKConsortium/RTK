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

#ifndef rtkDeconstructSoftThresholdReconstructImageFilter_h
#define rtkDeconstructSoftThresholdReconstructImageFilter_h

// ITK includes
#include "itkMacro.h"
#include "itkProgressReporter.h"

// rtk includes
#include "rtkDeconstructImageFilter.h"
#include "rtkReconstructImageFilter.h"
#include "rtkSoftThresholdImageFilter.h"

namespace rtk
{

/**
 * \class DeconstructSoftThresholdReconstructImageFilter
 * \brief Deconstructs an image, soft thresholds its wavelets coefficients,
 * then reconstructs
 *
 * This filter is inspired from Dan Mueller's GIFT package
 * https://www.insight-journal.org/browse/publication/103
 *
 * \author Cyril Mory
 *
 * \ingroup RTK
 */
template <class TImage>
class ITK_TEMPLATE_EXPORT DeconstructSoftThresholdReconstructImageFilter
  : public itk::ImageToImageFilter<TImage, TImage>
{
public:
  ITK_DISALLOW_COPY_AND_MOVE(DeconstructSoftThresholdReconstructImageFilter);

  /** Standard class type alias. */
  using Self = DeconstructSoftThresholdReconstructImageFilter;
  using Superclass = itk::ImageToImageFilter<TImage, TImage>;
  using Pointer = itk::SmartPointer<Self>;
  using ConstPointer = itk::SmartPointer<const Self>;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkOverrideGetNameOfClassMacro(DeconstructSoftThresholdReconstructImageFilter);

  /** ImageDimension enumeration. */
  static constexpr unsigned int ImageDimension = TImage::ImageDimension;

  /** Inherit types from Superclass. */
  using InputImageType = typename Superclass::InputImageType;
  using OutputImageType = typename Superclass::OutputImageType;
  using InputImagePointer = typename Superclass::InputImagePointer;
  using OutputImagePointer = typename Superclass::OutputImagePointer;
  using InputImageConstPointer = typename Superclass::InputImageConstPointer;
  using PixelType = typename TImage::PixelType;
  using InternalPixelType = typename TImage::InternalPixelType;

  /** Define the types of subfilters */
  using DeconstructFilterType = rtk::DeconstructImageFilter<InputImageType>;
  using ReconstructFilterType = rtk::ReconstructImageFilter<InputImageType>;
  using SoftThresholdFilterType = rtk::SoftThresholdImageFilter<InputImageType, InputImageType>;

  /** Set the number of levels of the deconstruction and reconstruction */
  void
  SetNumberOfLevels(unsigned int levels);

  /** Sets the order of the Daubechies wavelet used to deconstruct/reconstruct the image pyramid */
  itkGetMacro(Order, unsigned int);
  itkSetMacro(Order, unsigned int);

  /** Sets the threshold used in soft thresholding */
  itkGetMacro(Threshold, float);
  itkSetMacro(Threshold, float);

protected:
  DeconstructSoftThresholdReconstructImageFilter();
  ~DeconstructSoftThresholdReconstructImageFilter() override = default;
  void
  PrintSelf(std::ostream & os, itk::Indent indent) const override;

  /** Generate the output data. */
  void
  GenerateData() override;

  /** Compute the information on output's size and index */
  void
  GenerateOutputInformation() override;

  void
  GenerateInputRequestedRegion() override;

private:
  unsigned int m_Order;
  float        m_Threshold;
  bool         m_PipelineConstructed;

  typename DeconstructFilterType::Pointer m_DeconstructionFilter;
  typename ReconstructFilterType::Pointer m_ReconstructionFilter;
  std::vector<typename SoftThresholdFilterType::Pointer>
    m_SoftTresholdFilters; // Holds an array of soft threshold filters
};

} // namespace rtk

// Include CXX
#ifndef rtk_MANUAL_INSTANTIATION
#  include "rtkDeconstructSoftThresholdReconstructImageFilter.hxx"
#endif

#endif
