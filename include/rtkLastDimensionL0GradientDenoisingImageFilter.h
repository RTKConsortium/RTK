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
#ifndef rtkLastDimensionL0GradientDenoisingImageFilter_h
#define rtkLastDimensionL0GradientDenoisingImageFilter_h

#include "itkInPlaceImageFilter.h"

#include <itkImageRegionSplitterDirection.h>

namespace rtk
{
/** \class LastDimensionL0GradientDenoisingImageFilter
 * \brief Denoises along the last dimension, reducing the L0 norm of the gradient
 *
 * This filter implements the "Fast and Effective L0 Gradient Minimization by Region Fusion"
 * method, developped by Nguyen and Brown. Their method is computationally demanding, but its
 * restriction to 1D can be implemented efficiently. This is what this filter does.
 *
 * \test rtkl0gradientnormtest
 *
 * \author Cyril Mory
 *
 * \ingroup RTK
 *
 */
template <class TInputImage>

class LastDimensionL0GradientDenoisingImageFilter : public itk::InPlaceImageFilter<TInputImage, TInputImage>
{
public:
#if ITK_VERSION_MAJOR == 5 && ITK_VERSION_MINOR == 1
  ITK_DISALLOW_COPY_AND_ASSIGN(LastDimensionL0GradientDenoisingImageFilter);
#else
  ITK_DISALLOW_COPY_AND_MOVE(LastDimensionL0GradientDenoisingImageFilter);
#endif

  /** Standard class type alias. */
  using Self = LastDimensionL0GradientDenoisingImageFilter;
  using Superclass = itk::InPlaceImageFilter<TInputImage, TInputImage>;
  using Pointer = itk::SmartPointer<Self>;
  using InputPixelType = typename TInputImage::PixelType;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(LastDimensionL0GradientDenoisingImageFilter, itk::InPlaceImageFilter);

  /** Get / Set the threshold. Default is 0.001 */
  itkGetMacro(Lambda, double);
  itkSetMacro(Lambda, double);

  /** Get / Set the number of iterations. Default is 10 */
  itkGetMacro(NumberOfIterations, unsigned int);
  itkSetMacro(NumberOfIterations, unsigned int);

protected:
  LastDimensionL0GradientDenoisingImageFilter();
  ~LastDimensionL0GradientDenoisingImageFilter() override = default;

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

  virtual void
  OneDimensionMinimizeL0NormOfGradient(InputPixelType * input,
                                       unsigned int     length,
                                       double           lambda,
                                       unsigned int     nbIters);

  double       m_Lambda;
  unsigned int m_NumberOfIterations;
};
} // namespace rtk


#ifndef ITK_MANUAL_INSTANTIATION
#  include "rtkLastDimensionL0GradientDenoisingImageFilter.hxx"
#endif

#endif
