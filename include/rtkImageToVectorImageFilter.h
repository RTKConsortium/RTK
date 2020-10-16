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
#ifndef rtkImageToVectorImageFilter_h
#define rtkImageToVectorImageFilter_h

#include "rtkMacro.h"

#include <itkImageToImageFilter.h>
#include <itkImageRegionSplitterDirection.h>

namespace rtk
{
/** \class ImageToVectorImageFilter
 * \brief Re-writes an image as a vector image
 *
 * Depending on the dimensions of the input and output images, the filter
 * can have two different behaviors:
 *  - if the dimensions match, the channels of the input image are
 * obtained by slicing the last dimension. With an input image of size (X, Y*N),
 * the output is an N-components vector image of size (X,Y)
 *  - if the input image dimension equals the output dimension plus one,
 * the additional dimension of the input image is assumed to contain the channels.
 * With an input image of size (X,Y,N), the output is an N-components
 * vector image of size (X, Y).
 *
 * \author Cyril Mory
 *
 * \ingroup RTK
 */

template <typename InputImageType, typename OutputImageType>
class ImageToVectorImageFilter : public itk::ImageToImageFilter<InputImageType, OutputImageType>
{
public:
#if ITK_VERSION_MAJOR == 5 && ITK_VERSION_MINOR == 1
  ITK_DISALLOW_COPY_AND_ASSIGN(ImageToVectorImageFilter);
#else
  ITK_DISALLOW_COPY_AND_MOVE(ImageToVectorImageFilter);
#endif

  /** Standard class type alias. */
  using Self = ImageToVectorImageFilter;
  using Superclass = itk::ImageToImageFilter<InputImageType, OutputImageType>;
  using Pointer = itk::SmartPointer<Self>;

  using OutputImageRegionType = typename OutputImageType::RegionType;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(ImageToVectorImageFilter, itk::ImageToImageFilter);

  /** When the input and output dimensions are equal, the filter
   * cannot guess the number of channels. Set/Get methods to
   * pass it */
  itkSetMacro(NumberOfChannels, unsigned int);
  itkGetMacro(NumberOfChannels, unsigned int);

protected:
  ImageToVectorImageFilter();
  ~ImageToVectorImageFilter() override = default;

  void
  GenerateOutputInformation() override;
  void
  GenerateInputRequestedRegion() override;

  /** Does the real work. */
  void
  ThreadedGenerateData(const OutputImageRegionType & outputRegionForThread,
                       itk::ThreadIdType             itkNotUsed(threadId)) override;

  /** Splits the OutputRequestedRegion along the first direction, not the last */
  const itk::ImageRegionSplitterBase *
                                             GetImageRegionSplitter() const override;
  itk::ImageRegionSplitterDirection::Pointer m_Splitter;

  unsigned int m_NumberOfChannels;
};
} // namespace rtk


#ifndef ITK_MANUAL_INSTANTIATION
#  include "rtkImageToVectorImageFilter.hxx"
#endif

#endif
