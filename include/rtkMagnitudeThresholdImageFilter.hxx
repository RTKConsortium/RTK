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

#ifndef rtkMagnitudeThresholdImageFilter_hxx
#define rtkMagnitudeThresholdImageFilter_hxx

#include "rtkMagnitudeThresholdImageFilter.h"

#include <itkImageRegionIterator.h>
#include <itkImageRegionConstIterator.h>
#include <itkProgressReporter.h>

namespace rtk
{

template <typename TInputImage, typename TRealType, typename TOutputImage>
MagnitudeThresholdImageFilter<TInputImage, TRealType, TOutputImage>::MagnitudeThresholdImageFilter()
{
  m_Threshold = 0;
}

template <typename TInputImage, typename TRealType, typename TOutputImage>
void
MagnitudeThresholdImageFilter<TInputImage, TRealType, TOutputImage>::DynamicThreadedGenerateData(
  const OutputImageRegionType & outputRegionForThread)
{
  itk::ImageRegionConstIterator<TInputImage> InputIt;
  itk::ImageRegionIterator<TOutputImage>     OutputIt;

  InputIt = itk::ImageRegionConstIterator<TInputImage>(this->GetInput(), outputRegionForThread);
  OutputIt = itk::ImageRegionIterator<TOutputImage>(this->GetOutput(), outputRegionForThread);

  double norm;
  while (!InputIt.IsAtEnd())
  {
    norm = InputIt.Get().GetNorm();

    if (norm > m_Threshold)
      OutputIt.Set(m_Threshold * InputIt.Get() / norm);

    ++InputIt;
    ++OutputIt;
  }
}

} // namespace rtk

#endif
