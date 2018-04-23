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
#ifndef rtkConditionalMedianImageFilter_hxx
#define rtkConditionalMedianImageFilter_hxx

#include "rtkConditionalMedianImageFilter.h"
#include <itkImageRegionIterator.h>
#include <numeric>

namespace rtk
{
//
// Constructor
//
template< typename TInputImage >
ConditionalMedianImageFilter< TInputImage >
::ConditionalMedianImageFilter()
{
  // Default parameters
  m_Radius.Fill(1);
  m_ThresholdMultiplier = 1;
}


template< typename TInputImage >
void
ConditionalMedianImageFilter< TInputImage >
::GenerateInputRequestedRegion()
{
  //Call the superclass' implementation of this method
  Superclass::GenerateInputRequestedRegion();

  // Compute the requested region
  typename TInputImage::RegionType inputRequested = this->GetOutput()->GetRequestedRegion();
  typename TInputImage::SizeType requestedSize = inputRequested.GetSize();
  typename TInputImage::IndexType requestedIndex = inputRequested.GetIndex();

  // We need the previous projection to extract the noise,
  // and the neighboring pixels to compute its variance
  for (unsigned int dim = 0; dim < TInputImage::ImageDimension; dim++)
    {
    requestedSize[dim] += 2 * m_Radius[dim];
    requestedIndex[dim] -= m_Radius[dim];
    }
  inputRequested.SetSize(requestedSize);
  inputRequested.SetIndex(requestedIndex);

  // Crop the requested region to the LargestPossibleRegion
  inputRequested.Crop(this->GetInput()->GetLargestPossibleRegion());

  // Get a pointer to the input and set the requested region
  typename TInputImage::Pointer  inputPtr  = const_cast<TInputImage *>(this->GetInput());
  inputPtr->SetRequestedRegion(inputRequested);
}

template< typename TInputImage >
void
ConditionalMedianImageFilter< TInputImage >
::ThreadedGenerateData(const typename TInputImage::RegionType& outputRegionForThread, itk::ThreadIdType itkNotUsed(threadId))
{
  // Compute the centered difference with the previous and next frames, store it into the intermediate image
  itk::ConstNeighborhoodIterator<TInputImage> nIt(m_Radius, this->GetInput(), outputRegionForThread);
  itk::ImageRegionIterator<TInputImage> outIt(this->GetOutput(), outputRegionForThread);

  // Build a vector in which all pixel of the neighborhood will be temporarily stored
  std::vector< typename TInputImage::PixelType > pixels;
  pixels.resize(nIt.Size());

  // Walk the output image
  while(!outIt.IsAtEnd())
    {
    // Walk the neighborhood in the input image, store the pixels into a vector
    for (unsigned int i=0; i<nIt.Size(); i++)
      pixels[i] = nIt.GetPixel(i);

    // Compute the standard deviation
    double sum = std::accumulate(pixels.begin(), pixels.end(), 0.0);
    double mean = sum / pixels.size();
    double sq_sum = std::inner_product(pixels.begin(), pixels.end(), pixels.begin(), 0.0);
    double stdev = std::sqrt(sq_sum / pixels.size() - mean * mean);

    // Compute the median of the neighborhood
    std::nth_element( pixels.begin(), pixels.begin() + pixels.size()/2, pixels.end() );

    // If the pixel value is too far from the median, replace it by the median
    if(vnl_math_abs(pixels[pixels.size()/2] - nIt.GetCenterPixel()) > (m_ThresholdMultiplier * stdev) )
      outIt.Set(pixels[pixels.size()/2]);
    else // Otherwise, leave it as is
      outIt.Set(nIt.GetCenterPixel());

    ++nIt;
    ++outIt;
    }
}

} // end namespace itk

#endif
