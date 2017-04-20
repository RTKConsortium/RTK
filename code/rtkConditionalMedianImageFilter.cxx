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

#include "rtkConditionalMedianImageFilter.h"

template <>
void
rtk::ConditionalMedianImageFilter<itk::VectorImage<float, 3> >
::ThreadedGenerateData(const itk::VectorImage<float, 3>::RegionType& outputRegionForThread, itk::ThreadIdType itkNotUsed(threadId))
{
typedef itk::VectorImage<float, 3> TInputImage;

// Compute the centered difference with the previous and next frames, store it into the intermediate image
itk::ConstNeighborhoodIterator<TInputImage> nIt(m_Radius, this->GetInput(), outputRegionForThread);
itk::ImageRegionIterator<TInputImage> outIt(this->GetOutput(), outputRegionForThread);

// Build a vector in which all pixel of the neighborhood will be temporarily stored
std::vector< std::vector< TInputImage::InternalPixelType > > pixels;
pixels.resize(this->GetInput()->GetVectorLength());
for (unsigned int mat = 0; mat<pixels.size(); mat++)
  pixels[mat].resize(nIt.Size());

itk::VariableLengthVector<float> vlv;
vlv.SetSize(this->GetInput()->GetVectorLength());

// Walk the output image
while(!outIt.IsAtEnd())
  {
  // Walk the neighborhood in the input image, store the pixels into a vector
  for (unsigned int i=0; i<nIt.Size(); i++)
    {
    for(unsigned int mat = 0; mat<pixels.size(); mat++)
      pixels[mat][i] = nIt.GetPixel(i)[mat];
    }

  for(unsigned int mat = 0; mat<pixels.size(); mat++)
    {
    // Compute the standard deviation
    double sum = std::accumulate(pixels[mat].begin(), pixels[mat].end(), 0.0);
    double mean = sum / pixels[mat].size();
    double sq_sum = std::inner_product(pixels[mat].begin(), pixels[mat].end(), pixels[mat].begin(), 0.0);
    double stdev = std::sqrt(sq_sum / pixels[mat].size() - mean * mean);

    // Compute the median of the neighborhood
    std::nth_element( pixels[mat].begin(), pixels[mat].begin() + pixels[mat].size()/2, pixels[mat].end() );

    // If the pixel value is too far from the median, replace it by the median
    if(fabs(pixels[mat][pixels[mat].size()/2] - nIt.GetCenterPixel()[mat]) > (m_ThresholdMultiplier * stdev) )
      vlv[mat] = pixels[mat][pixels[mat].size()/2];
    else // Otherwise, leave it as is
      vlv[mat] = nIt.GetCenterPixel()[mat];
    }
  outIt.Set(vlv);

  ++nIt;
  ++outIt;
  }
}
