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

#ifndef __rtkMedianImageFilter_cxx
#define __rtkMedianImageFilter_cxx

#include "rtkMedianImageFilter.h"

#include <itkImageRegionConstIterator.h>
#include <itkImageRegionIterator.h>

typedef itk::Image<unsigned short, 2> TImage;
namespace rtk
{

//template<class TInputImage, class TOutputImage>
MedianImageFilter::MedianImageFilter()
{
  m_MedianWindow[0]=3;
  m_MedianWindow[1]=3;
}

//template <class TInputImage, class TOutputImage>
void MedianImageFilter
::ThreadedGenerateData(const OutputImageRegionType& itkNotUsed(outputRegionForThread), ThreadIdType itkNotUsed(threadId) )
{

  int inputSize[2];
  inputSize[0] = this->GetInput()->GetLargestPossibleRegion().GetSize()[0];
  inputSize[1] = this->GetInput()->GetLargestPossibleRegion().GetSize()[1];

  typedef   TImage::PixelType inputPixel;
  typedef   TImage::PixelType outputPixel;

  inputPixel  *bufferIn  = const_cast<inputPixel*>( this->GetInput()->GetBufferPointer() );
  outputPixel *bufferOut = const_cast<outputPixel*>(this->GetOutput()->GetBufferPointer() );

  // Median 3x3 and 3x2
  if(m_MedianWindow[0]==3 && (m_MedianWindow[1]==3 || m_MedianWindow[1]==2) )
  {
    int m  = m_MedianWindow[0];
    int n  = m_MedianWindow[1];
    int th = 4;
    int histBins = (1<<16);

    int inY, inX, winX, winY, kr;
    int k, median, ltmedian;
    int sum, lx, rx, g1;

    std::vector<unsigned short> hist(histBins, 0);

    // Boundaries not taken into account yet but set to original value
    for(inY = 0; inY <= (inputSize[1]-1); inY += (inputSize[1]-1))
      for(inX = 0; inX < inputSize[0]; inX++)
        bufferOut[inY*inputSize[0] + inX] = bufferIn[inY*inputSize[0] + inX];

    inX=0;
    for(inY = 1; inY < (inputSize[1]-1); inY++)
    {
      // Initialize histogram for this lines
      for(winY = 0; winY < n; winY++)
        for(winX = 0; winX < m; winX++)
          hist[ bufferIn[(inY+winY-1)*inputSize[0]+winX] ]++;

      k = 0;
      sum = 0;
      while(sum <= th)
      {
        sum += hist[k];
        k++;
      }
      median = k;
      if(n%2)
        bufferOut[inY*inputSize[0]]   = k;
      bufferOut[inY*inputSize[0]+1] = k;
      ltmedian = sum;

      for(inX = 2; inX < (inputSize[0]-1); inX++)
      {
        // leftmost x position
        lx = inX-2;
        // rightmost x position
        rx = inX+1;
        for(kr = 0; kr < n; kr++)
        {
          g1 = bufferIn[(inY-1+kr)*inputSize[0] + lx];
          hist[g1]--;
          if(g1 < median)
            --ltmedian;

          g1 = bufferIn[(inY-1+kr)*inputSize[0] + rx];
          hist[g1]++;
          if(g1 < median)
            ++ltmedian;
        }

        if(ltmedian > th)
        {
          --median;
          ltmedian -= hist[median];
          while((ltmedian>th) && (median>1))
          {
            --median;
            ltmedian -= hist[median];
          }
        }
        else
        {
          while((ltmedian + (int)hist[median]) <= th)
          {
            ltmedian += hist[median];
            ++median;
          }
        }
        bufferOut[inY*inputSize[0]+inX] = median;
      }
      if(n%2)
        bufferOut[inY*inputSize[0]+inputSize[0]-1] = median;

      // Clear histogram
      for (winY = 0; winY < n; winY++)
        for (winX = 0; winX < m; winX++)
        {
          g1 = bufferIn[(inY+winY-1)*inputSize[0] + winX + (inputSize[0]-m)];
          hist[g1]--;
        }
    }
  }

  else
  {
    itkExceptionMacro(<< "Median Window mismatch! Current Window: "
                      << m_MedianWindow[0] << "x"
                      << m_MedianWindow[1] << " "
                      << "accepted modes [3x3] and [3x2]")
  }
}

} // end namespace rtk

#endif // __rtkMedianImageFilter_cxx
