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
#ifndef rtkMultiplyByVectorImageFilter_hxx
#define rtkMultiplyByVectorImageFilter_hxx

#include "rtkMultiplyByVectorImageFilter.h"
#include "itkImageRegionIterator.h"


namespace rtk
{
//
// Constructor
//
template< class TInputImage >
MultiplyByVectorImageFilter< TInputImage >
::MultiplyByVectorImageFilter()
{
}

template< class TInputImage >
void
MultiplyByVectorImageFilter< TInputImage >
::SetVector(std::vector<float> vect)
{
  m_Vector = vect;
  this->Modified();
}

template< class TInputImage >
void
MultiplyByVectorImageFilter< TInputImage >
::ThreadedGenerateData(const typename TInputImage::RegionType& outputRegionForThread, itk::ThreadIdType itkNotUsed(threadId))
{
  int Dimension = this->GetInput()->GetImageDimension();

  for (unsigned int i=outputRegionForThread.GetIndex()[Dimension-1];
       i<outputRegionForThread.GetSize()[Dimension-1]
          + outputRegionForThread.GetIndex()[Dimension-1]; i++)
    {
    typename TInputImage::RegionType SubRegion;
    SubRegion=outputRegionForThread;
    SubRegion.SetSize(Dimension-1, 1);
    SubRegion.SetIndex(Dimension-1, i);

    itk::ImageRegionIterator<TInputImage> outputIterator(this->GetOutput(), SubRegion);
    itk::ImageRegionConstIterator<TInputImage> inputIterator(this->GetInput(), SubRegion);

    while(!inputIterator.IsAtEnd())
      {
      outputIterator.Set(inputIterator.Get() * m_Vector[i]);
      ++outputIterator;
      ++inputIterator;
      }
    }
}

} // end namespace itk

#endif
