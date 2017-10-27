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

#ifndef rtkBoellaardScatterCorrectionImageFilter_hxx
#define rtkBoellaardScatterCorrectionImageFilter_hxx

#include <itkImageRegionIteratorWithIndex.h>

namespace rtk
{

template <class TInputImage, class TOutputImage>
BoellaardScatterCorrectionImageFilter<TInputImage, TOutputImage>
::BoellaardScatterCorrectionImageFilter():
m_AirThreshold(32000),
m_ScatterToPrimaryRatio(0.),
m_NonNegativityConstraintThreshold(20)
{
}

// Requires full projection images to estimate scatter.
template <class TInputImage, class TOutputImage>
void
BoellaardScatterCorrectionImageFilter<TInputImage, TOutputImage>
::EnlargeOutputRequestedRegion(itk::DataObject *)
{
  typename Superclass::OutputImagePointer outputPtr = this->GetOutput();
  if ( !outputPtr )
    return;

  const unsigned int Dimension = TInputImage::ImageDimension;
  typename TOutputImage::RegionType orr = outputPtr->GetRequestedRegion();
  typename TOutputImage::RegionType lpr = outputPtr->GetLargestPossibleRegion();

  for(unsigned int i=0; i<Dimension-1; i++)
    {
    orr.SetIndex( i, lpr.GetIndex(i) );
    orr.SetSize( i, lpr.GetSize(i) );
    }

  outputPtr->SetRequestedRegion( orr );
}

template <class TInputImage, class TOutputImage>
void
BoellaardScatterCorrectionImageFilter<TInputImage, TOutputImage>
::ThreadedGenerateData(const OutputImageRegionType& outputRegionForThread, ThreadIdType itkNotUsed(threadId) )
{
  // Input / ouput iterators
  itk::ImageRegionConstIterator<InputImageType> itIn(this->GetInput(), outputRegionForThread);
  itk::ImageRegionIterator<OutputImageType>     itOut(this->GetOutput(), outputRegionForThread);

  const unsigned int Dimension = TInputImage::ImageDimension;
  unsigned int npixelPerSlice = 1;
  for(unsigned int i=0; i<Dimension-1; i++)
    npixelPerSlice *= outputRegionForThread.GetSize(i);

  unsigned int start = outputRegionForThread.GetIndex(Dimension - 1);
  unsigned int stop = start + outputRegionForThread.GetSize(Dimension - 1);
  for (unsigned int slice = start; slice<stop; slice++)
    {
    itk::ImageRegionConstIterator<InputImageType> itInSlice = itIn;

    // Retrieve useful characteristics of current slice
    double averageBehindPatient = 0.;
    double smallestValue = itk::NumericTraits<double>::max();
    for(unsigned int i=0; i<npixelPerSlice; i++)
      {
      smallestValue = std::min(smallestValue, (double)itInSlice.Get() );
      if(itInSlice.Get()>=m_AirThreshold)
        {
        averageBehindPatient += itInSlice.Get();
        }
      ++itInSlice;
      }
    averageBehindPatient /= npixelPerSlice;

    // Compute constant correction
    double correction = averageBehindPatient * m_ScatterToPrimaryRatio;

    // Apply non-negativity constraint
    if(smallestValue-correction<m_NonNegativityConstraintThreshold)
      correction = smallestValue - m_NonNegativityConstraintThreshold;

    // Remove constant factor
    for(unsigned int i=0; i<npixelPerSlice; i++)
      {
      itOut.Set( itIn.Get() - correction );
      ++itIn;
      ++itOut;
      }
    }
}

template <class TInputImage, class TOutputImage>
unsigned int
BoellaardScatterCorrectionImageFilter<TInputImage, TOutputImage>
::SplitRequestedRegion(unsigned int i, unsigned int num, OutputImageRegionType& splitRegion)
{
  return SplitRequestedRegion((int)i, (int)num, splitRegion);
}

template <class TInputImage, class TOutputImage>
int
BoellaardScatterCorrectionImageFilter<TInputImage, TOutputImage>
::SplitRequestedRegion(int i, int num, OutputImageRegionType& splitRegion)
{
  // Get the output pointer
  OutputImageType * outputPtr = this->GetOutput();
  const typename TOutputImage::SizeType& requestedRegionSize
    = outputPtr->GetRequestedRegion().GetSize();

  int splitAxis;
  typename TOutputImage::IndexType splitIndex;
  typename TOutputImage::SizeType splitSize;

  // Initialize the splitRegion to the output requested region
  splitRegion = outputPtr->GetRequestedRegion();
  splitIndex = splitRegion.GetIndex();
  splitSize = splitRegion.GetSize();

  // split on the outermost dimension available
  splitAxis = outputPtr->GetImageDimension() - 1;
  if (requestedRegionSize[splitAxis] == 1)
    { // cannot split
    itkDebugMacro("  Cannot Split");
    return 1;
    }

  // determine the actual number of pieces that will be generated
  typename TOutputImage::SizeType::SizeValueType range = requestedRegionSize[splitAxis];
  int valuesPerThread = itk::Math::Ceil<int>(range/(double)num);
  int maxThreadIdUsed = itk::Math::Ceil<int>(range/(double)valuesPerThread) - 1;

  // Split the region
  if (i < maxThreadIdUsed)
    {
    splitIndex[splitAxis] += i*valuesPerThread;
    splitSize[splitAxis] = valuesPerThread;
    }
  if (i == maxThreadIdUsed)
    {
    splitIndex[splitAxis] += i*valuesPerThread;
    // last thread needs to process the "rest" dimension being split
    splitSize[splitAxis] = splitSize[splitAxis] - i*valuesPerThread;
    }

  // set the split region ivars
  splitRegion.SetIndex( splitIndex );
  splitRegion.SetSize( splitSize );

  itkDebugMacro("  Split Piece: " << splitRegion );

  return maxThreadIdUsed + 1;
}

} // end namespace rtk
#endif
