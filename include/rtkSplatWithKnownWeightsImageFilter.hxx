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
#ifndef rtkSplatWithKnownWeightsImageFilter_hxx
#define rtkSplatWithKnownWeightsImageFilter_hxx

#include "rtkSplatWithKnownWeightsImageFilter.h"

#include "itkObjectFactory.h"
#include "itkImageRegionIterator.h"
#include "itkImageRegionConstIterator.h"

namespace rtk
{

template< typename VolumeSeriesType, typename VolumeType>
SplatWithKnownWeightsImageFilter<VolumeSeriesType, VolumeType>::SplatWithKnownWeightsImageFilter()
{
  this->SetNumberOfRequiredInputs(2);
  this->SetInPlace(true);
  m_ProjectionNumber = 0;

#if ITK_VERSION_MAJOR > 4 || (ITK_VERSION_MAJOR == 4 && ITK_VERSION_MINOR >= 4)
  // Set the direction along which the output requested region should NOT be split
  m_Splitter = itk::ImageRegionSplitterDirection::New();
  m_Splitter->SetDirection(VolumeSeriesType::ImageDimension - 1);
#else
  // Old versions of ITK (before 4.4) do not have the ImageRegionSplitterDirection
  // and should run this filter with only one thread
  this->SetNumberOfThreads(1);
#endif
}

template< typename VolumeSeriesType, typename VolumeType>
void SplatWithKnownWeightsImageFilter<VolumeSeriesType, VolumeType>::SetInputVolumeSeries(const VolumeSeriesType* VolumeSeries)
{
  this->SetNthInput(0, const_cast<VolumeSeriesType*>(VolumeSeries));
}

template< typename VolumeSeriesType, typename VolumeType>
void SplatWithKnownWeightsImageFilter<VolumeSeriesType, VolumeType>::SetInputVolume(const VolumeType* Volume)
{
  this->SetNthInput(1, const_cast<VolumeType*>(Volume));
}

template< typename VolumeSeriesType, typename VolumeType>
typename VolumeSeriesType::ConstPointer SplatWithKnownWeightsImageFilter<VolumeSeriesType, VolumeType>::GetInputVolumeSeries()
{
  return static_cast< const VolumeSeriesType * >
          ( this->itk::ProcessObject::GetInput(0) );
}

template< typename VolumeSeriesType, typename VolumeType>
typename VolumeType::Pointer SplatWithKnownWeightsImageFilter<VolumeSeriesType, VolumeType>::GetInputVolume()
{
  return static_cast< VolumeType * >
          ( this->itk::ProcessObject::GetInput(1) );
}

template< typename VolumeType, typename VolumeSeriesType>
void SplatWithKnownWeightsImageFilter<VolumeType, VolumeSeriesType>::SetProjectionNumber(int n)
{
  // Check whether the weights change from the old projection number to the new one
  for (unsigned int row=0; row < m_Weights.rows(); row++)
    {
    if (m_Weights[row][m_ProjectionNumber] != m_Weights[row][n])
      this->Modified();
    }

  // Change the member variable whatever the result
  m_ProjectionNumber = n;
}

#if ITK_VERSION_MAJOR > 4 || (ITK_VERSION_MAJOR == 4 && ITK_VERSION_MINOR >= 4)
  template< typename VolumeSeriesType, typename VolumeType>
  const itk::ImageRegionSplitterBase*
  SplatWithKnownWeightsImageFilter< VolumeSeriesType, VolumeType >
  ::GetImageRegionSplitter(void) const
  {
    return m_Splitter;
  }
#endif

template< typename VolumeSeriesType, typename VolumeType>
void SplatWithKnownWeightsImageFilter<VolumeSeriesType, VolumeType>
::BeforeThreadedGenerateData()
{
#if !(ITK_VERSION_MAJOR > 4 || (ITK_VERSION_MAJOR == 4 && ITK_VERSION_MINOR >= 4))
  if (this->GetNumberOfThreads() > 1)
    {
    itkWarningMacro(<< "Splat filter cannot use multiple threads with ITK versions older than v4.4. Reverting to single thread behavior");
    this->SetNumberOfThreads(1);
    }
#endif
}

template< typename VolumeSeriesType, typename VolumeType>
void SplatWithKnownWeightsImageFilter<VolumeSeriesType, VolumeType>
::ThreadedGenerateData(const typename VolumeSeriesType::RegionType& outputRegionForThread, itk::ThreadIdType itkNotUsed(threadId))
{
  typename VolumeType::Pointer volume = this->GetInputVolume();

  unsigned int Dimension = volume->GetImageDimension();

  typename VolumeType::RegionType volumeRegion;
  typename VolumeType::SizeType volumeSize;
  typename VolumeType::IndexType volumeIndex;

  typename VolumeSeriesType::RegionType volumeSeriesRegion;

  float weight;

  // Initialize output region with input region in case the filter is not in
  // place
  if(this->GetInput() != this->GetOutput() )
    {
    itk::ImageRegionIterator<VolumeSeriesType>        itOut(this->GetOutput(), outputRegionForThread);
    itk::ImageRegionConstIterator<VolumeSeriesType>   itIn(this->GetInputVolumeSeries(), outputRegionForThread);
    while(!itOut.IsAtEnd())
      {
      itOut.Set(itIn.Get());
      ++itOut;
      ++itIn;
      }
    }

  // Update each phase
  for (unsigned int phase=0; phase<m_Weights.rows(); phase++)
    {

    weight = m_Weights[phase][m_ProjectionNumber];
    volumeRegion = volume->GetLargestPossibleRegion();

    for (unsigned int i=0; i<Dimension; i++)
      {
      volumeSize[i] = outputRegionForThread.GetSize()[i];
      volumeIndex[i] = outputRegionForThread.GetIndex()[i];
      }
    volumeRegion.SetSize(volumeSize);
    volumeRegion.SetIndex(volumeIndex);

    volumeSeriesRegion = outputRegionForThread;
    volumeSeriesRegion.SetSize(Dimension, 1);
    volumeSeriesRegion.SetIndex(Dimension, phase);

    itk::ImageRegionIterator<VolumeSeriesType> outputIterator(this->GetOutput(), volumeSeriesRegion);
    itk::ImageRegionIterator<VolumeType> volumeIterator(volume, volumeRegion);

    while(!volumeIterator.IsAtEnd())
      {
      outputIterator.Set(outputIterator.Get() + weight * volumeIterator.Get());
      ++volumeIterator;
      ++outputIterator;
      }

    }

}

}// end namespace


#endif
