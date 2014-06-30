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
#ifndef __rtkSplatWithKnownWeightsImageFilter_txx
#define __rtkSplatWithKnownWeightsImageFilter_txx

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

template< typename VolumeSeriesType, typename VolumeType>
unsigned int  SplatWithKnownWeightsImageFilter<VolumeSeriesType, VolumeType>
::SplitRequestedRegion(unsigned int i, unsigned int num, OutputImageRegionType &splatRegion)
{
  // Get the output pointer
  VolumeSeriesType *outputPtr = this->GetOutput();

  const typename VolumeSeriesType::SizeType & requestedRegionSize =
          outputPtr->GetRequestedRegion().GetSize();


  // Initialize the splatRegion to the output requested region
  splatRegion = outputPtr->GetRequestedRegion();
  typename VolumeSeriesType::IndexType splatIndex = splatRegion.GetIndex();
  typename VolumeSeriesType::SizeType splatSize = splatRegion.GetSize();

  // splat on the innermost dimension available
  unsigned int splatAxis = 0;
  while ( requestedRegionSize[splatAxis] == 1 )
    {
    if ( splatAxis == outputPtr->GetImageDimension()-1 )
      { // cannot splat
      itkDebugMacro("  Cannot Splat");
      return 1;
      }
    ++splatAxis;
    }

  // determine the actual number of pieces that will be generated
  const double range=static_cast<double>(requestedRegionSize[splatAxis]);
  const unsigned int valuesPerThread = itk::Math::Ceil< unsigned int >(range / static_cast<double>(num));
  const unsigned int maxThreadIdUsed = itk::Math::Ceil< unsigned int >(range / static_cast<double>(valuesPerThread)) - 1;

  // Splat the region
  if ( i < maxThreadIdUsed )
    {
    splatIndex[splatAxis] += i * valuesPerThread;
    splatSize[splatAxis] = valuesPerThread;
    }
  if ( i == maxThreadIdUsed )
    {
    splatIndex[splatAxis] += i * valuesPerThread;
    // last thread needs to process the "rest" dimension being splat
    splatSize[splatAxis] = splatSize[splatAxis] - i * valuesPerThread;
    }

  // set the splat region ivars
  splatRegion.SetIndex(splatIndex);
  splatRegion.SetSize(splatSize);

  itkDebugMacro("  Splat Piece: " << splatRegion);

  return maxThreadIdUsed + 1;
}

template< typename VolumeSeriesType, typename VolumeType>
void SplatWithKnownWeightsImageFilter<VolumeSeriesType, VolumeType>
::ThreadedGenerateData(const typename VolumeSeriesType::RegionType& outputRegionForThread, itk::ThreadIdType itkNotUsed(threadId))
{

  //    typename VolumeSeriesType::Pointer output = this->GetOutput();
  //    typename VolumeSeriesType::ConstPointer volumeSeries = this->GetInputVolumeSeries();
  typename VolumeType::Pointer volume = this->GetInputVolume();

  unsigned int Dimension = volume->GetImageDimension();

  typename VolumeType::RegionType volumeRegion;
  typename VolumeType::SizeType volumeSize;
  typename VolumeType::IndexType volumeIndex;

  typename VolumeSeriesType::RegionType volumeSeriesRegion;

  float weight;

  // Update each phase
  for (unsigned int phase=0; phase<m_Weights.rows(); phase++)
    {

    weight = m_Weights[phase][m_ProjectionNumber];
    if (weight != 0)
      {
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

}

}// end namespace


#endif
