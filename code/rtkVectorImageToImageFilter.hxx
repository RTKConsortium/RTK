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
#ifndef rtkVectorImageToImageFilter_hxx
#define rtkVectorImageToImageFilter_hxx

#include "rtkVectorImageToImageFilter.h"

#include "itkObjectFactory.h"
#include "itkImageRegionIterator.h"
#include "itkImageRegionConstIterator.h"

namespace rtk
{

template< typename InputImageType, typename OutputImageType>
VectorImageToImageFilter<InputImageType, OutputImageType>::VectorImageToImageFilter()
{
#if ITK_VERSION_MAJOR > 4 || (ITK_VERSION_MAJOR == 4 && ITK_VERSION_MINOR >= 4)
  // Set the direction along which the output requested region should NOT be split
  m_Splitter = itk::ImageRegionSplitterDirection::New();
  m_Splitter->SetDirection(OutputImageType::ImageDimension - 1);
#else
  // Old versions of ITK (before 4.4) do not have the ImageRegionSplitterDirection
  // and should run this filter with only one thread
  this->SetNumberOfThreads(1);
#endif
}

#if ITK_VERSION_MAJOR > 4 || (ITK_VERSION_MAJOR == 4 && ITK_VERSION_MINOR >= 4)
  template< typename InputImageType, typename OutputImageType>
  const itk::ImageRegionSplitterBase*
  VectorImageToImageFilter< InputImageType, OutputImageType >
  ::GetImageRegionSplitter(void) const
  {
    return m_Splitter;
  }
#endif

template< typename InputImageType, typename OutputImageType>
void VectorImageToImageFilter<InputImageType, OutputImageType>
::GenerateOutputInformation()
{
  const unsigned int InputDimension = InputImageType::ImageDimension;
  const unsigned int OutputDimension = OutputImageType::ImageDimension;

  if (!( (OutputDimension == InputDimension) || (OutputDimension == (InputDimension + 1)) ))
    itkGenericExceptionMacro(<< "In VectorImageToImageFilter : input and output image dimensions are not compatible");

  typename OutputImageType::SizeType outputSize;
  typename OutputImageType::IndexType outputIndex;
  typename OutputImageType::RegionType outputRegion;
  typename OutputImageType::SpacingType outputSpacing;
  typename OutputImageType::PointType outputOrigin;
  typename OutputImageType::DirectionType outputDirection;

  // If the output has an additional dimension, set the parameters
  // for this new dimension. Fill() is used, but only the last
  // dimension will not be overwritten by the next part
  if (OutputDimension == (InputDimension + 1))
    {
    outputSize.Fill(this->GetInput()->GetVectorLength());
    outputIndex.Fill(0);
    outputSpacing.Fill(1);
    outputOrigin.Fill(0);
    outputDirection.SetIdentity();
    }

  // In all cases
  for (unsigned int dim=0; dim < InputDimension; dim++)
    {
    outputSize[dim] = this->GetInput()->GetLargestPossibleRegion().GetSize()[dim];
    outputIndex[dim] = this->GetInput()->GetLargestPossibleRegion().GetIndex()[dim];
    outputSpacing[dim] = this->GetInput()->GetSpacing()[dim];
    outputOrigin[dim] = this->GetInput()->GetOrigin()[dim];
    for (unsigned int j=0; j < InputDimension; j++)
      outputDirection[dim][j] = this->GetInput()->GetDirection()[dim][j];
    }

  // If dimensions match, overwrite the size on last dimension
  if (OutputDimension == InputDimension)
    {
    outputSize[OutputDimension - 1] = this->GetInput()->GetLargestPossibleRegion().GetSize()[OutputDimension - 1] * this->GetInput()->GetVectorLength();
    }

  outputRegion.SetSize(outputSize);
  outputRegion.SetIndex(outputIndex);
  this->GetOutput()->SetLargestPossibleRegion(outputRegion);
  this->GetOutput()->SetSpacing(outputSpacing);
  this->GetOutput()->SetOrigin(outputOrigin);
  this->GetOutput()->SetDirection(outputDirection);
}

template< typename InputImageType, typename OutputImageType>
void VectorImageToImageFilter<InputImageType, OutputImageType>
::GenerateInputRequestedRegion()
{
  typename InputImageType::Pointer  inputPtr  = const_cast<InputImageType *>(this->GetInput());
  inputPtr->SetRequestedRegionToLargestPossibleRegion();
}

template< typename InputImageType, typename OutputImageType>
void VectorImageToImageFilter<InputImageType, OutputImageType>
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

template< typename InputImageType, typename OutputImageType>
void VectorImageToImageFilter<InputImageType, OutputImageType>
::ThreadedGenerateData(const OutputImageRegionType& outputRegionForThread, itk::ThreadIdType itkNotUsed(threadId))
{
  const unsigned int InputDimension = InputImageType::ImageDimension;
  const unsigned int OutputDimension = OutputImageType::ImageDimension;

  typename InputImageType::SizeType inputSize;
  typename InputImageType::IndexType inputIndex;
  typename InputImageType::RegionType inputRegion;

  for (unsigned int dim=0; dim < InputDimension; dim++)
    {
    inputSize[dim] = outputRegionForThread.GetSize()[dim];
    inputIndex[dim] = outputRegionForThread.GetIndex()[dim];
    }
  if (OutputDimension == InputDimension)
    {
    inputSize[OutputDimension - 1] = this->GetInput()->GetLargestPossibleRegion().GetSize()[OutputDimension -1];
    inputIndex[OutputDimension - 1] = this->GetInput()->GetLargestPossibleRegion().GetIndex()[OutputDimension -1];
    }
  inputRegion.SetSize(inputSize);
  inputRegion.SetIndex(inputIndex);

  // Actual copy is the same in both cases
  itk::ImageRegionConstIterator<InputImageType> inIt(this->GetInput(), inputRegion);
  itk::ImageRegionIterator<OutputImageType> outIt(this->GetOutput(), outputRegionForThread);
  for (unsigned int channel=0; channel < this->GetInput()->GetVectorLength(); channel++)
    {
    inIt.GoToBegin();
    while(!inIt.IsAtEnd())
      {
      outIt.Set(inIt.Get()[channel]);
      ++inIt;
      ++outIt;
      }
    }
}

}// end namespace


#endif
