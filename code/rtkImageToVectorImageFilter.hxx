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
#ifndef rtkImageToVectorImageFilter_hxx
#define rtkImageToVectorImageFilter_hxx

#include "rtkImageToVectorImageFilter.h"

#include <itkObjectFactory.h>
#include <itkImageRegionIterator.h>
#include <itkImageRegionConstIterator.h>
#include <itkVariableLengthVector.h>

namespace rtk
{

template< typename InputImageType, typename OutputImageType>
ImageToVectorImageFilter<InputImageType, OutputImageType>::ImageToVectorImageFilter()
{
  m_NumberOfChannels = 1;

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
  ImageToVectorImageFilter< InputImageType, OutputImageType >
  ::GetImageRegionSplitter(void) const
  {
    return m_Splitter;
  }
#endif

template< typename InputImageType, typename OutputImageType>
void ImageToVectorImageFilter<InputImageType, OutputImageType>
::GenerateOutputInformation()
{
  const unsigned int InputDimension = InputImageType::ImageDimension;
  const unsigned int OutputDimension = OutputImageType::ImageDimension;

  if (!( (OutputDimension == InputDimension) || ((OutputDimension + 1) == InputDimension) ))
    itkGenericExceptionMacro(<< "In ImageToVectorImageFilter : input and output image dimensions are not compatible");

  typename OutputImageType::SizeType outputSize;
  typename OutputImageType::IndexType outputIndex;
  typename OutputImageType::RegionType outputRegion;
  typename OutputImageType::SpacingType outputSpacing;
  typename OutputImageType::PointType outputOrigin;
  typename OutputImageType::DirectionType outputDirection;

  // In all cases
  for (unsigned int dim=0; dim < OutputDimension; dim++)
    {
    outputSize[dim] = this->GetInput()->GetLargestPossibleRegion().GetSize()[dim];
    outputIndex[dim] = this->GetInput()->GetLargestPossibleRegion().GetIndex()[dim];
    outputSpacing[dim] = this->GetInput()->GetSpacing()[dim];
    outputOrigin[dim] = this->GetInput()->GetOrigin()[dim];
    for (unsigned int j=0; j < OutputDimension; j++)
      outputDirection[dim][j] = this->GetInput()->GetDirection()[dim][j];
    }

  // Set vector length and size of output, depending on the input and output dimensions
  if ((OutputDimension + 1) == InputDimension)
    {
    // No need to set the size, it should already be correct, thanks to the previous block of code
    m_NumberOfChannels = this->GetInput()->GetLargestPossibleRegion().GetSize()[InputDimension - 1];
    this->GetOutput()->SetVectorLength(m_NumberOfChannels);
    }
  else
    {
    // Can't guess vector length, must use the user-provided one
    outputSize[OutputDimension - 1] = this->GetInput()->GetLargestPossibleRegion().GetSize()[OutputDimension - 1] / this->m_NumberOfChannels;
    this->GetOutput()->SetVectorLength(this->m_NumberOfChannels);
    }

  outputRegion.SetSize(outputSize);
  outputRegion.SetIndex(outputIndex);
  this->GetOutput()->SetLargestPossibleRegion(outputRegion);
  this->GetOutput()->SetSpacing(outputSpacing);
  this->GetOutput()->SetOrigin(outputOrigin);
  this->GetOutput()->SetDirection(outputDirection);
}

template< typename InputImageType, typename OutputImageType>
void ImageToVectorImageFilter<InputImageType, OutputImageType>
::GenerateInputRequestedRegion()
{
  typename InputImageType::Pointer  inputPtr  = const_cast<InputImageType *>(this->GetInput());
  inputPtr->SetRequestedRegionToLargestPossibleRegion();
}

template< typename InputImageType, typename OutputImageType>
void ImageToVectorImageFilter<InputImageType, OutputImageType>
::BeforeThreadedGenerateData()
{
#if !(ITK_VERSION_MAJOR > 4 || (ITK_VERSION_MAJOR == 4 && ITK_VERSION_MINOR >= 4))
  if (this->GetNumberOfThreads() > 1)
    {
    itkWarningMacro(<< "ImageToVectorImage filter cannot use multiple threads with ITK versions older than v4.4. Reverting to single thread behavior");
    this->SetNumberOfThreads(1);
    }
#endif
}

template< typename InputImageType, typename OutputImageType>
void ImageToVectorImageFilter<InputImageType, OutputImageType>
::ThreadedGenerateData(const OutputImageRegionType& outputRegionForThread, itk::ThreadIdType itkNotUsed(threadId))
{
  const unsigned int InputDimension = InputImageType::ImageDimension;
  const unsigned int OutputDimension = OutputImageType::ImageDimension;

  typename InputImageType::SizeType inputSize;
  typename InputImageType::IndexType inputIndex;
  typename InputImageType::RegionType inputRegion;

  inputSize.Fill(0);
  inputIndex.Fill(0);
  for (unsigned int dim=0; dim < OutputDimension; dim++)
    {
    inputSize[dim] = outputRegionForThread.GetSize()[dim];
    inputIndex[dim] = outputRegionForThread.GetIndex()[dim];
    }

  // In all cases, the input's last dimension should be fully traversed
  inputSize[InputDimension - 1] = this->GetInput()->GetLargestPossibleRegion().GetSize()[InputDimension -1];

  inputRegion.SetSize(inputSize);
  inputRegion.SetIndex(inputIndex);

  // Actual copy is the same in both cases
  itk::ImageRegionConstIterator<InputImageType> inIt(this->GetInput(), inputRegion);
  itk::ImageRegionIterator<OutputImageType> outIt(this->GetOutput(), outputRegionForThread);
  for (unsigned int channel=0; channel < this->m_NumberOfChannels; channel++)
    {
    outIt.GoToBegin();
    while(!outIt.IsAtEnd())
      {
      itk::VariableLengthVector<typename InputImageType::PixelType> vector = outIt.Get();
      vector[channel] = inIt.Get();
      outIt.Set(vector);
      ++inIt;
      ++outIt;
      }
    }
}

}// end namespace


#endif
