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

#ifndef rtkLagCorrectionImageFilter_hxx
#define rtkLagCorrectionImageFilter_hxx

#include "itkImageRegionIterator.h"
#include "itkImageRegionConstIterator.h"
#include <iterator>

namespace rtk
{

template<typename TImage, unsigned ModelOrder>
LagCorrectionImageFilter<TImage, ModelOrder>::LagCorrectionImageFilter()
{
  this->SetNumberOfRequiredInputs(1);
  m_A.Fill(0.0f);
  m_B.Fill(0.0f);
  m_ExpmA.Fill(0.0f);
  m_NewParamJustReceived = false;
}

template<typename TImage, unsigned ModelOrder>
void LagCorrectionImageFilter<TImage, ModelOrder>
::GenerateOutputInformation()
{
  // get pointers to the input and output
  typename TImage::Pointer  inputPtr = const_cast<TImage *>(this->GetInput());
  typename TImage::Pointer outputPtr = this->GetOutput();

  if (!outputPtr || !inputPtr) {
    return;
  }

  this->SetInPlace(false);

  // Copy the meta data for this data type
  outputPtr->SetSpacing(inputPtr->GetSpacing());
  outputPtr->SetOrigin(inputPtr->GetOrigin());
  outputPtr->SetDirection(inputPtr->GetDirection());
  outputPtr->SetNumberOfComponentsPerPixel(inputPtr->GetNumberOfComponentsPerPixel());

  typename TImage::RegionType outputLargestPossibleRegion;
  outputLargestPossibleRegion = inputPtr->GetLargestPossibleRegion();

  // Compute the X coordinates of the corners of the image (images offset incl. in origin?)
  typename Superclass::InputImageType::PointType origin = inputPtr->GetOrigin();

  outputPtr->SetOrigin(origin);
  outputPtr->SetRegions(outputLargestPossibleRegion);
  outputPtr->SetSpacing(inputPtr->GetSpacing());

  // Initialization at first frame
  if (m_NewParamJustReceived && (m_B[0] != 0.f))
  {
    m_SumB = 1.f;
    for (unsigned int n = 0; n < ModelOrder; n++) {
      m_ExpmA[n] = expf(-m_A[n]);
      m_SumB += m_B[n];
    }

    m_StartIdx = this->GetInput()->GetLargestPossibleRegion().GetIndex();
    ImageSizeType SizeInput = this->GetInput()->GetLargestPossibleRegion().GetSize();
    m_S.assign(SizeInput[0] * SizeInput[1] * ModelOrder, 0.f);
    m_NewParamJustReceived = false;
  }
}

template<typename TImage, unsigned ModelOrder>
void LagCorrectionImageFilter<TImage, ModelOrder>
::GenerateInputRequestedRegion()
{
  typename Superclass::InputImagePointer  inputPtr = const_cast< TImage * >(this->GetInput());
  typename Superclass::OutputImagePointer outputPtr = this->GetOutput();

  if (!inputPtr || !outputPtr)
    return;

  typename TImage::RegionType inputRequestedRegion = outputPtr->GetRequestedRegion();
  inputRequestedRegion.Crop(inputPtr->GetLargestPossibleRegion());     // Because Largest region has been updated
  inputPtr->SetRequestedRegion(inputRequestedRegion);
}

template<typename TImage, unsigned ModelOrder>
void LagCorrectionImageFilter<TImage, ModelOrder>::
ThreadedGenerateData(const ImageRegionType & thRegion, itk::ThreadIdType threadId)
{
  // Input / ouput iterators
  itk::ImageRegionConstIterator<TImage> itIn(this->GetInput(), thRegion);
  itk::ImageRegionIterator<TImage>     itOut(this->GetOutput(), thRegion);

  itIn.GoToBegin();
  itOut.GoToBegin();

  if (m_B[0] == 0.f) {
    while (!itIn.IsAtEnd())
    {
      itOut.Set(itIn.Get());
      ++itIn;
      ++itOut;
    }
    return;
  }

  ImageSizeType SizeInput = this->GetInput()->GetLargestPossibleRegion().GetSize();

  for (unsigned int k = 0; k < thRegion.GetSize(2); ++k)
  {
    unsigned int jj = (thRegion.GetIndex()[1] - m_StartIdx[1]) * SizeInput[0];
    for (unsigned int j = 0; j < thRegion.GetSize(1); ++j)
    {
      unsigned int ii = thRegion.GetIndex()[0] - m_StartIdx[0];
      for (unsigned int i = 0; i < thRegion.GetSize(0); ++i, ++ii)
      {
        unsigned idx_s = (jj + ii)*ModelOrder;

        // Get measured pixel value y[k]
        float yk = static_cast<float>(itIn.Get());

        // Computes correction
        float xk = yk;         // Initial corrected pixel
        VectorType Sa;         // Update of the state
        for (unsigned int n = 0; n < ModelOrder; n++)
        {
          // Compute the update of internal state for nth exponential
          Sa[n] = m_ExpmA[n] * m_S[idx_s + n];

          // Update x[k] by removing contribution of the nth exponential
          xk -= m_B[n] * Sa[n];
        }

        // Apply normalization factor
        xk = xk / m_SumB;

        // Update internal state Snk
        for (unsigned int n = 0; n < ModelOrder; n++) {
          m_S[idx_s + n] = xk + Sa[n];
        }

        // Avoid negative values
        xk = (xk < 0.0f) ? 0.f : xk;

        itOut.Set(static_cast<PixelType>(xk));

        ++itIn;
        ++itOut;
      }
      jj += SizeInput[0];
    }
  }
}

template<typename TImage, unsigned ModelOrder>
unsigned int LagCorrectionImageFilter<TImage, ModelOrder>
::SplitRequestedRegion(unsigned int i, unsigned int num, OutputImageRegionType& splitRegion)
{
  return SplitRequestedRegion((int)i, (int)num, splitRegion);
}

template<typename TImage, unsigned ModelOrder>
int LagCorrectionImageFilter<TImage, ModelOrder>
::SplitRequestedRegion(int i, int num, OutputImageRegionType& splitRegion)
{
  // Split along the "second" direction

  // Get the output pointer
  TImage * outputPtr = this->GetOutput();
  const typename TImage::SizeType& requestedRegionSize
    = outputPtr->GetRequestedRegion().GetSize();

  int splitAxis;
  typename TImage::IndexType splitIndex;
  typename TImage::SizeType splitSize;

  // Initialize the splitRegion to the output requested region
  splitRegion = outputPtr->GetRequestedRegion();
  splitIndex = splitRegion.GetIndex();
  splitSize = splitRegion.GetSize();

  // split on the outermost dimension available
  splitAxis = 1;
  if (requestedRegionSize[splitAxis] == 1) {
    splitAxis = 0;
  }

  // determine the actual number of pieces that will be generated
  typename TImage::SizeType::SizeValueType range = requestedRegionSize[splitAxis];
  int valuesPerThread = itk::Math::Ceil<int>(range / (double)num);
  int maxThreadIdUsed = itk::Math::Ceil<int>(range / (double)valuesPerThread) - 1;

  // Split the region
  if (i < maxThreadIdUsed) {
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
  splitRegion.SetIndex(splitIndex);
  splitRegion.SetSize(splitSize);

  return maxThreadIdUsed + 1;
}

}

#endif
