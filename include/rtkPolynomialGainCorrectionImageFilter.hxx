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

#ifndef rtkPolynomialGainCorrectionImageFilter_hxx
#define rtkPolynomialGainCorrectionImageFilter_hxx

#include <itkImageFileReader.h>
#include <itkImageRegionConstIterator.h>
#include <itkImageRegionIterator.h>

namespace rtk
{

template<class TInputImage, class TOutputImage>
PolynomialGainCorrectionImageFilter<TInputImage, TOutputImage>
::PolynomialGainCorrectionImageFilter():
    m_MapsLoaded(false),
    m_ModelOrder(1),
    m_K(1.0f)
{
}

template<class TInputImage, class TOutputImage>
void PolynomialGainCorrectionImageFilter<TInputImage, TOutputImage>
::SetDarkImage(const InputImagePointer darkImage)
{
  m_DarkImage = darkImage;
}

template<class TInputImage, class TOutputImage>
void
PolynomialGainCorrectionImageFilter<TInputImage, TOutputImage>
::SetGainCoefficients(const OutputImagePointer gain)
{
  m_GainImage = gain;
}

template<class TInputImage, class TOutputImage>
void
PolynomialGainCorrectionImageFilter<TInputImage, TOutputImage>
::GenerateOutputInformation()
{
  // get pointers to the input and output
  InputImagePointer  inputPtr = const_cast<InputImageType *>(this->GetInput());
  OutputImagePointer outputPtr = this->GetOutput();

  if (!outputPtr || !inputPtr)
    {
    return;
    }

  // Copy the meta data for this data type
  outputPtr->SetSpacing(inputPtr->GetSpacing());
  outputPtr->SetOrigin(inputPtr->GetOrigin());
  outputPtr->SetDirection(inputPtr->GetDirection());
  outputPtr->SetNumberOfComponentsPerPixel(inputPtr->GetNumberOfComponentsPerPixel());

  InputImageRegionType outputLargestPossibleRegion;
  outputLargestPossibleRegion = inputPtr->GetLargestPossibleRegion();
  outputPtr->SetRegions(outputLargestPossibleRegion);

  // TODO: Do something if input not unsigned 16-bits
  //if (TInputImage::PixelType != USHORT)
  //{
  //    itkWarningMacro(<<"Polynomial gain calibration only allow unsigned short pixel format as input" );
  //    m_K = 0.0; // To disable processing
  //}

  //TInputImage::PixelType
  if (!m_MapsLoaded && m_K != 0.0)
    {
    m_GainSize = m_GainImage->GetLargestPossibleRegion().GetSize();

    m_ModelOrder = m_GainSize[2];
    m_MapsLoaded = true;

    // Create power LUT: the values for the different orders for the same pixel value are close to each other
    int npixValues = 65536;   // Input values are 16-bit unsigned
    for (int pid = 0; pid < npixValues; ++pid)
      {
      float value = static_cast<float>(pid);
      for (int order = 0; order < m_ModelOrder; ++order)
        {
        m_PowerLut.push_back(value);
        value = value*value;
        }
      }
    }
}

template<class TInputImage, class TOutputImage>
void
PolynomialGainCorrectionImageFilter<TInputImage, TOutputImage>
::GenerateInputRequestedRegion()
{
  typename Superclass::InputImagePointer  inputPtr = const_cast< InputImageType * >(this->GetInput());
  typename Superclass::OutputImagePointer outputPtr = this->GetOutput();

  if (!inputPtr || !outputPtr)
    return;

  InputImageRegionType inputRequestedRegion = outputPtr->GetLargestPossibleRegion();
  inputRequestedRegion.Crop(inputPtr->GetLargestPossibleRegion());     // Because Largest region has been updated
  inputPtr->SetRegions(inputRequestedRegion);
}

template<class TInputImage, class TOutputImage>
void
PolynomialGainCorrectionImageFilter<TInputImage, TOutputImage>
::ThreadedGenerateData(const OutputImageRegionType& outputRegionForThread, itk::ThreadIdType itkNotUsed(threadId))
{
  itk::ImageRegionConstIterator<InputImageType> itIn(this->GetInput(), outputRegionForThread);
  itk::ImageRegionIterator<OutputImageType>     itOut(this->GetOutput(), outputRegionForThread);
  itIn.GoToBegin();
  itOut.GoToBegin();

  // K==0 = no weighting requested
  if (m_K == 0.)
    {
    while (!itIn.IsAtEnd())
      {
      itOut.Set(static_cast<typename TOutputImage::PixelType>(itIn.Get()));
      ++itIn;
      ++itOut;
      }
    return;
    }

  InputImageRegionType darkRegion = outputRegionForThread;
  darkRegion.SetSize(2, 1);
  darkRegion.SetIndex(2, 0);
  itk::ImageRegionConstIterator<InputImageType> itDark(m_DarkImage, darkRegion);

  // Get gain map buffer
  const float *gainBuffer = m_GainImage->GetBufferPointer();

  int startk = static_cast<int>(outputRegionForThread.GetIndex(2));
  for (int k = startk; k < startk + static_cast<int>(outputRegionForThread.GetSize(2)); k++)
    {
    itDark.GoToBegin();

    int startj = outputRegionForThread.GetIndex(1);
    for (int j = startj; j < startj + static_cast<int>(outputRegionForThread.GetSize(1)); j++)
      {
      int starti = outputRegionForThread.GetIndex(0);
      for (int i = starti; i < starti + static_cast<int>(outputRegionForThread.GetSize(0)); i++)
        {
        int darkx = static_cast<int>(itDark.Get());
        int px = static_cast<int>(itIn.Get()) - darkx;
        px = (px >= 0) ? px : 0;

        float correctedValue = 0.f;
        int lutidx = px*m_ModelOrder;
        for (int m = 0; m < m_ModelOrder; ++m)
          {
          int gainidx = m*m_GainSize[1]*m_GainSize[0]+j*m_GainSize[0] + i;
          float gainM = gainBuffer[gainidx];
          correctedValue += gainM *m_PowerLut[lutidx + m];
          }
        correctedValue *= m_K;

        itOut.Set(static_cast<typename TOutputImage::PixelType>(correctedValue));

        ++itIn;
        ++itOut;
        ++itDark;
        }
      }
    }
}

}

#endif
