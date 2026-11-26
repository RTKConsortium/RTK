/*=========================================================================
 *
 *  Copyright RTK Consortium
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *         https://www.apache.org/licenses/LICENSE-2.0.txt
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
#include <itkInputDataObjectIterator.h>

namespace rtk
{

template <class TInputImage, class TOutputImage>
PolynomialGainCorrectionImageFilter<TInputImage, TOutputImage>::PolynomialGainCorrectionImageFilter() = default;

template <class TInputImage, class TOutputImage>
void
PolynomialGainCorrectionImageFilter<TInputImage, TOutputImage>::VerifyPreconditions() const
{
  this->Superclass::VerifyPreconditions();

  if (m_K != 0. && (this->GetDarkImage() == nullptr || this->GetGainCoefficients() == nullptr))
    itkExceptionMacro(<< "DarkImage and GainCoefficients must be set.");
}

template <class TInputImage, class TOutputImage>
void
PolynomialGainCorrectionImageFilter<TInputImage, TOutputImage>::GenerateOutputInformation()
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
  // if (TInputImage::PixelType != USHORT)
  //{
  //    itkWarningMacro(<<"Polynomial gain calibration only allow unsigned short pixel format as input" );
  //    m_K = 0.0; // To disable processing
  //}

  // TInputImage::PixelType
  if (m_K != 0.0)
  {
    m_GainSize = this->GetGainCoefficients()->GetLargestPossibleRegion().GetSize();
    m_VModelOrder = m_GainSize[2];

    // Create power LUT: the values for the different orders for the same pixel value are close to each other
    // The power LUT only depends on the model order
    constexpr int npixValues = 65536; // Input values are 16-bit unsigned
    if (m_PowerLut.size() != static_cast<size_t>(npixValues * m_VModelOrder))
    {
      m_PowerLut.clear();
      for (int pid = 0; pid < npixValues; ++pid)
      {
        auto value = static_cast<float>(pid);
        for (int order = 0; order < m_VModelOrder; ++order)
        {
          m_PowerLut.push_back(value);
          value = value * value;
        }
      }
    }
  }
}

template <class TInputImage, class TOutputImage>
void
PolynomialGainCorrectionImageFilter<TInputImage, TOutputImage>::GenerateInputRequestedRegion()
{
  // The whole projections, dark image and gain coefficients are required
  for (itk::InputDataObjectIterator it(this); !it.IsAtEnd(); ++it)
    it.GetInput()->SetRequestedRegionToLargestPossibleRegion();
}

template <class TInputImage, class TOutputImage>
void
PolynomialGainCorrectionImageFilter<TInputImage, TOutputImage>::DynamicThreadedGenerateData(
  const OutputImageRegionType & outputRegionForThread)
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
  itk::ImageRegionConstIterator<InputImageType> itDark(this->GetDarkImage(), darkRegion);

  // Get gain map buffer
  const typename OutputImageType::PixelType * gainBuffer = this->GetGainCoefficients()->GetBufferPointer();

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
        int   lutidx = px * m_VModelOrder;
        for (int m = 0; m < m_VModelOrder; ++m)
        {
          int   gainidx = m * m_GainSize[1] * m_GainSize[0] + j * m_GainSize[0] + i;
          float gainM = gainBuffer[gainidx];
          correctedValue += gainM * m_PowerLut[lutidx + m];
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

} // namespace rtk

#endif
