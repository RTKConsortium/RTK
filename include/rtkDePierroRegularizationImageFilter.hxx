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

#ifndef rtkDePierroRegularizationImageFilter_hxx
#define rtkDePierroRegularizationImageFilter_hxx

#include "rtkDePierroRegularizationImageFilter.h"

namespace rtk
{
template <class TInputImage, class TOutputImage>
DePierroRegularizationImageFilter<TInputImage, TOutputImage>::DePierroRegularizationImageFilter()
{
  this->SetNumberOfRequiredInputs(2);

  // Create each filter of the composite filter
  m_MultiplyConstant1ImageFilter = MultiplyImageFilterType::New();
  m_MultiplyConstant2ImageFilter = MultiplyImageFilterType::New();
  m_KernelImage = ConstantVolumeSourceType::New();
  m_DefaultNormalizationVolume = ConstantVolumeSourceType::New();
  m_SubtractImageFilter = SubtractImageFilterType::New();
  m_ConvolutionFilter = NOIFType::New();
  m_CustomBinaryFilter = CustomBinaryFilterType::New();

  // Set Lambda function
  auto customLambda = [](const typename InputImageType::PixelType & input1,
                         const typename InputImageType::PixelType & input2) -> typename OutputImageType::PixelType
  {
    return static_cast<typename OutputImageType::PixelType>((input1 + std::sqrt(input1 * input1 + input2)) / 2);
  };
  m_CustomBinaryFilter->SetFunctor(customLambda);

  // Permanent internal connections
  m_SubtractImageFilter->SetInput2(m_MultiplyConstant1ImageFilter->GetOutput());
  m_CustomBinaryFilter->SetInput1(m_SubtractImageFilter->GetOutput());
  m_CustomBinaryFilter->SetInput2(m_MultiplyConstant2ImageFilter->GetOutput());

  // Set the kernel image
  typename ConstantVolumeSourceType::PointType   origin;
  typename ConstantVolumeSourceType::SizeType    size;
  typename ConstantVolumeSourceType::SpacingType spacing;
  origin.Fill(-1);
  size.Fill(3);
  spacing.Fill(1);
  m_KernelImage->SetOrigin(origin);
  m_KernelImage->SetSpacing(spacing);
  m_KernelImage->SetSize(size);
  m_KernelImage->SetConstant(1.);
}

template <class TInputImage, class TOutputImage>
void
DePierroRegularizationImageFilter<TInputImage, TOutputImage>::GenerateInputRequestedRegion()
{
  // Input 0 is the k uptade of classic MLEM/OSEM algorithm
  typename TInputImage::Pointer inputPtr0 = const_cast<TInputImage *>(this->GetInput(0));
  if (!inputPtr0)
    return;
  inputPtr0->SetRequestedRegion(this->GetOutput()->GetRequestedRegion());

  // Input 0 is the k+1 uptade of classic MLEM/OSEM algorithm
  typename TInputImage::Pointer inputPtr1 = const_cast<TInputImage *>(this->GetInput(1));
  if (!inputPtr1)
    return;
  inputPtr1->SetRequestedRegion(inputPtr1->GetLargestPossibleRegion());

  // Input 3 is the normalization volume (optional)
  typename TInputImage::Pointer inputPtr3 = const_cast<TInputImage *>(this->GetInput(2));
  if (inputPtr3)
    inputPtr3->SetRequestedRegion(inputPtr0->GetRequestedRegion());
}

template <class TInputImage, class TOutputImage>
void
DePierroRegularizationImageFilter<TInputImage, TOutputImage>::GenerateOutputInformation()
{
  m_MultiplyConstant1ImageFilter->SetInput1(this->GetInput(0));
  m_MultiplyConstant2ImageFilter->SetInput1(this->GetInput(1));
  if (this->GetInput(2) != nullptr)
  {
    m_SubtractImageFilter->SetInput1(this->GetInput(2));
  }
  else
  {
    m_DefaultNormalizationVolume->SetInformationFromImage(const_cast<TInputImage *>(this->GetInput(0)));
    m_DefaultNormalizationVolume->SetConstant(1);
    m_SubtractImageFilter->SetInput1(m_DefaultNormalizationVolume->GetOutput());
  }
  m_MultiplyConstant1ImageFilter->SetConstant2(m_Beta);
  m_MultiplyConstant2ImageFilter->SetConstant2(4 * 2 * m_Beta * (pow(3, InputImageDimension) - 1));

  m_CustomBinaryFilter->UpdateOutputInformation();
  this->GetOutput()->SetOrigin(m_CustomBinaryFilter->GetOutput()->GetOrigin());
  this->GetOutput()->SetSpacing(m_CustomBinaryFilter->GetOutput()->GetSpacing());
  this->GetOutput()->SetDirection(m_CustomBinaryFilter->GetOutput()->GetDirection());
  this->GetOutput()->SetLargestPossibleRegion(m_CustomBinaryFilter->GetOutput()->GetLargestPossibleRegion());
}

template <class TInputImage, class TOutputImage>
void
DePierroRegularizationImageFilter<TInputImage, TOutputImage>::GenerateData()
{
  m_KernelImage->Update();
  typename TInputImage::IndexType pixelIndex;
  pixelIndex.Fill(1);
  m_KernelImage->GetOutput()->SetPixel(pixelIndex, pow(3, InputImageDimension) - 1);
  m_KernelOperator.SetImageKernel(m_KernelImage->GetOutput());
  // The radius of the kernel must be the radius of the patch, NOT the size of the patch
  itk::Size<InputImageDimension> radius;
  radius.Fill(1);
  m_KernelOperator.CreateToRadius(radius);
  m_ConvolutionFilter->OverrideBoundaryCondition(&m_BoundsCondition);
  m_ConvolutionFilter->SetOperator(m_KernelOperator);
  m_ConvolutionFilter->SetInput(this->GetInput(0));
  m_MultiplyConstant1ImageFilter->SetInput1(m_ConvolutionFilter->GetOutput());

  m_CustomBinaryFilter->Update();
  this->GraftOutput(m_CustomBinaryFilter->GetOutput());
}

} // end namespace rtk

#endif // rtkDePierroRegularizationImageFilter_hxx
