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

#ifndef rtkLaplacianImageFilter_hxx
#define rtkLaplacianImageFilter_hxx


namespace rtk
{

template <typename TOutputImage, typename TGradientImage>
LaplacianImageFilter<TOutputImage, TGradientImage>::LaplacianImageFilter()
{
  // Create the filters
  m_Gradient = GradientFilterType::New();
  m_Divergence = DivergenceFilterType::New();

  // Set memory management parameters
  m_Gradient->ReleaseDataFlagOn();
}

template <typename TOutputImage, typename TGradientImage>
void
LaplacianImageFilter<TOutputImage, TGradientImage>::GenerateOutputInformation()
{
  Superclass::GenerateOutputInformation();

  // Set runtime connections
  m_Gradient->SetInput(this->GetInput());

  // Set internal connection between filter
  if (this->GetWeights().IsNotNull())
  {
    m_Multiply = MultiplyImageFilterType::New();
    m_Multiply->SetInput1(m_Gradient->GetOutput());
    m_Multiply->SetInput2(this->GetWeights());
    m_Divergence->SetInput(m_Multiply->GetOutput());
  }
  else
    m_Divergence->SetInput(m_Gradient->GetOutput());

  // Update the last filter
  m_Divergence->UpdateOutputInformation();

  // Copy it as the output information of the composite filter
  this->GetOutput()->CopyInformation(m_Divergence->GetOutput());
}

template <typename TOutputImage, typename TGradientImage>
void
LaplacianImageFilter<TOutputImage, TGradientImage>::GenerateData()
{
  // Update the last filter
  m_Divergence->Update();

  // Graft its output to the composite filter's output
  this->GraftOutput(m_Divergence->GetOutput());
}

template <typename TOutputImage, typename TGradientImage>
void
LaplacianImageFilter<TOutputImage, TGradientImage>::SetWeights(const TOutputImage * weights)
{
  this->SetInput("Weights", const_cast<TOutputImage *>(weights));
}

template <typename TOutputImage, typename TGradientImage>
typename TOutputImage::ConstPointer
LaplacianImageFilter<TOutputImage, TGradientImage>::GetWeights()
{
  return static_cast<const TOutputImage *>(this->itk::ProcessObject::GetInput("Weights"));
}

} // namespace rtk


#endif
