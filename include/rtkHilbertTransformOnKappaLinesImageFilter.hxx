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

#ifndef rtkHilbertTransformOnKappaLinesImageFilter_hxx
#define rtkHilbertTransformOnKappaLinesImageFilter_hxx

#include "rtkHilbertTransformOnKappaLinesImageFilter.h"


namespace rtk
{

template <typename TInputImage, typename TOutputImage>
void
HilbertTransformOnKappaLinesImageFilter<TInputImage, TOutputImage>::GenerateOutputInformation()
{
  // Call the superclass' implementation of this method
  Superclass::GenerateOutputInformation();
}

template <class TInputImage, class TOutputImage>
HilbertTransformOnKappaLinesImageFilter<TInputImage, TOutputImage>::HilbertTransformOnKappaLinesImageFilter()
{
  m_ForwardFilter = ForwardBinningType::New();
  m_HilbertFilter = FFTHilbertType::New();
  m_HilbertFilter->SetInput(m_ForwardFilter->GetOutput());
  m_BackwardFilter = BackwardBinningType::New();
  m_BackwardFilter->SetInput(m_HilbertFilter->GetOutput());
}

template <typename TInputImage, typename TOutputImage>
void
HilbertTransformOnKappaLinesImageFilter<TInputImage, TOutputImage>::GenerateData()
{
  typename InputImageType::Pointer input = InputImageType::New();
  input->Graft(const_cast<InputImageType *>(this->GetInput()));
  m_ForwardFilter->SetInput(input);
  m_ForwardFilter->SetGeometry(this->m_Geometry);

  m_BackwardFilter->SetGeometry(this->m_Geometry);
  m_BackwardFilter->GraftOutput(this->GetOutput());
  m_BackwardFilter->Update();
  this->GraftOutput(m_BackwardFilter->GetOutput());
}

template <typename TInputImage, typename TOutputImage>
void
HilbertTransformOnKappaLinesImageFilter<TInputImage, TOutputImage>::PrintSelf(std::ostream & os,
                                                                              itk::Indent    indent) const
{
  Superclass::PrintSelf(os, indent);
  os << indent << "This is a Hilbert Image Filter" << std::endl;
}

} // namespace rtk
#endif
