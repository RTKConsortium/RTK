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
#ifndef rtkGetNewtonUpdateImageFilter_hxx
#define rtkGetNewtonUpdateImageFilter_hxx

#include "rtkGetNewtonUpdateImageFilter.h"
#include "itkImageRegionIteratorWithIndex.h"
#include "itkImageRegionIterator.h"
#include "vnl/vnl_inverse.h"

namespace rtk
{
//
// Constructor
//
template <class TGradient, class THessian>
GetNewtonUpdateImageFilter<TGradient, THessian>::GetNewtonUpdateImageFilter()
{
  this->SetNumberOfRequiredInputs(2);
}

template <class TGradient, class THessian>
void
GetNewtonUpdateImageFilter<TGradient, THessian>::SetInputGradient(const TGradient * gradient)
{
  this->SetNthInput(0, const_cast<TGradient *>(gradient));
}

template <class TGradient, class THessian>
void
GetNewtonUpdateImageFilter<TGradient, THessian>::SetInputHessian(const THessian * hessian)
{
  this->SetNthInput(1, const_cast<THessian *>(hessian));
}

template <class TGradient, class THessian>
typename TGradient::ConstPointer
GetNewtonUpdateImageFilter<TGradient, THessian>::GetInputGradient()
{
  return static_cast<const TGradient *>(this->itk::ProcessObject::GetInput(0));
}

template <class TGradient, class THessian>
typename THessian::ConstPointer
GetNewtonUpdateImageFilter<TGradient, THessian>::GetInputHessian()
{
  return static_cast<const THessian *>(this->itk::ProcessObject::GetInput(1));
}

template <class TGradient, class THessian>
void
GetNewtonUpdateImageFilter<TGradient, THessian>::GenerateInputRequestedRegion()
{
  // Call the superclass' implementation of this method
  Superclass::GenerateInputRequestedRegion();

  // Get the requested regions on both outputs (should be identical)
  typename TGradient::RegionType outputRequested = this->GetOutput()->GetRequestedRegion();

  // Get pointers to the inputs
  typename TGradient::Pointer input1Ptr = const_cast<TGradient *>(this->GetInputGradient().GetPointer());
  typename THessian::Pointer  input2Ptr = const_cast<THessian *>(this->GetInputHessian().GetPointer());

  // The first and second input must have the same requested region as the outputs
  input1Ptr->SetRequestedRegion(outputRequested);
  input2Ptr->SetRequestedRegion(outputRequested);
}

template <class TGradient, class THessian>
void
GetNewtonUpdateImageFilter<TGradient, THessian>::DynamicThreadedGenerateData(
  const typename TGradient::RegionType & outputRegionForThread)
{
  // Create iterators for all inputs and outputs
  itk::ImageRegionIterator<TGradient>      outIt(this->GetOutput(), outputRegionForThread);
  itk::ImageRegionConstIterator<TGradient> gradIt(this->GetInputGradient(), outputRegionForThread);
  itk::ImageRegionConstIterator<THessian>  hessIt(this->GetInputHessian(), outputRegionForThread);

  itk::Vector<dataType, nChannels> forOutput;

  while (!outIt.IsAtEnd())
  {
    // Make a vnl matrix out of the values read in input 2 (the hessian, but stored in a vector)
    vnl_matrix<dataType> hessian = vnl_matrix<dataType>(hessIt.Get().GetDataPointer(), nChannels, nChannels);
    vnl_matrix<dataType> regul = vnl_matrix<dataType>(nChannels, nChannels, 0);
    regul.fill_diagonal(1e-8);

    // Invert the hessian, multiply by the gradient, and write it in output
    forOutput.SetVnlVector(vnl_inverse(hessian + regul) * gradIt.Get().GetVnlVector());
    outIt.Set(forOutput);

    ++outIt;
    ++gradIt;
    ++hessIt;
  }
}

} // namespace rtk

#endif
