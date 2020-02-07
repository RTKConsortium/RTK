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
#ifndef rtkAddMatrixAndDiagonalImageFilter_hxx
#define rtkAddMatrixAndDiagonalImageFilter_hxx

#include "rtkAddMatrixAndDiagonalImageFilter.h"
#include "itkImageRegionIteratorWithIndex.h"
#include "itkImageRegionIterator.h"
#include "vnl/vnl_inverse.h"

namespace rtk
{
//
// Constructor
//
template <class TDiagonal, class TMatrix>
AddMatrixAndDiagonalImageFilter<TDiagonal, TMatrix>::AddMatrixAndDiagonalImageFilter()
{
  this->SetNumberOfRequiredInputs(2);
}

template <class TDiagonal, class TMatrix>
void
AddMatrixAndDiagonalImageFilter<TDiagonal, TMatrix>::SetInputMatrix(const TMatrix * hessian)
{
  this->SetNthInput(0, const_cast<TMatrix *>(hessian));
}

template <class TDiagonal, class TMatrix>
void
AddMatrixAndDiagonalImageFilter<TDiagonal, TMatrix>::SetInputDiagonal(const TDiagonal * gradient)
{
  this->SetNthInput(1, const_cast<TDiagonal *>(gradient));
}

template <class TDiagonal, class TMatrix>
typename TMatrix::ConstPointer
AddMatrixAndDiagonalImageFilter<TDiagonal, TMatrix>::GetInputMatrix()
{
  return static_cast<const TMatrix *>(this->itk::ProcessObject::GetInput(0));
}

template <class TDiagonal, class TMatrix>
typename TDiagonal::ConstPointer
AddMatrixAndDiagonalImageFilter<TDiagonal, TMatrix>::GetInputDiagonal()
{
  return static_cast<const TDiagonal *>(this->itk::ProcessObject::GetInput(1));
}

template <class TDiagonal, class TMatrix>
void
AddMatrixAndDiagonalImageFilter<TDiagonal, TMatrix>::GenerateInputRequestedRegion()
{
  // Call the superclass' implementation of this method
  Superclass::GenerateInputRequestedRegion();

  // Get the requested regions on both outputs (should be identical)
  typename TDiagonal::RegionType outputRequested = this->GetOutput()->GetRequestedRegion();

  // Get pointers to the inputs
  typename TMatrix::Pointer   input0Ptr = const_cast<TMatrix *>(this->GetInputMatrix().GetPointer());
  typename TDiagonal::Pointer input1Ptr = const_cast<TDiagonal *>(this->GetInputDiagonal().GetPointer());

  // The first and second input must have the same requested region as the outputs
  input1Ptr->SetRequestedRegion(outputRequested);
  input0Ptr->SetRequestedRegion(outputRequested);
}

template <class TDiagonal, class TMatrix>
void
AddMatrixAndDiagonalImageFilter<TDiagonal, TMatrix>::DynamicThreadedGenerateData(
  const typename TDiagonal::RegionType & outputRegionForThread)
{
  // Create iterators for all inputs and outputs
  itk::ImageRegionIterator<TMatrix>        outIt(this->GetOutput(), outputRegionForThread);
  itk::ImageRegionConstIterator<TDiagonal> diagIt(this->GetInputDiagonal(), outputRegionForThread);
  itk::ImageRegionConstIterator<TMatrix>   matIt(this->GetInputMatrix(), outputRegionForThread);

  itk::Vector<dataType, nChannels * nChannels> forOutput;

  while (!outIt.IsAtEnd())
  {
    // Make a vnl matrix out of the values read in input 2 (the hessian, but stored in a vector)
    forOutput = matIt.Get();
    for (unsigned int i = 0; i < nChannels; i++)
      forOutput[i * (nChannels + 1)] += diagIt.Get()[i];

    outIt.Set(forOutput);

    ++outIt;
    ++diagIt;
    ++matIt;
  }
}

} // namespace rtk

#endif
