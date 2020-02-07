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
#ifndef rtkBlockDiagonalMatrixVectorMultiplyImageFilter_hxx
#define rtkBlockDiagonalMatrixVectorMultiplyImageFilter_hxx

#include "rtkBlockDiagonalMatrixVectorMultiplyImageFilter.h"
#include "itkImageRegionIteratorWithIndex.h"
#include "itkImageRegionIterator.h"
#include "vnl/vnl_inverse.h"

namespace rtk
{
//
// Constructor
//
template <class TVectorImage, class TMatrixImage>
BlockDiagonalMatrixVectorMultiplyImageFilter<TVectorImage, TMatrixImage>::BlockDiagonalMatrixVectorMultiplyImageFilter()
{
  this->SetNumberOfRequiredInputs(2);
}

template <class TVectorImage, class TMatrixImage>
void
BlockDiagonalMatrixVectorMultiplyImageFilter<TVectorImage, TMatrixImage>::SetInput1(const TVectorImage * vector)
{
  this->SetNthInput(0, const_cast<TVectorImage *>(vector));
}

template <class TVectorImage, class TMatrixImage>
void
BlockDiagonalMatrixVectorMultiplyImageFilter<TVectorImage, TMatrixImage>::SetInput2(const TMatrixImage * matrix)
{
  this->SetNthInput(1, const_cast<TMatrixImage *>(matrix));
}

template <class TVectorImage, class TMatrixImage>
typename TVectorImage::ConstPointer
BlockDiagonalMatrixVectorMultiplyImageFilter<TVectorImage, TMatrixImage>::GetInput1()
{
  return static_cast<const TVectorImage *>(this->itk::ProcessObject::GetInput(0));
}

template <class TVectorImage, class TMatrixImage>
typename TMatrixImage::ConstPointer
BlockDiagonalMatrixVectorMultiplyImageFilter<TVectorImage, TMatrixImage>::GetInput2()
{
  return static_cast<const TMatrixImage *>(this->itk::ProcessObject::GetInput(1));
}

template <class TVectorImage, class TMatrixImage>
void
BlockDiagonalMatrixVectorMultiplyImageFilter<TVectorImage, TMatrixImage>::GenerateInputRequestedRegion()
{
  // Call the superclass' implementation of this method
  Superclass::GenerateInputRequestedRegion();

  // Get the requested regions on both outputs (should be identical)
  typename TVectorImage::RegionType outputRequested = this->GetOutput()->GetRequestedRegion();

  // Get pointers to the inputs
  typename TVectorImage::Pointer input1Ptr = const_cast<TVectorImage *>(this->GetInput1().GetPointer());
  typename TMatrixImage::Pointer input2Ptr = const_cast<TMatrixImage *>(this->GetInput2().GetPointer());

  // The first and second input must have the same requested region as the outputs
  input1Ptr->SetRequestedRegion(outputRequested);
  input2Ptr->SetRequestedRegion(outputRequested);
}

template <class TVectorImage, class TMatrixImage>
void
BlockDiagonalMatrixVectorMultiplyImageFilter<TVectorImage, TMatrixImage>::DynamicThreadedGenerateData(
  const typename TVectorImage::RegionType & outputRegionForThread)
{
  // Create iterators for all inputs and outputs
  itk::ImageRegionIterator<TVectorImage>      outIt(this->GetOutput(), outputRegionForThread);
  itk::ImageRegionConstIterator<TVectorImage> vectorIt(this->GetInput1(), outputRegionForThread);
  itk::ImageRegionConstIterator<TMatrixImage> matrixIt(this->GetInput2(), outputRegionForThread);

  itk::Vector<dataType, nChannels> forOutput;

  while (!outIt.IsAtEnd())
  {
    // Make a vnl matrix out of the values read in input 2 (the matrix mat, but stored in a vector)
    vnl_matrix<dataType> mat = vnl_matrix<dataType>(matrixIt.Get().GetDataPointer(), nChannels, nChannels);

    // Invert the mat, multiply by the vector, and write it in output
    forOutput.SetVnlVector(mat * vectorIt.Get().GetVnlVector());
    outIt.Set(forOutput);

    ++outIt;
    ++vectorIt;
    ++matrixIt;
  }
}

} // namespace rtk

#endif
