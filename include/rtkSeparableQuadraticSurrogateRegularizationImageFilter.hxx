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

#ifndef rtkSeparableQuadraticSurrogateRegularizationImageFilter_hxx
#define rtkSeparableQuadraticSurrogateRegularizationImageFilter_hxx

#include "rtkSeparableQuadraticSurrogateRegularizationImageFilter.h"

namespace rtk
{

template <typename TImage>
SeparableQuadraticSurrogateRegularizationImageFilter<TImage>::SeparableQuadraticSurrogateRegularizationImageFilter()
{
  // Create the outputs
  this->SetNthOutput(0, this->MakeOutput(0));
  this->SetNthOutput(1, this->MakeOutput(1));

  // Default radius is 0
  m_Radius.Fill(0);

  // Set default regularization weights to 0
  m_RegularizationWeights.Fill(0);

  // Constants used in Green's prior, not modifiable by the user
  m_c1 = 27.0 / 128.0;
  m_c2 = 16.0 / (3.0 * std::sqrt(3.0));
}

template <typename TImage>
itk::ProcessObject::DataObjectPointer
SeparableQuadraticSurrogateRegularizationImageFilter<TImage>::MakeOutput(
  itk::ProcessObject::DataObjectPointerArraySizeType idx)
{
  itk::DataObject::Pointer output;

  switch (idx)
  {
    case 0:
      output = (TImage::New()).GetPointer();
      break;
    case 1:
      output = (TImage::New()).GetPointer();
      break;
    default:
      std::cerr << "No output " << idx << std::endl;
      output = nullptr;
      break;
  }
  return output.GetPointer();
}

template <typename TImage>
itk::ProcessObject::DataObjectPointer
SeparableQuadraticSurrogateRegularizationImageFilter<TImage>::MakeOutput(
  const itk::ProcessObject::DataObjectIdentifierType & idx)
{
  return Superclass::MakeOutput(idx);
}

template <typename TImage>
void
SeparableQuadraticSurrogateRegularizationImageFilter<TImage>::GenerateInputRequestedRegion()
{
  // Call the superclass' implementation of this method
  Superclass::GenerateInputRequestedRegion();

  // Get the requested regions on both outputs (should be identical)
  typename TImage::RegionType outputRequested1 = this->GetOutput(0)->GetRequestedRegion();
  typename TImage::RegionType outputRequested2 = this->GetOutput(1)->GetRequestedRegion();
  if (outputRequested1 != outputRequested2)
    itkGenericExceptionMacro(
      << "In rtkWeidingerForwardModelImageFilter: requested regions for outputs 1 and 2 should be identical");

  // Get pointer to the input
  typename TImage::ConstPointer smart = this->GetInput();
  typename TImage::Pointer      inputPtr = const_cast<TImage *>(smart.GetPointer());
  //  typename TImage::Pointer inputPtr = const_cast<TImage*>(this->GetInput().GetPointer());

  // Pad by the radius of the neighborhood, and crop to the largest possible region
  typename TImage::RegionType inputRequested = outputRequested1;
  inputRequested.PadByRadius(m_Radius);
  inputRequested.Crop(inputPtr->GetLargestPossibleRegion());

  // Set the requested region
  inputPtr->SetRequestedRegion(inputRequested);
}

template <typename TImage>
typename TImage::PixelType
SeparableQuadraticSurrogateRegularizationImageFilter<TImage>::GreenPriorFirstDerivative(typename TImage::PixelType pix)
{
  typename TImage::PixelType out;
  for (unsigned int i = 0; i < TImage::PixelType::Dimension; i++)
    out[i] = 2 * m_RegularizationWeights[i] * m_c1 * m_c2 * tanh(m_c2 * pix[i]);

  return out;
}

template <typename TImage>
typename TImage::PixelType
SeparableQuadraticSurrogateRegularizationImageFilter<TImage>::GreenPriorSecondDerivative(typename TImage::PixelType pix)
{
  typename TImage::PixelType out;
  for (unsigned int i = 0; i < TImage::PixelType::Dimension; i++)
    out[i] = 4 * m_RegularizationWeights[i] * m_c1 * m_c2 * m_c2 / (cosh(m_c2 * pix[i]) * cosh(m_c2 * pix[i]));

  return out;
}

template <typename TImage>
void
SeparableQuadraticSurrogateRegularizationImageFilter<TImage>::DynamicThreadedGenerateData(
  const typename TImage::RegionType & outputRegionForThread)
{
  // Create iterators for all inputs and outputs
  itk::ImageRegionIterator<TImage>       out1It(this->GetOutput(0), outputRegionForThread);
  itk::ImageRegionIterator<TImage>       out2It(this->GetOutput(1), outputRegionForThread);
  itk::ConstNeighborhoodIterator<TImage> nIt(m_Radius, this->GetInput(), outputRegionForThread);

  // Precompute the offset of center pixel
  itk::SizeValueType c = (itk::SizeValueType)(nIt.Size() / 2);

  // Declare intermediate variables
  typename TImage::PixelType diff, out1, out2;

  // Walk the output image
  while (!out1It.IsAtEnd())
  {
    out1 = itk::NumericTraits<typename TImage::PixelType>::ZeroValue();
    out2 = itk::NumericTraits<typename TImage::PixelType>::ZeroValue();

    // Walk the neighborhood of the current pixel in the input image
    for (unsigned int i = 0; i < nIt.Size(); i++)
    {
      // Compute the difference between central pixel and neighbor pixel
      diff = nIt.GetPixel(c) - nIt.GetPixel(i);

      // Compute the first and second derivatives of Green's prior at diff
      // and accumulate them in the output
      out1 += GreenPriorFirstDerivative(diff);
      out2 += GreenPriorSecondDerivative(diff);
    }

    // Set the outputs
    out1It.Set(out1);
    out2It.Set(out2);

    // Move to next pixel
    ++nIt;
    ++out1It;
    ++out2It;
  }
}

} // namespace rtk


#endif
