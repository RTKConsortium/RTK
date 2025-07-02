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

#ifndef rtkDaubechiesWaveletsConvolutionImageFilter_hxx
#define rtkDaubechiesWaveletsConvolutionImageFilter_hxx

// Includes

#include <vector>
#include <algorithm>
#include <itkImageRegionIterator.h>

namespace rtk
{

template <typename TImage>
DaubechiesWaveletsConvolutionImageFilter<TImage>::DaubechiesWaveletsConvolutionImageFilter()
{
  this->SetDeconstruction();
}

template <typename TImage>
DaubechiesWaveletsConvolutionImageFilter<TImage>::~DaubechiesWaveletsConvolutionImageFilter() = default;

template <typename TImage>
void
DaubechiesWaveletsConvolutionImageFilter<TImage>::SetDeconstruction()
{
  m_Type = Self::Deconstruct;
}

template <typename TImage>
void
DaubechiesWaveletsConvolutionImageFilter<TImage>::SetReconstruction()
{
  m_Type = Self::Reconstruct;
}

template <typename TImage>
void
DaubechiesWaveletsConvolutionImageFilter<TImage>::PrintSelf(std::ostream & os, itk::Indent i) const
{
  os << i << "DaubechiesWaveletsConvolutionImageFilter { this=" << this << " }" << std::endl;

  os << i << "m_Order=" << this->GetOrder() << std::endl;
  os << i << "m_Pass=" << std::endl;
  for (unsigned int dim = 0; dim < TImage::ImageDimension; dim++)
  {
    os << i << i << this->m_Pass[dim] << std::endl;
  }
  os << i << "m_Type=" << this->m_Type << std::endl;

  Superclass::PrintSelf(os, i.GetNextIndent());
}

template <typename TImage>
typename DaubechiesWaveletsConvolutionImageFilter<TImage>::CoefficientVector
DaubechiesWaveletsConvolutionImageFilter<TImage>::GenerateCoefficientsLowpassDeconstruct()
{
  CoefficientVector coeff;
  switch (this->GetOrder())
  {
    case 1:
      coeff.push_back(1.0 / itk::Math::sqrt2);
      coeff.push_back(1.0 / itk::Math::sqrt2);
      break;
    case 2:
      coeff.push_back(-0.1830127 / itk::Math::sqrt2);
      coeff.push_back(0.3169873 / itk::Math::sqrt2);
      coeff.push_back(1.1830127 / itk::Math::sqrt2);
      coeff.push_back(0.6830127 / itk::Math::sqrt2);
      break;
    case 3:
      coeff.push_back(0.0498175 / itk::Math::sqrt2);
      coeff.push_back(-0.12083221 / itk::Math::sqrt2);
      coeff.push_back(-0.19093442 / itk::Math::sqrt2);
      coeff.push_back(0.650365 / itk::Math::sqrt2);
      coeff.push_back(1.14111692 / itk::Math::sqrt2);
      coeff.push_back(0.47046721 / itk::Math::sqrt2);
      break;
    case 4:
      coeff.push_back(-0.01498699 / itk::Math::sqrt2);
      coeff.push_back(0.0465036 / itk::Math::sqrt2);
      coeff.push_back(0.0436163 / itk::Math::sqrt2);
      coeff.push_back(-0.26450717 / itk::Math::sqrt2);
      coeff.push_back(-0.03957503 / itk::Math::sqrt2);
      coeff.push_back(0.8922014 / itk::Math::sqrt2);
      coeff.push_back(1.01094572 / itk::Math::sqrt2);
      coeff.push_back(0.32580343 / itk::Math::sqrt2);
      break;
    case 5:
      coeff.push_back(0.00471742793 / itk::Math::sqrt2);
      coeff.push_back(-0.01779187 / itk::Math::sqrt2);
      coeff.push_back(-0.00882680 / itk::Math::sqrt2);
      coeff.push_back(0.10970265 / itk::Math::sqrt2);
      coeff.push_back(-0.04560113 / itk::Math::sqrt2);
      coeff.push_back(-0.34265671 / itk::Math::sqrt2);
      coeff.push_back(0.19576696 / itk::Math::sqrt2);
      coeff.push_back(1.02432694 / itk::Math::sqrt2);
      coeff.push_back(0.85394354 / itk::Math::sqrt2);
      coeff.push_back(0.22641898 / itk::Math::sqrt2);
      break;
    case 6:
      coeff.push_back(-0.00152353381 / itk::Math::sqrt2);
      coeff.push_back(0.00675606236 / itk::Math::sqrt2);
      coeff.push_back(0.000783251152 / itk::Math::sqrt2);
      coeff.push_back(-0.04466375 / itk::Math::sqrt2);
      coeff.push_back(0.03892321 / itk::Math::sqrt2);
      coeff.push_back(0.13788809 / itk::Math::sqrt2);
      coeff.push_back(-0.18351806 / itk::Math::sqrt2);
      coeff.push_back(-0.31998660 / itk::Math::sqrt2);
      coeff.push_back(0.44583132 / itk::Math::sqrt2);
      coeff.push_back(1.06226376 / itk::Math::sqrt2);
      coeff.push_back(0.69950381 / itk::Math::sqrt2);
      coeff.push_back(0.15774243 / itk::Math::sqrt2);
      break;
    case 7:
      coeff.push_back(0.000500226853 / itk::Math::sqrt2);
      coeff.push_back(-0.00254790472 / itk::Math::sqrt2);
      coeff.push_back(0.000607514995 / itk::Math::sqrt2);
      coeff.push_back(0.01774979 / itk::Math::sqrt2);
      coeff.push_back(-0.02343994 / itk::Math::sqrt2);
      coeff.push_back(-0.05378245 / itk::Math::sqrt2);
      coeff.push_back(0.11400345 / itk::Math::sqrt2);
      coeff.push_back(0.1008467 / itk::Math::sqrt2);
      coeff.push_back(-0.31683501 / itk::Math::sqrt2);
      coeff.push_back(-0.20351382 / itk::Math::sqrt2);
      coeff.push_back(0.66437248 / itk::Math::sqrt2);
      coeff.push_back(1.03114849 / itk::Math::sqrt2);
      coeff.push_back(0.56079128 / itk::Math::sqrt2);
      coeff.push_back(0.11009943 / itk::Math::sqrt2);
      break;
    default:
      itkGenericExceptionMacro(<< "In rtkDaubechiesWaveletsConvolutionImageFilter.hxx: Order should be <= 7.");
  } // end case(Order)
  return coeff;
}

template <typename TImage>
typename DaubechiesWaveletsConvolutionImageFilter<TImage>::CoefficientVector
DaubechiesWaveletsConvolutionImageFilter<TImage>::GenerateCoefficientsHighpassDeconstruct()
{
  CoefficientVector coeff = this->GenerateCoefficientsLowpassDeconstruct();
  std::reverse(coeff.begin(), coeff.end());
  unsigned int it = 0;

  double factor = -1;
  for (it = 0; it < coeff.size(); it++)
  {
    coeff[it] *= factor;
    factor *= -1;
  }
  return coeff;
}

template <typename TImage>
typename DaubechiesWaveletsConvolutionImageFilter<TImage>::CoefficientVector
DaubechiesWaveletsConvolutionImageFilter<TImage>::GenerateCoefficientsLowpassReconstruct()
{
  CoefficientVector coeff = this->GenerateCoefficientsLowpassDeconstruct();
  std::reverse(coeff.begin(), coeff.end());
  return coeff;
}

template <typename TImage>
typename DaubechiesWaveletsConvolutionImageFilter<TImage>::CoefficientVector
DaubechiesWaveletsConvolutionImageFilter<TImage>::GenerateCoefficientsHighpassReconstruct()
{
  CoefficientVector coeff = this->GenerateCoefficientsHighpassDeconstruct();
  std::reverse(coeff.begin(), coeff.end());
  return coeff;
}

template <typename TImage>
void
DaubechiesWaveletsConvolutionImageFilter<TImage>::GenerateOutputInformation()
{
  unsigned int                                         dim = TImage::ImageDimension;
  std::vector<typename TImage::Pointer>                kernelImages;
  std::vector<typename ConvolutionFilterType::Pointer> convolutionFilters;

  // Create and connect as many Convolution filters as the number of dimensions
  for (unsigned int d = 0; d < dim; d++)
  {
    // Create the 1-D kernel image
    typename TImage::SizeType kernelSize;
    kernelSize.Fill(1);
    kernelSize[d] = 2 * m_Order;

    typename TImage::IndexType kernelIndex;
    kernelIndex.Fill(0);

    typename TImage::RegionType kernelRegion;
    kernelRegion.SetSize(kernelSize);
    kernelRegion.SetIndex(kernelIndex);

    kernelImages.push_back(TImage::New());
    kernelImages[d]->SetRegions(kernelRegion);

    // Create the convolution filters
    convolutionFilters.push_back(ConvolutionFilterType::New());
    convolutionFilters[d]->SetKernelImage(kernelImages[d]);
    convolutionFilters[d]->SetOutputRegionModeToValid();

    if (d == 0)
      convolutionFilters[d]->SetInput(this->GetInput());
    else
      convolutionFilters[d]->SetInput(convolutionFilters[d - 1]->GetOutput());
  }

  // Generate output information
  convolutionFilters[dim - 1]->UpdateOutputInformation();

  // Copy it to the output
  this->GetOutput()->CopyInformation(convolutionFilters[dim - 1]->GetOutput());
}

template <typename TImage>
void
DaubechiesWaveletsConvolutionImageFilter<TImage>::GenerateData()
{
  int dim = TImage::ImageDimension;

  // Create a vector holding the coefficients along each direction
  auto * coeffs = new CoefficientVector[dim];
  for (int d = 0; d < dim; d++)
  {
    if (m_Type == Self::Deconstruct)
    {
      switch (m_Pass[d])
      {
        case Self::Low:
          coeffs[d] = GenerateCoefficientsLowpassDeconstruct();
          break;
        case Self::High:
          coeffs[d] = GenerateCoefficientsHighpassDeconstruct();
          break;
        default:
          itkGenericExceptionMacro("In rtkDaubechiesWaveletsConvolutionImageFilter : unknown pass");
      }
    }
    if (m_Type == Self::Reconstruct)
    {
      switch (m_Pass[d])
      {
        case Self::Low:
          coeffs[d] = GenerateCoefficientsLowpassReconstruct();
          break;
        case Self::High:
          coeffs[d] = GenerateCoefficientsHighpassReconstruct();
          break;
        default:
          itkGenericExceptionMacro("In rtkDaubechiesWaveletsConvolutionImageFilter : unknown pass");
      }
    }
  }

  std::vector<typename TImage::Pointer>                kernelImages;
  std::vector<typename ConvolutionFilterType::Pointer> convolutionFilters;

  // Convolve the input "Dimension" times, each time with a 1-D kernel
  for (int d = 0; d < dim; d++)
  {
    // Create the 1-D kernel image
    typename TImage::SizeType kernelSize;
    kernelSize.Fill(1);
    kernelSize[d] = 2 * m_Order;

    typename TImage::IndexType kernelIndex;
    kernelIndex.Fill(0);

    typename TImage::RegionType kernelRegion;
    kernelRegion.SetSize(kernelSize);
    kernelRegion.SetIndex(kernelIndex);

    kernelImages.push_back(TImage::New());
    kernelImages[d]->SetRegions(kernelRegion);
    kernelImages[d]->Allocate();
    itk::ImageRegionIterator<TImage> kernelIt(kernelImages[d], kernelImages[d]->GetLargestPossibleRegion());

    int pos = 0;
    while (!kernelIt.IsAtEnd())
    {
      kernelIt.Set(coeffs[d][pos]);
      pos++;
      ++kernelIt;
    }

    // Create the convolution filters
    convolutionFilters.push_back(ConvolutionFilterType::New());
    convolutionFilters[d]->SetKernelImage(kernelImages[d]);
    convolutionFilters[d]->SetOutputRegionModeToValid();

    if (d == 0)
      convolutionFilters[d]->SetInput(this->GetInput());
    else
      convolutionFilters[d]->SetInput(convolutionFilters[d - 1]->GetOutput());
  }

  // Generate output information
  convolutionFilters[dim - 1]->Update();

  this->GraftOutput(convolutionFilters[dim - 1]->GetOutput());

  // Clean up
  delete[] coeffs;
}

} // end namespace rtk

#endif
