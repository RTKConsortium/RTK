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

#ifndef rtkDaubechiesWaveletsConvolutionImageFilter_h
#define rtkDaubechiesWaveletsConvolutionImageFilter_h

// Includes
#include <itkImageToImageFilter.h>
#include <itkConvolutionImageFilter.h>

#include "rtkMacro.h"

namespace rtk
{

/**
 * \class DaubechiesWaveletsConvolutionImageFilter
 * \brief Creates a Daubechies wavelets kernel image with the requested
 * attributes (order, type, pass along each dimension)
 *
 * This filter is inspired from Dan Mueller's GIFT package
 * http://www.insight-journal.org/browse/publication/103
 *
 * \author Cyril Mory
 *
 * \ingroup RTK
 *
 */
template <typename TImage>
class DaubechiesWaveletsConvolutionImageFilter : public itk::ImageToImageFilter<TImage, TImage>
{
public:
#if ITK_VERSION_MAJOR == 5 && ITK_VERSION_MINOR == 1
  ITK_DISALLOW_COPY_AND_ASSIGN(DaubechiesWaveletsConvolutionImageFilter);
#else
  ITK_DISALLOW_COPY_AND_MOVE(DaubechiesWaveletsConvolutionImageFilter);
#endif

  enum Pass
  {
    Low = 0x0, // Indicates to return the low-pass filter coefficients
    High = 0x1 // Indicates to return the high-pass filter coefficients
  };

  enum Type
  {
    Deconstruct = 0x0, // Indicates to deconstruct the image into levels/bands
    Reconstruct = 0x1  // Indicates to reconstruct the image from levels/bands
  };


  /** Standard class type alias. */
  using Self = DaubechiesWaveletsConvolutionImageFilter;
  using Superclass = itk::ImageToImageFilter<TImage, TImage>;
  using Pointer = itk::SmartPointer<Self>;
  using ConstPointer = itk::SmartPointer<const Self>;

  /** Typedef for the output image type. */
  using OutputImageType = TImage;

  /** Typedef for the output image PixelType. */
  using OutputImagePixelType = typename TImage::PixelType;

  /** Typedef to describe the output image region type. */
  using OutputImageRegionType = typename TImage::RegionType;

  /** Typedef for the "pass" vector (high pass or low pass along each dimension). */
  using PassVector = typename itk::Vector<typename Self::Pass, TImage::ImageDimension>;

  /** Run-time type information (and related methods). */
  itkTypeMacro(DaubechiesWaveletsConvolutionImageFilter, itk::ImageSource);

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Typedef for the internal convolution filter */
  using ConvolutionFilterType = typename itk::ConvolutionImageFilter<TImage>;

  /** Sets the filter to return coefficients for low pass, deconstruct. */
  void
  SetDeconstruction();

  /** Sets the filter to return coefficients for low pass, reconstruct. */
  void
  SetReconstruction();

  /** Prints some debugging information. */
  void
  PrintSelf(std::ostream & os, itk::Indent i) const override;

  /** Set and Get macro for the wavelet order */
  itkSetMacro(Order, unsigned int);
  itkGetConstMacro(Order, unsigned int);

  /** Set and Get macro for the pass vector */
  itkSetMacro(Pass, PassVector);
  itkGetMacro(Pass, PassVector);

protected:
  DaubechiesWaveletsConvolutionImageFilter();
  ~DaubechiesWaveletsConvolutionImageFilter() override;

  using CoefficientVector = std::vector<typename TImage::PixelType>;

  /** Calculates CoefficientsVector coefficients. */
  CoefficientVector
  GenerateCoefficients();

  /** Does the real work */
  void
  GenerateData() override;

  /** Defines the size, spacing, ... of the output kernel image */
  void
  GenerateOutputInformation() override;

private:
  /** Returns the wavelet coefficients for each type*/
  CoefficientVector
  GenerateCoefficientsLowpassDeconstruct();
  CoefficientVector
  GenerateCoefficientsHighpassDeconstruct();
  CoefficientVector
  GenerateCoefficientsLowpassReconstruct();
  CoefficientVector
  GenerateCoefficientsHighpassReconstruct();

  /** Specifies the wavelet type name */
  unsigned int m_Order{ 3 };

  /** Specifies the filter pass along each dimension */
  PassVector m_Pass{ PassVector(typename PassVector::ComponentType(0)) };

  /** Specifies the filter type */
  Type m_Type;
};

} // namespace rtk

// Include CXX
#ifndef rtk_MANUAL_INSTANTIATION
#  include "rtkDaubechiesWaveletsConvolutionImageFilter.hxx"
#endif

#endif
