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

#ifndef rtkPolynomialGainCorrectionImageFilter_h
#define rtkPolynomialGainCorrectionImageFilter_h

#include <itkImageToImageFilter.h>

#include <vector>

#include "rtkMacro.h"

/** \class PolynomialGainCorrection
 * \brief Pixel-wize polynomial gain calibration
 *
 * Based on 'An improved method for flat-field correction of flat panel x-ray detector'
 *          Kwan, Med. Phys 33 (2), 2006
 * Only allow unsigned short as input format
 *
 * \author Sebastien Brousmiche
 *
 * \ingroup RTK ImageToImageFilter
 */

namespace rtk
{

template <class TInputImage, class TOutputImage>
class ITK_TEMPLATE_EXPORT PolynomialGainCorrectionImageFilter
  : public itk::ImageToImageFilter<TInputImage, TOutputImage>
{
public:
  ITK_DISALLOW_COPY_AND_MOVE(PolynomialGainCorrectionImageFilter);

  /** Standard class type alias. */
  using Self = PolynomialGainCorrectionImageFilter;
  using Superclass = itk::ImageToImageFilter<TInputImage, TOutputImage>;
  using Pointer = itk::SmartPointer<Self>;
  using ConstPointer = itk::SmartPointer<const Self>;

  /** Some convenient type alias. */
  using InputImageType = TInputImage;
  using OutputImageType = TOutputImage;
  using InputImagePointer = typename InputImageType::Pointer;
  using OutputImagePointer = typename OutputImageType::Pointer;
  using InputImageRegionType = typename InputImageType::RegionType;
  using OutputImageRegionType = typename TOutputImage::RegionType;
  using VectorType = typename std::vector<float>;
  using OutputSizeType = typename OutputImageType::SizeType;

  /** Standard New method. */
  itkNewMacro(Self);

  /** Runtime information support. */
  itkOverrideGetNameOfClassMacro(PolynomialGainCorrectionImageFilter);

  /** Dark image, an input with a single projection of the same size as the
   * input projections. */
  itkSetInputMacro(DarkImage, InputImageType);
  itkGetInputMacro(DarkImage, InputImageType);

  /** Weights, matrix A from reference paper, an input 3D image: 2D x order. */
  itkSetInputMacro(GainCoefficients, OutputImageType);
  itkGetInputMacro(GainCoefficients, OutputImageType);

  /* if K==0, the filter is bypassed */
  itkSetMacro(K, float);
  itkGetMacro(K, float);

protected:
  PolynomialGainCorrectionImageFilter();
  ~PolynomialGainCorrectionImageFilter() override = default;

  /** Checks that inputs are correctly set. */
  void
  VerifyPreconditions() const override;

  void
  GenerateOutputInformation() override;

  void
  GenerateInputRequestedRegion() override;

  /** The dark image and the gain coefficients are not in the same space as
   * the projections so there is nothing to verify. */
  void
  VerifyInputInformation() const override
  {}

  void
  DynamicThreadedGenerateData(const OutputImageRegionType & outputRegionForThread) override;

  int            m_VModelOrder{ 1 }; // Polynomial correction order
  float          m_K{ 1.0F };        // Scaling constant, a 0 means no correction
  VectorType     m_PowerLut;         // Vector containing I^n
  OutputSizeType m_GainSize;         // Gain map size
}; // end of class

} // namespace rtk

#ifndef ITK_MANUAL_INSTANTIATION
#  include "rtkPolynomialGainCorrectionImageFilter.hxx"
#endif

#endif
