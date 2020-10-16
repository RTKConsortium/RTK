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

#ifndef rtkConjugateGradientGetP_kPlusOneImageFilter_h
#define rtkConjugateGradientGetP_kPlusOneImageFilter_h

#include <itkImageToImageFilter.h>
#include <itkAddImageFilter.h>
#include <itkMultiplyImageFilter.h>

namespace rtk
{
/** \class ConjugateGradientGetP_kPlusOneImageFilter
 *
 * \author Cyril Mory
 *
 * \ingroup RTK
 */
template <typename TInputImage>
class ConjugateGradientGetP_kPlusOneImageFilter : public itk::ImageToImageFilter<TInputImage, TInputImage>
{
public:
#if ITK_VERSION_MAJOR == 5 && ITK_VERSION_MINOR == 1
  ITK_DISALLOW_COPY_AND_ASSIGN(ConjugateGradientGetP_kPlusOneImageFilter);
#else
  ITK_DISALLOW_COPY_AND_MOVE(ConjugateGradientGetP_kPlusOneImageFilter);
#endif

  /** Standard class type alias. */
  using Self = ConjugateGradientGetP_kPlusOneImageFilter;
  using Superclass = itk::ImageToImageFilter<TInputImage, TInputImage>;
  using Pointer = itk::SmartPointer<Self>;
  using OutputImageRegionType = typename TInputImage::RegionType;
  using BetaImage = itk::Image<typename TInputImage::InternalPixelType, TInputImage::ImageDimension>;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(ConjugateGradientGetP_kPlusOneImageFilter, itk::ImageToImageFilter);

  /** Functions to set the inputs */
  void
  SetR_kPlusOne(const TInputImage * R_kPlusOne);
  void
  SetRk(const TInputImage * Rk);
  void
  SetPk(const TInputImage * Pk);

  itkSetMacro(SquaredNormR_k, double);
  itkSetMacro(SquaredNormR_kPlusOne, double);

  /** Typedefs for sub filters */
  using AddFilterType = itk::AddImageFilter<TInputImage>;
  using MultiplyFilterType = itk::MultiplyImageFilter<TInputImage, BetaImage, TInputImage>;

protected:
  ConjugateGradientGetP_kPlusOneImageFilter();
  ~ConjugateGradientGetP_kPlusOneImageFilter() override = default;

  typename TInputImage::Pointer
  GetR_kPlusOne();
  typename TInputImage::Pointer
  GetRk();
  typename TInputImage::Pointer
  GetPk();

  /** Does the real work. */
  void
  GenerateData() override;

  void
  GenerateOutputInformation() override;

private:
  double m_SquaredNormR_k;
  double m_SquaredNormR_kPlusOne;
  double m_Betak;

  /** Pointers to sub filters */
  typename AddFilterType::Pointer      m_AddFilter;
  typename MultiplyFilterType::Pointer m_MultiplyFilter;
};
} // namespace rtk


#ifndef ITK_MANUAL_INSTANTIATION
#  include "rtkConjugateGradientGetP_kPlusOneImageFilter.hxx"
#endif

#endif
