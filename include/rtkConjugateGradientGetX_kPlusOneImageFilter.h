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

#ifndef rtkConjugateGradientGetX_kPlusOneImageFilter_h
#define rtkConjugateGradientGetX_kPlusOneImageFilter_h

#include <itkImageToImageFilter.h>
#include <itkAddImageFilter.h>
#include <itkMultiplyImageFilter.h>

namespace rtk
{
/** \class ConjugateGradientGetX_kPlusOneImageFilter
 *
 * \author Cyril Mory
 *
 * \ingroup RTK
 */
template <typename TInputImage>
class ConjugateGradientGetX_kPlusOneImageFilter : public itk::ImageToImageFilter<TInputImage, TInputImage>
{
public:
#if ITK_VERSION_MAJOR == 5 && ITK_VERSION_MINOR == 1
  ITK_DISALLOW_COPY_AND_ASSIGN(ConjugateGradientGetX_kPlusOneImageFilter);
#else
  ITK_DISALLOW_COPY_AND_MOVE(ConjugateGradientGetX_kPlusOneImageFilter);
#endif

  /** Standard class type alias. */
  using Self = ConjugateGradientGetX_kPlusOneImageFilter;
  using Superclass = itk::ImageToImageFilter<TInputImage, TInputImage>;
  using Pointer = itk::SmartPointer<Self>;
  using OutputImageRegionType = typename TInputImage::RegionType;
  using AlphaImage = itk::Image<typename TInputImage::InternalPixelType, TInputImage::ImageDimension>;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(ConjugateGradientGetX_kPlusOneImageFilter, itk::ImageToImageFilter);

  /** Functions to set the inputs */
  void
  SetXk(const TInputImage * Xk);
  void
  SetPk(const TInputImage * Pk);

  itkGetMacro(Alphak, double);
  itkSetMacro(Alphak, double);

  /** Typedefs for sub filters */
  using AddFilterType = itk::AddImageFilter<TInputImage>;
  using MultiplyFilterType = itk::MultiplyImageFilter<TInputImage, AlphaImage, TInputImage>;

protected:
  ConjugateGradientGetX_kPlusOneImageFilter();
  ~ConjugateGradientGetX_kPlusOneImageFilter() override = default;

  typename TInputImage::Pointer
  GetXk();
  typename TInputImage::Pointer
  GetPk();

  /** Does the real work. */
  void
  GenerateData() override;

  void
  GenerateOutputInformation() override;

private:
  double m_Alphak;

  /** Pointers to sub filters */
  typename AddFilterType::Pointer      m_AddFilter;
  typename MultiplyFilterType::Pointer m_MultiplyFilter;
};
} // namespace rtk


#ifndef ITK_MANUAL_INSTANTIATION
#  include "rtkConjugateGradientGetX_kPlusOneImageFilter.hxx"
#endif

#endif
