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

#ifndef rtkConjugateGradientGetR_kPlusOneImageFilter_h
#define rtkConjugateGradientGetR_kPlusOneImageFilter_h

#include <itkImageToImageFilter.h>
#include <itkVectorImage.h>

#include "rtkConfiguration.h"
#include "rtkMacro.h"

namespace rtk
{
/** \class ConjugateGradientGetR_kPlusOneImageFilter
 *
 * \author Cyril Mory
 *
 * \ingroup RTK
 */
template <typename TInputImage>
class ConjugateGradientGetR_kPlusOneImageFilter : public itk::ImageToImageFilter<TInputImage, TInputImage>
{
public:
#if ITK_VERSION_MAJOR == 5 && ITK_VERSION_MINOR == 1
  ITK_DISALLOW_COPY_AND_ASSIGN(ConjugateGradientGetR_kPlusOneImageFilter);
#else
  ITK_DISALLOW_COPY_AND_MOVE(ConjugateGradientGetR_kPlusOneImageFilter);
#endif

  /** Standard class type alias. */
  using Self = ConjugateGradientGetR_kPlusOneImageFilter;
  using Superclass = itk::ImageToImageFilter<TInputImage, TInputImage>;
  using Pointer = itk::SmartPointer<Self>;
  using OutputImageRegionType = typename TInputImage::RegionType;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(ConjugateGradientGetR_kPlusOneImageFilter, itk::ImageToImageFilter);

  /** Functions to set the inputs */
  void
  SetRk(const TInputImage * Rk);
  void
  SetPk(const TInputImage * Pk);
  void
  SetAPk(const TInputImage * APk);

  itkGetMacro(Alphak, double);
  itkGetMacro(SquaredNormR_k, double);
  itkGetMacro(SquaredNormR_kPlusOne, double);

protected:
  ConjugateGradientGetR_kPlusOneImageFilter();
  ~ConjugateGradientGetR_kPlusOneImageFilter() override = default;

  typename TInputImage::Pointer
  GetRk();
  typename TInputImage::Pointer
  GetPk();
  typename TInputImage::Pointer
  GetAPk();

  void
  GenerateData() override;

private:
  double m_Alphak{ 0. };
  double m_SquaredNormR_k{ 0. };
  double m_SquaredNormR_kPlusOne{ 0. };
};

} // namespace rtk


#ifndef ITK_MANUAL_INSTANTIATION
#  include "rtkConjugateGradientGetR_kPlusOneImageFilter.hxx"
#endif

#endif
