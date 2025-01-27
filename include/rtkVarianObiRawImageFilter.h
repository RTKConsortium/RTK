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

#ifndef rtkVarianObiRawImageFilter_h
#define rtkVarianObiRawImageFilter_h

#include <itkUnaryFunctorImageFilter.h>
#include <itkConceptChecking.h>
#include <itkNumericTraits.h>

#include "rtkMacro.h"

namespace rtk
{

namespace Function
{

/** \class ObiAttenuation
 * \brief Converts a raw value measured by the Varian OBI system to attenuation
 *
 * The user can specify I0 and IDark values. The defaults are 139000 and 0,
 * respectively.
 *
 * \author Simon Rit
 *
 * \ingroup RTK Functions
 */
template <class TInput, class TOutput>
class ITK_TEMPLATE_EXPORT ObiAttenuation
{
public:
  ObiAttenuation() = default;
  ~ObiAttenuation() = default;
  bool
  operator!=(const ObiAttenuation &) const
  {
    return false;
  }
  bool
  operator==(const ObiAttenuation & other) const
  {
    return !(*this != other);
  }
  inline TOutput
  operator()(const TInput & A) const
  {
    return (!A) ? 0. : TOutput(std::log((m_I0 - m_IDark) / (A - m_IDark)));
  }
  void
  SetI0(double i0)
  {
    m_I0 = i0;
  }
  void
  SetIDark(double idark)
  {
    m_IDark = idark;
  }

private:
  double m_I0;
  double m_IDark;
};
} // namespace Function

/** \class VarianObiRawImageFilter
 * \brief Converts raw images measured by the Varian OBI system to attenuation
 *
 * Uses ObiAttenuation.
 *
 * \author Simon Rit
 *
 * \ingroup RTK ImageToImageFilter
 */
template <class TInputImage, class TOutputImage>
class ITK_TEMPLATE_EXPORT VarianObiRawImageFilter
  : public itk::UnaryFunctorImageFilter<
      TInputImage,
      TOutputImage,
      Function::ObiAttenuation<typename TInputImage::PixelType, typename TOutputImage::PixelType>>
{
public:
  ITK_DISALLOW_COPY_AND_MOVE(VarianObiRawImageFilter);

  /** Standard class type alias. */
  using Self = VarianObiRawImageFilter;
  using Superclass = itk::UnaryFunctorImageFilter<
    TInputImage,
    TOutputImage,
    Function::ObiAttenuation<typename TInputImage::PixelType, typename TOutputImage::PixelType>>;
  using Pointer = itk::SmartPointer<Self>;
  using ConstPointer = itk::SmartPointer<const Self>;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Runtime information support. */
  itkOverrideGetNameOfClassMacro(VarianObiRawImageFilter);

  itkGetMacro(I0, double);
  itkSetMacro(I0, double);

  itkGetMacro(IDark, double);
  itkSetMacro(IDark, double);

  void
  BeforeThreadedGenerateData() override;

protected:
  VarianObiRawImageFilter();
  ~VarianObiRawImageFilter() override = default;

private:
  double m_I0{ 139000. };
  double m_IDark{ 0. };
};

} // end namespace rtk

#ifndef ITK_MANUAL_INSTANTIATION
#  include "rtkVarianObiRawImageFilter.hxx"
#endif

#endif
