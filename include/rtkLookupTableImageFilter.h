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

#ifndef rtkLookupTableImageFilter_h
#define rtkLookupTableImageFilter_h

#include <itkUnaryFunctorImageFilter.h>
#include <itkLinearInterpolateImageFunction.h>

namespace rtk
{

/** \class LookupTableImageFilter
 * \brief Function to do the lookup operation.
 *
 * The lookup table is a 1D image. Two cases:
 * - if the pixel type is float or double, the meta information of the image
 * (spacing and origin) is used to select the pixel position and interpolate
 * linearly at this position.
 * - otherwise, it reads the value at the integer position, without
 * interpolation.
 *
 * \author Simon Rit
 *
 * \ingroup RTK Functions
 */

namespace Functor
{

template <class TInput, class TOutput>
class LUT
{
public:
  using LookupTableType = itk::Image<TOutput, 1>;
  using LookupTablePointer = typename LookupTableType::Pointer;
  using LookupTableDataPointer = typename LookupTableType::PixelType *;
  using InterpolatorType = typename itk::LinearInterpolateImageFunction<LookupTableType, double>;
  using InterpolatorPointer = typename InterpolatorType::Pointer;

  LUT()
    : m_LookupTableDataPointer(nullptr)
    , m_Interpolator(InterpolatorType::New()){};
  ~LUT() = default;

  /** Get/Set the lookup table. */
  LookupTablePointer
  GetLookupTable()
  {
    return m_LookupTablePointer;
  }
  void
  SetLookupTable(LookupTablePointer lut)
  {
    m_LookupTablePointer = lut;
    m_LookupTableDataPointer = lut->GetBufferPointer();
    m_InverseLUTSpacing = 1. / m_LookupTablePointer->GetSpacing()[0];
    m_Interpolator->SetInputImage(lut);
  }

  LookupTableDataPointer
  GetLookupTable() const
  {
    return m_LookupTableDataPointer;
  }

  bool
  operator!=(const LUT & lut) const
  {
    return m_LookupTableDataPointer != lut.GetLookupTable();
  }
  bool
  operator==(const LUT & lut) const
  {
    return m_LookupTableDataPointer == lut.GetLookupTable();
  }

  inline TOutput
  operator()(const TInput & val) const;

private:
  LookupTablePointer     m_LookupTablePointer;
  LookupTableDataPointer m_LookupTableDataPointer;
  InterpolatorPointer    m_Interpolator;
  double                 m_InverseLUTSpacing;
};

template <class TInput, class TOutput>
TOutput
LUT<TInput, TOutput>::operator()(const TInput & val) const
{
  return m_LookupTableDataPointer[val];
}

template <>
inline float
LUT<float, float>::operator()(const float & val) const
{
  InterpolatorType::ContinuousIndexType index;
  index[0] = m_InverseLUTSpacing * (val - m_LookupTablePointer->GetOrigin()[0]);
  return float(m_Interpolator->EvaluateAtContinuousIndex(index));
}

template <>
inline double
LUT<double, double>::operator()(const double & val) const
{
  InterpolatorType::ContinuousIndexType index;
  index[0] = m_InverseLUTSpacing * (val - m_LookupTablePointer->GetOrigin()[0]);
  return double(m_Interpolator->EvaluateAtContinuousIndex(index));
}

} // end namespace Functor

/** \class LookupTableImageFilter
 * \brief Converts values of an input image using lookup table.
 *
 * The lookup table is passed via a functor of type Functor::LUT. If the image
 * is of type integer, it directly reads the corresponding value in the lookup
 * table. If the lookup table is of type double or float, it uses the meta-
 * information of the lookup table (origin and spacing) to locate a continuous
 * index and interpolate at the corresponding location.
 *
 * \author Simon Rit
 *
 * \ingroup RTK ImageToImageFilter
 */
template <class TInputImage, class TOutputImage>
class ITK_EXPORT LookupTableImageFilter
  : public itk::UnaryFunctorImageFilter<TInputImage,
                                        TOutputImage,
                                        Functor::LUT<typename TInputImage::PixelType, typename TOutputImage::PixelType>>
{
public:
#if ITK_VERSION_MAJOR == 5 && ITK_VERSION_MINOR == 1
  ITK_DISALLOW_COPY_AND_ASSIGN(LookupTableImageFilter);
#else
  ITK_DISALLOW_COPY_AND_MOVE(LookupTableImageFilter);
#endif

  /** Lookup table type definition. */
  using FunctorType = Functor::LUT<typename TInputImage::PixelType, typename TOutputImage::PixelType>;
  using LookupTableType = typename FunctorType::LookupTableType;

  /** Standard class type alias. */
  using Self = LookupTableImageFilter;
  using Superclass = itk::UnaryFunctorImageFilter<TInputImage, TOutputImage, FunctorType>;
  using Pointer = itk::SmartPointer<Self>;
  using ConstPointer = itk::SmartPointer<const Self>;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Runtime information support. */
  itkTypeMacro(LookupTableImageFilter, itk::UnaryFunctorImageFilter);

  /** Set lookup table. */
  virtual void
  SetLookupTable(LookupTableType * _arg)
  {
    // Idem as itkSetObjectMacro + call to functor SetLookupTableDataPointer
    itkDebugMacro("setting "
                  << "LookupTable"
                     " to "
                  << _arg);
    if (this->m_LookupTable != _arg || (_arg && _arg->GetTimeStamp() > this->GetTimeStamp()))
    {
      this->m_LookupTable = _arg;
      this->Modified();
      this->GetFunctor().SetLookupTable(_arg);
    }
  }

  /** Get lookup table. */
  itkGetModifiableObjectMacro(LookupTable, LookupTableType);

  /** Update the LUT before using it to process the data in case it is the
   * result of a pipeline. */
  void
  BeforeThreadedGenerateData() override;

protected:
  LookupTableImageFilter() = default;
  ~LookupTableImageFilter() override = default;
  typename LookupTableType::Pointer m_LookupTable;
};

} // end namespace rtk

#ifndef ITK_MANUAL_INSTANTIATION
#  include "rtkLookupTableImageFilter.hxx"
#endif

#endif
