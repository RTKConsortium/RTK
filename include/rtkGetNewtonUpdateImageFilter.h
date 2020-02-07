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
#ifndef rtkGetNewtonUpdateImageFilter_h
#define rtkGetNewtonUpdateImageFilter_h

#include "itkImageToImageFilter.h"
#include "rtkMacro.h"

namespace rtk
{
/** \class GetNewtonUpdateImageFilter
 * \brief Computes update from gradient and Hessian in Newton's method
 *
 * This filter takes in inputs the gradient G (input 1) and the Hessian H (input 2)
 * of a cost function, and computes the update U (output), by U = H^{-1} * G.
 * In Newton's method, the quantity to add to the current iterate in order to get the next
 * iterate is actually -U, so the minus operation has to be handled downstream.
 * It is assumed that the cost function is separable, so that each pixel can be processed
 * independently and has its own small G, H and U
 *
 * \author Cyril Mory
 *
 * \ingroup RTK
 *
 */
template <class TGradient,
          class THessian = itk::Image<itk::Vector<typename TGradient::PixelType::ValueType,
                                                  TGradient::PixelType::Dimension * TGradient::PixelType::Dimension>,
                                      TGradient::ImageDimension>>
class GetNewtonUpdateImageFilter : public itk::ImageToImageFilter<TGradient, TGradient>
{
public:
  ITK_DISALLOW_COPY_AND_ASSIGN(GetNewtonUpdateImageFilter);

  /** Standard class type alias. */
  using Self = GetNewtonUpdateImageFilter;
  using Superclass = itk::ImageToImageFilter<TGradient, TGradient>;
  using Pointer = itk::SmartPointer<Self>;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(GetNewtonUpdateImageFilter, itk::ImageToImageFilter);

  /** Convenient parameters extracted from template types */
  static constexpr unsigned int nChannels = TGradient::PixelType::Dimension;

  /** Convenient type alias */
  using dataType = typename TGradient::PixelType::ValueType;

  /** Set methods for all inputs, since they have different types */
  void
  SetInputGradient(const TGradient * gradient);
  void
  SetInputHessian(const THessian * hessian);

protected:
  GetNewtonUpdateImageFilter();
  ~GetNewtonUpdateImageFilter() override = default;

  void
  GenerateInputRequestedRegion() override;

  /** Does the real work. */
  void
  DynamicThreadedGenerateData(const typename TGradient::RegionType & outputRegionForThread) override;

  /** Getters for the inputs */
  typename TGradient::ConstPointer
  GetInputGradient();
  typename THessian::ConstPointer
  GetInputHessian();
};
} // namespace rtk


#ifndef ITK_MANUAL_INSTANTIATION
#  include "rtkGetNewtonUpdateImageFilter.hxx"
#endif

#endif
