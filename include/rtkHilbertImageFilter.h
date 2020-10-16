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

#ifndef rtkHilbertImageFilter_h
#define rtkHilbertImageFilter_h

#include <itkImageToImageFilter.h>

namespace rtk
{
/** \class HilbertImageFilter
 *
 * \brief Computes the complex analytic signal of a 1D signal
 *
 * The function reproduces the Matlab function "hilbert" which computes the
 * analytic signal using the Hilbert transform. In Matlab's code, references
 * are made to:
 * [Oppenheim and Schafer, Discrete-Time signal processing, 1998]
 * [Marple, IEEE Trans Sig Proc, 1999]
 *
 * \test rtkamsterdamshroudtest
 *
 * \author Simon Rit
 *
 * \ingroup RTK ImageToImageFilter
 */

template <class TInputImage, class TOutputImage>
class ITK_EXPORT HilbertImageFilter : public itk::ImageToImageFilter<TInputImage, TOutputImage>
{
public:
#if ITK_VERSION_MAJOR == 5 && ITK_VERSION_MINOR == 1
  ITK_DISALLOW_COPY_AND_ASSIGN(HilbertImageFilter);
#else
  ITK_DISALLOW_COPY_AND_MOVE(HilbertImageFilter);
#endif

  /** Standard class type alias. */
  using Self = HilbertImageFilter;
  using Superclass = itk::ImageToImageFilter<TInputImage, TOutputImage>;
  using Pointer = itk::SmartPointer<Self>;
  using ConstPointer = itk::SmartPointer<const Self>;

  /** Standard New method. */
  itkNewMacro(Self);

  /** Runtime information support. */
  itkTypeMacro(HilbertImageFilter, itk::ImageToImageFilter);

protected:
  HilbertImageFilter() = default;
  ~HilbertImageFilter() override = default;

  void
  GenerateData() override;

}; // end of class

} // end of namespace rtk

#ifndef ITK_MANUAL_INSTANTIATION
#  include "rtkHilbertImageFilter.hxx"
#endif

#endif
