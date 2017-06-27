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
   * \ingroup ImageToImageFilter
   */

template<class TInputImage, class TOutputImage>
class ITK_EXPORT HilbertImageFilter :
  public itk::ImageToImageFilter<TInputImage, TOutputImage>
{
public:
  /** Standard class typedefs. */
  typedef HilbertImageFilter                                 Self;
  typedef itk::ImageToImageFilter<TInputImage, TOutputImage> Superclass;
  typedef itk::SmartPointer<Self>                            Pointer;
  typedef itk::SmartPointer<const Self>                      ConstPointer;

  /** Standard New method. */
  itkNewMacro(Self);

  /** Runtime information support. */
  itkTypeMacro(HilbertImageFilter, itk::ImageToImageFilter);

protected:
  HilbertImageFilter(){}
  ~HilbertImageFilter() {}

  void GenerateData() ITK_OVERRIDE;

private:
  HilbertImageFilter(const Self&);  //purposely not implemented
  void operator=(const Self&);      //purposely not implemented
}; // end of class

} // end of namespace rtk

#ifndef ITK_MANUAL_INSTANTIATION
#include "rtkHilbertImageFilter.hxx"
#endif

#endif
