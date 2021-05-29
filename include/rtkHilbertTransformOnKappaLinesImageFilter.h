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

#ifndef rtkHilbertTransformOnKappaLinesImageFilter_h
#define rtkHilbertTransformOnKappaLinesImageFilter_h

#include "rtkKatsevichForwardBinningImageFilter.h"
#include "rtkFFTHilbertImageFilter.h"
#include "rtkKatsevichBackwardBinningImageFilter.h"

#include <itkInPlaceImageFilter.h>

#include <itkNumericTraits.h>


namespace rtk
{

/** \class katsHilbertTransformOnKappaLinesImageFilter
 * \brief Implements the Hilbert transform as described in Noo et al., PMB, 2003.
 * It is 3-step :
 *  - Resample the data in v on the psi angle values
 *  - Apply 1D Hilbert transform on the resampled psi-direction.
 *  - Resample the data back to the usual vertical coordinates.
 *
 *  * \dot
 * digraph HilbertTransformOnKappaLinesImageFilter {
 * node [shape=box];
 * 1 [ label="rtk::KatsevichForwardBinningImageFilter" URL="\ref rtk::KatsevichForwardBinningImageFilter"];
 * 2 [ label="rtk::FFTHilbertImageFilter" URL="\ref rtk::FFTHilbertImageFilter"];
 * 3 [ label="rtk::KatsevichBackwardBinningImageFilter" URL="\ref rtk::KatsevichBackwardBinningImageFilter"];
 * 1 -> 2;
 * 2 -> 3;
 * }
 * \enddot
 *
 *
 * \author Jerome Lesaint and Alexandre Esa
 *
 * \test
 *
 * \ingroup
 */
template <class TInputImage, class TOutputImage>
class ITK_EXPORT HilbertTransformOnKappaLinesImageFilter : public itk::ImageToImageFilter<TInputImage, TOutputImage>
{
public:
  ITK_DISALLOW_COPY_AND_ASSIGN(HilbertTransformOnKappaLinesImageFilter);

  using Self = HilbertTransformOnKappaLinesImageFilter;
  using Superclass = itk::ImageToImageFilter<TInputImage, TOutputImage>;
  using Pointer = itk::SmartPointer<Self>;
  using ConstPointer = itk::SmartPointer<const Self>;


  /** Method for creation through object factory */
  itkNewMacro(Self);

  /** Run-time type information */
  itkTypeMacro(HilbertTransformOnKappaLinesImageFilter, ImageToImageFilter);

  using InputImageType = TInputImage;
  using OutputImageType = TOutputImage;

  using GeometryType = rtk::ThreeDHelicalProjectionGeometry;
  using GeometryPointer = GeometryType::Pointer;

  itkGetMacro(Geometry, GeometryPointer);
  itkSetMacro(Geometry, GeometryPointer);

protected:
  HilbertTransformOnKappaLinesImageFilter();
  ~HilbertTransformOnKappaLinesImageFilter() override = default;

protected:
  using ForwardBinningType = KatsevichForwardBinningImageFilter<InputImageType, OutputImageType>;
  using FFTHilbertType = FFTHilbertImageFilter<InputImageType, OutputImageType>;
  using BackwardBinningType = KatsevichBackwardBinningImageFilter<InputImageType, OutputImageType>;

  void
  GenerateData() override;

  void
  GenerateOutputInformation() override;

  /** Display */
  void
  PrintSelf(std::ostream & os, itk::Indent indent) const override;

private:
  typename ForwardBinningType::Pointer  m_ForwardFilter;
  typename FFTHilbertType::Pointer      m_HilbertFilter;
  typename BackwardBinningType::Pointer m_BackwardFilter;

  GeometryPointer m_Geometry;

}; // end of class

} // end namespace rtk

#ifndef ITK_MANUAL_INSTANTIATION
#  include "rtkHilbertTransformOnKappaLinesImageFilter.hxx"
#endif

#endif
