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

#ifndef rtkSheppLoganPhantomFilter_h
#define rtkSheppLoganPhantomFilter_h

#include "rtkProjectGeometricPhantomImageFilter.h"

namespace rtk
{

/** \class SheppLoganPhantomFilter
 * \brief Analytical projection of a SheppLoganPhantom with a 128 (default) scale.
 *
 * \test rtkRaycastInterpolatorForwardProjectionTest.cxx,
 * rtkprojectgeometricphantomtest.cxx, rtkfdktest.cxx, rtkrampfiltertest.cxx,
 * rtkforwardprojectiontest.cxx, rtkdisplaceddetectortest.cxx,
 * rtkshortscantest.cxx
 *
 * \author Marc Vila, Simon Rit
 *
 * \ingroup RTK InPlaceImageFilter
 */
template <class TInputImage, class TOutputImage>
class ITK_EXPORT SheppLoganPhantomFilter : public ProjectGeometricPhantomImageFilter<TInputImage, TOutputImage>
{
public:
#if ITK_VERSION_MAJOR == 5 && ITK_VERSION_MINOR == 1
  ITK_DISALLOW_COPY_AND_ASSIGN(SheppLoganPhantomFilter);
#else
  ITK_DISALLOW_COPY_AND_MOVE(SheppLoganPhantomFilter);
#endif

  /** Standard class type alias. */
  using Self = SheppLoganPhantomFilter;
  using Superclass = ProjectGeometricPhantomImageFilter<TInputImage, TOutputImage>;
  using Pointer = itk::SmartPointer<Self>;
  using ConstPointer = itk::SmartPointer<const Self>;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(SheppLoganPhantomFilter, ProjectGeometricPhantomImageFilter);

protected:
  SheppLoganPhantomFilter();
  ~SheppLoganPhantomFilter() override = default;

  void
  GenerateData() override;
};

} // end namespace rtk

#ifndef ITK_MANUAL_INSTANTIATION
#  include "rtkSheppLoganPhantomFilter.hxx"
#endif

#endif
