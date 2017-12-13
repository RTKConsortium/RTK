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
 * \brief Computes intersection between source rays and ellipsoids,
 * in order to create the projections of a Shepp-Logan phantom resized
 * to m_PhantoScale ( default 128 ).
 *
 * \test rtkRaycastInterpolatorForwardProjectionTest.cxx,
 * rtkprojectgeometricphantomtest.cxx, rtkfdktest.cxx, rtkrampfiltertest.cxx,
 * rtkforwardprojectiontest.cxx, rtkdisplaceddetectortest.cxx,
 * rtkshortscantest.cxx
 *
 * \author Marc Vila
 *
 * \ingroup InPlaceImageFilter
 */
template <class TInputImage, class TOutputImage>
class ITK_EXPORT SheppLoganPhantomFilter:
  public ProjectGeometricPhantomImageFilter<TInputImage,TOutputImage>
{
public:
  /** Standard class typedefs. */
  typedef SheppLoganPhantomFilter                                      Self;
  typedef ProjectGeometricPhantomImageFilter<TInputImage,TOutputImage> Superclass;
  typedef itk::SmartPointer<Self>                                      Pointer;
  typedef itk::SmartPointer<const Self>                                ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(SheppLoganPhantomFilter, ProjectGeometricPhantomImageFilter);


protected:
  SheppLoganPhantomFilter();
  ~SheppLoganPhantomFilter() {}

  void GenerateData() ITK_OVERRIDE;

private:
  SheppLoganPhantomFilter(const Self&); //purposely not implemented
  void operator=(const Self&);          //purposely not implemented

};

} // end namespace rtk

#ifndef ITK_MANUAL_INSTANTIATION
#include "rtkSheppLoganPhantomFilter.hxx"
#endif

#endif
