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

#ifndef rtkDrawSheppLoganFilter_h
#define rtkDrawSheppLoganFilter_h

#include "rtkDrawGeometricPhantomImageFilter.h"

namespace rtk
{

/** \class DrawSheppLoganFilter
 * \brief Draws a SheppLoganPhantom in a 3D image with a default scale of 128.
 *
 * \test rtkRaycastInterpolatorForwardProjectionTest.cxx,
 * rtkprojectgeometricphantomtest.cxx, rtkfdktest.cxx, rtkrampfiltertest.cxx,
 * rtkforwardprojectiontest.cxx, rtkdisplaceddetectortest.cxx,
 * rtkshortscantest.cxx, rtkforbildtest.cxx
 *
 * \author Marc Vila, Simon Rit
 *
 * \ingroup RTK InPlaceImageFilter
 */
template <class TInputImage, class TOutputImage>
class ITK_EXPORT DrawSheppLoganFilter:
  public DrawGeometricPhantomImageFilter<TInputImage,TOutputImage>
{
public:
  /** Standard class typedefs. */
  typedef DrawSheppLoganFilter                                      Self;
  typedef DrawGeometricPhantomImageFilter<TInputImage,TOutputImage> Superclass;
  typedef itk::SmartPointer<Self>                                   Pointer;
  typedef itk::SmartPointer<const Self>                             ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(DrawSheppLoganFilter, DrawGeometricPhantomImageFilter);

protected:
  DrawSheppLoganFilter();
  virtual ~DrawSheppLoganFilter() ITK_OVERRIDE {}

  void GenerateData() ITK_OVERRIDE;

private:
  DrawSheppLoganFilter(const Self&); //purposely not implemented
  void operator=(const Self&);          //purposely not implemented
};

} // end namespace rtk

#ifndef ITK_MANUAL_INSTANTIATION
#include "rtkDrawSheppLoganFilter.hxx"
#endif

#endif
