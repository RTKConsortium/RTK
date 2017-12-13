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

#ifndef rtkDrawCylinderImageFilter_h
#define rtkDrawCylinderImageFilter_h

#include "rtkDrawEllipsoidImageFilter.h"
#include "rtkConfiguration.h"

namespace rtk
{

/** \class DrawCylinderImageFilter
 * \brief Draws in a 3D image user defined Cylinder.
 *
 * \test rtkdrawgeometricphantomtest.cxx
 *
 * \author Marc Vila
 *
 * \ingroup InPlaceImageFilter
 */
template <class TInputImage, class TOutputImage>
class DrawCylinderImageFilter:
public DrawEllipsoidImageFilter< TInputImage, TOutputImage >
{
public:
  /** Standard class typedefs. */
  typedef DrawCylinderImageFilter                            Self;
  typedef DrawEllipsoidImageFilter<TInputImage,TOutputImage> Superclass;
  typedef itk::SmartPointer<Self>                            Pointer;
  typedef itk::SmartPointer<const Self>                      ConstPointer;

  /** Convenient typedefs. */
  typedef ConvexObject::ScalarType ScalarType;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(DrawCylinderImageFilter, DrawCylinderImageFilter);

protected:
  DrawCylinderImageFilter() {}
  ~DrawCylinderImageFilter() {}

  void BeforeThreadedGenerateData() ITK_OVERRIDE;

private:
  DrawCylinderImageFilter(const Self&); //purposely not implemented
  void operator=(const Self&);         //purposely not implemented
};

} // end namespace rtk

#ifndef ITK_MANUAL_INSTANTIATION
#include "rtkDrawCylinderImageFilter.hxx"
#endif

#endif
