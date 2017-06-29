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

#ifndef rtkForwardProjectionImageFilter_h
#define rtkForwardProjectionImageFilter_h

#include <itkInPlaceImageFilter.h>
#include "rtkThreeDCircularProjectionGeometry.h"
#include "rtkMacro.h"

namespace rtk
{

/** \class ForwardProjectionImageFilter
 * \brief Base class for forward projection, i.e. accumulation along x-ray lines.
 *
 * \author Simon Rit
 *
 * \ingroup Projector
 */
template <class TInputImage, class TOutputImage>
class ForwardProjectionImageFilter :
  public itk::InPlaceImageFilter<TInputImage,TOutputImage>
{
public:
  /** Standard class typedefs. */
  typedef ForwardProjectionImageFilter                      Self;
  typedef itk::InPlaceImageFilter<TInputImage,TOutputImage> Superclass;
  typedef itk::SmartPointer<Self>                           Pointer;
  typedef itk::SmartPointer<const Self>                     ConstPointer;

  typedef rtk::ThreeDCircularProjectionGeometry             GeometryType;
  typedef typename GeometryType::Pointer                    GeometryPointer;

  /** Run-time type information (and related methods). */
  itkTypeMacro(ForwardProjectionImageFilter, itk::InPlaceImageFilter);

  /** Get / Set the object pointer to projection geometry */
  itkGetMacro(Geometry, GeometryPointer);
  itkSetMacro(Geometry, GeometryPointer);

protected:
  ForwardProjectionImageFilter() : m_Geometry(ITK_NULLPTR) {
    this->SetNumberOfRequiredInputs(2); this->SetInPlace( true );
  };

  ~ForwardProjectionImageFilter() {}

  /** Apply changes to the input image requested region. */
  void GenerateInputRequestedRegion() ITK_OVERRIDE;

  /** The two inputs should not be in the same space so there is nothing
   * to verify. */
  void VerifyInputInformation() ITK_OVERRIDE {}

private:
  ForwardProjectionImageFilter(const Self&); //purposely not implemented
  void operator=(const Self&);            //purposely not implemented

  /** RTK geometry object */
  GeometryPointer m_Geometry;
};

} // end namespace rtk

#ifndef ITK_MANUAL_INSTANTIATION
#include "rtkForwardProjectionImageFilter.hxx"
#endif

#endif
