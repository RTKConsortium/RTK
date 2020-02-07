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

#ifndef rtkDrawConvexImageFilter_h
#define rtkDrawConvexImageFilter_h


#include <itkInPlaceImageFilter.h>
#include "rtkConvexShape.h"
#include "rtkMacro.h"

namespace rtk
{

/** \class DrawConvexImageFilter
 * \brief Draws a rtk::ConvexShape in a 3D image.
 *
 * \test rtkforbildtest.cxx
 *
 * \author Mathieu Dupont, Simon Rit
 *
 * \ingroup RTK
 *
 */

template <class TInputImage, class TOutputImage>
class DrawConvexImageFilter : public itk::InPlaceImageFilter<TInputImage, TOutputImage>
{
public:
  ITK_DISALLOW_COPY_AND_ASSIGN(DrawConvexImageFilter);

  /** Standard class type alias. */
  using Self = DrawConvexImageFilter;
  using Superclass = itk::InPlaceImageFilter<TInputImage, TOutputImage>;
  using Pointer = itk::SmartPointer<Self>;
  using ConstPointer = itk::SmartPointer<const Self>;

  /** Convenient type alias. */
  using OutputImageRegionType = typename TOutputImage::RegionType;
  using ConvexShapePointer = ConvexShape::Pointer;
  using ScalarType = ConvexShape::ScalarType;
  using PointType = ConvexShape::PointType;
  using VectorType = ConvexShape::VectorType;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(DrawConvexImageFilter, itk::InPlaceImageFilter);

  /** Get / Set the object pointer to the ConvexShape. */
  itkGetModifiableObjectMacro(ConvexShape, ConvexShape);
  itkSetObjectMacro(ConvexShape, ConvexShape);

protected:
  DrawConvexImageFilter();
  ~DrawConvexImageFilter() override = default;

  /** ConvexShape must be created in the BeforeThreadedGenerateData in the
   * daugter classes. */
  void
  BeforeThreadedGenerateData() override;

  /** Apply changes to the input image requested region. */
  void
  DynamicThreadedGenerateData(const OutputImageRegionType & outputRegionForThread) override;

private:
  ConvexShapePointer m_ConvexShape;
};


} // end namespace rtk

#ifndef ITK_MANUAL_INSTANTIATION
#  include "rtkDrawConvexImageFilter.hxx"
#endif

#endif
