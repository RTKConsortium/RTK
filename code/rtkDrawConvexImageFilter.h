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
 * \brief Base Class for drawing a 3D image by using a DrawConvex. Uses a functor to fill the image.
 *
 * \author Mathieu Dupont, Simon Rit
 *
 */

template < class TInputImage,
           class TOutputImage >
class DrawConvexImageFilter :
  public itk::InPlaceImageFilter<TInputImage,TOutputImage>
{

public:
  /** Standard class typedefs. */
  typedef DrawConvexImageFilter                             Self;
  typedef itk::InPlaceImageFilter<TInputImage,TOutputImage> Superclass;
  typedef itk::SmartPointer<Self>                           Pointer;
  typedef itk::SmartPointer<const Self>                     ConstPointer;

  /** Convenient typedefs. */
  typedef typename TOutputImage::RegionType OutputImageRegionType;
  typedef ConvexShape::Pointer             ConvexShapePointer;
  typedef ConvexShape::ScalarType          ScalarType;
  typedef ConvexShape::PointType           PointType;
  typedef ConvexShape::VectorType          VectorType;

  /** Method for creation through the object factory. */
  itkNewMacro ( Self );

  /** Run-time type information (and related methods). */
  itkTypeMacro ( DrawConvexImageFilter, itk::InPlaceImageFilter );

  /** Get / Set the object pointer to the geometry. */
  itkGetObjectMacro(ConvexShape, ConvexShape);
  itkSetObjectMacro(ConvexShape, ConvexShape);

protected:
  DrawConvexImageFilter();
  ~DrawConvexImageFilter() {}

  /** ConvexShape must be created in the BeforeThreadedGenerateData in the
   * daugter classes. */
  void BeforeThreadedGenerateData() ITK_OVERRIDE;

  /** Apply changes to the input image requested region. */
  void ThreadedGenerateData( const OutputImageRegionType& outputRegionForThread,
                             ThreadIdType threadId ) ITK_OVERRIDE;

private:
  DrawConvexImageFilter ( const Self& ); //purposely not implemented
  void operator=(const Self&);           //purposely not implemented

  ConvexShapePointer      m_ConvexShape;
  ScalarType              m_Density;
  std::vector<VectorType> m_PlaneDirections;
  std::vector<PointType>  m_PlanePositions;
};


} // end namespace rtk

#ifndef ITK_MANUAL_INSTANTIATION
#include "rtkDrawConvexImageFilter.hxx"
#endif

#endif
