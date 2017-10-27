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

#ifndef rtkDrawCubeImageFilter_h
#define rtkDrawCubeImageFilter_h

#include <itkInPlaceImageFilter.h>
#include <itkVector.h>
#include <itkAddImageFilter.h>

#include "rtkDrawImageFilter.h"
#include "rtkDrawSpatialObject.h"
#include "rtkThreeDCircularProjectionGeometry.h"
#include "rtkConvertEllipsoidToQuadricParametersFunction.h"
#include "rtkConfiguration.h"

#include <vector>

namespace rtk
{
class DrawCubeSpatialObject : public DrawSpatialObject
{
public:
  typedef double                            ScalarType;
  typedef itk::Point< ScalarType, 3 >       PointType;
  typedef itk::Vector<double,6>             AxisType;
  typedef itk::Vector<double,3>             VectorType;
  typedef std::string                       StringType;


  DrawCubeSpatialObject()
  {
    m_Axis.Fill ( 50. );
    m_Center.Fill ( 0. );
    m_Angle = 0.;
  }

  /** Returns true if a point is inside the object. */
  bool IsInside ( const PointType & point ) const ITK_OVERRIDE;
  void UpdateParameters();


  VectorType      m_Axis;
  VectorType      m_Center;
  ScalarType      m_Angle;
private:
  AxisType        m_Semiprincipalaxis;

};

/** \class DrawCubeImageFilter
 * \brief Draws in a 3D image a user defined cube/box.
 *
 * \test rtksarttest.cxx, rtkmotioncompensatedfdktest.cxx
 *
 * \author Marc Vila
 *
 * \ingroup InPlaceImageFilter
 */
template <class TInputImage,
          class TOutputImage,
          class TSpatialObject = DrawCubeSpatialObject,
          typename TFunction = itk::Functor::Add2<typename TInputImage::PixelType,
                                                  typename TInputImage::PixelType,
                                                  typename TOutputImage::PixelType>
         >
class ITK_EXPORT DrawCubeImageFilter :
public DrawImageFilter< TInputImage,
                        TOutputImage,
                        DrawCubeSpatialObject,
                        TFunction >
{
public:
  /** Standard class typedefs. */
  typedef DrawCubeImageFilter                                 Self;
  typedef DrawImageFilter < TInputImage,TOutputImage,
                            DrawCubeSpatialObject,
                            TFunction  >                      Superclass;
  typedef itk::SmartPointer<Self>                             Pointer;
  typedef itk::SmartPointer<const Self>                       ConstPointer;
  typedef typename TOutputImage::RegionType                   OutputImageRegionType;

  typedef itk::Vector<double,3>                               VectorType;
  typedef std::string                                         StringType;


  /** Method for creation through the object factory. */
  itkNewMacro ( Self );

  /** Run-time type information (and related methods). */
  itkTypeMacro ( DrawCubeImageFilter, DrawImageFilter );

  void SetAxis ( VectorType Axis )
    {
    if ( Axis == this->m_SpatialObject.m_Axis )
      {
        return;
      }
    this->m_SpatialObject.m_Axis = Axis;
    this->m_SpatialObject.UpdateParameters();
    this->Modified();
    }

  VectorType GetAxis()
    {
    return this->m_SpatialObject.m_Axis;
    }


  void SetCenter ( VectorType Center )
    {
    if ( Center == this->m_SpatialObject.m_Center )
      {
        return;
      }
    this->m_SpatialObject.m_Center = Center;
    this->m_SpatialObject.UpdateParameters();
    this->Modified();
    }

  VectorType GetCenter()
    {
    return this->m_SpatialObject.m_Center;
    }

  void SetAngle ( double Angle )
    {
    if ( Angle == this->m_SpatialObject.m_Angle )
      {
        return;
      }
    this->m_SpatialObject.m_Angle = Angle;
    this->m_SpatialObject.UpdateParameters();
    this->Modified();
    }

  double GetAngle()
    {
    return this->m_SpatialObject.m_Angle;
    }

protected:
  DrawCubeImageFilter();
  ~DrawCubeImageFilter() {}

private:
  DrawCubeImageFilter ( const Self& ); //purposely not implemented
  void operator= ( const Self& );      //purposely not implemented
};

} // end namespace rtk

#ifndef ITK_MANUAL_INSTANTIATION
#include "rtkDrawCubeImageFilter.hxx"
#endif

#endif
