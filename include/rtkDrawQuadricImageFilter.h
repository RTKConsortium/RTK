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

#ifndef rtkDrawQuadricImageFilter_h
#define rtkDrawQuadricImageFilter_h

#include <itkAddImageFilter.h>

#include "rtkDrawQuadricSpatialObject.h"
#include "rtkDrawImageFilter.h"
#include "rtkConvertEllipsoidToQuadricParametersFunction.h"
#include "rtkConfiguration.h"
#include "rtkDrawQuadricImageFilter.h"

namespace rtk
{

/** \class DrawQuadricImageFilter
 * \brief Draws in a 3D image user defined Quadric.
 *
 * \test rtkdrawgeometricphantomtest.cxx
 *
 * \author Marc Vila
 *
 * \ingroup InPlaceImageFilter
 */
template <class TInputImage,
          class TOutputImage,
          class TSpatialObject = DrawQuadricSpatialObject,
          typename TFunction = itk::Functor::Add2<typename TInputImage::PixelType,
                                                  typename TInputImage::PixelType,
                                                  typename TOutputImage::PixelType>
                                                  >
class ITK_EXPORT DrawQuadricImageFilter :
public DrawImageFilter< TInputImage,
                        TOutputImage,
                        DrawQuadricSpatialObject,
                        TFunction >
{
public:
  /** Standard class typedefs. */
  typedef DrawQuadricImageFilter                           Self;
  typedef DrawImageFilter<TInputImage,TOutputImage,
                          DrawQuadricSpatialObject,
                          TFunction  >                     Superclass;
  typedef itk::SmartPointer<Self>                          Pointer;
  typedef itk::SmartPointer<const Self>                    ConstPointer;
  typedef typename TOutputImage::RegionType                OutputImageRegionType;

  typedef itk::Vector<double,3>                            VectorType;
  typedef std::string                                      StringType;
  typedef rtk::ConvertEllipsoidToQuadricParametersFunction EQPFunctionType;


  void SetFigure(StringType Figure)
    {
    if (Figure ==  this->m_SpatialObject.m_Figure)
      {
      return;
      }
    this->m_SpatialObject.m_Figure = Figure;
    this->m_SpatialObject.UpdateParameters();
    this->Modified();
    }

  void SetAxis(VectorType Axis)
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


   void SetCenter(VectorType Center)
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

  void SetAngle(double Angle)
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

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(DrawQuadricImageFilter, DrawImageFilter);

protected:
  DrawQuadricImageFilter();
  ~DrawQuadricImageFilter() {}

private:
  DrawQuadricImageFilter(const Self&); //purposely not implemented
  void operator=(const Self&);         //purposely not implemented
};

} // end namespace rtk

#ifndef ITK_MANUAL_INSTANTIATION
#include "rtkDrawQuadricImageFilter.hxx"
#endif

#endif
