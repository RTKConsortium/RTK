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

#ifndef __rtkDrawCylinderImageFilter_h
#define __rtkDrawCylinderImageFilter_h


#include "rtkDrawImageFilter.h"
#include "rtkConvertEllipsoidToQuadricParametersFunction.h"
#include "rtkConfiguration.h"
#include <vector>

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
  template <class TInputImage, 
            class TOutputImage, 
	    typename TFunction = itk::Functor::Add2<typename TInputImage::PixelType,
                                                    typename TInputImage::PixelType,
                                                    typename TOutputImage::PixelType>
                                                    >
  class DrawCylinderImageFilter : 
  public DrawImageFilter< TInputImage,
                         TOutputImage, 
			 DrawCylinderSpatialObject,
			 TFunction
			 >
  {
       public:
  /** Standard class typedefs. */
  typedef DrawCylinderImageFilter                               Self;
  typedef itk::InPlaceImageFilter<TInputImage,TOutputImage> Superclass;
  typedef itk::SmartPointer<Self>                           Pointer;
  typedef itk::SmartPointer<const Self>                     ConstPointer;
  typedef typename TOutputImage::RegionType                 OutputImageRegionType;
  
  typedef itk::Vector<double,3>                             VectorType;
  typedef rtk::ConvertEllipsoidToQuadricParametersFunction  EQPFunctionType;
 

  
  void SetAxis(VectorType Axis)
  {
    if ( Axis == this->m_spatialObject.m_Axis )
      {
      return;
      }
    this->m_spatialObject.m_Axis = Axis;
    this->m_spatialObject.sqpFunctor->Translate(this->m_spatialObject.m_Axis);
    this->m_spatialObject.sqpFunctor->Rotate(this->m_spatialObject.m_Angle, this->m_spatialObject.m_Center);
    this->Modified();
  }  
  
  
   void SetCenter(VectorType Center)
  {
    if ( Center == this->m_spatialObject.m_Center )
      {
      return;
      }
    this->m_spatialObject.m_Center = Center;
    this->m_spatialObject.sqpFunctor->Translate(this->m_spatialObject.m_Axis);
    this->m_spatialObject.sqpFunctor->Rotate(this->m_spatialObject.m_Angle, this->m_spatialObject.m_Center);
    this->Modified();
  }
  
  void SetAngle(double Angle)
  {
    if ( Angle == this->m_spatialObject.m_Angle )
      {
      return;
      }
    this->m_spatialObject.m_Angle = Angle;
    this->m_spatialObject.sqpFunctor->Translate(this->m_spatialObject.m_Axis);
    this->m_spatialObject.sqpFunctor->Rotate(this->m_spatialObject.m_Angle, this->m_spatialObject.m_Center);
    this->Modified();
  }
  
  
  
  
  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(DrawCylinderImageFilter, DrawImageFilter);
  
  protected:
  DrawCylinderImageFilter();
  virtual ~DrawCylinderImageFilter() {};
//   virtual void ThreadedGenerateData( const OutputImageRegionType& outputRegionForThread, ThreadIdType threadId ) ITK_OVERRIDE;
    
  
  
  
  private:
      DrawCylinderImageFilter(const Self&); //purposely not implemented
      void operator=(const Self&);            //purposely not implemented
 
  
  
  }; 

} // end namespace rtk

#ifndef ITK_MANUAL_INSTANTIATION
#include "rtkDrawCylinderImageFilter.txx"
#endif

#endif
