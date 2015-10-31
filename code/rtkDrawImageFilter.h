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

#ifndef __rtkDrawImageFilter_h
#define __rtkDrawImageFilter_h


#include <itkInPlaceImageFilter.h>
#include "rtkConvertEllipsoidToQuadricParametersFunction.h"
#include "rtkDrawSpatialObject.h"


namespace rtk
{

  template <class TInputImage, class TOutputImage, class TSpatialObject>
class ITK_EXPORT DrawImageFilter :
  public itk::InPlaceImageFilter<TInputImage,TOutputImage>  
  {
    
    public:
  /** Standard class typedefs. */
  typedef DrawImageFilter                               Self;
  typedef itk::InPlaceImageFilter<TInputImage,TOutputImage> Superclass;
  typedef itk::SmartPointer<Self>                           Pointer;
  typedef itk::SmartPointer<const Self>                     ConstPointer;
  typedef typename TOutputImage::RegionType                 OutputImageRegionType;
  
  
  typedef TSpatialObject                              SpatialObject;
  
  
  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(DrawImageFilter, InPlaceImageFilter);
  
  itkSetMacro(Density, double);
  itkGetMacro(Density, double);
    
    
    protected:
  DrawImageFilter();
  virtual ~DrawImageFilter() {};  
  virtual void ThreadedGenerateData( const OutputImageRegionType& outputRegionForThread, ThreadIdType threadId ) ITK_OVERRIDE;
  TSpatialObject m_spatialObject;
  
  
  
  private:
  DrawImageFilter(const Self&); //purposely not implemented
  
  
  double         m_Density;    
    
  };
  
  
  template <class TInputImage, class TOutputImage>
  class myDrawCylinderImageFilter : 
  public DrawImageFilter<TInputImage, TOutputImage, DrawCylinderSpatialObject>
  {
       public:
  /** Standard class typedefs. */
  typedef myDrawCylinderImageFilter                               Self;
  typedef itk::InPlaceImageFilter<TInputImage,TOutputImage> Superclass;
  typedef itk::SmartPointer<Self>                           Pointer;
  typedef itk::SmartPointer<const Self>                     ConstPointer;
  typedef typename TOutputImage::RegionType                 OutputImageRegionType;
  
  typedef itk::Vector<double,3>                             VectorType;
  typedef rtk::ConvertEllipsoidToQuadricParametersFunction  EQPFunctionType;
  
  struct FigureType
  {
    FigureType():angle(0.),density(0.){};
    VectorType semiprincipalaxis;
    VectorType center;
    double     angle;
    double     density;
  }; 
  
  
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
  
  
  
  
  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(myDrawCylinderImageFilter, DrawImageFilter);
  
  protected:
  myDrawCylinderImageFilter();
  virtual ~myDrawCylinderImageFilter() {};
//   virtual void ThreadedGenerateData( const OutputImageRegionType& outputRegionForThread, ThreadIdType threadId ) ITK_OVERRIDE;
    
  
  
  
  private:
      myDrawCylinderImageFilter(const Self&); //purposely not implemented
      void operator=(const Self&);            //purposely not implemented
 
  
  
  };
  
//template <typename T, typename D>
//  using CylinderDrawImageFilter = DrawImageFilter<T,D,CylinderDrawSpatialObject<>>;
  
} // end namespace rtk

#ifndef ITK_MANUAL_INSTANTIATION
#include "rtkDrawImageFilter.txx"
#endif

#endif