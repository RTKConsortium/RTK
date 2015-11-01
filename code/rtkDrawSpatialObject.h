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

#ifndef __rtkDrawSpatialObject_h
#define __rtkDrawSpatialObject_h

#include <itkPoint.h>
#include "rtkConvertEllipsoidToQuadricParametersFunction.h"


namespace rtk
{

class DrawSpatialObject //: public itk::DataObject
{
  
public:  
  typedef double ScalarType;
  
  DrawSpatialObject(){};
  
  
  typedef itk::Point< ScalarType, 3 > PointType; 
  
  
  
  /** Returns true if a point is inside the object. */
  virtual bool IsInside(const PointType & point) const;    
  
};

class DrawCylinderSpatialObject : public DrawSpatialObject
{
  public:
  
  typedef double ScalarType;
  typedef rtk::ConvertEllipsoidToQuadricParametersFunction  EQPFunctionType;
  
  
  DrawCylinderSpatialObject();
 
  
  typedef itk::Point< ScalarType, 3 > PointType;
  typedef itk::Vector<double,3>                             VectorType;


/** Returns true if a point is inside the object. */
  virtual bool IsInside(const PointType & point) const;    
  
public:
  EQPFunctionType::Pointer sqpFunctor;
  VectorType     m_Axis;
  VectorType     m_Center;
  ScalarType     m_Angle;  
  
};


};



#endif