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

#ifndef rtkDrawQuadricSpatialObject_h
#define rtkDrawQuadricSpatialObject_h

#include "rtkWin32Header.h"

#include <itkPoint.h>
#include <rtkMacro.h>

#include "rtkConvertEllipsoidToQuadricParametersFunction.h"
#include "rtkDrawSpatialObject.h"

namespace rtk
{

class RTK_EXPORT DrawQuadricSpatialObject : public DrawSpatialObject
{
  public:

    DrawQuadricSpatialObject();

    typedef double                                              ScalarType;
    typedef rtk::ConvertEllipsoidToQuadricParametersFunction    EQPFunctionType;
    typedef itk::Point< ScalarType, 3 >                         PointType;
    typedef itk::Vector<double,3>                               VectorType;
    typedef std::string                                         StringType;

  /** Returns true if a point is inside the object. */
  bool IsInside(const PointType & point) const ITK_OVERRIDE;

  void UpdateParameters();

public:
  EQPFunctionType::Pointer m_SqpFunctor;
  VectorType               m_Axis;
  VectorType               m_Center;
  ScalarType               m_Angle;
  StringType               m_Figure;

};

} // end namespace rtk

#endif
