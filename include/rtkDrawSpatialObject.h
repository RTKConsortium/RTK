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

#ifndef rtkDrawSpatialObject_h
#define rtkDrawSpatialObject_h

#include <itkPoint.h>
#include "rtkConvertEllipsoidToQuadricParametersFunction.h"
#include <iostream>

namespace rtk
{
/** \class DrawSpatialObject
 * \brief Base class for a 3D object. rtk::DrawImageFilter fills uses it (and IsInside) to fill a volume.
 *
 * \author Mathieu Dupont
 *
 */

class DrawSpatialObject
{

public:

  typedef double ScalarType;
  DrawSpatialObject(){}
  typedef itk::Point< ScalarType, 3 > PointType;

  /** Returns true if a point is inside the object. */
  virtual bool IsInside(const PointType & point) const = 0;

};

}

#endif
