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

#include "rtkReg23ProjectionGeometry.h"

rtk::Reg23ProjectionGeometry::Reg23ProjectionGeometry()
  : rtk::ThreeDCircularProjectionGeometry()
{
}

rtk::Reg23ProjectionGeometry::~Reg23ProjectionGeometry()
{
}

bool rtk::Reg23ProjectionGeometry::AddReg23Projection(
    const PointType &sourcePosition, const PointType &detectorPosition,
    const VectorType &detectorRowVector, const VectorType &detectorColumnVector)
{
  return Superclass::AddProjection(sourcePosition, detectorPosition,
                                   detectorRowVector, detectorColumnVector);
}
