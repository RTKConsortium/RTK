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
#ifndef rtkProjectionsRegionConstIteratorRayBasedWithCylindricalPanel_hxx
#define rtkProjectionsRegionConstIteratorRayBasedWithCylindricalPanel_hxx

#include "rtkProjectionsRegionConstIteratorRayBasedWithCylindricalPanel.h"
#include "rtkHomogeneousMatrix.h"
#include "rtkMacro.h"

namespace rtk
{
template< typename TImage >
ProjectionsRegionConstIteratorRayBasedWithCylindricalPanel< TImage >
::ProjectionsRegionConstIteratorRayBasedWithCylindricalPanel(const TImage *ptr,
                                                             const RegionType & region,
                                                             ThreeDCircularProjectionGeometry *geometry,
                                                             const MatrixType &postMat):
  ProjectionsRegionConstIteratorRayBased< TImage >(ptr, region, geometry, postMat),
  m_Radius(geometry->GetRadiusCylindricalDetector()),
  m_InverseRadius(1./geometry->GetRadiusCylindricalDetector())
{
  NewProjection();
  NewPixel();
}

template< typename TImage >
void
ProjectionsRegionConstIteratorRayBasedWithCylindricalPanel< TImage >
::NewProjection()
{
  // Index of the projection in the stack
  IndexValueType iProj = this->m_PositionIndex[2];

  m_SourceToIsocenterDistance = this->m_Geometry->GetSourceToIsocenterDistances()[iProj];

  // Set source position in volume indices
  // GetSourcePosition() returns coordinates in mm. Multiplying by
  // volPPToIndex gives the corresponding volume index
  this->m_SourcePosition = this->m_PostMultiplyMatrix * this->m_Geometry->GetSourcePosition(iProj);

  // Compute matrix to transform projection index to position on a flat panel
  // if the panel were flat, before accounting for the curvature
  m_ProjectionIndexTransformMatrix =
      this->m_Geometry->GetProjectionCoordinatesToDetectorSystemMatrix(iProj).GetVnlMatrix() *
      GetIndexToPhysicalPointMatrix( this->m_Image.GetPointer() ).GetVnlMatrix();

  // Get transformation from coordinate in the (u,v,u^v) coordinate system to
  // the tomography (fixed) coordinate system
  m_VolumeTransformMatrix =
      this->m_PostMultiplyMatrix.GetVnlMatrix() *
      this->m_Geometry-> GetRotationMatrices()[iProj].GetInverse();
}

template< typename TImage >
void
ProjectionsRegionConstIteratorRayBasedWithCylindricalPanel< TImage >
::NewPixel()
{
  // Position on the projection before applying rotations and m_PostMultiplyMatrix
  PointType posProj;

  // Compute point coordinate in volume depending on projection index
  for(unsigned int i=0; i<this->GetImageDimension(); i++)
    {
    posProj[i] = m_ProjectionIndexTransformMatrix[i][this->GetImageDimension()];
    for(unsigned int j=0; j<this->GetImageDimension(); j++)
      posProj[i] += m_ProjectionIndexTransformMatrix[i][j] * this->m_PositionIndex[j];
    }

  // Convert cylindrical angle to coordinates in the (u,v,u^v) coordinate system
  double a = m_InverseRadius * posProj[0];
  posProj[0] = vcl_sin(a) * m_Radius;
  posProj[2] += (1. - vcl_cos(a)) * m_Radius;

  // Rotate and apply m_PostMultiplyMatrix
  for(unsigned int i=0; i<this->GetImageDimension(); i++)
    {
    this->m_PixelPosition[i] = m_VolumeTransformMatrix[i][this->GetImageDimension()];
    for(unsigned int j=0; j<this->GetImageDimension(); j++)
      this->m_PixelPosition[i] += m_VolumeTransformMatrix[i][j] * posProj[j];
    }
}

} // end namespace itk

#endif
