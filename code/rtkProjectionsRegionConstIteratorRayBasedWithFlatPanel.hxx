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
#ifndef rtkProjectionsRegionConstIteratorRayBasedWithFlatPanel_hxx
#define rtkProjectionsRegionConstIteratorRayBasedWithFlatPanel_hxx

#include "rtkProjectionsRegionConstIteratorRayBasedWithFlatPanel.h"
#include "rtkHomogeneousMatrix.h"
#include "rtkMacro.h"

namespace rtk
{
template< typename TImage >
ProjectionsRegionConstIteratorRayBasedWithFlatPanel< TImage >
::ProjectionsRegionConstIteratorRayBasedWithFlatPanel(const TImage *ptr,
                                         const RegionType & region,
                                         ThreeDCircularProjectionGeometry *geometry,
                                         const MatrixType &postMat):
  ProjectionsRegionConstIteratorRayBased< TImage >(ptr, region, geometry, postMat)
{
  NewProjection();
  NewPixel();
}

template< typename TImage >
void
ProjectionsRegionConstIteratorRayBasedWithFlatPanel< TImage >
::NewProjection()
{
  // Set source position in volume indices
  // GetSourcePosition() returns coordinates in mm. Multiplying by
  // volPPToIndex gives the corresponding volume index
  this->m_SourcePosition = this->m_PostMultiplyMatrix * this->m_Geometry->GetSourcePosition(this->m_PositionIndex[2]);

  // Compute matrix to transform projection index to volume index
  // IndexToPhysicalPointMatrix maps the 2D index of a projection's pixel to its 2D position on the detector (in mm)
  // ProjectionCoordinatesToFixedSystemMatrix maps the 2D position of a pixel on the detector to its 3D coordinates in volume's coordinates (still in mm)
  // volPPToIndex maps 3D volume coordinates to a 3D index
  m_ProjectionIndexTransformMatrix =
      this->m_PostMultiplyMatrix.GetVnlMatrix() *
      this->m_Geometry->GetProjectionCoordinatesToFixedSystemMatrix(this->m_PositionIndex[2]).GetVnlMatrix() *
      GetIndexToPhysicalPointMatrix( this->m_Image.GetPointer() ).GetVnlMatrix();
}

template< typename TImage >
void
ProjectionsRegionConstIteratorRayBasedWithFlatPanel< TImage >
::NewPixel()
{
  // Compute point coordinate in volume depending on projection index
  for(unsigned int i=0; i<this->GetImageDimension(); i++)
    {
    this->m_PixelPosition[i] = m_ProjectionIndexTransformMatrix[i][this->GetImageDimension()];
    for(unsigned int j=0; j<this->GetImageDimension(); j++)
      this->m_PixelPosition[i] += m_ProjectionIndexTransformMatrix[i][j] * this->m_PositionIndex[j];
    }
}

} // end namespace itk

#endif
