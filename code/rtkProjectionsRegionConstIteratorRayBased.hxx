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
#ifndef rtkProjectionsRegionConstIteratorRayBased_hxx
#define rtkProjectionsRegionConstIteratorRayBased_hxx

#include "rtkProjectionsRegionConstIteratorRayBased.h"
#include "rtkProjectionsRegionConstIteratorRayBasedWithFlatPanel.h"
#include "rtkProjectionsRegionConstIteratorRayBasedWithCylindricalPanel.h"
#include "rtkHomogeneousMatrix.h"
#include "rtkMacro.h"

namespace rtk
{
template< typename TImage >
ProjectionsRegionConstIteratorRayBased< TImage >
::ProjectionsRegionConstIteratorRayBased(const TImage *ptr,
                                         const RegionType & region,
                                         ThreeDCircularProjectionGeometry *geometry,
                                         const MatrixType &postMat):
  itk::ImageConstIteratorWithIndex< TImage >(ptr, region),
  m_Geometry(geometry),
  m_PostMultiplyMatrix(postMat)
{
}

template< typename TImage >
ProjectionsRegionConstIteratorRayBased< TImage > &
ProjectionsRegionConstIteratorRayBased< TImage >
::operator++()
{
  // This code is copy pasted from itkProjectionsRegionConstIteratorRayBased since
  // operators are not virtual.
  this->m_Remaining = false;
  unsigned int in = 0;
  for ( in = 0; in < TImage::ImageDimension; in++ )
    {
    this->m_PositionIndex[in]++;
    if ( this->m_PositionIndex[in] < this->m_EndIndex[in] )
      {
      this->m_Position += this->m_OffsetTable[in];
      this->m_Remaining = true;
      break;
      }
    else
      {
      this->m_Position -= this->m_OffsetTable[in]
                          * ( static_cast< OffsetValueType >( this->m_Region.GetSize()[in] ) - 1 );
      this->m_PositionIndex[in] = this->m_BeginIndex[in];
      }
    }

  if ( !this->m_Remaining ) // It will not advance here otherwise
    {
    this->m_Position = this->m_End;
    return *this;
    }

  if(in == 2)
    {
    NewProjection();
    }
  NewPixel();

  return *this;
}

template< typename TImage >
ProjectionsRegionConstIteratorRayBased< TImage > *
ProjectionsRegionConstIteratorRayBased< TImage >
::New(const TImage *ptr,
      const RegionType & region,
      ThreeDCircularProjectionGeometry *geometry,
      const MatrixType &postMat)
{
  if(geometry->GetRadiusCylindricalDetector() == 0.)
    {
    typedef ProjectionsRegionConstIteratorRayBasedWithFlatPanel<TImage> IteratorType;
    return new IteratorType(ptr, region, geometry, postMat);
    }
  else
    {
    typedef ProjectionsRegionConstIteratorRayBasedWithCylindricalPanel<TImage> IteratorType;
    return new IteratorType(ptr, region, geometry, postMat);
    }
}

template< typename TImage >
ProjectionsRegionConstIteratorRayBased< TImage > *
ProjectionsRegionConstIteratorRayBased< TImage >
::New(const TImage *ptr,
      const RegionType & region,
      ThreeDCircularProjectionGeometry *geometry,
      const HomogeneousMatrixType &postMat)
{
  MatrixType pm;
  for(unsigned int i=0; i<MatrixType::RowDimensions; i++)
    for(unsigned int j=0; j<MatrixType::ColumnDimensions; j++)
        pm[i][j] = postMat[i][j];
  return New(ptr, region, geometry, pm);
}

template<class TImage>
rtk::ProjectionsRegionConstIteratorRayBased<TImage>*
ProjectionsRegionConstIteratorRayBased< TImage >
::New(const TImage *ptr,
      const RegionType & region,
      ThreeDCircularProjectionGeometry *geometry)
{
  MatrixType postMat;
  postMat.SetIdentity();
  return New(ptr, region, geometry, postMat);
}

} // end namespace itk

#endif
