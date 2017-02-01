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

#ifndef rtkProjectionsRegionConstIteratorRayBased_h
#define rtkProjectionsRegionConstIteratorRayBased_h

#include <itkImageConstIteratorWithIndex.h>

#include "rtkThreeDCircularProjectionGeometry.h"

namespace rtk
{
/** \class ProjectionsRegionConstIteratorRayBased
 * \brief Iterate over a projection stack with corresponding ray information
 *
 * Base class for iterating over the pixels of projection images with the
 * and having at the same time the geometry information for the corresponding
 * source to pixel ray, defined by a 3D source position and a 3D pixel position.
 * The iterator provides this information in mm, unless a 3D matrix is provided
 * to convert the mm value of the coordinates of the two points to some other
 * coordinate system. A typical example is the mm to voxel matrix to work in
 * voxel coordinates. The iterator only works with the
 * ThreeDCircularProjectionGeometry is purely virtual because this geometry
 * can handle parallel geometry with flat panels and cone-beam geometries with
 * flat and curved detectors. */
template< typename TImage >
class ProjectionsRegionConstIteratorRayBased:
    public itk::ImageConstIteratorWithIndex< TImage >
{
public:
  /** Standard class typedefs. */
  typedef ProjectionsRegionConstIteratorRayBased      Self;
  typedef itk::ImageConstIteratorWithIndex< TImage >  Superclass;

  /**
   * Index typedef support. While these were already typdef'ed in the superclass
   * they need to be redone here for this subclass to compile properly with gcc.
   */
  /** Types inherited from the Superclass */
  typedef typename Superclass::OffsetValueType OffsetValueType;
  typedef typename Superclass::RegionType      RegionType;
  typedef typename itk::Vector<double, 3>      PointType;
  typedef typename Superclass::IndexValueType  IndexValueType;

  typedef itk::Matrix< double, 3, 4 >          MatrixType;
  typedef itk::Matrix< double, 4, 4 >          HomogeneousMatrixType;

  /** Constructor establishes an iterator to walk a particular image and a
   * particular region of that image.
   * Set the matrix by which the 3D coordinates of the projection can be
   * multiplied. A typical example is the conversion from 3D physical
   * coordinates to voxel indices in an itk Image. */
  ProjectionsRegionConstIteratorRayBased(const TImage *ptr,
                                         const RegionType & region,
                                         ThreeDCircularProjectionGeometry *geometry,
                                         const MatrixType &postMat);

  static Self *
  New(const TImage *ptr,
      const RegionType & region,
      ThreeDCircularProjectionGeometry *geometry,
      const MatrixType &postMat);

  static Self *
  New(const TImage *ptr,
      const RegionType & region,
      ThreeDCircularProjectionGeometry *geometry,
      const HomogeneousMatrixType &postMat);

  static Self *
  New(const TImage *ptr,
      const RegionType & region,
      ThreeDCircularProjectionGeometry *geometry);

  /** Increment (prefix) the fastest moving dimension of the iterator's index.
   * This operator will constrain the iterator within the region (i.e. the
   * iterator will automatically wrap from the end of the row of the region
   * to the beginning of the next row of the region) up until the iterator
   * tries to moves past the last pixel of the region.  Here, the iterator
   * will be set to be one pixel past the end of the region.
   * \sa operator-- */
  Self & operator++();

  /** Go to the next pixel by simply calling the ++ operator. Should not be
   * the ++ operator should be. The function is provided for cosmetic
   * reasons, because pointers to these iterators will be used more than the
   * iterator itself. */
  void Next() {++*this;}

  /** Get ray information. A ray is described by the 3D coordinates of two points,
   * the (current) SourcePosition and the (current) PixelPosition in the
   * projection stack. */
  const PointType &GetSourcePosition()
    {
    return this->m_SourcePosition;
    }
  const PointType &GetPixelPosition()
    {
    return this->m_PixelPosition;
    }

  /** Computes and returns the vector from the source to the pixel positions,
   * i.e., PixelPosition - SourcePosition. */
  const PointType GetSourceToPixel()
    {
    return this->m_PixelPosition - this->m_SourcePosition;
    }

  /** Computes and returns a unit vector pointing from the source to the
   * current pixel, i.e., GetSourceToPixel()/||GetSourceToPixel()||. */
  const PointType GetDirection()
    {
    const PointType sourceToPixel = this->GetSourceToPixel();
    return sourceToPixel / sourceToPixel.GetNorm();
    }

protected:
  /** Init the parameters common to a new 2D projection in the 3D stack. */
  virtual void NewProjection() = 0;

  /** Init a new pixel position in a 2D projection, assuming that the
   * NewProjection method has already been called. */
  virtual void NewPixel() = 0;

  ThreeDCircularProjectionGeometry * m_Geometry;
  MatrixType                         m_PostMultiplyMatrix;
  PointType                          m_SourcePosition;
  PointType                          m_PixelPosition;
};
} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "rtkProjectionsRegionConstIteratorRayBased.hxx"
#endif

#endif
