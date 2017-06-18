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

#ifndef rtkProjectionsRegionConstIteratorRayBasedWithFlatPanel_h
#define rtkProjectionsRegionConstIteratorRayBasedWithFlatPanel_h

#include "rtkProjectionsRegionConstIteratorRayBased.h"

namespace rtk
{
/** \class ProjectionsRegionConstIteratorRayBasedWithFlatPanel
 * \brief Implements a ray-based iterator for a point source and a flat panel 
 */
template< typename TImage >
class ProjectionsRegionConstIteratorRayBasedWithFlatPanel:
    public ProjectionsRegionConstIteratorRayBased< TImage >
{
public:
  /** Standard class typedefs. */
  typedef ProjectionsRegionConstIteratorRayBasedWithFlatPanel Self;
  typedef ProjectionsRegionConstIteratorRayBased< TImage >    Superclass;

  /**
   * Index typedef support. While these were already typdef'ed in the superclass
   * they need to be redone here for this subclass to compile properly with gcc.
   */
  /** Types inherited from the Superclass */
  typedef typename Superclass::OffsetValueType OffsetValueType;
  typedef typename Superclass::RegionType      RegionType;
  typedef typename itk::Vector<double, 3>      PointType;
  typedef typename Superclass::MatrixType      MatrixType;
  typedef itk::Matrix< double, 4, 4 >          HomogeneousMatrixType;

  /** Constructor establishes an iterator to walk a particular image and a
   * particular region of that image.
   * Set the matrix by which the 3D coordinates of the projection can be
   * multiplied. A typical example is the conversion from 3D physical
   * coordinates to voxel indices in an itk Image. */
  ProjectionsRegionConstIteratorRayBasedWithFlatPanel(const TImage *ptr,
                                                      const RegionType & region,
                                                      ThreeDCircularProjectionGeometry *geometry,
                                                      const MatrixType &postMat);

protected:
  /** Init the parameters common to a new 2D projection in the 3D stack. */
  inline virtual void NewProjection();

  /** Init a new pixel position in a 2D projection, assuming that the
   * NewProjection method has already been called. */
  inline virtual void NewPixel();

  MatrixType m_ProjectionIndexTransformMatrix;
};
} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "rtkProjectionsRegionConstIteratorRayBasedWithFlatPanel.hxx"
#endif

#endif
