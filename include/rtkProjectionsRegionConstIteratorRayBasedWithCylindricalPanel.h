/*=========================================================================
 *
 *  Copyright RTK Consortium
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *         https://www.apache.org/licenses/LICENSE-2.0.txt
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 *=========================================================================*/

#ifndef rtkProjectionsRegionConstIteratorRayBasedWithCylindricalPanel_h
#define rtkProjectionsRegionConstIteratorRayBasedWithCylindricalPanel_h

#include "rtkProjectionsRegionConstIteratorRayBased.h"

namespace rtk
{
/** \class ProjectionsRegionConstIteratorRayBasedWithCylindricalPanel
 *
 * \brief Implements a ray-based iterator for a point source and a cylindrical panel
 *
 * \author Simon Rit
 *
 * \ingroup RTK
 */
template <typename TImage>
class ITK_TEMPLATE_EXPORT ProjectionsRegionConstIteratorRayBasedWithCylindricalPanel
  : public ProjectionsRegionConstIteratorRayBased<TImage>
{
public:
  /** Standard class type alias. */
  using Self = ProjectionsRegionConstIteratorRayBasedWithCylindricalPanel;
  using Superclass = ProjectionsRegionConstIteratorRayBased<TImage>;

  /**
   * Index type alias support While these were already typdef'ed in the superclass
   * they need to be redone here for this subclass to compile properly with gcc.
   */
  /** Types inherited from the Superclass */
  using OffsetValueType = typename Superclass::OffsetValueType;
  using RegionType = typename Superclass::RegionType;
  using MatrixType = typename Superclass::MatrixType;
  using IndexValueType = typename Superclass::IndexValueType;

  using PointType = typename itk::Point<double, 3>;
  using VectorType = typename itk::Vector<double, 3>;
  using HomogeneousMatrixType = itk::Matrix<double, 4, 4>;

  /** Constructor establishes an iterator to walk a particular image and a
   * particular region of that image.
   * Set the matrix by which the 3D coordinates of the projection can be
   * multiplied. A typical example is the conversion from 3D physical
   * coordinates to voxel indices in an itk Image. */
  ProjectionsRegionConstIteratorRayBasedWithCylindricalPanel(const TImage *                           ptr,
                                                             const RegionType &                       region,
                                                             const ThreeDCircularProjectionGeometry * geometry,
                                                             const MatrixType &                       postMat);

protected:
  /** Init the parameters common to a new 2D projection in the 3D stack. */
  inline void
  NewProjection() override;

  /** Init a new pixel position in a 2D projection, assuming that the
   * NewProjection method has already been called. */
  inline void
  NewPixel() override;

  HomogeneousMatrixType m_ProjectionIndexTransformMatrix;
  MatrixType            m_VolumeTransformMatrix;
  double                m_Radius;
  double                m_InverseRadius;
  double                m_SourceToIsocenterDistance;
};
} // namespace rtk

#ifndef ITK_MANUAL_INSTANTIATION
#  include "rtkProjectionsRegionConstIteratorRayBasedWithCylindricalPanel.hxx"
#endif

#endif
