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

#ifndef rtkThreeDHelicalProjectionGeometry_h
#define rtkThreeDHelicalProjectionGeometry_h

#include "RTKExport.h"
#include "rtkThreeDCircularProjectionGeometry.h"

namespace rtk
{
/** \class ThreeDHelicalProjectionGeometry
 * \brief Projection geometry for a source and a 2-D flat panel.
 *
 * The source and the detector rotate around a helix paremeterized
 * with the SourceToDetectorDistance and the SourceToIsocenterDistance and a helical pitch.
 * The position of each projection along the helix is parameterized
 * by the HelicalAngle.
 * The detector is assumed to have no horizontal offset (ProjectionOffsetX = 0). It can be also rotated with
 * InPlaneAngles and OutOfPlaneAngles. All angles are in radians except for the function AddProjection that takes angles
 * in degrees. The source is assumed to have no horizontal offset (SourceOffsetX = 0).
 *
 * If SDD equals 0., then one is dealing with a parallel geometry.
 * If m_RadiusCylindricalDetector != 0 : the geometry assumes a cylindrical detector.
 *
 * More information is provided in \ref DocGeo3D.
 *
 * \author Jerome Lesaint
 *
 * \ingroup RTK ProjectionGeometry
 */

class RTK_EXPORT ThreeDHelicalProjectionGeometry : public ThreeDCircularProjectionGeometry
{
public:
#if ITK_VERSION_MAJOR == 5 && ITK_VERSION_MINOR == 1
  ITK_DISALLOW_COPY_AND_ASSIGN(ThreeDHelicalProjectionGeometry);
#else
  ITK_DISALLOW_COPY_AND_MOVE(ThreeDHelicalProjectionGeometry);
#endif

  using Self = ThreeDHelicalProjectionGeometry;
  using Superclass = ThreeDCircularProjectionGeometry;
  using Pointer = itk::SmartPointer<Self>;
  using ConstPointer = itk::SmartPointer<const Self>;

  using VectorType = itk::Vector<double, 3>;
  using HomogeneousVectorType = itk::Vector<double, 4>;
  using TwoDHomogeneousMatrixType = itk::Matrix<double, 3, 3>;
  using ThreeDHomogeneousMatrixType = itk::Matrix<double, 4, 4>;
  using PointType = itk::Point<double, 3>;
  using Matrix3x3Type = itk::Matrix<double, 3, 3>;
  using HomogeneousProjectionMatrixType = Superclass::MatrixType;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Add projection to geometry. One projection is defined with the rotation
   * angle in degrees and the in-plane translation of the detector in physical
   * units (e.g. mm). The rotation axis is assumed to be (0,1,0).
   */

  /** Only the AddProjection in radians is modified to populate the helical angle vector. */
  virtual void
  AddProjectionInRadians(const double sid,
                         const double sdd,
                         const double gantryAngle,
                         const double projOffsetX = 0.,
                         const double projOffsetY = 0.,
                         const double outOfPlaneAngle = 0.,
                         const double inPlaneAngle = 0.,
                         const double sourceOffsetX = 0.,
                         const double sourceOffsetY = 0.) override;


  /** Empty the geometry object. */
  void
  Clear() override;

  /** Get the vector of geometry parameters (one per projection). Angles are
   * in radians.*/
  /** Get the vector of geometry parameters (one per projection). Angles are
   * in radians.*/
  const std::vector<double> &
  GetHelicalAngles() const
  {
    return this->m_HelicalAngles;
  }

  const double &
  GetHelixPitch() const
  {
    return this->m_HelixPitch;
  }

  const double &
  GetHelixVerticalGap() const
  {
    return this->m_HelixVerticalGap;
  }

  const double &
  GetHelixAngularGap() const
  {
    return this->m_HelixAngularGap;
  }

  const double &
  GetHelixRadius() const
  {
    return this->m_HelixRadius;
  }

  const double &
  GetHelixSourceToDetectorDist() const
  {
    return this->m_HelixSourceToDetectorDist;
  }

  const bool &
  GetTheGeometryIsVerified() const
  {
    return this->m_TheGeometryIsVerified;
  }

  bool
  VerifyHelixParameters();

protected:
  ThreeDHelicalProjectionGeometry();
  ~ThreeDHelicalProjectionGeometry() override = default;

  /** Some additional helix-specific global parameteres.
   *  Helical angles are NOT converted between 0 and 2*pi.
   */
  double              m_HelixRadius;
  double              m_HelixSourceToDetectorDist;
  double              m_HelixVerticalGap;
  double              m_HelixAngularGap;
  double              m_HelixPitch;
  std::vector<double> m_HelicalAngles;
  bool                m_TheGeometryIsVerified;
};
} // namespace rtk


#endif // __rtkThreeDHelicalProjectionGeometry_h
