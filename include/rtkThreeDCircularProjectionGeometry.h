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

#ifndef rtkThreeDCircularProjectionGeometry_h
#define rtkThreeDCircularProjectionGeometry_h

#include "RTKExport.h"
#include "rtkProjectionGeometry.h"

namespace rtk
{
/** \class ThreeDCircularProjectionGeometry
 * \brief Projection geometry for a source and a 2-D flat panel.
 *
 * The source and the detector rotate around a circle paremeterized
 * with the SourceToDetectorDistance and the SourceToIsocenterDistance.
 * The position of each projection along this circle is parameterized
 * by the GantryAngle.
 * The detector can be shifted in plane with the ProjectionOffsetsX
 * and the ProjectionOffsetsY. It can be also rotated with InPlaneAngles
 * and OutOfPlaneAngles. All angles are in radians except for the function
 * AddProjection that takes angles in degrees.
 * The source can be shifted in plane with the SourceOffsetsX
 * and the SourceOffsetsY.
 *
 * If SDD equals 0., then one is dealing with a parallel geometry.
 *
 * See https://docs.openrtk.org/en/latest/documentation/docs/Geometry.html
 * for more information.
 *
 * \author Simon Rit
 *
 * \ingroup RTK ProjectionGeometry
 */

class RTK_EXPORT ThreeDCircularProjectionGeometry : public ProjectionGeometry<3>
{
public:
  ITK_DISALLOW_COPY_AND_MOVE(ThreeDCircularProjectionGeometry);

  using Self = ThreeDCircularProjectionGeometry;
  using Superclass = ProjectionGeometry<3>;
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
  void
  AddProjection(const double sid,
                const double sdd,
                const double gantryAngle,
                const double projOffsetX = 0.,
                const double projOffsetY = 0.,
                const double outOfPlaneAngle = 0.,
                const double inPlaneAngle = 0.,
                const double sourceOffsetX = 0.,
                const double sourceOffsetY = 0.);

  /** Idem with angles in radians. */
  virtual void
  AddProjectionInRadians(const double sid,
                         const double sdd,
                         const double gantryAngle,
                         const double projOffsetX = 0.,
                         const double projOffsetY = 0.,
                         const double outOfPlaneAngle = 0.,
                         const double inPlaneAngle = 0.,
                         const double sourceOffsetX = 0.,
                         const double sourceOffsetY = 0.);

  /**
   * @brief Add a REG23-based geometry set to the RTK projections list.
   * @param sourcePosition absolute position of the point source S in WCS
   * @param detectorPosition absolute position of the detector origin R in WCS
   * @param detectorRowVector absolute direction vector indicating the
   * orientation of the detector's rows r (sometimes referred to as v1)
   * @param detectorColumnVector absolute direction vector indicating the
   * orientation of the detector's columns c (sometimes referred to as v2)
   * @return TRUE if the projection could be added to the RTK projections list
   */
  bool
  AddProjection(const PointType &  sourcePosition,
                const PointType &  detectorPosition,
                const VectorType & detectorRowVector,
                const VectorType & detectorColumnVector);


  /** Add projection from a projection matrix. A projection matrix is defined
   * up to a scaling factor. The function here Assumes that the input matrix
   * pMat is normalized such that pMat*(x,y,z,1)'=(u,v,1)'.
   * This code assumes that the SourceToDetectorDistance is positive.
   * If the projection is for parallel geometry, then
   * the SourceToIsocenterDistance is set to 1000 mm.
   *
   * \note The projection matrix has a special form for a parallel geometry.
   *
   * | Column-1 | Column-2 | Column-3 |      Column-4      |
   * |:--------:|:--------:|:--------:|:------------------:|
   * | R11      | R12      | R13      | -ProjectionOffsetX |
   * | R21      | R22      | R23      | -ProjectionOffsetY |
   * |  0       |  0       |  0       |          1         |
   *
   * where
   *
   * | Column-1 | Column-2 | Column-3 |
   * |:--------:|:--------:|:--------:|
   * |    R11   |    R12   |   R13    |
   * |    R21   |    R22   |   R23    |
   * |     0    |     0    |    0     |
   *
   * is a cropped 3x3 rotation matrix.
   *
   * \warning The outOfPlaneAngle for a parallel geometry must be
   * within (-90, 90) degrees range, otherwise the rotation
   * angles could be miscalculated.
   *
   */
  bool
  AddProjection(const HomogeneousProjectionMatrixType & pMat);

  /** Empty the geometry object. */
  void
  Clear() override;

  /** Get the vector of geometry parameters (one per projection). Angles are
   * in radians.*/
  const std::vector<double> &
  GetGantryAngles() const
  {
    return this->m_GantryAngles;
  }
  const std::vector<double> &
  GetOutOfPlaneAngles() const
  {
    return this->m_OutOfPlaneAngles;
  }
  const std::vector<double> &
  GetInPlaneAngles() const
  {
    return this->m_InPlaneAngles;
  }
  const std::vector<double> &
  GetSourceToIsocenterDistances() const
  {
    return this->m_SourceToIsocenterDistances;
  }
  const std::vector<double> &
  GetSourceOffsetsX() const
  {
    return this->m_SourceOffsetsX;
  }
  const std::vector<double> &
  GetSourceOffsetsY() const
  {
    return this->m_SourceOffsetsY;
  }
  const std::vector<double> &
  GetSourceToDetectorDistances() const
  {
    return this->m_SourceToDetectorDistances;
  }
  const std::vector<double> &
  GetProjectionOffsetsX() const
  {
    return this->m_ProjectionOffsetsX;
  }
  const std::vector<double> &
  GetProjectionOffsetsY() const
  {
    return this->m_ProjectionOffsetsY;
  }

  /** Get a vector containing the source angles in radians. The source angle is
   * defined as the angle between the z-axis and the isocenter-source line
   * projected on the central plane. */
  const std::vector<double> &
  GetSourceAngles() const
  {
    return this->m_SourceAngles;
  }

  /** Get a vector containing the tilt angles in radians. The tilt angle is
   * defined as the difference between -GantryAngle and the SourceAngle. */
  const std::vector<double>
  GetTiltAngles() const;

  /** Get a multimap containing all sorted angles in radians and corresponding
   * index. */
  const std::multimap<double, unsigned int>
  GetSortedAngles(const std::vector<double> & angles) const;

  /** Get a map containing unique sorted angles in radians and corresponding
   * index. */
  const std::map<double, unsigned int>
  GetUniqueSortedAngles(const std::vector<double> & angles) const;

  /** Get for each projection the angular gaps with next projection in radians. */
  const std::vector<double>
  GetAngularGapsWithNext(const std::vector<double> & angles) const;

  /** Get for each projection half the angular distance between the previous
   *  and the next projection in radians. */
  const std::vector<double>
  GetAngularGaps(const std::vector<double> & angles);

  /** Compute rotation matrix in homogeneous coordinates from 3 angles in
   * degrees. The convention is the default in itk, i.e. ZXY of Euler angles.*/
  static ThreeDHomogeneousMatrixType
  ComputeRotationHomogeneousMatrix(double angleX, double angleY, double angleZ);

  /** Compute translation matrix in homogeneous coordinates from translation parameters.*/
  static TwoDHomogeneousMatrixType
  ComputeTranslationHomogeneousMatrix(double transX, double transY);
  static ThreeDHomogeneousMatrixType
  ComputeTranslationHomogeneousMatrix(double transX, double transY, double transZ);

  /** Compute the magnification matrix from 3D to 2D given a source to detector
   * and to isocenter distance. */
  static Superclass::MatrixType
  ComputeProjectionMagnificationMatrix(double sdd, double sid);

  /** Get the vector containing the sub matrices used to compute the main
   * projection matrix. */
  const std::vector<TwoDHomogeneousMatrixType> &
  GetProjectionTranslationMatrices() const
  {
    return this->m_ProjectionTranslationMatrices;
  }
  TwoDHomogeneousMatrixType
  GetProjectionTranslationMatrix(const unsigned int i) const
  {
    return this->m_ProjectionTranslationMatrices[i];
  }

  const std::vector<ThreeDHomogeneousMatrixType> &
  GetRotationMatrices() const
  {
    return this->m_RotationMatrices;
  }
  ThreeDHomogeneousMatrixType
  GetRotationMatrix(const unsigned int i) const
  {
    return this->m_RotationMatrices[i];
  }

  const std::vector<ThreeDHomogeneousMatrixType> &
  GetSourceTranslationMatrices() const
  {
    return this->m_SourceTranslationMatrices;
  }
  ThreeDHomogeneousMatrixType
  GetSourceTranslationMatrices(const unsigned int i) const
  {
    return this->m_SourceTranslationMatrices[i];
  }

  const std::vector<Superclass::MatrixType> &
  GetMagnificationMatrices() const
  {
    return this->m_MagnificationMatrices;
  }
  Superclass::MatrixType
  GetMagnificationMatrices(const unsigned int i) const
  {
    return this->m_MagnificationMatrices[i];
  }

  /** Get the vector containing the collimation jaw parameters. */
  const std::vector<double> &
  GetCollimationUInf() const
  {
    return this->m_CollimationUInf;
  }
  const std::vector<double> &
  GetCollimationUSup() const
  {
    return this->m_CollimationUSup;
  }
  const std::vector<double> &
  GetCollimationVInf() const
  {
    return this->m_CollimationVInf;
  }
  const std::vector<double> &
  GetCollimationVSup() const
  {
    return this->m_CollimationVSup;
  }

  /** Set the collimation of the latest added projection (to be called after
   * AddProjection). */
  void
  SetCollimationOfLastProjection(const double uinf, const double usup, const double vinf, const double vsup);

  /** Get the source position for the ith projection in the fixed reference
   * system and in homogeneous coordinates. */
  const HomogeneousVectorType
  GetSourcePosition(const unsigned int i) const;

  /** Compute the ith matrix to convert projection coordinates to coordinates
   * in the detector coordinate system (u,v,u^v). Note that the matrix is square but the
   * third element of the projection coordinates is ignored because projection
   * coordinates are 2D. This is meant to manipulate more easily stack of
   * projection images. */
  const ThreeDHomogeneousMatrixType
  GetProjectionCoordinatesToDetectorSystemMatrix(const unsigned int i) const;

  /** Compute the ith matrix to convert projection coordinates to coordinates
   * in the fixed coordinate system. Note that the matrix is square but the
   * third element of the projection coordinates is ignored because projection
   * coordinates are 2D. This is meant to manipulate more easily stack of
   * projection images. */
  const ThreeDHomogeneousMatrixType
  GetProjectionCoordinatesToFixedSystemMatrix(const unsigned int i) const;

  /** This function wraps an angle value between 0 and 360 degrees. */
  static double
  ConvertAngleBetween0And360Degrees(const double a);

  /** This function wraps an angle value between 0 and 2*PI radians. */
  static double
  ConvertAngleBetween0And2PIRadians(const double a);

  /** This function wraps an angle value between -PI and PI radians. */
  static double
  ConvertAngleBetweenMinusAndPlusPIRadians(const double a);

  /** Changes the coordinate on the projection image to the coordinate on a
   * virtual detector that is perpendicular to the source to isocenter line and
   * positioned at the isocenter.
   * It is assumed that OutOfPlaneAngle=0 and InPlaneAngle=0.*/
  double
  ToUntiltedCoordinateAtIsocenter(const unsigned int noProj, const double tiltedCoord) const;

  /** Accessor for the radius of curved detector. The default is 0 and it means
   * a flat detector. */
  itkGetConstMacro(RadiusCylindricalDetector, double);
  itkSetMacro(RadiusCylindricalDetector, double);

  /** Accessor for the internal tolerance on angles. */
  itkGetConstMacro(VerifyAnglesTolerance, double);
  itkSetMacro(VerifyAnglesTolerance, double);

  itkGetConstMacro(FixAnglesTolerance, double);
  itkSetMacro(FixAnglesTolerance, double);

protected:
  ThreeDCircularProjectionGeometry();
  ~ThreeDCircularProjectionGeometry() override = default;

  virtual void
  AddProjectionTranslationMatrix(const TwoDHomogeneousMatrixType & m)
  {
    this->m_ProjectionTranslationMatrices.push_back(m);
    this->Modified();
  }
  virtual void
  AddRotationMatrix(const ThreeDHomogeneousMatrixType & m)
  {
    this->m_RotationMatrices.push_back(m);
    this->Modified();
  }
  virtual void
  AddSourceTranslationMatrix(const ThreeDHomogeneousMatrixType & m)
  {
    this->m_SourceTranslationMatrices.push_back(m);
    this->Modified();
  }
  virtual void
  AddMagnificationMatrix(const Superclass::MatrixType & m)
  {
    this->m_MagnificationMatrices.push_back(m);
    this->Modified();
  }

  /** Verify that the specified Euler angles in ZXY result in a rotation matrix
   * which corresponds to the specified detector orientation. Rationale for this
   * utility method is that in some situations numerical instabilities (e.g. if
   * gantry=+90deg,in-plane=-90deg or vice versa, "invalid" angles may be
   * computed using the standard ITK Euler transform) may occur.
   * @param outOfPlaneAngleRAD out-of-plane angle of the detector in radians
   * @param gantryAngleRAD gantry angle of the detector in radians
   * @param inPlaneAngleRAD in-plane angle of the detector in radians
   * @param referenceMatrix reference matrix which reflects detector orientation
   * in WCS
   * @return TRUE if the angles correspond the implicitly specified final
   * rotation matrix; if FALSE is returned, the angles should be fixed
   * (@see FixAngles())
   * @warning {Internally, the matrix check is performed with a tolerance level
   * of 1e-6!}
   */
  bool
  VerifyAngles(const double          outOfPlaneAngleRAD,
               const double          gantryAngleRAD,
               const double          inPlaneAngleRAD,
               const Matrix3x3Type & referenceMatrix) const;

  /** Try to fix Euler angles, which were found incorrect, to match the specified
   * reference matrix.
   * @param [out] outOfPlaneAngleRAD out-of-plane angle of the detector in radians;
   * if this method returns TRUE, this angle can be safely considered
   * @param [out] gantryAngleRAD gantry angle of the detector in radians;
   * if this method returns TRUE, this angle can be safely considered
   * @param [out] inPlaneAngleRAD in-plane angle of the detector in radians;
   * if this method returns TRUE, this angle can be safely considered
   * @param referenceMatrix reference matrix which reflects detector orientation
   * in WCS
   * @return TRUE if the angles were fixed and can be safely considered;
   * if FALSE is returned, the method could not find angles which generate the
   * desired matrix with respect to ZXY Euler order and the internal tolerance
   * level
   * @see VerifyAngles()
   * @warning {Internally, the matrix check is performed with a tolerance level
   * of 1e-6!}
   */
  bool
  FixAngles(double &              outOfPlaneAngleRAD,
            double &              gantryAngleRAD,
            double &              inPlaneAngleRAD,
            const Matrix3x3Type & referenceMatrix) const;

  /** Clone the geometry object in a new one. */
  itk::LightObject::Pointer
  InternalClone() const override;

  /** Circular geometry parameters per projection (angles in degrees between 0
    and 360). */
  std::vector<double> m_GantryAngles;
  std::vector<double> m_OutOfPlaneAngles;
  std::vector<double> m_InPlaneAngles;
  std::vector<double> m_SourceAngles;
  std::vector<double> m_SourceToIsocenterDistances;
  std::vector<double> m_SourceOffsetsX;
  std::vector<double> m_SourceOffsetsY;
  std::vector<double> m_SourceToDetectorDistances;
  std::vector<double> m_ProjectionOffsetsX;
  std::vector<double> m_ProjectionOffsetsY;

  /** Radius of curved detector. The default is 0 and it means a flat detector. */
  double m_RadiusCylindricalDetector{ 0. };

  /** Parameters of the collimation jaws.
   * The collimation position is with respect to the distance of the m_RotationCenter along
   * - the m_RotationAxis for the m_CollimationVInf and m_CollimationVSup,
   * - the m_SourceCenter ^ m_RotationAxis for the m_CollimationUInf and m_CollimationUSup.
   * The default is +infinity (itk::NumericTraits<double>::max) is completely
   * opened, negative values are allowed if the collimation travels beyond the m_RotationCenter.
   */
  std::vector<double> m_CollimationUInf;
  std::vector<double> m_CollimationUSup;
  std::vector<double> m_CollimationVInf;
  std::vector<double> m_CollimationVSup;

  /** Matrices to change coordiate systems. */
  std::vector<TwoDHomogeneousMatrixType>   m_ProjectionTranslationMatrices;
  std::vector<Superclass::MatrixType>      m_MagnificationMatrices;
  std::vector<ThreeDHomogeneousMatrixType> m_RotationMatrices;
  std::vector<ThreeDHomogeneousMatrixType> m_SourceTranslationMatrices;

  /** Internal tolerance parameters. */
  double m_VerifyAnglesTolerance{ 1e-4 };
  double m_FixAnglesTolerance{ 1e-6 };
};
} // namespace rtk


#endif // __rtkThreeDCircularProjectionGeometry_h
