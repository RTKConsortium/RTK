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

#ifndef __rtkThreeDCircularProjectionGeometry_h
#define __rtkThreeDCircularProjectionGeometry_h

#include "rtkProjectionGeometry.h"

namespace rtk
{
/** \class ThreeDCircularProjectionGeometry
 * \brief Projection geometry for a source and a 2-D flat panel.
 *
 * The source and the detector rotate around a circle paremeterized
 * with the SourceToDetectorDistance and the SourceToIsocenterDistance.
 * The position of each projection along this circle is parameterized
 * by the RotationAngle.
 * The detector can be shifted in plane with the ProjectionOffsetsX
 * and the ProjectionOffsetsY. It can be also rotated with InPlaneAngles
 * and OutOfPlaneAngles.
 * The source can be shifted in plane with the SourceOffsetsX
 * and the SourceOffsetsY.
 * 
 * If SDD equals 0., then one is dealing with a parallel geometry.
 *
 * \author Simon Rit
 *
 * \ingroup ProjectionGeometry
 */
class ThreeDCircularProjectionGeometry : public ProjectionGeometry<3>
{
public:
  typedef ThreeDCircularProjectionGeometry Self;
  typedef ProjectionGeometry<3>            Superclass;
  typedef itk::SmartPointer< Self >        Pointer;
  typedef itk::SmartPointer< const Self >  ConstPointer;

  typedef itk::Vector<double, 3>           VectorType;
  typedef itk::Vector<double, 4>           HomogeneousVectorType;
  typedef itk::Matrix< double, 3, 3 >      TwoDHomogeneousMatrixType;
  typedef itk::Matrix< double, 4, 4 >      ThreeDHomogeneousMatrixType;

  /** Method for creation through the object factory. */
  itkNewMacro( Self );

  /** Add projection to geometry. One projection is defined with the rotation
   * angle in degrees and the in-plane translation of the detector in physical
   * units (e.g. mm). The rotation axis is assumed to be (0,1,0).
   */
  void AddProjection(const double sid, const double sdd, const double gantryAngle,
                     const double projOffsetX=0., const double projOffsetY=0.,
                     const double outOfPlaneAngle=0., const double inPlaneAngle=0.,
                     const double sourceOffsetX=0., const double sourceOffsetY=0.);

  /** Empty the geometry object. */
  void Clear();

  /** Get the vector of geometry parameters (one per projection) */
  const std::vector<double> &GetGantryAngles() const {
    return this->m_GantryAngles;
  }
  const std::vector<double> &GetOutOfPlaneAngles() const {
    return this->m_OutOfPlaneAngles;
  }
  const std::vector<double> &GetInPlaneAngles() const {
    return this->m_InPlaneAngles;
  }
  const std::vector<double> &GetSourceToIsocenterDistances() const {
    return this->m_SourceToIsocenterDistances;
  }
  const std::vector<double> &GetSourceOffsetsX() const {
    return this->m_SourceOffsetsX;
  }
  const std::vector<double> &GetSourceOffsetsY() const {
    return this->m_SourceOffsetsY;
  }
  const std::vector<double> &GetSourceToDetectorDistances() const {
    return this->m_SourceToDetectorDistances;
  }
  const std::vector<double> &GetProjectionOffsetsX() const {
    return this->m_ProjectionOffsetsX;
  }
  const std::vector<double> &GetProjectionOffsetsY() const {
    return this->m_ProjectionOffsetsY;
  }

  /** Get a multimap containing all sorted angles and corresponding index. */
  const std::multimap<double,unsigned int> GetSortedAngles();

  /** Get for each projection the angular gaps with next projection. */
  const std::vector<double> GetAngularGapsWithNext();

  /** Get for each projection half the angular distance between the previous
   *  and the next projection. */
  const std::vector<double> GetAngularGaps();

  /** Compute rotation matrix in homogeneous coordinates from 3 angles in
   * degrees. The convention is the default in itk, i.e. ZXY of Euler angles.*/
  static ThreeDHomogeneousMatrixType
  ComputeRotationHomogeneousMatrix(double angleX,
                                   double angleY,
                                   double angleZ);

  /** Compute translation matrix in homogeneous coordinates from translation parameters.*/
  static TwoDHomogeneousMatrixType
  ComputeTranslationHomogeneousMatrix(double transX,
                                      double transY);
  static ThreeDHomogeneousMatrixType
  ComputeTranslationHomogeneousMatrix(double transX,
                                      double transY,
                                      double transZ);

  /** Compute the magnification matrix from 3D to 2D given a source to detector
   * and to detector distance. */
  static Superclass::MatrixType ComputeProjectionMagnificationMatrix(double sdd,
                                                                     double sid);

  /** Get the vector containing the sub matrices used to compute the main
   * projection matrix. */
  const std::vector<TwoDHomogeneousMatrixType> &GetProjectionTranslationMatrices() const {
    return this->m_ProjectionTranslationMatrices;
  }
  const std::vector<ThreeDHomogeneousMatrixType> &GetRotationMatrices() const {
    return this->m_RotationMatrices;
  }
  const std::vector<ThreeDHomogeneousMatrixType> &GetSourceTranslationMatrices() const {
    return this->m_SourceTranslationMatrices;
  }
  const std::vector<Superclass::MatrixType> &GetMagnificationMatrices() const {
    return this->m_MagnificationMatrices;
  }

  /** Get the source position for the ith projection in the fixed reference
   * system and in homogeneous coordinates. */
  const HomogeneousVectorType GetSourcePosition(const unsigned int i) const;

  /** Compute the ith matrix to convert projection coordinates to coordinates
   * in the fixed coordinate system. Note that the matrix is square but the
   * third element of the projection coordinates is ignored because projection
   * coordinates are 2D. This is meant to manipulate more easily stack of
   * projection images. */
  const ThreeDHomogeneousMatrixType GetProjectionCoordinatesToFixedSystemMatrix(const unsigned int i) const;

  /** This function wraps an angle value between 0 and 360 degrees. */
  static double ConvertAngleBetween0And360Degrees(const double a);

protected:
  ThreeDCircularProjectionGeometry() {};
  virtual ~ThreeDCircularProjectionGeometry() {};

  virtual void AddProjectionTranslationMatrix(const TwoDHomogeneousMatrixType &m){
    this->m_ProjectionTranslationMatrices.push_back(m);
    this->Modified();
  }
  virtual void AddRotationMatrix(const ThreeDHomogeneousMatrixType &m){
    this->m_RotationMatrices.push_back(m);
    this->Modified();
  }
  virtual void AddSourceTranslationMatrix(const ThreeDHomogeneousMatrixType &m){
    this->m_SourceTranslationMatrices.push_back(m);
    this->Modified();
  }
  virtual void AddMagnificationMatrix(const Superclass::MatrixType &m){
    this->m_MagnificationMatrices.push_back(m);
    this->Modified();
  }

  /** Circular geometry parameters per projection (angles in degrees between 0
    and 360). */
  std::vector<double> m_GantryAngles;
  std::vector<double> m_OutOfPlaneAngles;
  std::vector<double> m_InPlaneAngles;
  std::vector<double> m_SourceToIsocenterDistances;
  std::vector<double> m_SourceOffsetsX;
  std::vector<double> m_SourceOffsetsY;
  std::vector<double> m_SourceToDetectorDistances;
  std::vector<double> m_ProjectionOffsetsX;
  std::vector<double> m_ProjectionOffsetsY;

  std::vector<TwoDHomogeneousMatrixType>         m_ProjectionTranslationMatrices;
  std::vector<Superclass::MatrixType>            m_MagnificationMatrices;
  std::vector<ThreeDHomogeneousMatrixType>       m_RotationMatrices;
  std::vector<ThreeDHomogeneousMatrixType>       m_SourceTranslationMatrices;

private:
  ThreeDCircularProjectionGeometry(const Self&); //purposely not implemented
  void operator=(const Self&);                   //purposely not implemented
};
}

#endif // __rtkThreeDCircularProjectionGeometry_h
