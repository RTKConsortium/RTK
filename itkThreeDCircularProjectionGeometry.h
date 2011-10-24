#ifndef ITKTHREEDCIRCULARPROJECTIONGEOMETRY_H
#define ITKTHREEDCIRCULARPROJECTIONGEOMETRY_H

#include "itkProjectionGeometry.h"

namespace itk
{
/** \class ThreeDCircularProjectionGeometry
 * \brief Projection geometry for a point source and a 2-D flat panel.
 * The source and the detector rotate around a circle paremeterized
 * with the SourceToDetectorDistance and the SourceToIsocenterDistance.
 * The position of each projection along this circle is parameterized
 * by the RotationAngle.
 * The detector can be shifted in plane with the ProjectionOffsetsX
 * and the ProjectionOffsetsY.
 */

class ThreeDCircularProjectionGeometry : public ProjectionGeometry<3>
{
public:
  typedef ThreeDCircularProjectionGeometry Self;
  typedef ProjectionGeometry<3>            Superclass;
  typedef SmartPointer< Self >             Pointer;
  typedef SmartPointer< const Self >       ConstPointer;

  typedef Vector<double, 3>                VectorType;
  typedef Vector<double, 4>                HomogeneousVectorType;
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

protected:
  ThreeDCircularProjectionGeometry() {};
  virtual ~ThreeDCircularProjectionGeometry() {};

  double ConvertAngleBetween0And360Degrees(const double a);

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

#endif // ITKTHREEDCIRCULARPROJECTIONGEOMETRY_H
