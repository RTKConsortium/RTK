#ifndef ITKTHREEDCIRCULARPROJECTIONGEOMETRY_H
#define ITKTHREEDCIRCULARPROJECTIONGEOMETRY_H

#include "itkProjectionGeometry.h"
#include "rtkHomogeneousMatrix.h"

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

  typedef Vector<double, 3> VectorType;

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
  const std::vector<double> &GetGantryAngles() {
    return this->m_GantryAngles;
  }
  const std::vector<double> &GetOutOfPlaneAngles() {
    return this->m_OutOfPlaneAngles;
  }
  const std::vector<double> &GetInPlaneAngles() {
    return this->m_InPlaneAngles;
  }
  const std::vector<double> &GetSourceToIsocenterDistances() {
    return this->m_SourceToIsocenterDistances;
  }
  const std::vector<double> &GetSourceOffsetsX() {
    return this->m_SourceOffsetsX;
  }
  const std::vector<double> &GetSourceOffsetsY() {
    return this->m_SourceOffsetsY;
  }
  const std::vector<double> &GetSourceToDetectorDistances() {
    return this->m_SourceToDetectorDistances;
  }
  const std::vector<double> &GetProjectionOffsetsX() {
    return this->m_ProjectionOffsetsX;
  }
  const std::vector<double> &GetProjectionOffsetsY() {
    return this->m_ProjectionOffsetsY;
  }

  /** Get for each projection the angular gaps with next projection. */
  const std::vector<double> GetAngularGapsWithNext();

  /** Get for each projection half the angular distance between the previous
   *  and the next projection. */
  const std::vector<double> GetAngularGaps();

protected:
  ThreeDCircularProjectionGeometry() {};
  virtual ~ThreeDCircularProjectionGeometry() {};

  double ConvertAngleBetween0And360Degrees(const double a);

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

private:
  ThreeDCircularProjectionGeometry(const Self&); //purposely not implemented
  void operator=(const Self&);                   //purposely not implemented
};
}

#endif // ITKTHREEDCIRCULARPROJECTIONGEOMETRY_H
