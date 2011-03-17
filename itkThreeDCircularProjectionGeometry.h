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

class ThreeDCircularProjectionGeometry: public ProjectionGeometry<3>
{
public:
  typedef ThreeDCircularProjectionGeometry    Self;
  typedef ProjectionGeometry<3>               Superclass;
  typedef SmartPointer< Self >                Pointer;
  typedef SmartPointer< const Self >          ConstPointer;

  typedef Vector<double, 3>                   VectorType;

  /** Method for creation through the object factory. */
  itkNewMacro( Self );

  /** Get / Set radius for the circular trajectory of the source
   * and the detector in physical units (e.g. mm).
   */
  itkSetMacro(SourceToDetectorDistance, double);
  itkGetMacro(SourceToDetectorDistance, double);
  itkSetMacro(SourceToIsocenterDistance, double);
  itkGetMacro(SourceToIsocenterDistance, double);
  
  /** Get / Set projection scaling to account for missing spacing in projections information. */
  itkSetMacro(ProjectionScalingX, double);
  itkGetMacro(ProjectionScalingX, double);
  itkSetMacro(ProjectionScalingY, double);
  itkGetMacro(ProjectionScalingY, double);
  
  /** Get / Set rotation axis, default is (0,1,0). */
  itkSetMacro(RotationCenter, VectorType);
  itkGetMacro(RotationCenter, VectorType);

  /** Get / Set rotation axis, default is (0,1,0). */
  itkSetMacro(RotationAxis, VectorType);
  itkGetMacro(RotationAxis, VectorType);

  /** Add projection to geometry. One projection is defined with the rotation
   * angle in degrees and the in-plane translation of the detector in physical
   * units (e.g. mm). The rotation axis is assumed to be (0,1,0).
   */
  void AddProjection(const double angle, const double offsetX, const double offsetY);

  /** Get the vector of rotation angles */
  const std::vector<double> &GetRotationAngles(){
    return this->m_RotationAngles;
  }

  /** Get the vector of projection offsets */
  const std::vector<double> &GetProjectionOffsetsX(){
    return this->m_ProjOffsetsX;
  }
  const std::vector<double> &GetProjectionOffsetsY(){
    return this->m_ProjOffsetsY;
  }

  /** Get for each projection the angular gaps with next projection. */
  const std::vector<double> GetAngularGapsWithNext();

  /** Get for each projection half the angular distance between the previous
   *  and the next projection. */
  const std::vector<double> GetAngularGaps();

protected:
  ThreeDCircularProjectionGeometry();
  virtual ~ThreeDCircularProjectionGeometry(){};

private:
  ThreeDCircularProjectionGeometry(const Self&); //purposely not implemented
  void operator=(const Self&); //purposely not implemented

private:
  /** Circular geometry global parameters */
  double m_SourceToDetectorDistance;
  double m_SourceToIsocenterDistance;
  double m_ProjectionScalingX;
  double m_ProjectionScalingY;
  VectorType m_RotationCenter;
  VectorType m_RotationAxis;

  /** Circular geometry parameters per projection (angles in degrees between 0 and 360). */
  std::vector<double> m_RotationAngles;
  std::vector<double> m_ProjOffsetsX;
  std::vector<double> m_ProjOffsetsY;
};
}

#endif // ITKTHREEDCIRCULARPROJECTIONGEOMETRY_H
