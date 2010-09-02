#ifndef RTKTHREEDCIRCULARGEOMETRY_H
#define RTKTHREEDCIRCULARGEOMETRY_H

#include "rtkGeometry.h"
#include "rtkHomogeneousMatrix.h"

namespace rtk
{
class ThreeDCircularGeometry: public Geometry<3>
{
public:
  typedef ThreeDCircularGeometry              Self;
  typedef Geometry<3>                         Superclass;
  typedef itk::SmartPointer< Self >           Pointer;
  typedef itk::SmartPointer< const Self >     ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro( Self );

  /** Get/ Set ThreeDCircularGeometry global parameters */
  itkSetMacro(SourceToDetectorDistance, double);
  itkGetMacro(SourceToDetectorDistance, double);
  itkSetMacro(SourceToIsocenterDistance, double);
  itkGetMacro(SourceToIsocenterDistance, double);

  /** Add projection to geometry */
  void AddProjection(const double angle, const double offsetX, const double offsetY);

  /** Get the vector of angles */
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

protected:

  ThreeDCircularGeometry(): m_SourceToDetectorDistance(-1), m_SourceToIsocenterDistance(-1) {};
  virtual ~ThreeDCircularGeometry(){};

  virtual void PrintSelf( std::ostream& os, itk::Indent indent ) const;

private:
  ThreeDCircularGeometry(const Self&); //purposely not implemented
  void operator=(const Self&); //purposely not implemented

private:
  /** Circular geometry global parameters */
  double m_SourceToDetectorDistance;
  double m_SourceToIsocenterDistance;

  /** Circular geometry parameters per projection */
  std::vector<double> m_RotationAngles;
  std::vector<double> m_ProjOffsetsX;
  std::vector<double> m_ProjOffsetsY;
};
}

#endif // RTKTHREEDCIRCULARGEOMETRY_H
