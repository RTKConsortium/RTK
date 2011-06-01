#include "itkThreeDCircularProjectionGeometry.h"

double itk::ThreeDCircularProjectionGeometry::ConvertAngleBetween0And360Degrees(const double a)
{
  double result = a-360*floor(a/360); // between -360 and 360
  if(result<0) result += 360;         // between 0    and 360
  return result;
}

void itk::ThreeDCircularProjectionGeometry::AddProjection(
  const double sid, const double sdd, const double gantryAngle,
  const double projOffsetX, const double projOffsetY,
  const double outOfPlaneAngle, const double inPlaneAngle,
  const double sourceOffsetX, const double sourceOffsetY)
{
  // Detector orientation parameters
  m_GantryAngles.push_back( ConvertAngleBetween0And360Degrees(gantryAngle) );
  m_OutOfPlaneAngles.push_back( ConvertAngleBetween0And360Degrees(outOfPlaneAngle) );
  m_InPlaneAngles.push_back( ConvertAngleBetween0And360Degrees(inPlaneAngle) );

  // Source position parameters
  m_SourceToIsocenterDistances.push_back( sid );
  m_SourceOffsetsX.push_back( sourceOffsetX );
  m_SourceOffsetsY.push_back( sourceOffsetY );

  // Detector position parameters
  m_SourceToDetectorDistances.push_back( sdd );
  m_ProjectionOffsetsX.push_back( projOffsetX );
  m_ProjectionOffsetsY.push_back( projOffsetY );
  
  Superclass::MatrixType matrix;
  matrix =
    Get2DScalingHomogeneousMatrix(0.184118024374599, 0.184118024374599).GetVnlMatrix() *
    Get2DRigidTransformationHomogeneousMatrix(0, projOffsetX-sourceOffsetX, projOffsetY-sourceOffsetY).GetVnlMatrix() *
    GetProjectionMagnificationMatrix<3>(sdd, sid).GetVnlMatrix() *
    Get3DRigidTransformationHomogeneousMatrix(outOfPlaneAngle, gantryAngle, inPlaneAngle,
                                              sourceOffsetX, sourceOffsetY, 0.).GetVnlMatrix();
  this->AddMatrix(matrix);
}

const std::vector<double> itk::ThreeDCircularProjectionGeometry::GetAngularGapsWithNext()
{
  std::vector<double> angularGaps;
  unsigned int        nProj = this->GetGantryAngles().size();
  angularGaps.resize(nProj);

  // Special management of single or empty dataset
  const double degreesToRadians = vcl_atan(1.0) / 45.0;
  if(nProj==1)
    angularGaps[0] = degreesToRadians * 360;
  if(nProj<2)
    return angularGaps;

  // Otherwise we sort the angles in a multimap
  std::multimap<double,unsigned int> angles;
  for(unsigned int iProj=0; iProj<nProj; iProj++)
    {
    double angle = this->GetGantryAngles()[iProj];
    angles.insert(std::pair<double, unsigned int>(angle, iProj) );
    }

  // We then go over the sorted angles and deduce the angular weight
  std::multimap<double,unsigned int>::const_iterator curr = angles.begin(), next = angles.begin();
  next++;

  // All but the last projection
  while(next!=angles.end() )
    {
    angularGaps[curr->second] = degreesToRadians * ( next->first - curr->first );
    curr++; next++;
    }

  //Last projection wraps the angle of the first one
  angularGaps[curr->second] = 0.5 * degreesToRadians * ( angles.begin()->first + 360 - curr->first );

  return angularGaps;
}

const std::vector<double> itk::ThreeDCircularProjectionGeometry::GetAngularGaps()
{
  std::vector<double> angularGaps;
  unsigned int        nProj = this->GetGantryAngles().size();
  angularGaps.resize(nProj);

  // Special management of single or empty dataset
  const double degreesToRadians = vcl_atan(1.0) / 45.0;
  if(nProj==1)
    angularGaps[0] = degreesToRadians * 180;
  if(nProj<2)
    return angularGaps;

  // Otherwise we sort the angles in a multimap
  std::multimap<double,unsigned int> angles;
  for(unsigned int iProj=0; iProj<nProj; iProj++)
    {
    double angle = this->GetGantryAngles()[iProj];
    angles.insert(std::pair<double, unsigned int>(angle, iProj) );
    }

  // We then go over the sorted angles and deduce the angular weight
  std::multimap<double,unsigned int>::const_iterator prev = angles.begin(),
                                                     curr = angles.begin(),
                                                     next = angles.begin();
  next++;

  //First projection wraps the angle of the last one
  angularGaps[curr->second] = 0.5 * degreesToRadians * ( next->first - angles.rbegin()->first + 360 );
  curr++; next++;

  //Rest of the angles
  while(next!=angles.end() )
    {
    angularGaps[curr->second] = 0.5 * degreesToRadians * ( next->first - prev->first );
    prev++; curr++; next++;
    }

  //Last projection wraps the angle of the first one
  angularGaps[curr->second] = 0.5 * degreesToRadians * ( angles.begin()->first + 360 - prev->first );

  return angularGaps;
}
