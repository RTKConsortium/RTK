#include "itkThreeDCircularProjectionGeometry.h"

void itk::ThreeDCircularProjectionGeometry::AddProjection(const double angle, const double offsetX, const double offsetY)
{
  Superclass::MatrixType matrix;
  double storedAngle = angle-360*floor(angle/360); // between -360 and 360
  if(storedAngle<0) storedAngle += 360;            // between 0    and 360
  m_RotationAngles.push_back(storedAngle);
  m_ProjOffsetsX.push_back(offsetX);
  m_ProjOffsetsY.push_back(offsetY);
  matrix = Get2DRigidTransformationHomogeneousMatrix(0, offsetX, offsetY).GetVnlMatrix() *
           GetProjectionMagnificationMatrix<3>(m_SourceToDetectorDistance, m_SourceToIsocenterDistance).GetVnlMatrix() *
           Get3DRigidTransformationHomogeneousMatrix(0, angle, 0, 0, 0, 0).GetVnlMatrix();
  this->AddMatrix(matrix);
}

const std::vector<double> itk::ThreeDCircularProjectionGeometry::GetAngularGaps()
{
  std::vector<double> angularGaps;
  unsigned int nProj = this->GetRotationAngles().size();
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
    double angle = this->GetRotationAngles()[iProj];
    angles.insert(std::pair<double, unsigned int>(angle, iProj));
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
  while(next!=angles.end())
  {
    angularGaps[curr->second] = 0.5 * degreesToRadians * ( next->first - prev->first );
    prev++; curr++; next++;
  }

  //Last projection wraps the angle of the first one
  angularGaps[curr->second] = 0.5 * degreesToRadians * ( angles.begin()->first + 360 - prev->first );

  //Detect short scans, warn the user until implementation of adequate weighting
  for(unsigned int iProj=0; iProj<nProj; iProj++)
    {
    if( angularGaps[iProj] > 10*degreesToRadians )
      {
      itkWarningMacro(<<"Short scan detected, current implementation does not account for it"<<std::endl);
      angularGaps[iProj] = vnl_math_min(angularGaps[(nProj+iProj-1)%nProj], angularGaps[(iProj+1)%nProj]); 
      }
    }
  return angularGaps;
}
