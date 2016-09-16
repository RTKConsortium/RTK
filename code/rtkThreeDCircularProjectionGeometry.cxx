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

#include "rtkThreeDCircularProjectionGeometry.h"

#include <itkCenteredEuler3DTransform.h>

rtk::ThreeDCircularProjectionGeometry::ThreeDCircularProjectionGeometry():
    m_RadiusCylindricalDetector(0.)
{
}

double rtk::ThreeDCircularProjectionGeometry::ConvertAngleBetween0And360Degrees(const double a)
{
  double result = a-360*floor(a/360); // between -360 and 360
  if(result<0) result += 360;         // between 0    and 360
  return result;
}

double rtk::ThreeDCircularProjectionGeometry::ConvertAngleBetween0And2PIRadians(const double a)
{
  double result = a-2*vnl_math::pi*floor( a / (2*vnl_math::pi) ); // between -2*PI and 2*PI
  if(result<0) result += 2*vnl_math::pi;                          // between 0     and 2*PI
  return result;
}

void rtk::ThreeDCircularProjectionGeometry::AddProjection(
  const double sid, const double sdd, const double gantryAngle,
  const double projOffsetX, const double projOffsetY,
  const double outOfPlaneAngle, const double inPlaneAngle,
  const double sourceOffsetX, const double sourceOffsetY)
{
  const double degreesToRadians = vcl_atan(1.0) / 45.0;
  AddProjectionInRadians(sid, sdd, degreesToRadians * gantryAngle,
                         projOffsetX, projOffsetY, degreesToRadians * outOfPlaneAngle,
                         degreesToRadians * inPlaneAngle,
                         sourceOffsetX, sourceOffsetY);
}

void rtk::ThreeDCircularProjectionGeometry::AddProjectionInRadians(
  const double sid, const double sdd, const double gantryAngle,
  const double projOffsetX, const double projOffsetY,
  const double outOfPlaneAngle, const double inPlaneAngle,
  const double sourceOffsetX, const double sourceOffsetY)
{
  // Detector orientation parameters
  m_GantryAngles.push_back( ConvertAngleBetween0And2PIRadians(gantryAngle) );
  m_OutOfPlaneAngles.push_back( ConvertAngleBetween0And2PIRadians(outOfPlaneAngle) );
  m_InPlaneAngles.push_back( ConvertAngleBetween0And2PIRadians(inPlaneAngle) );

  // Source position parameters
  m_SourceToIsocenterDistances.push_back( sid );
  m_SourceOffsetsX.push_back( sourceOffsetX );
  m_SourceOffsetsY.push_back( sourceOffsetY );

  // Detector position parameters
  m_SourceToDetectorDistances.push_back( sdd );
  m_ProjectionOffsetsX.push_back( projOffsetX );
  m_ProjectionOffsetsY.push_back( projOffsetY );

  // Compute sub-matrices
  AddProjectionTranslationMatrix( ComputeTranslationHomogeneousMatrix(sourceOffsetX-projOffsetX, sourceOffsetY-projOffsetY) );
  AddMagnificationMatrix( ComputeProjectionMagnificationMatrix(-sdd, -sid) );
  AddRotationMatrix( ComputeRotationHomogeneousMatrix(-outOfPlaneAngle, -gantryAngle, -inPlaneAngle) );
  AddSourceTranslationMatrix( ComputeTranslationHomogeneousMatrix(-sourceOffsetX, -sourceOffsetY, 0.) );

  Superclass::MatrixType matrix;
  matrix =
    this->GetProjectionTranslationMatrices().back().GetVnlMatrix() *
    this->GetMagnificationMatrices().back().GetVnlMatrix() *
    this->GetSourceTranslationMatrices().back().GetVnlMatrix()*
    this->GetRotationMatrices().back().GetVnlMatrix();
  this->AddMatrix(matrix);

  // Calculate source angle
  VectorType z;
  z.Fill(0.);
  z[2] = 1.;
  HomogeneousVectorType sph = GetSourcePosition( m_GantryAngles.size()-1 );
  sph[1] = 0.; // Project position to central plane
  VectorType sp( &(sph[0]) );
  sp.Normalize();
  double a = acos(sp*z);
  if(sp[0] > 0.)
    a = 2. * vnl_math::pi - a;
  m_SourceAngles.push_back( ConvertAngleBetween0And2PIRadians(a) );

  this->Modified();
}

void rtk::ThreeDCircularProjectionGeometry::Clear()
{
  Superclass::Clear();

  m_GantryAngles.clear();
  m_OutOfPlaneAngles.clear();
  m_InPlaneAngles.clear();
  m_SourceAngles.clear();
  m_SourceToIsocenterDistances.clear();
  m_SourceOffsetsX.clear();
  m_SourceOffsetsY.clear();
  m_SourceToDetectorDistances.clear();
  m_ProjectionOffsetsX.clear();
  m_ProjectionOffsetsY.clear();

  m_ProjectionTranslationMatrices.clear();
  m_MagnificationMatrices.clear();
  m_RotationMatrices.clear();
  m_SourceTranslationMatrices.clear();
  this->Modified();
}

const std::vector<double> rtk::ThreeDCircularProjectionGeometry::GetTiltAngles()
{
  const std::vector<double> sangles = this->GetSourceAngles();
  const std::vector<double> gangles = this->GetGantryAngles();
  std::vector<double> tang;
  for(unsigned int iProj=0; iProj<gangles.size(); iProj++)
    {
    double angle = -1. * gangles[iProj] - sangles[iProj];
    tang.push_back( ConvertAngleBetween0And2PIRadians(angle) );
    }
  return tang;
}

const std::multimap<double,unsigned int> rtk::ThreeDCircularProjectionGeometry::GetSortedAngles(const std::vector<double> &angles)
{
  unsigned int nProj = angles.size();
  std::multimap<double,unsigned int> sangles;
  for(unsigned int iProj=0; iProj<nProj; iProj++)
    {
    double angle = angles[iProj];
    sangles.insert(std::pair<double, unsigned int>(angle, iProj) );
    }
  return sangles;
}

const std::map<double,unsigned int> rtk::ThreeDCircularProjectionGeometry::GetUniqueSortedAngles(const std::vector<double> &angles)
{
  unsigned int nProj = angles.size();
  std::map<double,unsigned int> sangles;
  for(unsigned int iProj=0; iProj<nProj; iProj++)
    {
    double angle = angles[iProj];
    sangles.insert(std::pair<double, unsigned int>(angle, iProj) );
    }
  return sangles;
}

const std::vector<double> rtk::ThreeDCircularProjectionGeometry::GetAngularGapsWithNext(const std::vector<double> &angles)
{
  std::vector<double> angularGaps;
  unsigned int        nProj = angles.size();
  angularGaps.resize(nProj);

  // Special management of single or empty dataset
  if(nProj==1)
    angularGaps[0] = 2*vnl_math::pi;
  if(nProj<2)
    return angularGaps;

  // Otherwise we sort the angles in a multimap
  std::multimap<double,unsigned int> sangles = this->GetSortedAngles( angles );

  // We then go over the sorted angles and deduce the angular weight
  std::multimap<double,unsigned int>::const_iterator curr = sangles.begin(),
                                                     next = sangles.begin();
  next++;

  // All but the last projection
  while(next!=sangles.end() )
    {
    angularGaps[curr->second] = ( next->first - curr->first );
    curr++;
    next++;
    }

  //Last projection wraps the angle of the first one
  angularGaps[curr->second] = sangles.begin()->first + 2*vnl_math::pi - curr->first;

  return angularGaps;
}

const std::vector<double> rtk::ThreeDCircularProjectionGeometry::GetAngularGaps(const std::vector<double> &angles)
{
  std::vector<double> angularGaps;
  unsigned int        nProj = angles.size();
  angularGaps.resize(nProj);

  // Special management of single or empty dataset
  if(nProj==1)
    angularGaps[0] = vnl_math::pi;
  if(nProj<2)
    return angularGaps;

  // Otherwise we sort the angles in a multimap
  std::multimap<double,unsigned int> sangles = this->GetSortedAngles(angles);

  // We then go over the sorted angles and deduce the angular weight
  std::multimap<double,unsigned int>::const_iterator prev = sangles.begin(),
                                                     curr = sangles.begin(),
                                                     next = sangles.begin();
  next++;

  //First projection wraps the angle of the last one
  angularGaps[curr->second] = 0.5 * ( next->first - sangles.rbegin()->first + 2*vnl_math::pi );
  curr++; next++;

  //Rest of the angles
  while(next!=sangles.end() )
    {
    angularGaps[curr->second] = 0.5 * ( next->first - prev->first );
    prev++; curr++; next++;
    }

  //Last projection wraps the angle of the first one
  angularGaps[curr->second] = 0.5 * ( sangles.begin()->first + 2*vnl_math::pi - prev->first );

  // FIXME: Trick for the half scan in parallel geometry case
  if(m_SourceToDetectorDistances[0]==0.)
    {
    std::vector<double>::iterator it = std::max_element(angularGaps.begin(), angularGaps.end());
    if(*it>itk::Math::pi_over_2)
      {
      for(it=angularGaps.begin(); it<angularGaps.end(); it++)
        {
        if(*it>itk::Math::pi_over_2)
          *it -= itk::Math::pi_over_2;
        *it *= 2.;
        }
      }
    }

  return angularGaps;
}

rtk::ThreeDCircularProjectionGeometry::ThreeDHomogeneousMatrixType
rtk::ThreeDCircularProjectionGeometry::
ComputeRotationHomogeneousMatrix(double angleX,
                                 double angleY,
                                 double angleZ)
{
  typedef itk::CenteredEuler3DTransform<double> ThreeDTransformType;
  ThreeDTransformType::Pointer xfm = ThreeDTransformType::New();
  xfm->SetIdentity();
  xfm->SetRotation( angleX, angleY, angleZ );

  ThreeDHomogeneousMatrixType matrix;
  matrix.SetIdentity();
  for(int i=0; i<3; i++)
    for(int j=0; j<3; j++)
      matrix[i][j] = xfm->GetMatrix()[i][j];

  return matrix;
}

rtk::ThreeDCircularProjectionGeometry::TwoDHomogeneousMatrixType
rtk::ThreeDCircularProjectionGeometry::
ComputeTranslationHomogeneousMatrix(double transX,
                                    double transY)
{
  TwoDHomogeneousMatrixType matrix;
  matrix.SetIdentity();
  matrix[0][2] = transX;
  matrix[1][2] = transY;
  return matrix;
}

rtk::ThreeDCircularProjectionGeometry::ThreeDHomogeneousMatrixType
rtk::ThreeDCircularProjectionGeometry::
ComputeTranslationHomogeneousMatrix(double transX,
                                    double transY,
                                    double transZ)
{
  ThreeDHomogeneousMatrixType matrix;
  matrix.SetIdentity();
  matrix[0][3] = transX;
  matrix[1][3] = transY;
  matrix[2][3] = transZ;
  return matrix;
}

rtk::ThreeDCircularProjectionGeometry::Superclass::MatrixType
rtk::ThreeDCircularProjectionGeometry::
ComputeProjectionMagnificationMatrix(double sdd, const double sid)
{
  Superclass::MatrixType matrix;
  matrix.Fill(0.0);
  for(unsigned int i=0; i<2; i++)
    matrix[i][i] = (sdd==0.)?1.:sdd;
  matrix[2][2] = (sdd==0.)?0.:1.;
  matrix[2][3] = (sdd==0.)?1.:sid;
  return matrix;
}

const rtk::ThreeDCircularProjectionGeometry::HomogeneousVectorType
rtk::ThreeDCircularProjectionGeometry::
GetSourcePosition(const unsigned int i) const
{
  HomogeneousVectorType sourcePosition;
  sourcePosition[0] = this->GetSourceOffsetsX()[i];
  sourcePosition[1] = this->GetSourceOffsetsY()[i];
  sourcePosition[2] = this->GetSourceToIsocenterDistances()[i];
  sourcePosition[3] = 1.;

  // Rotate
  sourcePosition.SetVnlVector(GetRotationMatrices()[i].GetInverse() * sourcePosition.GetVnlVector());
  return sourcePosition;
}

const rtk::ThreeDCircularProjectionGeometry::ThreeDHomogeneousMatrixType
rtk::ThreeDCircularProjectionGeometry::
GetProjectionCoordinatesToDetectorSystemMatrix(const unsigned int i) const
{
  // Compute projection inverse and distance to source
  ThreeDHomogeneousMatrixType matrix;
  matrix.SetIdentity();
  matrix[0][3] = this->GetProjectionOffsetsX()[i];
  matrix[1][3] = this->GetProjectionOffsetsY()[i];
  matrix[2][3] = this->GetSourceToIsocenterDistances()[i]-this->GetSourceToDetectorDistances()[i];
  matrix[2][2] = 0.; // Force z to axis to detector distance
  return matrix;
}

const rtk::ThreeDCircularProjectionGeometry::ThreeDHomogeneousMatrixType
rtk::ThreeDCircularProjectionGeometry::
GetProjectionCoordinatesToFixedSystemMatrix(const unsigned int i) const
{
  ThreeDHomogeneousMatrixType matrix;
  matrix = this->GetRotationMatrices()[i].GetInverse() *
           GetProjectionCoordinatesToDetectorSystemMatrix(i).GetVnlMatrix();
  return matrix;
}


double
rtk::ThreeDCircularProjectionGeometry::
ToUntiltedCoordinateAtIsocenter(const unsigned int noProj,
                                const double tiltedCoord) const
{
  // Aliases / constant
  const double sid  = this->GetSourceToIsocenterDistances()[noProj];
  const double sid2 = sid*sid;
  const double sdd  = this->GetSourceToDetectorDistances()[noProj];
  const double sx   = this->GetSourceOffsetsX()[noProj];
  const double px   = this->GetProjectionOffsetsX()[noProj];

  // sidu is the distance between the source and the virtual untilted detector
  const double sidu = sqrt(sid2 + sx*sx);
  // l is the coordinate on the virtual detector parallel to the real detector
  // and passing at the isocenter
  const double l    = (tiltedCoord + px - sx) * sid / sdd + sx;

  // a is the angle between the virtual detector and the real detector
  const double cosa = sx/sidu;

  // the following relation refers to a note by R. Clackdoyle, title
 // "Samping a tilted detector"
  return l * std::abs(sid) / (sidu - l*cosa);
}
