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
#include "rtkMacro.h"

#include <algorithm>
#include <math.h>

#include <itkCenteredEuler3DTransform.h>
#include <itkEuler3DTransform.h>

rtk::ThreeDCircularProjectionGeometry::ThreeDCircularProjectionGeometry():
    m_RadiusCylindricalDetector(0.)
{
}

double rtk::ThreeDCircularProjectionGeometry::ConvertAngleBetween0And360Degrees(const double a)
{
  return a-360*floor(a/360);
}

double rtk::ThreeDCircularProjectionGeometry::ConvertAngleBetween0And2PIRadians(const double a)
{
  return a-2*vnl_math::pi*floor( a / (2*vnl_math::pi) );
}

double rtk::ThreeDCircularProjectionGeometry::ConvertAngleBetweenMinusAndPlusPIRadians(const double a)
{
  double d = ConvertAngleBetween0And2PIRadians(a);
  if(d>vnl_math::pi)
    d -= 2*vnl_math::pi;
  return d;
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

  // Default collimation (uncollimated)
  m_CollimationUInf.push_back(std::numeric_limits< double >::max());
  m_CollimationUSup.push_back(std::numeric_limits< double >::max());
  m_CollimationVInf.push_back(std::numeric_limits< double >::max());
  m_CollimationVSup.push_back(std::numeric_limits< double >::max());

  this->Modified();
}

bool rtk::ThreeDCircularProjectionGeometry::
AddProjection(const PointType &sourcePosition,
              const PointType &detectorPosition,
              const VectorType &detectorRowVector,
              const VectorType &detectorColumnVector)
{
  typedef itk::Euler3DTransform<double> EulerType;

  // these parameters relate absolutely to the WCS (IEC-based):
  const VectorType &r = detectorRowVector; // row dir
  const VectorType &c = detectorColumnVector; // column dir
  VectorType n = itk::CrossProduct(r, c); // normal
  const PointType &S = sourcePosition; // source pos
  const PointType &R = detectorPosition; // detector pos

  if (fabs(r * c) > 1e-6) // non-orthogonal row/column vectors
    return false;

  // Euler angles (ZXY convention) from detector orientation in IEC-based WCS:
  double ga; // gantry angle
  double oa; // out-of-plane angle
  double ia; // in-plane angle
  // extract Euler angles from the orthogonal matrix which is established
  // by the detector orientation; however, we would like RTK to internally
  // store the inverse of this rotation matrix, therefore the corresponding
  // angles are computed here:
  Matrix3x3Type rm; // reference matrix
  // NOTE: This transposed matrix should internally
  // set by rtk::ThreeDProjectionGeometry (inverse)
  // of external rotation!
  rm[0][0] = r[0]; rm[0][1] = r[1]; rm[0][2] = r[2];
  rm[1][0] = c[0]; rm[1][1] = c[1]; rm[1][2] = c[2];
  rm[2][0] = n[0]; rm[2][1] = n[1]; rm[2][2] = n[2];
  // extract Euler angles by using the standard ITK implementation:
  EulerType::Pointer euler = EulerType::New();
  euler->SetComputeZYX(false); // ZXY order
  // workaround: Orthogonality tolerance problem when using
  // Euler3DTransform->SetMatrix() due to error magnification.
  // Parent class MatrixOffsetTransformBase does not perform an
  // orthogonality check on the matrix!
  euler->itk::MatrixOffsetTransformBase<double>::SetMatrix(rm);
  oa = euler->GetAngleX(); // delivers radians
  ga = euler->GetAngleY();
  ia = euler->GetAngleZ();
  // verify that extracted ZXY angles result in the *desired* matrix:
  // (at some angle constellations we may run into numerical troubles, therefore,
  // verify angles and try to fix instabilities)
  if (!VerifyAngles(oa, ga, ia, rm))
    {
    if (!FixAngles(oa, ga, ia, rm))
      {
      itkWarningMacro(<< "Failed to AddProjection");
      return false;
      }
    }
  // since rtk::ThreeDCircularProjectionGeometry::AddProjection() mirrors the
  // angles (!) internally, let's invert the computed ones in order to
  // get at the end what we would like (see above); convert rad->deg:
  ga *= -1.;
  oa *= -1.;
  ia *= -1.;

  // SID: distance from source to isocenter along detector normal
  double SID = n[0] * S[0] + n[1] * S[1] + n[2] * S[2];
  // SDD: distance from source to detector along detector normal
  double SDD = n[0] * (S[0] - R[0]) + n[1] * (S[1] - R[1]) + n[2] * (S[2] - R[2]);
  if (fabs(SDD) < 1e-6) // source is in detector plane
    return false;

  // source offset: compute source's "in-plane" x/y shift off isocenter
  VectorType Sv;
  Sv[0] = S[0];
  Sv[1] = S[1];
  Sv[2] = S[2];
  double oSx = Sv * r;
  double oSy = Sv * c;

  // detector offset: compute detector's in-plane x/y shift off isocenter
  VectorType Rv;
  Rv[0] = R[0];
  Rv[1] = R[1];
  Rv[2] = R[2];
  double oRx = Rv * r;
  double oRy = Rv * c;

  // configure native RTK geometry
  this->AddProjectionInRadians(SID, SDD, ga, oRx, oRy, oa, ia, oSx, oSy);

  return true;
}

bool rtk::ThreeDCircularProjectionGeometry::
AddProjection(const HomogeneousProjectionMatrixType &pMat)
{
  Matrix3x3Type A;
  for (unsigned int i = 0; i < 3; i++)
    for (unsigned int j = 0; j < 3; j++)
      {
      A(i,j) = pMat(i,j);
      }

  VectorType p;
  p[0] = pMat(0,3);
  p[1] = pMat(1,3);
  p[2] = pMat(2,3);

  // Compute determinant of A
  double d = pMat(0,0)*pMat(1,1)*pMat(2,2) +
             pMat(0,1)*pMat(1,2)*pMat(2,0) +
             pMat(0,2)*pMat(1,0)*pMat(2,1) -
             pMat(0,0)*pMat(1,2)*pMat(2,1) -
             pMat(0,1)*pMat(1,0)*pMat(2,2) -
             pMat(0,2)*pMat(1,1)*pMat(2,0);
  d = -1.*d/std::abs(d);

  // Extract intrinsic parameters u0, v0 and f (f is chosen to be positive at that point)
  // The extraction of u0 and v0 is independant of KR-decomp.
  double u0 = (pMat(0, 0)*pMat(2, 0)) + (pMat(0, 1)*pMat(2, 1)) + (pMat(0, 2)*pMat(2, 2));
  double v0 = (pMat(1, 0)*pMat(2, 0)) + (pMat(1, 1)*pMat(2, 1)) + (pMat(1, 2)*pMat(2, 2));
  double aU = sqrt(pMat(0, 0)*pMat(0, 0) + pMat(0, 1)*pMat(0, 1) + pMat(0, 2)*pMat(0, 2) - u0*u0);
  double aV = sqrt(pMat(1, 0)*pMat(1, 0) + pMat(1, 1)*pMat(1, 1) + pMat(1, 2)*pMat(1, 2) - v0*v0);
  double sdd = 0.5 * (aU + aV);

  // Def matrix K so that detK = det P[:,:3]
  Matrix3x3Type K;
  K.Fill(0.0f);
  K(0,0) = sdd;
  K(1,1) = sdd;
  K(2,2) = -1.0;
  K(0,2) = -1.*u0;
  K(1,2) = -1.*v0;
  K *= d;

  // Compute R (since det K = det P[:,:3], detR = 1 is enforced)
  Matrix3x3Type invK(K.GetInverse());
  Matrix3x3Type R = invK*A;

  //Declare a 3D euler transform in order to properly extract angles
  typedef itk::Euler3DTransform<double> EulerType;
  EulerType::Pointer euler = EulerType::New();
  euler->SetComputeZYX(false); // ZXY order

  //Extract angle using parent method without orthogonality check
  euler->itk::MatrixOffsetTransformBase<double>::SetMatrix(R);
  double oa = euler->GetAngleX();
  double ga = euler->GetAngleY();
  double ia = euler->GetAngleZ();

  // verify that extracted ZXY angles result in the *desired* matrix:
  // (at some angle constellations we may run into numerical troubles, therefore,
  // verify angles and try to fix instabilities)
  if (!VerifyAngles(oa, ga, ia, R))
    {
    if (!FixAngles(oa, ga, ia, R))
      {
      itkWarningMacro(<< "Failed to AddProjection");
      return false;
      }
    }

  // Coordinates of source in oriented coord sys :
  // (sx,sy,sid) = RS = R(-A^{-1}P[:,3]) = -K^{-1}P[:,3]
  Matrix3x3Type invA(A.GetInverse());
  VectorType v = invK*p;
  v *= -1.;
  double sx = v[0];
  double sy = v[1];
  double sid = v[2];

  // Add to geometry
  this->AddProjectionInRadians(sid, sdd, -1.*ga, sx-u0, sy-v0, -1.*oa, -1.*ia, sx, sy);

  return true;
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
  m_CollimationUInf.clear();
  m_CollimationUSup.clear();
  m_CollimationVInf.clear();
  m_CollimationVSup.clear();

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
    angularGaps[0] = 2*vnl_math::pi;
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

void
rtk::ThreeDCircularProjectionGeometry::
SetCollimationOfLastProjection(const double uinf,
                               const double usup,
                               const double vinf,
                               const double vsup)
{
  m_CollimationUInf.back() = uinf;
  m_CollimationUSup.back() = usup;
  m_CollimationVInf.back() = vinf;
  m_CollimationVSup.back() = vsup;
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

bool rtk::ThreeDCircularProjectionGeometry::
VerifyAngles(const double outOfPlaneAngleRAD,
             const double gantryAngleRAD,
             const double inPlaneAngleRAD,
             const Matrix3x3Type &referenceMatrix) const
{
  // Check if parameters are Nan. Fails if they are.
  if(outOfPlaneAngleRAD != outOfPlaneAngleRAD ||
     gantryAngleRAD != gantryAngleRAD ||
     inPlaneAngleRAD != inPlaneAngleRAD)
    return false;

  typedef itk::Euler3DTransform<double> EulerType;

  const Matrix3x3Type &rm = referenceMatrix; // shortcut
  const double EPSILON = 1e-5; // internal tolerance for comparison

  EulerType::Pointer euler = EulerType::New();
  euler->SetComputeZYX(false); // ZXY order
  euler->SetRotation(outOfPlaneAngleRAD, gantryAngleRAD, inPlaneAngleRAD);
  Matrix3x3Type m = euler->GetMatrix(); // resultant matrix

  for (int i = 0; i < 3; i++) // check whether matrices match
    for (int j = 0; j < 3; j++)
      if (fabs(rm[i][j] - m[i][j]) > EPSILON)
        return false;

  return true;
}

bool rtk::ThreeDCircularProjectionGeometry::
FixAngles(double &outOfPlaneAngleRAD,
          double &gantryAngleRAD,
          double &inPlaneAngleRAD,
          const Matrix3x3Type &referenceMatrix) const
{
  const Matrix3x3Type &rm = referenceMatrix; // shortcut
  const double EPSILON = 1e-6; // internal tolerance for comparison

  if (fabs(fabs(rm[2][1]) - 1.) > EPSILON)
    {
    double oa, ga, ia;

    // @see Slabaugh, GG, "Computing Euler angles from a rotation matrix"
    // but their convention is XYZ where we use the YXZ convention

    // first trial:
    oa = asin(rm[2][1]);
    double coa = cos(oa);
    ga = atan2(-rm[2][0] / coa, rm[2][2] / coa);
    ia = atan2(-rm[0][1] / coa, rm[1][1] / coa);
    if (VerifyAngles(oa, ga, ia, rm))
      {
      outOfPlaneAngleRAD = oa;
      gantryAngleRAD = ga;
      inPlaneAngleRAD = ia;
      return true;
      }

    // second trial:
    oa = 3.1415926535897932384626433832795 /*PI*/ - asin(rm[2][1]);
    coa = cos(oa);
    ga = atan2(-rm[2][0] / coa, rm[2][2] / coa);
    ia = atan2(-rm[0][1] / coa, rm[1][1] / coa);
    if (VerifyAngles(oa, ga, ia, rm))
      {
      outOfPlaneAngleRAD = oa;
      gantryAngleRAD = ga;
      inPlaneAngleRAD = ia;
      return true;
      }
    }
  else
    {
    // Gimbal lock, one angle in {ia,oa} has to be set randomly
    double ia;
    ia = 0.;
    if (rm[2][1] < 0.)
      {
      double oa = -itk::Math::pi_over_2;
      double ga = atan2(rm[0][2], rm[0][0]);
      if (VerifyAngles(oa, ga, ia, rm))
        {
        outOfPlaneAngleRAD = oa;
        gantryAngleRAD = ga;
        inPlaneAngleRAD = ia;
        return true;
        }
      }
    else
      {
      double oa = itk::Math::pi_over_2;
      double ga = atan2(rm[0][2], rm[0][0]);
      if (VerifyAngles(oa, ga, ia, rm))
        {
        outOfPlaneAngleRAD = oa;
        gantryAngleRAD = ga;
        inPlaneAngleRAD = ia;
        return true;
        }
      }
    }
  return false;
}

itk::LightObject::Pointer
rtk::ThreeDCircularProjectionGeometry::InternalClone() const
{
  LightObject::Pointer loPtr = Superclass::InternalClone();
  Self::Pointer clone = dynamic_cast<Self *>(loPtr.GetPointer());
  for(unsigned int iProj=0; iProj<this->GetGantryAngles().size(); iProj++)
    {
    clone->AddProjectionInRadians(this->GetSourceToIsocenterDistances()[iProj],
                                  this->GetSourceToDetectorDistances()[iProj],
                                  this->GetGantryAngles()[iProj],
                                  this->GetProjectionOffsetsX()[iProj],
                                  this->GetProjectionOffsetsY()[iProj],
                                  this->GetOutOfPlaneAngles()[iProj],
                                  this->GetInPlaneAngles()[iProj],
                                  this->GetSourceOffsetsX()[iProj],
                                  this->GetSourceOffsetsY()[iProj]);
    clone->SetCollimationOfLastProjection( this->GetCollimationUInf()[iProj],
                                           this->GetCollimationUSup()[iProj],
                                           this->GetCollimationVInf()[iProj],
                                           this->GetCollimationVSup()[iProj]);
    }
  clone->SetRadiusCylindricalDetector( this->GetRadiusCylindricalDetector() );
  return loPtr;
}
