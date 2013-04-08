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

#include "rtkReg23ProjectionGeometry.h"

//std
#include <math.h>
//ITK
#include <itkVector.h>
#include <itkEuler3DTransform.h>

rtk::Reg23ProjectionGeometry::Reg23ProjectionGeometry()
  : rtk::ThreeDCircularProjectionGeometry()
{
}

rtk::Reg23ProjectionGeometry::~Reg23ProjectionGeometry()
{
}

bool rtk::Reg23ProjectionGeometry::VerifyAngles(const double outOfPlaneAngleRAD,
  const double gantryAngleRAD, const double inPlaneAngleRAD,
  const Matrix3x3Type &referenceMatrix) const
{
  typedef itk::Euler3DTransform<double> EulerType;

  const Matrix3x3Type &rm = referenceMatrix; // shortcut
  const double EPSILON = 1e-6; // internal tolerance for comparison

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

bool rtk::Reg23ProjectionGeometry::FixAngles(double &outOfPlaneAngleRAD,
  double &gantryAngleRAD, double &inPlaneAngleRAD,
  const Matrix3x3Type &referenceMatrix) const
{
  const Matrix3x3Type &rm = referenceMatrix; // shortcut
  const double EPSILON = 1e-6; // internal tolerance for comparison

  if (fabs(fabs(rm[2][0]) - 1.) > EPSILON)
  {
    double oa, ga, ia;

    // @see Slabaugh, GG, "Computing Euler angles from a rotation matrix"

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

    return false;
  }
  return false;
}

bool rtk::Reg23ProjectionGeometry::AddReg23Projection(
    const PointType &sourcePosition, const PointType &detectorPosition,
    const VectorType &detectorRowVector, const VectorType &detectorColumnVector)
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
  euler->SetMatrix(rm);
  oa = euler->GetAngleX(); // delivers radians
  ga = euler->GetAngleY();
  ia = euler->GetAngleZ();
  // verify that extracted ZXY angles result in the *desired* matrix:
  // (at some angle constellations we may run into numerical troubles, therefore,
  // verify angles and try to fix instabilities)
  if (!VerifyAngles(oa, ga, ia, rm))
  {
    if (!FixAngles(oa, ga, ia, rm))
      return false;
  }
  // since rtk::ThreeDCircularProjectionGeometry::AddProjection() mirrors the
  // angles (!) internally, let's invert the computed ones in order to
  // get at the end what we would like (see above); convert rad->deg:
  ga *= -57.29577951308232;
  oa *= -57.29577951308232;
  ia *= -57.29577951308232;

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
  this->Superclass::AddProjection(SID, SDD, ga, oRx, oRy, oa, ia, oSx, oSy);

  return true;
}
