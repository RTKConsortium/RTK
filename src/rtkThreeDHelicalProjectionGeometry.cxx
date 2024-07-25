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

#include "math.h"

#include "rtkThreeDHelicalProjectionGeometry.h"
#include "rtkMacro.h"

#include <algorithm>
#include <cmath>

#include <itkCenteredEuler3DTransform.h>
#include <itkEuler3DTransform.h>

rtk::ThreeDHelicalProjectionGeometry::ThreeDHelicalProjectionGeometry()
{
  m_HelixRadius = 0.;
  m_HelixSourceToDetectorDist = 0.;
  m_HelixVerticalGap = 0.;
  m_HelixAngularGap = 0.;
  m_HelixPitch = 0.;
  m_TheGeometryIsVerified = false;
}


void
rtk::ThreeDHelicalProjectionGeometry::AddProjectionInRadians(const double sid,
                                                             const double sdd,
                                                             const double gantryAngle,
                                                             const double projOffsetX,
                                                             const double projOffsetY,
                                                             const double outOfPlaneAngle,
                                                             const double inPlaneAngle,
                                                             const double sourceOffsetX,
                                                             const double sourceOffsetY)
{
  // Check parallel / divergent projections consistency
  if (!m_GantryAngles.empty())
  {
    if (sdd == 0. && m_SourceToDetectorDistances[0] != 0.)
    {
      itkGenericExceptionMacro(
        << "Cannot add a parallel projection in a 3D geometry object containing divergent projections");
    }
    if (sdd != 0. && m_SourceToDetectorDistances[0] == 0.)
    {
      itkGenericExceptionMacro(
        << "Cannot add a divergent projection in a 3D geometry object containing parallel projections");
    }
  }

  // Detector orientation parameters
  m_GantryAngles.push_back(ConvertAngleBetween0And2PIRadians(gantryAngle));
  m_HelicalAngles.push_back(gantryAngle); // No conversion mod 2*pi here for a helix.
  m_OutOfPlaneAngles.push_back(ConvertAngleBetween0And2PIRadians(outOfPlaneAngle));
  m_InPlaneAngles.push_back(ConvertAngleBetween0And2PIRadians(inPlaneAngle));

  // Source position parameters
  m_SourceToIsocenterDistances.push_back(sid);
  m_SourceOffsetsX.push_back(sourceOffsetX);
  m_SourceOffsetsY.push_back(sourceOffsetY);

  // Detector position parameters
  m_SourceToDetectorDistances.push_back(sdd);
  m_ProjectionOffsetsX.push_back(projOffsetX);
  m_ProjectionOffsetsY.push_back(projOffsetY);

  // Compute sub-matrices
  AddProjectionTranslationMatrix(
    ComputeTranslationHomogeneousMatrix(sourceOffsetX - projOffsetX, sourceOffsetY - projOffsetY));
  AddMagnificationMatrix(ComputeProjectionMagnificationMatrix(-sdd, -sid));
  AddRotationMatrix(ComputeRotationHomogeneousMatrix(-outOfPlaneAngle, -gantryAngle, -inPlaneAngle));
  AddSourceTranslationMatrix(ComputeTranslationHomogeneousMatrix(-sourceOffsetX, -sourceOffsetY, 0.));

  Superclass::MatrixType matrix;
  matrix = this->GetProjectionTranslationMatrices().back().GetVnlMatrix() *
           this->GetMagnificationMatrices().back().GetVnlMatrix() *
           this->GetSourceTranslationMatrices().back().GetVnlMatrix() *
           this->GetRotationMatrices().back().GetVnlMatrix();
  this->AddMatrix(matrix);

  // Calculate source angle
  VectorType z;
  z.Fill(0.);
  z[2] = 1.;
  HomogeneousVectorType sph = GetSourcePosition(m_GantryAngles.size() - 1);
  sph[1] = 0.; // Project position to central plane
  VectorType sp(&(sph[0]));
  sp.Normalize();
  double a = acos(sp * z);
  if (sp[0] > 0.)
    a = 2. * itk::Math::pi - a;
  m_SourceAngles.push_back(ConvertAngleBetween0And2PIRadians(a));

  // Default collimation (uncollimated)
  m_CollimationUInf.push_back(std::numeric_limits<double>::max());
  m_CollimationUSup.push_back(std::numeric_limits<double>::max());
  m_CollimationVInf.push_back(std::numeric_limits<double>::max());
  m_CollimationVSup.push_back(std::numeric_limits<double>::max());

  this->Modified();
}


void
rtk::ThreeDHelicalProjectionGeometry::Clear()
{
  Superclass::Clear();

  m_GantryAngles.clear();
  m_HelicalAngles.clear();
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

bool
rtk::ThreeDHelicalProjectionGeometry::VerifyHelixParameters()
{
  // Check that sid is constant
  std::vector<double> v = this->GetSourceToIsocenterDistances();
  for (size_t i = 0; i < v.size(); i++)
    if (v[i] != v[0])
    {
      itkGenericExceptionMacro(<< "Helical Traj. : SourceToIsocenterDistance must be constant.");
      return false;
    }
  m_HelixRadius = v[0];

  // Check that sdd is constant
  v = this->GetSourceToDetectorDistances();
  for (size_t i = 0; i < v.size(); i++)
    if (v[i] != v[0])
    {
      itkGenericExceptionMacro(<< "(Helical Traj. : SourceToDetectorDistance must be constant.");
      return false;
    }
  m_HelixSourceToDetectorDist = v[0];

  // Check that DetectorOffsetX is constant
  v = this->GetProjectionOffsetsX();
  for (size_t i = 0; i < v.size(); i++)
    if (v[i] != v[0])
    {
      itkGenericExceptionMacro(<< "Helical Traj. : ProjectionOffsetX must be constant");
      return false;
    }
  // Check that SourceOffsetX is constant
  v = this->GetSourceOffsetsX();
  for (size_t i = 0; i < v.size(); i++)
    if (v[i] != v[0])
    {
      itkGenericExceptionMacro(<< "Helical Traj. : SourceOffsetX must be constant");
      return false;
    }

  // Check that SourceOffsetsY and DetectorOffsetsY are equal
  v = this->GetProjectionOffsetsY();
  std::vector<double> w = this->GetSourceOffsetsY();
  for (size_t i = 0; i < w.size(); i++)
    if (!itk::Math::AlmostEquals(v[i], w[i]))
    {
      itkGenericExceptionMacro(<< "Helical Traj. : DetectorOffsetsY must all be equal to SourceOffsetsY");
      return false;
    }

  // Check that vertical gaps are constant
  for (size_t i = 1; i < v.size() - 1; i++)
    if (!itk::Math::FloatAlmostEqual(v[i + 1] - v[i], v[i] - v[i - 1], 4, 1e-10))
    {
      itkGenericExceptionMacro(<< "Helical Traj. : Vertical gaps must all be equal");
      return false;
    }
  m_HelixVerticalGap = v[1] - v[0];

  // Check that InPlaneAnles are al ZERO
  v = this->GetInPlaneAngles();
  for (size_t i = 0; i < v.size(); i++)
    if (v[i] != v[0])
    {
      itkGenericExceptionMacro(<< "Helical Traj. : InPlaneAngles must all be equal to ZERO");
      return false;
    }
  if (!itk::Math::AlmostEquals(v[0], 0.))
  {
    itkGenericExceptionMacro(<< "Helical Traj. : InPlaneAngles must all be equal to ZERO");
    return false;
  }
  // Check that OutOfPlaneAngles are all ZERO
  v = this->GetOutOfPlaneAngles();
  for (size_t i = 0; i < v.size(); i++)
    if (v[i] != v[0])
    {
      itkGenericExceptionMacro(<< "Helical Traj. : OutOfPlaneAngles must all be equal to ZERO");
      return false;
    }
  if (!itk::Math::AlmostEquals(v[0], 0.))
  {
    itkGenericExceptionMacro(<< "Helical Traj. : OutOfPlaneAngles must all be equal to ZERO");
    return false;
  }

  // Check that angluar gap is constant
  v = this->GetHelicalAngles();
  for (size_t i = 1; i < v.size(); i++)
    if (!itk::Math::FloatAlmostEqual(v[i] - v[i - 1], v[1] - v[0], 4, 1e-8))
    {
      std::cout << "Gap 1 : " << v[i] - v[i - 1] << " Gap 2 : " << v[i + 1] - v[i] << std::endl;
      itkGenericExceptionMacro(<< "Helical Traj. : Angular gaps must be constant");
      return false;
    }
  m_HelixAngularGap = v[1] - v[0];
  m_HelixPitch = m_HelixVerticalGap / m_HelixAngularGap * (2 * itk::Math::pi);

  m_TheGeometryIsVerified = true;

  return true;
}
