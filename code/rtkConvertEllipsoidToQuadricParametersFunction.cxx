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

#include <itkMath.h>
#include <vcl_cmath.h>
#include "rtkConvertEllipsoidToQuadricParametersFunction.h"

rtk::ConvertEllipsoidToQuadricParametersFunction
::ConvertEllipsoidToQuadricParametersFunction():
m_SemiPrincipalAxisX(0.),
m_SemiPrincipalAxisY(0.),
m_SemiPrincipalAxisZ(0.),
m_CenterX(0.),
m_CenterY(0.),
m_CenterZ(0.),
m_RotationAngle(0.), m_A(0.), m_B(0.), m_C(0.), m_D(0.),
m_E(0.), m_F(0.), m_G(0.), m_H(0.), m_I(0.), m_J(0.),
m_Figure("Ellipsoid")
{
}

bool rtk::ConvertEllipsoidToQuadricParametersFunction
::Translate( const VectorType& SemiPrincipalAxis )
{
  m_SemiPrincipalAxisX = SemiPrincipalAxis[0];
  m_SemiPrincipalAxisY = SemiPrincipalAxis[1];
  m_SemiPrincipalAxisZ = SemiPrincipalAxis[2];
  //Regular Ellipsoid Expression (No rotation, No Translation)
  // Parameter A
  if(m_SemiPrincipalAxisX > itk::NumericTraits<double>::ZeroValue())
    m_A = 1/vcl_pow(m_SemiPrincipalAxisX,2.0);
  else if (m_SemiPrincipalAxisX < itk::NumericTraits<double>::ZeroValue())
    m_A = -1/vcl_pow(m_SemiPrincipalAxisX,2.0);
  else
    m_A = 0.;
  // Parameter B
  if(m_SemiPrincipalAxisY > itk::NumericTraits<double>::ZeroValue())
    m_B = 1/vcl_pow(m_SemiPrincipalAxisY,2.0);
  else if (m_SemiPrincipalAxisY < itk::NumericTraits<double>::ZeroValue())
    m_B = -1/vcl_pow(m_SemiPrincipalAxisY,2.0);
  else
    m_B = 0.;
  // Parameter C
  if(m_SemiPrincipalAxisZ > itk::NumericTraits<double>::ZeroValue())
    m_C = 1/vcl_pow(m_SemiPrincipalAxisZ,2.0);
  else if (m_SemiPrincipalAxisZ < itk::NumericTraits<double>::ZeroValue())
    m_C = -1/vcl_pow(m_SemiPrincipalAxisZ,2.0);
  else
    m_C = 0.;

  m_D = 0.;
  m_E = 0.;
  m_F = 0.;
  m_G = 0.;
  m_H = 0.;
  m_I = 0.;

  // J Quadric value for surfaces different to Cylinders and Ellipsoids.
  if(m_Figure == "Cylinder" || m_Figure == "Ellipsoid")
    m_J = -1;
  else if(m_Figure == "Cone")
    m_J = 0;
  else
    m_J = -1;

  return true;
}

bool rtk::ConvertEllipsoidToQuadricParametersFunction
::Rotate( const double RotationAngle, const VectorType& Center )
{
  m_RotationAngle = RotationAngle;
  m_CenterX = Center[0];
  m_CenterY = Center[1];
  m_CenterZ = Center[2];

  //Temporary Quadric Parameters
  double TempA = m_A;
  double TempB = m_B;
  double TempC = m_C;
  double TempD = m_D;
  double TempE = m_E;
  double TempF = m_F;
  double TempG = m_G;
  double TempH = m_H;
  double TempI = m_I;
  double TempJ = m_J;

  //Applying Rotation on Y-axis
  m_A = TempA*vcl_pow(cos(m_RotationAngle*(itk::Math::pi/180)), 2.0) + TempC*vcl_pow(sin(m_RotationAngle*(itk::Math::pi/180)),2.0);
  m_B = TempB;
  m_C = TempA*vcl_pow(sin(m_RotationAngle*(itk::Math::pi/180)), 2.0) + TempC*vcl_pow(cos(m_RotationAngle*(itk::Math::pi/180)),2.0);
  m_D = 0.;
  m_E = 2*cos(m_RotationAngle*(itk::Math::pi/180))*sin(m_RotationAngle*(itk::Math::pi/180))*(TempA - TempC);
  m_F = 0.;
  m_G = TempG*cos(m_RotationAngle*(itk::Math::pi/180)) - TempI*sin(m_RotationAngle*(itk::Math::pi/180));
  m_H = TempH;
  m_I = TempG*sin(m_RotationAngle*(itk::Math::pi/180)) + TempI*cos(m_RotationAngle*(itk::Math::pi/180));
  m_J = TempJ;

  //Saving Quadric Parameters for Translation
  TempA = m_A;
  TempB = m_B;
  TempC = m_C;
  TempD = m_D;
  TempE = m_E;
  TempF = m_F;
  TempG = m_G;
  TempH = m_H;
  TempI = m_I;
  TempJ = m_J;

  //Translation Parameters
  double TransG = -2*TempA*m_CenterX - TempD*m_CenterY - TempE*m_CenterZ;
  double TransH = -2*TempB*m_CenterY - TempD*m_CenterX - TempF*m_CenterZ;
  double TransI = -2*TempC*m_CenterZ - TempE*m_CenterX - TempF*m_CenterY;
  double TransJ = TempA*vcl_pow(m_CenterX,2.0) + TempB*vcl_pow(m_CenterY,2.0)
                + TempC*vcl_pow(m_CenterZ,2.0) + TempD*m_CenterX*m_CenterY
                + TempE*m_CenterX*m_CenterZ + TempF*m_CenterY*m_CenterZ
                - TempG*m_CenterX - TempH*m_CenterY - TempI*m_CenterZ;

  //Applying Translation
  m_G += TransG;
  m_H += TransH;
  m_I += TransI;
  m_J += TransJ;
  return true;
}
