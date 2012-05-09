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

#include <fstream>

namespace rtk
{
SetQuadricParamFromRegularParamFunction
::SetQuadricParamFromRegularParamFunction():
m_SemiPrincipalAxisX(0.),
m_SemiPrincipalAxisY(0.),
m_SemiPrincipalAxisZ(0.),
m_CenterX(0.),
m_CenterY(0.),
m_CenterZ(0.),
m_RotationAngle(0.), m_A(0.), m_B(0.), m_C(0.), m_D(0.),
m_E(0.), m_F(0.), m_G(0.), m_H(0.), m_I(0.), m_J(0.)
{
}

bool SetQuadricParamFromRegularParamFunction
::Translate( const VectorType& SemiPrincipalAxis )
{
  SetSemiPrincipalAxisX(SemiPrincipalAxis[0]);
  m_SemiPrincipalAxisY = SemiPrincipalAxis[1];
  m_SemiPrincipalAxisZ = SemiPrincipalAxis[2];

  //Regular Ellipsoid Expression (No rotation, No Translation)
  m_A = 1/pow(m_SemiPrincipalAxisX,2.0);
  m_B = 1/pow(m_SemiPrincipalAxisY,2.0);
  m_C = 1/pow(m_SemiPrincipalAxisZ,2.0);
  m_D = 0.;
  m_E = 0.;
  m_F = 0.;
  m_G = 0.;
  m_H = 0.;
  m_I = 0.;
  m_J = -1.;
  return true;
}

bool SetQuadricParamFromRegularParamFunction
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

  //Applying Rotation
  m_A = TempA*pow(cos(m_RotationAngle*(itk::Math::pi/180)), 2.0) + TempB*pow(sin(m_RotationAngle*(itk::Math::pi/180)),2.0);
  m_B = TempA*pow(sin(m_RotationAngle*(itk::Math::pi/180)), 2.0) + TempB*pow(cos(m_RotationAngle*(itk::Math::pi/180)),2.0);
  m_C = TempC;
  m_D = 2*cos(m_RotationAngle*(itk::Math::pi/180))*sin(m_RotationAngle*(itk::Math::pi/180))*(TempB - TempA);
  m_E = 0.;
  m_F = 0.;
  m_G = TempG*cos(m_RotationAngle*(itk::Math::pi/180)) + TempH*sin(m_RotationAngle*(itk::Math::pi/180));
  m_H = TempG*(-1)*sin(m_RotationAngle*(itk::Math::pi/180)) + TempH*cos(m_RotationAngle*(itk::Math::pi/180));
  m_I = TempI;
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
  double TransJ = TempA*pow(m_CenterX,2.0) + TempB*pow(m_CenterY,2.0) + TempC*pow(m_CenterZ,2.0) + TempD*m_CenterX*m_CenterY + TempE*m_CenterX*m_CenterZ + TempF*m_CenterY*m_CenterZ - TempG*m_CenterX - TempH*m_CenterY - TempI*m_CenterZ;

  //Applying Translation
  m_G += TransG;
  m_H += TransH;
  m_I += TransI;
  m_J += TransJ;
  return true;
}

bool SetQuadricParamFromRegularParamFunction::Config(const std::string ConfigFile )
{
  const char *       search_fig = "Ellipsoid"; // Set search pattern
  int                offset = 0;
  std::string        line;
  std::ifstream      myFile;

  myFile.open( ConfigFile.c_str() );
  if ( !myFile.is_open() )
    {
    itkGenericExceptionMacro("Error opening File");
    return false;
    }

  while ( !myFile.eof() )
    {
    getline(myFile, line);
    if ( ( offset = line.find(search_fig, 0) ) != std::string::npos ) //Ellipsoid
                                                                      // found
      {
      const std::string parameterNames[8] = { "x", "y", "z", "A", "B", "C", "beta", "gray" };
      std::vector<double> parameters;
      for ( int j = 0; j < 8; j++ )
        {
        double val = 0.;
        if ( ( offset = line.find(parameterNames[j], 0) ) != std::string::npos )
          {
          offset += parameterNames[j].length()+1;
          std::string s = line.substr(offset,line.length()-offset);
          std::istringstream ss(s);
          ss >> val;
          //Saving all parameters for each ellipsoid
          }
        parameters.push_back(val);
        }
      m_Fig.push_back(parameters);
      }
    }
  myFile.close();
  return true;
}

} // namespace rtk
