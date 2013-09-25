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
#include "rtkGeometricPhantomFileReader.h"

namespace rtk
{
bool GeometricPhantomFileReader::Config(const std::string ConfigFile )
{
  //Admitted figures
  const std::string search_fig[4] = {"Ellipsoid", "Cylinder", "Cone", "Box"};
  size_t            offset        = 0;
  std::string       line;
  std::ifstream     myFile;

  myFile.open( ConfigFile.c_str() );
  if ( !myFile.is_open() )
    {
    itkGenericExceptionMacro("Error opening File");
    return false;
    }
  unsigned int k=0;
  while ( !myFile.eof() )
    {
    k++;
    getline(myFile, line);
    for(unsigned int i = 0; i < 4; i++)
      {
      if( (offset = line.find(search_fig[i], 0)) != std::string::npos )
        {
        const std::string parameterNames[9] = {"Figure", "A=", "B=", "C=", "x=", "y=", "z=", "beta=", "gray=" };
        VectorType parameters;
        parameters.push_back((double)i);
        for ( int j = 1; j < 9; j++ )
          {
          double val = 0.;
          if ( ( offset = line.find(parameterNames[j], 0) ) != std::string::npos )
            {
            offset += parameterNames[j].length();
            std::string s = line.substr(offset,line.length()-offset);
            std::istringstream ss(s);
            ss >> val;
            //Saving all parameters for each ellipsoid
            }
          parameters.push_back(val);
          }
        m_Fig.push_back(parameters);
        m_FigureTypes.push_back(search_fig[i]);
        }
      }
    }
    myFile.close();
    return true;
}

GeometricPhantomFileReader::VectorOfVectorType GeometricPhantomFileReader::GetFig ()
{
  itkDebugMacro("returning Fig.");
  return this->m_Fig;
}

void GeometricPhantomFileReader::SetFig (const VectorOfVectorType _arg)
{
  itkDebugMacro("setting Fig");
  if (this->m_Fig != _arg)
    {
    this->m_Fig = _arg;
    this->Modified();
    }
}

} // namespace rtk
