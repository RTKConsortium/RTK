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
  const char *       search_fig = "Ellipsoid"; // Set search pattern
  size_t             offset = 0;
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
      VectorType parameters;
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
