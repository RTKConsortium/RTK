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
#include "rtkQuadricShape.h"

namespace rtk
{
void
GeometricPhantomFileReader
::GenerateOutputInformation()
{
  m_GeometricPhantom = GeometricPhantom::New();

  //Admitted figures
  const unsigned int NFIGURES = 4;
  const std::string search_fig[NFIGURES] = {"Ellipsoid", "Cylinder", "Cone", "Box"};
  size_t            offset        = 0;
  std::string       line;
  std::ifstream     myFile;

  myFile.open( m_Filename.c_str() );
  if ( !myFile.is_open() )
    {
    itkGenericExceptionMacro("Error opening File");
    }
  while ( !myFile.eof() )
    {
    getline(myFile, line);
    for(unsigned int i = 0; i < NFIGURES; i++)
      {
      if( (offset = line.find(search_fig[i], 0)) != std::string::npos )
        {
        const std::string parameterNames[9] = {"Figure", "A=", "B=", "C=", "x=", "y=", "z=", "beta=", "gray=" };
        std::vector<ConvexShape::ScalarType> parameters;
        parameters.push_back((ConvexShape::ScalarType)i);
        for ( int j = 1; j < 9; j++ )
          {
          double val = 0.;
          offset = line.find(parameterNames[j], 0);
          if ( offset != std::string::npos )
            {
            offset += parameterNames[j].length();
            std::string s = line.substr(offset,line.length()-offset);
            std::istringstream ss(s);
            ss >> val;
            //Saving all parameters for each ellipsoid
            }
          parameters.push_back(val);
          }

        QuadricShape::Pointer qo = QuadricShape::New();
        QuadricShape::VectorType axis;
        QuadricShape::PointType center;
        for(int k=0; k<3; k++)
          {
          axis[k] = parameters[k+1];
          center[k] = parameters[k+4];
          }
        if(search_fig[i]=="Box")
          {
          // TODO
          }
        else
          {
          qo->SetEllipsoid(center, axis, parameters[7]);
          if(search_fig[i]=="Cone")
            qo->SetJ(0.);
          }

        qo->SetDensity(parameters[8]);
        m_GeometricPhantom->AddConvexShape(qo.GetPointer());
        }
      }
    }
  myFile.close();
}

} // namespace rtk
