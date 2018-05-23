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

#include "rtkOraXMLFileReader.h"

#include <itksys/SystemTools.hxx>
#include <itkMetaDataObject.h>

namespace rtk
{
OraXMLFileReader
::OraXMLFileReader()
{
  m_OutputObject = &m_Dictionary;
}

int
OraXMLFileReader
::CanReadFile(const char *name)
{
  if(!itksys::SystemTools::FileExists(name) ||
     itksys::SystemTools::FileIsDirectory(name) ||
     itksys::SystemTools::FileLength(name) == 0)
    return 0;
  return 1;
}

void
OraXMLFileReader
::StartElement(const char * itkNotUsed(name),const char ** itkNotUsed(atts))
{
  m_CurCharacterData = "";
}

void
OraXMLFileReader
::EndElement(const char *name)
{
  EncapsulatePoint("SourcePosition", name);
  EncapsulatePoint("Origin", name);
  EncapsulateMatrix3x3("Direction", name);
  EncapsulateDouble("table_axis_distance_cm", name);
  EncapsulateDouble("longitudinalposition_cm", name);
  EncapsulateDouble("rescale_slope", name);
  EncapsulateDouble("rescale_intercept", name);
  EncapsulateString("MHD_File", name);
  EncapsulateDouble("xrayx1_cm", name);
  EncapsulateDouble("xrayx2_cm", name);
  EncapsulateDouble("xrayy1_cm", name);
  EncapsulateDouble("xrayy2_cm", name);
}

void
OraXMLFileReader
::CharacterDataHandler(const char *inData, int inLength)
{
  for(int i = 0; i < inLength; i++)
    m_CurCharacterData = m_CurCharacterData + inData[i];
}

void
OraXMLFileReader
::EncapsulatePoint(const char *metaName, const char *name)
{
  if(itksys::SystemTools::Strucmp(name, metaName) == 0)
    {
    typedef itk::Vector<double, 3> PointType;
    PointType p;
    std::istringstream iss(m_CurCharacterData);
    for(int i=0; i<3; i++)
      {
      iss >> p[i];
      iss.ignore(1);
      }
    itk::EncapsulateMetaData<PointType>(m_Dictionary, metaName, p);
    }
}

void
OraXMLFileReader
::EncapsulateMatrix3x3(const char *metaName, const char *name)
{
  if(itksys::SystemTools::Strucmp(name, metaName) == 0)
    {
    typedef itk::Matrix<double, 3, 3> Matrix3x3Type;
    Matrix3x3Type m;
    std::istringstream iss(m_CurCharacterData);
    for(int i=0; i<3; i++)
      {
      for(int j=0; j<3; j++)
        {
        iss >> m[i][j];
        iss.ignore(1);
        }
      }
    itk::EncapsulateMetaData<Matrix3x3Type>(m_Dictionary, metaName, m);
    }
}

void
OraXMLFileReader
::EncapsulateDouble(const char *metaName, const char *name)
{
  if(itksys::SystemTools::Strucmp(name, metaName) == 0)
    {
    double d = atof(m_CurCharacterData.c_str());
    itk::EncapsulateMetaData<double>(m_Dictionary, metaName, d);
    }
}

void
OraXMLFileReader
::EncapsulateString(const char *metaName, const char *name)
{
  if(itksys::SystemTools::Strucmp(name, metaName) == 0)
    {
    itk::EncapsulateMetaData<std::string>(m_Dictionary, metaName, m_CurCharacterData);
    }
}

}
