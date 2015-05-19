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

#include "rtkElektaXVI5GeometryXMLFile.h"

#include <itksys/SystemTools.hxx>
#include <itkMetaDataObject.h>
#include <itkIOCommon.h>

#include <iomanip>

namespace rtk
{

ElektaXVI5GeometryXMLFileReader::
  ElektaXVI5GeometryXMLFileReader() :
  m_Geometry(GeometryType::New() ),
  m_CurCharacterData(""),
  m_InPlaneAngle(0.),
  m_OutOfPlaneAngle(0.),
  m_GantryAngle(0.),
  m_SourceToIsocenterDistance(1000.),
  m_SourceOffsetX(0.),
  m_SourceOffsetY(0.),
  m_SourceToDetectorDistance(1536.),
  m_ProjectionOffsetX(0.),
  m_ProjectionOffsetY(0.),
  m_Version(0)
{
  this->m_OutputObject = &(*m_Geometry);
}

int
ElektaXVI5GeometryXMLFileReader::
CanReadFile(const char *name)
{
  if(!itksys::SystemTools::FileExists(name) ||
     itksys::SystemTools::FileIsDirectory(name) ||
     itksys::SystemTools::FileLength(name) == 0)
    return 0;
  return 1;
}

void
ElektaXVI5GeometryXMLFileReader::
StartElement(const char * name,const char **atts)
{
  m_CurCharacterData = "";
  this->StartElement(name);
}

void
ElektaXVI5GeometryXMLFileReader::
StartElement(const char * itkNotUsed(name))
{
}

void
ElektaXVI5GeometryXMLFileReader::
EndElement(const char *name)
{
  if (itksys::SystemTools::Strucmp(name, "GantryAngle") == 0 ||
    itksys::SystemTools::Strucmp(name, "Angle") == 0) // Second one for backward compatibility
  {
    m_GantryAngle = atof(this->m_CurCharacterData.c_str());
    if (m_GantryAngle < 0)
      m_GantryAngle = m_GantryAngle + 360.0;
  }

  //Regarding PanelOffset, XVI5 specifies position of the center(UCentre, VCentre) instead of offset.
  //Therefore, negation is required to get classical m_ProjectionOffsetX and m_ProjectionOffsetY values.
  if (itksys::SystemTools::Strucmp(name, "UCentre") == 0)
    m_ProjectionOffsetX = atof(this->m_CurCharacterData.c_str()) * -1.0;

  if (itksys::SystemTools::Strucmp(name, "VCentre") == 0)
  {
    m_ProjectionOffsetY = atof(this->m_CurCharacterData.c_str()) * -1.0;
  }

  if (itksys::SystemTools::Strucmp(name, "Frame") == 0)
  {
    this->m_OutputObject->AddProjection(m_SourceToIsocenterDistance,
      m_SourceToDetectorDistance,
      m_GantryAngle,
      m_ProjectionOffsetX,
      m_ProjectionOffsetY,
      m_OutOfPlaneAngle,
      m_InPlaneAngle,
      m_SourceOffsetX,
      m_SourceOffsetY);
  }
}

void
ElektaXVI5GeometryXMLFileReader::
CharacterDataHandler(const char *inData, int inLength)
{
  for(int i = 0; i < inLength; i++)
    m_CurCharacterData = m_CurCharacterData + inData[i];
}

}
