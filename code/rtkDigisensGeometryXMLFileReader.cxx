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

#include "rtkDigisensGeometryXMLFileReader.h"

#include <itksys/SystemTools.hxx>
#include <itkMetaDataObject.h>
#include <itkVector.h>

namespace rtk
{

DigisensGeometryXMLFileReader::
DigisensGeometryXMLFileReader()
{
  m_OutputObject = &m_Dictionary;
  m_NumberOfFiles = 0;
  m_CurrentSection = NONE;
  m_TreeLevel = 0;
}

int
DigisensGeometryXMLFileReader::
CanReadFile(const char *name)
{
  if(!itksys::SystemTools::FileExists(name) ||
     itksys::SystemTools::FileIsDirectory(name) ||
     itksys::SystemTools::FileLength(name) == 0)
    return 0;
  return 1;
}

void
DigisensGeometryXMLFileReader::
StartElement(const char * name,const char ** itkNotUsed(atts))
{
  m_CurCharacterData = "";

  if(m_TreeLevel==1)
    {
    if(itksys::SystemTools::Strucmp(name, "Rotation") == 0)
      m_CurrentSection = ROTATION;
    else if(itksys::SystemTools::Strucmp(name, "XRay") == 0)
      m_CurrentSection = XRAY;
    else if(itksys::SystemTools::Strucmp(name, "Camera") == 0)
      m_CurrentSection = CAMERA;
    else if(itksys::SystemTools::Strucmp(name, "Radios") == 0)
      {
      m_CurrentSection = RADIOS;
      m_NumberOfFiles = 0;
      }
    else if(itksys::SystemTools::Strucmp(name, "Grid") == 0)
      m_CurrentSection = GRID;
    else if(itksys::SystemTools::Strucmp(name, "Processing") == 0)
      m_CurrentSection = PROCESSING;
    }
  m_TreeLevel++;
}

void
DigisensGeometryXMLFileReader::
EndElement(const char *name)
{
  typedef itk::Vector<double, 3> VectorThreeDType;
  typedef itk::Vector<double, 4> Vector4DType;

#define ENCAPLULATE_META_DATA_3D(section, metaName) \
  if(m_CurrentSection == section && itksys::SystemTools::Strucmp(name, metaName) == 0) \
    { \
    VectorThreeDType vec; \
    std::istringstream iss(m_CurCharacterData); \
    iss >> vec; \
    itk::EncapsulateMetaData<VectorThreeDType>(m_Dictionary, #section metaName, vec); \
    }

#define ENCAPLULATE_META_DATA_4D(section, metaName) \
  if(m_CurrentSection == section && itksys::SystemTools::Strucmp(name, metaName) == 0) \
    { \
    Vector4DType vec; \
    std::istringstream iss(m_CurCharacterData); \
    iss >> vec; \
    itk::EncapsulateMetaData<Vector4DType>(m_Dictionary, #section metaName, vec); \
    }

#define ENCAPLULATE_META_DATA_INTEGER(section, metaName) \
  if(m_CurrentSection == section && itksys::SystemTools::Strucmp(name, metaName) == 0) \
    { \
    int i = atoi(m_CurCharacterData.c_str() ); \
    itk::EncapsulateMetaData<int>(m_Dictionary, #section metaName, i); \
    }

#define ENCAPLULATE_META_DATA_DOUBLE(section, metaName) \
  if(m_CurrentSection == section && itksys::SystemTools::Strucmp(name, metaName) == 0) \
    { \
    double d = atof(m_CurCharacterData.c_str() ); \
    itk::EncapsulateMetaData<double>(m_Dictionary, #section metaName, d); \
    }

  ENCAPLULATE_META_DATA_4D(GRID, "rotation");

  ENCAPLULATE_META_DATA_3D(ROTATION, "axis");
  ENCAPLULATE_META_DATA_3D(ROTATION, "center");
  ENCAPLULATE_META_DATA_3D(XRAY, "source");
  ENCAPLULATE_META_DATA_3D(CAMERA, "reference");
  ENCAPLULATE_META_DATA_3D(CAMERA, "normal");
  ENCAPLULATE_META_DATA_3D(CAMERA, "horizontal");
  ENCAPLULATE_META_DATA_3D(CAMERA, "vertical");
  ENCAPLULATE_META_DATA_3D(GRID, "center");
  ENCAPLULATE_META_DATA_3D(GRID, "scale");
  ENCAPLULATE_META_DATA_3D(GRID, "resolution");

  ENCAPLULATE_META_DATA_INTEGER(CAMERA, "pixelWidth");
  ENCAPLULATE_META_DATA_INTEGER(CAMERA, "pixelHeight");

  ENCAPLULATE_META_DATA_DOUBLE(CAMERA, "totalWidth");
  ENCAPLULATE_META_DATA_DOUBLE(CAMERA, "totalHeight");
  ENCAPLULATE_META_DATA_DOUBLE(RADIOS, "angularRange");
  ENCAPLULATE_META_DATA_DOUBLE(RADIOS, "startAngle");

  if(m_CurrentSection == RADIOS && itksys::SystemTools::Strucmp(name, "files") == 0)
    itk::EncapsulateMetaData<int>(m_Dictionary, "RADIOSNumberOfFiles", m_NumberOfFiles);

  if(itksys::SystemTools::Strucmp(name, "file") == 0)
    m_NumberOfFiles++;

  m_TreeLevel--;
}

void
DigisensGeometryXMLFileReader::
CharacterDataHandler(const char *inData, int inLength)
{
  for(int i = 0; i < inLength; i++)
    m_CurCharacterData = m_CurCharacterData + inData[i];
}

}
