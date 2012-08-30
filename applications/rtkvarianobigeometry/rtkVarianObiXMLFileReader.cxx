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

#include "rtkVarianObiXMLFileReader.h"
#include "itkMacro.h"

#include <itksys/SystemTools.hxx>
#include <itkMetaDataObject.h>

namespace rtk
{

int
VarianObiXMLFileReader::
CanReadFile(const char *name)
{
  if(!itksys::SystemTools::FileExists(name) ||
     itksys::SystemTools::FileIsDirectory(name) ||
     itksys::SystemTools::FileLength(name) == 0)
    return 0;
  return 1;
}

void
VarianObiXMLFileReader::
StartElement(const char * itkNotUsed(name),const char ** itkNotUsed(atts))
{
  m_CurCharacterData = "";
}

void
VarianObiXMLFileReader::
EndElement(const char *name)
{
#define ENCAPLULATE_META_DATA_DOUBLE(metaName) \
  if(itksys::SystemTools::Strucmp(name, metaName) == 0) { \
    double d = atof(m_CurCharacterData.c_str() ); \
    itk::EncapsulateMetaData<double>(m_Dictionary, metaName, d); \
    }

#define ENCAPLULATE_META_DATA_STRING(metaName) \
  if(itksys::SystemTools::Strucmp(name, metaName) == 0) { \
    itk::EncapsulateMetaData<std::string>(m_Dictionary, metaName, m_CurCharacterData); \
    }

  ENCAPLULATE_META_DATA_DOUBLE("GantryRtnSpeed");
  ENCAPLULATE_META_DATA_DOUBLE("CalibratedSAD");
  ENCAPLULATE_META_DATA_DOUBLE("CalibratedSID");
  ENCAPLULATE_META_DATA_DOUBLE("CalibratedDetectorOffsetX");
  ENCAPLULATE_META_DATA_DOUBLE("CalibratedDetectorOffsetY");
  ENCAPLULATE_META_DATA_DOUBLE("DetectorSizeX");
  ENCAPLULATE_META_DATA_DOUBLE("DetectorSizeY");
  ENCAPLULATE_META_DATA_DOUBLE("DetectorPosLat");
  ENCAPLULATE_META_DATA_STRING("FanType");
}

void
VarianObiXMLFileReader::
CharacterDataHandler(const char *inData, int inLength)
{
  for(int i = 0; i < inLength; i++)
    m_CurCharacterData = m_CurCharacterData + inData[i];
}

}
