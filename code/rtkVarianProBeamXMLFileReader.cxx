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

#include "rtkVarianProBeamXMLFileReader.h"
#include "itkMacro.h"

#include <itksys/SystemTools.hxx>
#include <itkMetaDataObject.h>

namespace rtk
{

int
VarianProBeamXMLFileReader::
CanReadFile(const char *name)
{
  if(!itksys::SystemTools::FileExists(name) ||
     itksys::SystemTools::FileIsDirectory(name) ||
     itksys::SystemTools::FileLength(name) == 0)
    return 0;
  return 1;
}

void
VarianProBeamXMLFileReader::
StartElement(const char * itkNotUsed(name),const char ** itkNotUsed(atts))
{
  m_CurCharacterData = "";
}

void
VarianProBeamXMLFileReader::
EndElement(const char *name)
{
#define ENCAPLULATE_META_DATA_DOUBLE(metaName) \
  if(itksys::SystemTools::Strucmp(name, metaName) == 0) { \
    double d = atof(m_CurCharacterData.c_str() ); \
    itk::EncapsulateMetaData<double>(m_Dictionary, metaName, d); \
      }

#define ENCAPLULATE_META_DATA_DOUBLE_AS(metaName, encapsulatedName) \
  if(itksys::SystemTools::Strucmp(name, metaName) == 0) { \
    double d = atof(m_CurCharacterData.c_str() ); \
    itk::EncapsulateMetaData<double>(m_Dictionary, encapsulatedName, d); \
      }

#define MODIFY_META_DATA_DOUBLE_MULTIPLY(metaName, encapsulatedName) \
  if(itksys::SystemTools::Strucmp(name, metaName) == 0) { \
    double d = atof(m_CurCharacterData.c_str() ); \
    typedef itk::MetaDataObject< double > MetaDataDoubleType;\
    const double multiplier = dynamic_cast<MetaDataDoubleType *>(m_Dictionary[encapsulatedName].GetPointer())->GetMetaDataObjectValue(); \
    itk::EncapsulateMetaData<double>(m_Dictionary, encapsulatedName, d * multiplier); \
    }

#define ENCAPLULATE_META_DATA_STRING(metaName) \
  if(itksys::SystemTools::Strucmp(name, metaName) == 0) { \
    itk::EncapsulateMetaData<std::string>(m_Dictionary, metaName, m_CurCharacterData); \
    }

  ENCAPLULATE_META_DATA_DOUBLE("Velocity");
  ENCAPLULATE_META_DATA_DOUBLE("SAD");
  ENCAPLULATE_META_DATA_DOUBLE("SID");
  ENCAPLULATE_META_DATA_DOUBLE("SourceAngleOffset");
  ENCAPLULATE_META_DATA_DOUBLE_AS("ImagerSizeX", "DetectorSizeX");  // DetectorSize = ImagerSize * ImagerRes
  ENCAPLULATE_META_DATA_DOUBLE_AS("ImagerSizeY", "DetectorSizeY");
  MODIFY_META_DATA_DOUBLE_MULTIPLY("ImagerResX", "DetectorSizeX"); // Assumes ImagerSize is always read first!!
  MODIFY_META_DATA_DOUBLE_MULTIPLY("ImagerResY", "DetectorSizeY");
  ENCAPLULATE_META_DATA_DOUBLE("ImagerLat");
  ENCAPLULATE_META_DATA_STRING("Fan");
}

void
VarianProBeamXMLFileReader::
CharacterDataHandler(const char *inData, int inLength)
{
  for(int i = 0; i < inLength; i++)
    m_CurCharacterData = m_CurCharacterData + inData[i];
}

}
