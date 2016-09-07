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

#include "rtkImagXXMLFileReader.h"
#include "itkMacro.h"

#include <itksys/SystemTools.hxx>
#include <itkMetaDataObject.h>

namespace rtk
{

int
ImagXXMLFileReader::
CanReadFile(const char *name)
{
  if(!itksys::SystemTools::FileExists(name) ||
     itksys::SystemTools::FileIsDirectory(name) ||
     itksys::SystemTools::FileLength(name) == 0)
    return 0;
  return 1;
}

void
ImagXXMLFileReader::
StartElement(const char * name, const char ** atts)
{
#define ENCAPLULATE_META_DATA_INT(metaName) \
  if(itksys::SystemTools::Strucmp(atts[i], metaName) == 0) { \
    double d = atof(atts[i+1]); \
    itk::EncapsulateMetaData<int>(m_Dictionary, metaName, d); \
    }

#define ENCAPLULATE_META_DATA_STRING(metaName) \
  if(itksys::SystemTools::Strucmp(atts[i], metaName) == 0) { \
    itk::EncapsulateMetaData<std::string>(m_Dictionary, metaName, atts[i+1]); \
    }

  if(std::string(name) == std::string("image") )
    {
    for(int i=0; atts[i] != ITK_NULLPTR; i+=2)
      {
      ENCAPLULATE_META_DATA_STRING("name");
      ENCAPLULATE_META_DATA_INT("bitDepth");
      ENCAPLULATE_META_DATA_STRING("pixelFormat");
      ENCAPLULATE_META_DATA_STRING("byteOrder");
      ENCAPLULATE_META_DATA_STRING("modality");
      ENCAPLULATE_META_DATA_STRING("matrixTransform");
      ENCAPLULATE_META_DATA_INT("dimensions");
      ENCAPLULATE_META_DATA_INT("sequence");
      ENCAPLULATE_META_DATA_STRING("rawFile");
      }
    }
  if(std::string(name) == std::string("size") )
    {
    for(int i=0; atts[i] != ITK_NULLPTR; i+=2)
      {
      ENCAPLULATE_META_DATA_INT("x");
      ENCAPLULATE_META_DATA_INT("y");
      ENCAPLULATE_META_DATA_INT("z");
      }
    }
  if(std::string(name) == std::string("spacing") )
    {
#define ENCAPLULATE_META_DATA_DOUBLE(metaName) \
  if(itksys::SystemTools::Strucmp(atts[i], metaName) == 0) { \
    double d = atof(atts[i+1]); \
    itk::EncapsulateMetaData<double>(m_Dictionary, std::string("spacing_") + std::string(metaName), d); \
    }
    for(int i=0; atts[i] != ITK_NULLPTR; i+=2)
      {
      ENCAPLULATE_META_DATA_DOUBLE("x");
      ENCAPLULATE_META_DATA_DOUBLE("y");
      ENCAPLULATE_META_DATA_DOUBLE("z");
      }
    }
  m_CurCharacterData = "";
}

void
ImagXXMLFileReader::
EndElement( const char *itkNotUsed(name) )
{
}

void
ImagXXMLFileReader::
CharacterDataHandler(const char *inData, int inLength)
{
  for(int i = 0; i < inLength; i++)
    m_CurCharacterData = m_CurCharacterData + inData[i];
}

}
