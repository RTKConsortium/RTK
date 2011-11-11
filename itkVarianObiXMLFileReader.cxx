#include "itkVarianObiXMLFileReader.h"
#include "rtkMacro.h"

#include <itksys/SystemTools.hxx>
#include <itkMetaDataObject.h>

namespace itk
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
StartElement(const char * name,const char **atts)
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
