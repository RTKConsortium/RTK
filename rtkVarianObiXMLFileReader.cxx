#ifndef _rtkVarianObiXMLFileReader_cxx
#define _rtkVarianObiXMLFileReader_cxx

#include "rtkVarianObiXMLFileReader.h"
#include "rtkMacro.h"

#include <itksys/SystemTools.hxx>
#include <itkMetaDataObject.h>

namespace rtk
{

void 
VarianObiXMLFileReader::
StartElement(const char * name)
{
}

void 
VarianObiXMLFileReader::
EndElement(const char *name)
{
#define ENCAPLULATE_META_DATA_DOUBLE(metaName) \
  if(itksys::SystemTools::Strucmp(name, metaName) == 0) { \
    double d = atof(m_CurCharacterData.c_str()); \
    itk::EncapsulateMetaData<double>(m_Dictionary, metaName, d); \
  }

  ENCAPLULATE_META_DATA_DOUBLE("GantryRtnSpeed");
  ENCAPLULATE_META_DATA_DOUBLE("CalibratedSAD");
  ENCAPLULATE_META_DATA_DOUBLE("CalibratedSID");
  ENCAPLULATE_META_DATA_DOUBLE("CalibratedDetectorOffsetX");
  ENCAPLULATE_META_DATA_DOUBLE("CalibratedDetectorOffsetY");
  ENCAPLULATE_META_DATA_DOUBLE("DetectorSizeX");
  ENCAPLULATE_META_DATA_DOUBLE("DetectorSizeY");
}

}

#endif
