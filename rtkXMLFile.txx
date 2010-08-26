#ifndef _rtkXMLFile_txx
#define _rtkXMLFile_txx

#include "rtkXMLFile.h"
#include "rtkMacro.h"

#include <itksys/SystemTools.hxx>
#include <itkMetaDataObject.h>
#include <itkIOCommon.h>

namespace rtk
{

template< class T >
int
XMLFileReader< T >::
CanReadFile(const char *name)
{
  if(!itksys::SystemTools::FileExists(name) ||
     itksys::SystemTools::FileIsDirectory(name) ||
     itksys::SystemTools::FileLength(name) == 0)
    return 0;
  return 1;
}

template< class T >
void
XMLFileReader< T >::
StartElement(const char * name,const char **atts)
{
  m_CurCharacterData = "";
  this->StartElement(name);
}

template< class T >
void
XMLFileReader< T >::
CharacterDataHandler(const char *inData, int inLength)
{
  for(int i = 0; i < inLength; i++)
    m_CurCharacterData = m_CurCharacterData + inData[i];
}

template< class T >
int
XMLFileWriter< T >::
CanWriteFile(const char * name)
{
  std::ofstream output(name);
  if(output.fail())
    return false;
  return true;
}

}

#endif
