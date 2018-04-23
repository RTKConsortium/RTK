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

#include "rtkOraImageIO.h"
#include "rtkOraXMLFileReader.h"

#include <itksys/SystemTools.hxx>

void
rtk::OraImageIO::ReadImageInformation()
{
  rtk::OraXMLFileReader::Pointer xmlReader;

  std::string oraFileName = this->GetFileName();

  xmlReader = rtk::OraXMLFileReader::New();
  xmlReader->SetFilename(oraFileName);
  xmlReader->GenerateOutputInformation();

  this->SetMetaDataDictionary( *(xmlReader->GetOutputObject() ) );

  // Retrieve MHD file name
  typedef itk::MetaDataObject< std::string > MetaDataStringType;
  MetaDataStringType *mhdMeta = dynamic_cast<MetaDataStringType *>(this->GetMetaDataDictionary()["MHD_File"].GetPointer() );
  if(mhdMeta==ITK_NULLPTR)
    {
    itkExceptionMacro(<< "No MHD_File in " << oraFileName);
    }
  m_MetaFileName = itksys::SystemTools::GetFilenamePath(oraFileName);
  m_MetaFileName += '/';
  m_MetaFileName += mhdMeta->GetMetaDataObjectValue();

  this->SetFileName(m_MetaFileName);
  Superclass::ReadImageInformation();
  this->SetFileName(oraFileName);
}

bool
rtk::OraImageIO::CanReadFile(const char* FileNameToRead)
{
  std::string filename(FileNameToRead);
  if(filename.size()<8)
      return false;
  std::string extension = filename.substr(filename.size()-7, 7);

  if(extension != std::string("ora.xml") )
    return false;

  return true;
}

void
rtk::OraImageIO::Read(void *buffer)
{
  std::string oraFileName = this->GetFileName();
  this->SetFileName(m_MetaFileName);
  Superclass::Read(buffer);
  this->SetFileName(oraFileName);
}

bool
rtk::OraImageIO::CanWriteFile( const char* itkNotUsed(FileNameToWrite) )
{
  return false;
}
