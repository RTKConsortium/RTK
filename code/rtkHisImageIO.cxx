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

#define HEADER_INFO_SIZE 68

// Based on a true story by the Nederlands Kanker Instituut (AVS_HEIMANN.CPP
// from the 20090608)

// Includes
#include <fstream>
#include "rtkHisImageIO.h"

//--------------------------------------------------------------------
// Read Image Information
void rtk::HisImageIO::ReadImageInformation()
{
  // open file
  std::ifstream file(m_FileName.c_str(), std::ios::in | std::ios::binary);

  if ( file.fail() )
    itkGenericExceptionMacro(<< "Could not open file (for reading): "
                             << m_FileName);

  // read header
  char header[HEADER_INFO_SIZE];
  file.read(header, HEADER_INFO_SIZE);

  if (header[0]!=0 || header[1]!=112 || header[2]!=68 || header[3]!=0) {
    itkExceptionMacro(<< "rtk::HisImageIO::ReadImageInformation: file "
                      << m_FileName
                      << " not in Heimann HIS format version 100");
    return;
    }

  int nrframes, type, ulx, uly, brx, bry;
  m_HeaderSize  = header[10] + (header[11]<<8);
  ulx           = header[12] + (header[13]<<8);
  uly           = header[14] + (header[15]<<8);
  brx           = header[16] + (header[17]<<8);
  bry           = header[18] + (header[19]<<8);
  nrframes      = header[20] + (header[21]<<8);
  type          = header[32] + (header[34]<<8);

  switch(type)
    {
    case  4:
      SetComponentType(itk::ImageIOBase::USHORT);
      break;
//    case  8: SetComponentType(itk::ImageIOBase::INT);   break;
//    case 16: SetComponentType(itk::ImageIOBase::FLOAT); break;
//    case 32: SetComponentType(itk::ImageIOBase::INT);   break;
    default:
      SetComponentType(itk::ImageIOBase::USHORT);
      break;
    }

  switch(nrframes)
    {
    case 1:
      SetNumberOfDimensions(2);
      break;
    default:
      SetNumberOfDimensions(3);
      break;
    }

  SetDimensions(0, bry-uly+1);
  SetDimensions(1, brx-ulx+1);
  if (nrframes>1)
    SetDimensions(2, nrframes);

  SetSpacing(0, 409.6/GetDimensions(0) );
  SetSpacing(1, 409.6/GetDimensions(1) );

  SetOrigin(0, -0.5*(GetDimensions(0)-1)*GetSpacing(0) );
  SetOrigin(1, -0.5*(GetDimensions(1)-1)*GetSpacing(1) );
} ////

//--------------------------------------------------------------------
// Read Image Information
bool rtk::HisImageIO::CanReadFile(const char* FileNameToRead)
{
  std::string                  filename(FileNameToRead);
  const std::string::size_type it = filename.find_last_of( "." );
  std::string                  fileExt( filename, it+1, filename.length() );

  if (fileExt != std::string("his") )
    return false;
  return true;
} ////

//--------------------------------------------------------------------
// Read Image Content
void rtk::HisImageIO::Read(void * buffer)
{
  // open file
  std::ifstream file(m_FileName.c_str(), std::ios::in | std::ios::binary);

  if ( file.fail() )
    itkGenericExceptionMacro(<< "Could not open file (for reading): " << m_FileName);

  file.seekg(m_HeaderSize+HEADER_INFO_SIZE, std::ios::beg);
  if ( file.fail() )
    itkExceptionMacro(<<"File seek failed (His Read)");

  file.read( (char*)buffer, GetImageSizeInBytes() );
  if ( file.fail() )
    itkExceptionMacro(<<"Read failed: Wanted "
                      << GetImageSizeInBytes()
                      << " bytes, but read "
                      << file.gcount() << " bytes. The current state is: "
                      << file.rdstate() );
}

//--------------------------------------------------------------------
bool rtk::HisImageIO::CanWriteFile(const char* FileNameToWrite)
{
  return CanReadFile(FileNameToWrite);
}

//--------------------------------------------------------------------
// Write Image
void rtk::HisImageIO::Write(const void* buffer)
{
  std::ofstream file(m_FileName.c_str(), std::ios::out | std::ios::binary);

  if ( file.fail() )
    itkGenericExceptionMacro(<< "Could not open file (for writing): " << m_FileName);

  m_HeaderSize = HEADER_INFO_SIZE + 32;
  char szHeader[HEADER_INFO_SIZE + 32] = {
    0x00, 0x70, 0x44, 0x00, 0x64, 0x00, 0x64, 0x00, 0x20, 0x00, 0x20, 0x00, 0x01, 0x00, 0x01, 0x00,
    0x00, 0x04, 0x00, 0x04, 0x01, 0x00, 0x03, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x6A, 0x18, 0x41,
    0x04, 0x00, 0x40, 0x5F, 0x48, 0x01, 0x40, 0x00, 0x86, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x08, 0x63, 0x13, 0x00, 0xE8, 0x51, 0x13, 0x00, 0x5C, 0xE7, 0x12, 0x00,
    0xFE, 0x2A, 0x49, 0x5F, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00
    };

  /* Fill into the header the essentials
     The 'iheader' in previous module is fixed to 0x20, and is included in szHeader.
     The 'ulx' and 'uly' are fixed to 0x01, so that 'brx' and 'bry' reflect the dimensions of
     the image.
  */
  const unsigned int ndim = GetNumberOfDimensions();
  if ( (ndim < 2) || (ndim > 3) )
    itkExceptionMacro( <<"Only 2D or 3D support");

  szHeader[16] = (char)(GetDimensions(0) % 256);  // X-size    lsb
  szHeader[17] = (char)(GetDimensions(0) / 256);  // X-size    msb
  szHeader[18] = (char)(GetDimensions(1) % 256);  // Y-size    lsb
  szHeader[19] = (char)(GetDimensions(1) / 256);  // Y-size    msb
  if (ndim == 3) {
    szHeader[20] = (char)(GetDimensions(0) % 256);  // NbFrames    lsb
    szHeader[21] = (char)(GetDimensions(0) / 256);  // NbFrames    msb
    }

  switch (GetComponentType())
    {
    case itk::ImageIOBase::USHORT:
      szHeader[32] = 4;
      break;
    //case AVS_TYPE_INTEGER:
    //  szHeader[32] = 8;
    //  break;
    //case AVS_TYPE_REAL:
    //  szHeader[32] = 16;
    //  break;
    default:
      itkExceptionMacro(<< "Unsupported field type");
      break;
    }

  file.write(szHeader, m_HeaderSize);
  file.write( (const char *)buffer, GetImageSizeInBytes() );
  file.close();
} ////
