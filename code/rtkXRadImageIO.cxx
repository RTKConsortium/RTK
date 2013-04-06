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

#include "rtkXRadImageIO.h"

#include <itkRawImageIO.h>
#include <itkMetaDataObject.h>

//--------------------------------------------------------------------
// Read Image Information
void rtk::XRadImageIO::ReadImageInformation()
{
  std::ifstream is;
  is.open(m_FileName.c_str());

  if(!is.is_open())
    itkExceptionMacro(<<"Could not open file " << m_FileName);

  SetNumberOfDimensions(3);
  std::string section="";
  while(!is.eof())
    {
    std::string line;
    std::getline(is, line);
    if(line.find('[') != std::string::npos)
      {
      unsigned int pos1 = line.find('[');
      unsigned int pos2 = line.find(']');
      section = line.substr(pos1+1, pos2-pos1-1);
      }
    if(line.find('=') != std::string::npos)
      {
      unsigned int pos       = line.find('=');
      std::string paramName  = line.substr(0,pos);
      std::string paramValue = line.substr(pos+1, line.length()-pos-1);

      if(paramName == std::string("CBCT.DimensionalAttributes.IDim"))
        SetDimensions(0, atoi(paramValue.c_str()));
      else if(paramName == std::string("CBCT.DimensionalAttributes.JDim"))
        SetDimensions(1, atoi(paramValue.c_str()));
      else if(paramName == std::string("CBCT.DimensionalAttributes.KDim"))
        SetDimensions(2, atoi(paramValue.c_str()));
      else if(paramName == std::string("CBCT.DimensionalAttributes.DataSize"))
        {
        if(atoi(paramValue.c_str()) == 3)
          SetComponentType(itk::ImageIOBase::FLOAT);
        if(atoi(paramValue.c_str()) == 6)
          SetComponentType(itk::ImageIOBase::USHORT);
        }
      else if(paramName == std::string("CBCT.DimensionalAttributes.PixelDimension_I_cm"))
        {
        double spacing = 10*atof(paramValue.c_str());
        SetSpacing(0, (spacing==0.)?1.:spacing);
        }
      else if(paramName == std::string("CBCT.DimensionalAttributes.PixelDimension_J_cm"))
        {
        double spacing = 10*atof(paramValue.c_str());
        SetSpacing(1, (spacing==0.)?1.:spacing);
        }
      else if(paramName == std::string("CBCT.DimensionalAttributes.PixelDimension_K_cm"))
        {
        double spacing = 10*atof(paramValue.c_str());
        SetSpacing(2, (spacing==0.)?1.:spacing);
        }
      else
        {
        paramName = section + std::string("_") + paramName;
        itk::EncapsulateMetaData<std::string>(this->GetMetaDataDictionary(),
                                              paramName.c_str(),
                                              paramValue);
        }
      }

    }
} ////

//--------------------------------------------------------------------
// Read Image Information
bool rtk::XRadImageIO::CanReadFile(const char* FileNameToRead)
{
  std::string                  filename(FileNameToRead);
  const std::string::size_type it = filename.find_last_of( "." );
  std::string                  fileExt( filename, it+1, filename.length() );

  if (fileExt != std::string("header") ) return false;
  return true;
} ////

//--------------------------------------------------------------------
// Read Image Content
void rtk::XRadImageIO::Read(void * buffer)
{
  // Adapted from itkRawImageIO
  std::string rawFileName( m_FileName, 0, m_FileName.size()-6);
  rawFileName += "img";

  std::ifstream is(rawFileName.c_str(), std::ios::binary);
  if(!is.is_open() )
    itkExceptionMacro(<<"Could not open file " << rawFileName);

  unsigned long numberOfBytesToBeRead = GetComponentSize();
  for(unsigned int i=0; i<GetNumberOfDimensions(); i++) numberOfBytesToBeRead *= GetDimensions(i);

  if(!this->ReadBufferAsBinary(is, buffer, numberOfBytesToBeRead) ) {
    itkExceptionMacro(<<"Read failed: Wanted "
                      << numberOfBytesToBeRead
                      << " bytes, but read "
                      << is.gcount() << " bytes.");
    }
  itkDebugMacro(<< "Reading Done");

  // Adapted from itkRawImageIO
    {
    using namespace itk;
    // Swap bytes if necessary
    if itkReadRawBytesAfterSwappingMacro( unsigned short, USHORT )
    else if itkReadRawBytesAfterSwappingMacro( short, SHORT )
    else if itkReadRawBytesAfterSwappingMacro( char, CHAR )
    else if itkReadRawBytesAfterSwappingMacro( unsigned char, UCHAR )
    else if itkReadRawBytesAfterSwappingMacro( unsigned int, UINT )
    else if itkReadRawBytesAfterSwappingMacro( int, INT )
    else if itkReadRawBytesAfterSwappingMacro( unsigned int, ULONG )
    else if itkReadRawBytesAfterSwappingMacro( int, LONG )
    else if itkReadRawBytesAfterSwappingMacro( float, FLOAT )
    else if itkReadRawBytesAfterSwappingMacro( double, DOUBLE );
    }
}

//--------------------------------------------------------------------
// Write Image Information
void rtk::XRadImageIO::WriteImageInformation(bool itkNotUsed(keepOfStream))
{
}

//--------------------------------------------------------------------
// Write Image Information
bool rtk::XRadImageIO::CanWriteFile(const char* itkNotUsed(FileNameToWrite))
{
  return false;
}

//--------------------------------------------------------------------
// Write Image
void rtk::XRadImageIO::Write(const void * itkNotUsed(buffer))
{
} ////
