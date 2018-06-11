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

#include "rtkImagXImageIO.h"
#include "rtkImagXXMLFileReader.h"

// itk include (for itkReadRawBytesAfterSwappingMacro)
#include <itkRawImageIO.h>
#include <itksys/SystemTools.hxx>
#include <itkMetaDataObject.h>
#include <itkMatrix.h>

//--------------------------------------------------------------------
// Read Image Information
void rtk::ImagXImageIO::ReadImageInformation()
{
  rtk::ImagXXMLFileReader::Pointer xmlReader;

  xmlReader = rtk::ImagXXMLFileReader::New();
  xmlReader->SetFilename(m_FileName);
  xmlReader->GenerateOutputInformation();

  itk::MetaDataDictionary &dic = *(xmlReader->GetOutputObject() );

  typedef itk::MetaDataObject< double >      MetaDataDoubleType;
  typedef itk::MetaDataObject< std::string > MetaDataStringType;
  typedef itk::MetaDataObject< int >         MetaDataIntType;

  std::string pixelType = dynamic_cast<MetaDataStringType*>(dic["pixelFormat"].GetPointer() )->GetMetaDataObjectValue();
  if(pixelType=="Type_uint8")
    SetComponentType(itk::ImageIOBase::UCHAR);
  if(pixelType=="Type_sint8")
    SetComponentType(itk::ImageIOBase::CHAR);
  if(pixelType=="Type_uint16")
    SetComponentType(itk::ImageIOBase::USHORT);
  if(pixelType=="Type_sint16")
    SetComponentType(itk::ImageIOBase::SHORT);
  if(pixelType=="Type_uint32")
    SetComponentType(itk::ImageIOBase::UINT);
  if(pixelType=="Type_sint32")
    SetComponentType(itk::ImageIOBase::INT);
  if(pixelType=="Type_float")
    SetComponentType(itk::ImageIOBase::FLOAT);

  if( dic["dimensions"].GetPointer() == ITK_NULLPTR )
    SetNumberOfDimensions(3);
  else
    SetNumberOfDimensions( ( dynamic_cast<MetaDataIntType *>(dic["dimensions"].GetPointer() )->GetMetaDataObjectValue() ) );

  SetDimensions(0, dynamic_cast<MetaDataIntType *>(dic["x"].GetPointer() )->GetMetaDataObjectValue() );
  SetSpacing(0, dynamic_cast<MetaDataDoubleType *>(dic["spacing_x"].GetPointer() )->GetMetaDataObjectValue() );
  if(GetNumberOfDimensions()>1)
    {
    SetDimensions(1, dynamic_cast<MetaDataIntType *>(dic["y"].GetPointer() )->GetMetaDataObjectValue() );
    SetSpacing(1, dynamic_cast<MetaDataDoubleType *>(dic["spacing_y"].GetPointer() )->GetMetaDataObjectValue() );
    }
  if(GetNumberOfDimensions()>2)
    {
    SetDimensions(2, dynamic_cast<MetaDataIntType *>(dic["z"].GetPointer() )->GetMetaDataObjectValue() );
    SetSpacing(2, dynamic_cast<MetaDataDoubleType *>(dic["spacing_z"].GetPointer() )->GetMetaDataObjectValue() );
    if(GetSpacing(2) == 0)
      SetSpacing(2, 1);
    }

  itk::Matrix<double, 4, 4> matrix;
  if(dic["matrixTransform"].GetPointer() == ITK_NULLPTR)
    matrix.SetIdentity();
  else
    {
    std::istringstream iss(
      dynamic_cast<MetaDataStringType*>(dic["matrixTransform"].GetPointer() )->GetMetaDataObjectValue() );
    for(unsigned int j=0; j<4; j++)
      for(unsigned int i=0; i<4; i++)
        iss >> matrix[j][i];
    matrix /= matrix[3][3];
    }

  std::vector<double> direction;
  for(unsigned int i=0; i<GetNumberOfDimensions(); i++)
    {
    direction.clear();
    for(unsigned int j=0; j<GetNumberOfDimensions(); j++)
      direction.push_back(matrix[i][j]);
    SetDirection(i, direction);
    SetOrigin(i, matrix[i][3]);
    }

  if(std::string("LSB") == dynamic_cast<MetaDataStringType*>(dic["byteOrder"].GetPointer() )->GetMetaDataObjectValue() )
    this->SetByteOrder(LittleEndian);
  else
    this->SetByteOrder(BigEndian);

  // Prepare raw file name
  m_RawFileName = itksys::SystemTools::GetFilenamePath(m_FileName);
  if(m_RawFileName != "")
    m_RawFileName += std::string("/");
  m_RawFileName += dynamic_cast<MetaDataStringType*>(dic["rawFile"].GetPointer() )->GetMetaDataObjectValue();
} ////

//--------------------------------------------------------------------
// Read Image Information
bool rtk::ImagXImageIO::CanReadFile(const char* FileNameToRead)
{
  std::string ext = itksys::SystemTools::GetFilenameLastExtension(FileNameToRead);

  if( ext!=std::string(".xml") )
    return false;

  std::ifstream is(FileNameToRead);
  if(!is.is_open() )
    return false;

  // If the XML file has "<image name=" at the beginning of the first or second
  // line, we assume this is an ImagX file
  std::string line;

  std::getline(is, line);
  if(line.substr(0, 12) == std::string("<image name=") )
    return true;

  std::getline(is, line);
  if(line.substr(0, 12) == std::string("<image name=") )
    return true;

  return false;
} ////

//--------------------------------------------------------------------
// Read Image Content
void rtk::ImagXImageIO::Read(void * buffer)
{
  // Adapted from itkRawImageIO
  std::ifstream is(m_RawFileName.c_str(), std::ios::binary);

  if(!is.is_open() )
    itkExceptionMacro(<<"Could not open file " << m_RawFileName);

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
void rtk::ImagXImageIO::WriteImageInformation( bool itkNotUsed(keepOfStream) )
{
}

//--------------------------------------------------------------------
// Write Image Information
bool rtk::ImagXImageIO::CanWriteFile( const char* itkNotUsed(FileNameToWrite) )
{
  return false;
}

//--------------------------------------------------------------------
// Write Image
void rtk::ImagXImageIO::Write( const void * itkNotUsed(buffer) )
{
} ////
