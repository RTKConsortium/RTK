/*=========================================================================
 *
 *  Copyright RTK Consortium
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *         https://www.apache.org/licenses/LICENSE-2.0.txt
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

#include <itksys/SystemTools.hxx>
#include <itkMacro.h>
#include <itkMetaDataObject.h>
#include <itkMatrix.h>
#include <itkByteSwapper.h>
#include <itkRawImageIO.h>

//--------------------------------------------------------------------
// Read Image Information
void
rtk::ImagXImageIO::ReadImageInformation()
{
  rtk::ImagXXMLFileReader::Pointer xmlReader;

  xmlReader = rtk::ImagXXMLFileReader::New();
  xmlReader->SetFilename(m_FileName);
  xmlReader->GenerateOutputInformation();

  itk::MetaDataDictionary & dic = *(xmlReader->GetOutputObject());

  using MetaDataDoubleType = itk::MetaDataObject<double>;
  using MetaDataStringType = itk::MetaDataObject<std::string>;
  using MetaDataIntType = itk::MetaDataObject<int>;
  auto * pixelTypeMetaData = dynamic_cast<MetaDataStringType *>(dic["pixelFormat"].GetPointer());
  if (pixelTypeMetaData == nullptr)
    itkExceptionMacro(<< "Missing or invalid metadata \"pixelFormat\".");
  std::string pixelType = pixelTypeMetaData->GetMetaDataObjectValue();
  if (pixelType == "Type_uint8")
    SetComponentType(itk::ImageIOBase::IOComponentEnum::UCHAR);
  if (pixelType == "Type_sint8")
    SetComponentType(itk::ImageIOBase::IOComponentEnum::CHAR);
  if (pixelType == "Type_uint16")
    SetComponentType(itk::ImageIOBase::IOComponentEnum::USHORT);
  if (pixelType == "Type_sint16")
    SetComponentType(itk::ImageIOBase::IOComponentEnum::SHORT);
  if (pixelType == "Type_uint32")
    SetComponentType(itk::ImageIOBase::IOComponentEnum::UINT);
  if (pixelType == "Type_sint32")
    SetComponentType(itk::ImageIOBase::IOComponentEnum::INT);
  if (pixelType == "Type_float")
    SetComponentType(itk::ImageIOBase::IOComponentEnum::FLOAT);

  if (dic["dimensions"].GetPointer() == nullptr)
    SetNumberOfDimensions(3);
  else
  {
    auto * dimensionsMetaData = dynamic_cast<MetaDataIntType *>(dic["dimensions"].GetPointer());
    if (dimensionsMetaData == nullptr)
      itkExceptionMacro(<< "Missing or invalid metadata \"dimensions\".");
    SetNumberOfDimensions(dimensionsMetaData->GetMetaDataObjectValue());
  }

  auto * xMetaData = dynamic_cast<MetaDataIntType *>(dic["x"].GetPointer());
  if (xMetaData == nullptr)
    itkExceptionMacro(<< "Missing or invalid metadata \"x\".");
  SetDimensions(0, xMetaData->GetMetaDataObjectValue());
  auto * spacingXMetaData = dynamic_cast<MetaDataDoubleType *>(dic["spacing_x"].GetPointer());
  if (spacingXMetaData == nullptr)
    itkExceptionMacro(<< "Missing or invalid metadata \"spacing_x\".");
  SetSpacing(0, spacingXMetaData->GetMetaDataObjectValue());
  if (GetNumberOfDimensions() > 1)
  {
    auto * yMetaData = dynamic_cast<MetaDataIntType *>(dic["y"].GetPointer());
    if (yMetaData == nullptr)
      itkExceptionMacro(<< "Missing or invalid metadata \"y\".");
    SetDimensions(1, yMetaData->GetMetaDataObjectValue());
    auto * spacingYMetaData = dynamic_cast<MetaDataDoubleType *>(dic["spacing_y"].GetPointer());
    if (spacingYMetaData == nullptr)
      itkExceptionMacro(<< "Missing or invalid metadata \"spacing_y\".");
    SetSpacing(1, spacingYMetaData->GetMetaDataObjectValue());
  }
  if (GetNumberOfDimensions() > 2)
  {
    auto * zMetaData = dynamic_cast<MetaDataIntType *>(dic["z"].GetPointer());
    if (zMetaData == nullptr)
      itkExceptionMacro(<< "Missing or invalid metadata \"z\".");
    SetDimensions(2, zMetaData->GetMetaDataObjectValue());
    auto * spacingZMetaData = dynamic_cast<MetaDataDoubleType *>(dic["spacing_z"].GetPointer());
    if (spacingZMetaData == nullptr)
      itkExceptionMacro(<< "Missing or invalid metadata \"spacing_z\".");
    SetSpacing(2, spacingZMetaData->GetMetaDataObjectValue());
    if (GetSpacing(2) == 0)
      SetSpacing(2, 1);
  }

  itk::Matrix<double, 4, 4> matrix;
  if (dic["matrixTransform"].GetPointer() == nullptr)
    matrix.SetIdentity();
  else
  {
    auto * matrixTransformMetaData = dynamic_cast<MetaDataStringType *>(dic["matrixTransform"].GetPointer());
    if (matrixTransformMetaData == nullptr)
      itkExceptionMacro(<< "Missing or invalid metadata \"matrixTransform\".");
    std::istringstream iss(matrixTransformMetaData->GetMetaDataObjectValue());
    for (unsigned int j = 0; j < 4; j++)
      for (unsigned int i = 0; i < 4; i++)
        iss >> matrix[j][i];
    matrix /= matrix[3][3];
  }

  std::vector<double> direction;
  for (unsigned int i = 0; i < GetNumberOfDimensions(); i++)
  {
    direction.clear();
    for (unsigned int j = 0; j < GetNumberOfDimensions(); j++)
      direction.push_back(matrix[i][j]);
    SetDirection(i, direction);
    SetOrigin(i, matrix[i][3]);
  }

  auto * byteOrderMetaData = dynamic_cast<MetaDataStringType *>(dic["byteOrder"].GetPointer());
  if (byteOrderMetaData == nullptr)
    itkExceptionMacro(<< "Missing or invalid metadata \"byteOrder\".");
  if (std::string("LSB") == byteOrderMetaData->GetMetaDataObjectValue())
    this->SetByteOrder(IOByteOrderEnum::LittleEndian);
  else
    this->SetByteOrder(IOByteOrderEnum::BigEndian);

  // Prepare raw file name
  m_RawFileName = itksys::SystemTools::GetFilenamePath(m_FileName);
  if (!m_RawFileName.empty())
    m_RawFileName += std::string("/");
  auto * rawFileMetaData = dynamic_cast<MetaDataStringType *>(dic["rawFile"].GetPointer());
  if (rawFileMetaData == nullptr)
    itkExceptionMacro(<< "Missing or invalid metadata \"rawFile\".");
  m_RawFileName += rawFileMetaData->GetMetaDataObjectValue();
} ////

//--------------------------------------------------------------------
// Read Image Information
bool
rtk::ImagXImageIO::CanReadFile(const char * FileNameToRead)
{
  std::string ext = itksys::SystemTools::GetFilenameLastExtension(FileNameToRead);

  if (ext != std::string(".xml"))
    return false;

  std::ifstream is(FileNameToRead);
  if (!is.is_open())
    return false;

  // If the XML file has "<image name=" at the beginning of the first or second
  // line, we assume this is an ImagX file
  std::string line;

  std::getline(is, line);
  if (line.substr(0, 12) == std::string("<image name="))
    return true;

  std::getline(is, line);
  if (line.substr(0, 12) == std::string("<image name="))
    return true;

  return false;
} ////

//--------------------------------------------------------------------
// Read Image Content
void
rtk::ImagXImageIO::Read(void * buffer)
{
  // Adapted from itkRawImageIO
  std::ifstream is(m_RawFileName.c_str(), std::ios::binary);

  if (!is.is_open())
    itkExceptionMacro(<< "Could not open file " << m_RawFileName);

  unsigned long numberOfBytesToBeRead = GetComponentSize();
  for (unsigned int i = 0; i < GetNumberOfDimensions(); i++)
    numberOfBytesToBeRead *= GetDimensions(i);

  if (!this->ReadBufferAsBinary(is, buffer, numberOfBytesToBeRead))
  {
    itkExceptionMacro(<< "Read failed: Wanted " << numberOfBytesToBeRead << " bytes, but read " << is.gcount()
                      << " bytes.");
  }
  itkDebugMacro(<< "Reading Done");

  const auto          componentType = this->GetComponentType();
  const SizeValueType numberOfComponents = this->GetImageSizeInComponents();
  ReadRawBytesAfterSwapping(componentType, buffer, m_ByteOrder, numberOfComponents);
}

//--------------------------------------------------------------------
// Write Image Information
void
rtk::ImagXImageIO::WriteImageInformation(bool itkNotUsed(keepOfStream))
{}

//--------------------------------------------------------------------
// Write Image Information
bool
rtk::ImagXImageIO::CanWriteFile(const char * itkNotUsed(FileNameToWrite))
{
  return false;
}

//--------------------------------------------------------------------
// Write Image
void
rtk::ImagXImageIO::Write(const void * itkNotUsed(buffer))
{} ////
